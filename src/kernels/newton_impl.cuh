// Kernels and launch definitions for fixed-boundary Newton corrections.
// Included once per scalar type by newton_double.cu / newton_float.cu.
#ifndef CUMES_SRC_KERNELS_NEWTON_IMPL_CUH_
#define CUMES_SRC_KERNELS_NEWTON_IMPL_CUH_

#include "cumes/core/checked_size.hpp"
#include "cumes/numerics/newton_correction.hpp"
#include "cumes/runtime/cuda_status.hpp"
#include "cumes/solver/dump_windows.hpp"

#include <cfloat>
#include <climits>
#include <cmath>

namespace cumes {

namespace correction_detail {
template <class T>
__global__ void pack_kernel(const T* d_residual,
                            T* d_q,
                            int ns,
                            int mnmax,
                            int ntor,
                            bool fix_m1) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = ns * mnmax;
    if (i >= 6 * count) return;
    const int component = i / count, mode = (i % count) / ns, j = i % ns;
    const int m = mode / (ntor + 1), n = mode % (ntor + 1);
    d_q[i] = active(component, m, n, j, ns, fix_m1) ? d_residual[i] : T(0);
}

template <class T, bool Trial>
__global__ void physical_kernel(const T* d_base,
                                const T* d_q,
                                T* d_result,
                                int ns,
                                int mnmax,
                                int ntor,
                                bool fix_m1,
                                T scale,
                                const T* d_scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = ns * mnmax;
    if (i >= count) return;
    const int mode = i / ns, j = i % ns;
    const int m = mode / (ntor + 1), n = mode % (ntor + 1);
    const T basis = (m == 0 ? T(1) : sqrt(T(2))) * (n == 0 ? T(1) : sqrt(T(2)));
    T q[6];
    for (int c = 0; c < 6; ++c)
        q[c] = active(c, m, n, j, ns, fix_m1) ? d_q[c * count + i] : T(0);
    if (m == 1) {
        // Production descent undoes this pair without a 1/sqrt(2) factor.
        // A fixed mixed Zcs coordinate therefore permits equal PHYSICAL
        // Rss/Zcs increments; masking physical Zcs would freeze the wrong DOF.
        const T rss = q[3], zcs = q[4];
        q[3] = rss + zcs;
        q[4] = rss - zcs;
    }
    if constexpr (Trial) {
        if (d_scale) scale *= *d_scale;
    }
    for (int c = 0; c < 6; ++c) {
        const int index = c * count + i;
        if constexpr (Trial) {
            // Reproduce descent's increment order, including the basis last.
            // Direct copy for a zero increment preserves signed zeros and all
            // fixed boundary, dependent-axis and null-parity baseline values.
            d_result[index] = q[c] == T(0) || scale == T(0)
                                  ? d_base[index]
                                  : d_base[index] + (scale * q[c]) * basis;
        } else {
            d_result[index] = q[c] * basis;
        }
    }
}

}  // namespace correction_detail

namespace gmres_detail {

template <class T, bool MAXIMUM = false>
__device__ T reduce(T value) {
    __shared__ T values[256];
    values[threadIdx.x] = value;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (threadIdx.x < stride) {
            if constexpr (MAXIMUM)
                values[threadIdx.x] =
                    fmax(values[threadIdx.x], values[threadIdx.x + stride]);
            else
                values[threadIdx.x] += values[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const T result = values[0];
    __syncthreads();  // permit immediate reuse by the next reduction
    return result;
}

// Scaling avoids classifying a small nonzero RHS as zero when sum(x*x)
// underflows. Norm-based stopping does not consume the squared telemetry.
template <class T>
__device__ T norm(int size, const T* d_values) {
    T scale = T(0);
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        const T value = d_values[i];
        scale = isfinite(value) ? fmax(scale, fabs(value)) : T(INFINITY);
    }
    scale = reduce<T, true>(scale);
    if (scale == T(0) || !isfinite(scale)) return scale;
    T sum = T(0);
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        const T value = d_values[i] / scale;
        sum += value * value;
    }
    return scale * sqrt(reduce(sum));
}

template <class T>
__global__ void initialize(int size,
                           const T* d_b,
                           T* d_x,
                           T* d_residual,
                           GmresControl<T>* d_control,
                           T tolerance) {
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        d_x[i] = T(0);
        d_residual[i] = d_b[i];
    }
    const T rhs_norm = norm(size, d_b);
    if (threadIdx.x == 0) {
        *d_control = GmresControl<T>{};
        d_control->rhs_norm = d_control->residual_norm = rhs_norm;
        d_control->norm_squared = rhs_norm * rhs_norm;
        d_control->target_norm = rhs_norm * tolerance;
        d_control->target_squared =
            d_control->target_norm * d_control->target_norm;
        d_control->breakdown = isfinite(rhs_norm) ? 0 : 2;
        d_control->converged = rhs_norm == T(0);
        d_control->active = isfinite(rhs_norm) && rhs_norm > T(0);
    }
}

template <class T>
__global__ void begin_cycle(int size,
                            int max_basis,
                            const T* d_residual,
                            T* d_basis,
                            T* d_hessenberg,
                            T* d_g,
                            T* d_y,
                            GmresControl<T>* d_control) {
    for (int i = threadIdx.x; i < (max_basis + 1) * max_basis; i += blockDim.x)
        d_hessenberg[i] = T(0);
    for (int i = threadIdx.x; i <= max_basis; i += blockDim.x) d_g[i] = T(0);
    for (int i = threadIdx.x; i < max_basis; i += blockDim.x) d_y[i] = T(0);
    if (threadIdx.x == 0) {
        d_control->cycle_started = d_control->active;
        d_control->cycle_active = d_control->active;
        d_control->cycle_steps = 0;
        d_control->update_ready = 0;
        d_g[0] = d_control->active ? d_control->residual_norm : T(0);
    }
    __syncthreads();
    if (!d_control->active) return;
    const T beta = d_control->residual_norm;
    for (int i = threadIdx.x; i < size; i += blockDim.x)
        d_basis[i] = d_residual[i] / beta;
}

// CGS2: one CTA per existing basis column computes its projection, then a
// full-vector kernel subtracts all projections. Repeat both kernels to recover
// orthogonality after cancellation without serializing the vector on one SM.
template <class T>
__global__ void project(int size,
                        int max_basis,
                        int column,
                        const T* d_basis,
                        const T* d_work,
                        T* d_projection,
                        T* d_hessenberg,
                        const GmresControl<T>* d_control) {
    if (!d_control->cycle_active) return;
    const int k = blockIdx.x;
    const T* d_v = d_basis + static_cast<std::size_t>(k) * size;
    T dot = T(0);
    for (int i = threadIdx.x; i < size; i += blockDim.x)
        dot += d_v[i] * d_work[i];
    dot = reduce(dot);
    if (threadIdx.x == 0) {
        d_projection[k] = dot;
        d_hessenberg[column * (max_basis + 1) + k] += dot;
    }
}

template <class T>
__global__ void subtract_projection(int size,
                                    int column,
                                    const T* d_basis,
                                    const T* d_projection,
                                    T* d_work,
                                    const GmresControl<T>* d_control) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size || !d_control->cycle_active) return;
    T value = T(0);
    for (int k = 0; k <= column; ++k)
        value +=
            d_basis[static_cast<std::size_t>(k) * size + i] * d_projection[k];
    d_work[i] -= value;
}

template <class T>
__global__ void arnoldi(int size,
                        int max_basis,
                        int column,
                        T* d_basis,
                        T* d_work,
                        T* d_hessenberg,
                        T* d_cosine,
                        T* d_sine,
                        T* d_g,
                        GmresControl<T>* d_control) {
    if (!d_control->cycle_active) return;
    const int leading = max_basis + 1;
    T* d_h = d_hessenberg + column * leading;
    const T next_norm = norm(size, d_work);
    if (!isfinite(next_norm)) {
        if (threadIdx.x == 0) {
            d_control->breakdown = 2;
            d_control->cycle_active = 0;
        }
        return;
    }
    if (next_norm > T(0)) {
        T* d_next = d_basis + static_cast<std::size_t>(column + 1) * size;
        for (int i = threadIdx.x; i < size; i += blockDim.x)
            d_next[i] = d_work[i] / next_norm;
    }
    if (threadIdx.x == 0) {
        T reference_norm = next_norm;
        for (int i = 0; i <= column; ++i) {
            if (!isfinite(d_h[i])) {
                d_control->breakdown = 2;
                d_control->cycle_active = 0;
                return;
            }
            reference_norm = fmax(reference_norm, fabs(d_h[i]));
        }
        const T epsilon =
            sizeof(T) == sizeof(float) ? T(FLT_EPSILON) : T(DBL_EPSILON);
        const bool happy = next_norm <= epsilon * reference_norm;
        d_h[column + 1] = next_norm;
        for (int i = 0; i < column; ++i) {
            const T value = d_cosine[i] * d_h[i] + d_sine[i] * d_h[i + 1];
            d_h[i + 1] = -d_sine[i] * d_h[i] + d_cosine[i] * d_h[i + 1];
            d_h[i] = value;
        }
        const T a = d_h[column], b = d_h[column + 1];
        const T scale = fmax(fabs(a), fabs(b));
        const T radius = scale > T(0) ? scale * sqrt((a / scale) * (a / scale) +
                                                     (b / scale) * (b / scale))
                                      : T(0);
        if (!isfinite(radius) || radius == T(0)) {
            d_control->breakdown = isfinite(radius) ? 1 : 2;
            d_control->cycle_active = 0;
            return;
        }
        const T cosine = a / radius, sine = b / radius;
        d_cosine[column] = cosine;
        d_sine[column] = sine;
        d_h[column] = radius;
        d_h[column + 1] = T(0);
        d_g[column + 1] = -sine * d_g[column];
        d_g[column] *= cosine;
        ++d_control->steps;
        d_control->cycle_steps = column + 1;
        if (happy || fabs(d_g[column + 1]) <= d_control->target_norm)
            d_control->cycle_active = 0;
    }
}

template <class T>
__global__ void backsolve(int max_basis,
                          const T* d_hessenberg,
                          const T* d_g,
                          T* d_y,
                          GmresControl<T>* d_control) {
    const int count = d_control->cycle_steps;
    if (!d_control->cycle_started || count == 0) return;
    for (int i = count - 1; i >= 0; --i) {
        T value = d_g[i];
        for (int j = i + 1; j < count; ++j)
            value -= d_hessenberg[j * (max_basis + 1) + i] * d_y[j];
        const T diagonal = d_hessenberg[i * (max_basis + 1) + i];
        if (diagonal == T(0)) {
            d_control->breakdown = 1;
            return;
        }
        d_y[i] = value / diagonal;
        if (!isfinite(d_y[i])) {
            d_control->breakdown = 2;
            return;
        }
    }
    d_control->update_ready = 1;
}

template <class T>
__global__ void update(int size,
                       const T* d_basis,
                       const T* d_y,
                       T* d_x,
                       const GmresControl<T>* d_control) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size || !d_control->update_ready) return;
    T value = T(0);
    for (int j = 0; j < d_control->cycle_steps; ++j)
        value += d_basis[static_cast<std::size_t>(j) * size + i] * d_y[j];
    d_x[i] += value;
}

template <class T>
__global__ void residual(int size, const T* d_b, const T* d_ax, T* d_r) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) d_r[i] = d_b[i] - d_ax[i];
}

template <class T>
__global__ void finish_cycle(int size,
                             const T* d_residual,
                             GmresControl<T>* d_control) {
    const T residual_norm = norm(size, d_residual);
    if (threadIdx.x == 0) {
        d_control->residual_norm = residual_norm;
        d_control->norm_squared = residual_norm * residual_norm;
        d_control->cycles += d_control->cycle_started;
        if (!isfinite(residual_norm)) d_control->breakdown = 2;
        d_control->converged = d_control->breakdown == 0 &&
                               residual_norm <= d_control->target_norm;
        d_control->active = d_control->breakdown == 0 && !d_control->converged;
    }
}

}  // namespace gmres_detail

namespace newton_detail {

template <class T>
__global__ void difference_scale(int size,
                                 const T* d_physical,
                                 T step,
                                 T* d_scale,
                                 int* d_valid) {
    T maximum = T(0);
    int valid = 1;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        if (!isfinite(d_physical[i])) valid = 0;
        maximum = fmax(maximum, fabs(d_physical[i]));
    }
    __shared__ T maxima[256];
    __shared__ int validity[256];
    maxima[threadIdx.x] = maximum;
    validity[threadIdx.x] = valid;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (threadIdx.x < stride) {
            maxima[threadIdx.x] =
                fmax(maxima[threadIdx.x], maxima[threadIdx.x + stride]);
            validity[threadIdx.x] &= validity[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        d_scale[0] = maxima[0] > T(0) ? step / maxima[0] : T(0);
        d_valid[0] = validity[0] && isfinite(d_scale[0]);
    }
}

static __global__ void check_probe(const cumes::ControlStatus* d_status,
                                   int* d_valid) {
    if (threadIdx.x == 0)
        d_valid[0] &= d_status->jacobian_valid &&
                      !d_status->invariant_nonfinite &&
                      d_status->preconditioned_evaluated;
}

template <class T>
__global__ void difference(int size,
                           const T* d_positive,
                           const T* d_negative,
                           const T* d_scale,
                           const int* d_valid,
                           T* d_result,
                           bool central) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    // A=-J: the force has the sign of the descent direction.
    d_result[i] = !d_valid[0] ? T(NAN)
                  : d_scale[0] == T(0)
                      ? T(0)
                      : (d_negative[i] - d_positive[i]) /
                            ((central ? T(2) : T(1)) * d_scale[0]);
}

}  // namespace newton_detail

template <class T>
void CorrectionCoordinates<T>::enqueue_pack(const T* d_residual,
                                            T* d_q,
                                            cudaStream_t stream) const {
    correction_detail::pack_kernel<<<(size() + 255) / 256, 256, 0, stream>>>(
        d_residual, d_q, ns_, mnmax_, ntor_, fix_m1_);
    cumes::check_cuda(cudaGetLastError(), "correction coordinate mask");
}

template <class T>
void CorrectionCoordinates<T>::enqueue_physical(const T* d_q,
                                                T* d_physical,
                                                cudaStream_t stream) const {
    correction_detail::physical_kernel<T, false>
        <<<(ns_ * mnmax_ + 255) / 256, 256, 0, stream>>>(
            nullptr, d_q, d_physical, ns_, mnmax_, ntor_, fix_m1_, T(1),
            nullptr);
    cumes::check_cuda(cudaGetLastError(), "correction physical direction");
}

template <class T>
void CorrectionCoordinates<T>::enqueue_trial(const T* d_base,
                                             const T* d_q,
                                             T* d_trial,
                                             T scale,
                                             cudaStream_t stream,
                                             const T* d_scale) const {
    correction_detail::physical_kernel<T, true>
        <<<(ns_ * mnmax_ + 255) / 256, 256, 0, stream>>>(
            d_base, d_q, d_trial, ns_, mnmax_, ntor_, fix_m1_, scale, d_scale);
    cumes::check_cuda(cudaGetLastError(), "correction physical trial");
}

template <class T>
DeviceGmres<T>::DeviceGmres(int size, int max_basis)
    : size_(check_size(size)),
      max_basis_(check_basis(max_basis)),
      d_basis_(basis_count(size_, max_basis_)),
      d_work_(size_),
      d_residual_(size_),
      d_hessenberg_((max_basis_ + 1) * max_basis_),
      d_projection_(max_basis_),
      d_cosine_(max_basis_),
      d_sine_(max_basis_),
      d_g_(max_basis_ + 1),
      d_y_(max_basis_),
      d_control_(1) {}

template <class T>
int DeviceGmres<T>::check_size(int size) {
    if (size < 1 || size > INT_MAX - 255)
        throw cumes::CumesError("GMRES: size exceeds positive int indexing");
    return size;
}

template <class T>
int DeviceGmres<T>::check_basis(int basis) {
    if (basis < 1 || basis > 256)
        throw cumes::CumesError("GMRES: basis must be in [1,256]");
    return basis;
}

template <class T>
std::size_t DeviceGmres<T>::basis_count(int size, int basis) {
    const auto count = cumes::checked_mul(static_cast<std::size_t>(size),
                                          static_cast<std::size_t>(basis + 1));
    if (!count) throw cumes::CumesError("GMRES: basis size overflow");
    return *count;
}

template <class T>
void DeviceGmres<T>::enqueue_initialize(const T* d_b,
                                        T* d_x,
                                        T tolerance,
                                        cudaStream_t stream) {
    gmres_detail::initialize<<<1, 256, 0, stream>>>(
        size_, d_b, d_x, d_residual_.data(), d_control_.data(), tolerance);
}

template <class T>
void DeviceGmres<T>::enqueue_begin_cycle(cudaStream_t stream) {
    // Future Arnoldi columns are valid zero inputs even when a
    // converged/broken recurrence no longer generates new vectors.
    d_basis_.zero_async(stream);
    gmres_detail::begin_cycle<<<1, 256, 0, stream>>>(
        size_, max_basis_, d_residual_.data(), d_basis_.data(),
        d_hessenberg_.data(), d_g_.data(), d_y_.data(), d_control_.data());
}

template <class T>
void DeviceGmres<T>::enqueue_arnoldi(int j, cudaStream_t stream) {
    const int blocks = (size_ - 1) / 256 + 1;
    for (int sweep = 0; sweep < 2; ++sweep) {
        gmres_detail::project<<<j + 1, 256, 0, stream>>>(
            size_, max_basis_, j, d_basis_.data(), d_work_.data(),
            d_projection_.data(), d_hessenberg_.data(), d_control_.data());
        gmres_detail::subtract_projection<<<blocks, 256, 0, stream>>>(
            size_, j, d_basis_.data(), d_projection_.data(), d_work_.data(),
            d_control_.data());
    }
    gmres_detail::arnoldi<<<1, 256, 0, stream>>>(
        size_, max_basis_, j, d_basis_.data(), d_work_.data(),
        d_hessenberg_.data(), d_cosine_.data(), d_sine_.data(), d_g_.data(),
        d_control_.data());
}

template <class T>
void DeviceGmres<T>::enqueue_update(T* d_x, cudaStream_t stream) {
    const int blocks = (size_ - 1) / 256 + 1;
    gmres_detail::backsolve<<<1, 1, 0, stream>>>(
        max_basis_, d_hessenberg_.data(), d_g_.data(), d_y_.data(),
        d_control_.data());
    gmres_detail::update<<<blocks, 256, 0, stream>>>(
        size_, d_basis_.data(), d_y_.data(), d_x, d_control_.data());
}

template <class T>
void DeviceGmres<T>::enqueue_finish_cycle(const T* d_b, cudaStream_t stream) {
    const int blocks = (size_ - 1) / 256 + 1;
    gmres_detail::residual<<<blocks, 256, 0, stream>>>(
        size_, d_b, d_work_.data(), d_residual_.data());
    gmres_detail::finish_cycle<<<1, 256, 0, stream>>>(size_, d_residual_.data(),
                                                      d_control_.data());
}

template <class T>
NewtonCorrection<T>::NewtonCorrection(
    const DeviceParams<T>& p,
    cumes::EquilibriumOperator<T>& equilibrium,
    cumes::SpectralStorage<T>& storage,
    int capacity,
    DifferenceScheme scheme)
    : p_(p),
      equilibrium_(equilibrium),
      storage_(storage),
      coordinates_(p, true),
      scheme_(scheme),
      krylov_(6 * p.ns * p.mnmax, capacity),
      d_base_(size()),
      d_rhs_(size()),
      d_delta_(size()),
      d_physical_(size()),
      d_positive_(size()),
      d_negative_(size()),
      d_scale_(1),
      d_valid_(1) {}

template <class T>
void NewtonCorrection<T>::prepare(const cumes::EvaluationSchedule& schedule,
                                  double norm_rz,
                                  double norm_l,
                                  cudaStream_t stream) {
    schedule_ = schedule;
    schedule_.refresh_preconditioner = false;
    schedule_.reset_constraint_reference = false;
    schedule_.decay_rcon0_zcon0 = false;
    schedule_.update_iota_chi = p_.ncurr == 1;
    if (schedule.run_vacuum_block || schedule.apply_vacuum_edge_force)
        throw cumes::CumesError("Newton probe requires fixed boundary");
    norm_rz_ = norm_rz;
    norm_l_ = norm_l;
    coordinates_.set_fix_m1(schedule.zero_z_force_m1);
    d_base_.copy_from_async(storage_.state_buffer(), stream);
    coordinates_.enqueue_pack(equilibrium_.residual_const().data(),
                              d_rhs_.data(), stream);
}

template <class T>
void NewtonCorrection<T>::enqueue_jvp(const T* d_direction,
                                      T* d_result,
                                      T step,
                                      cudaStream_t stream) {
    coordinates_.enqueue_physical(d_direction, d_physical_.data(), stream);
    newton_detail::difference_scale<<<1, 256, 0, stream>>>(
        size(), d_physical_.data(), step, d_scale_.data(), d_valid_.data());
    coordinates_.enqueue_trial(d_base_.data(), d_direction,
                               storage_.state_slab(), T(1), stream,
                               d_scale_.data());
    evaluate(stream);
    newton_detail::check_probe<<<1, 1, 0, stream>>>(
        &equilibrium_.control_device()->status, d_valid_.data());
    coordinates_.enqueue_pack(equilibrium_.residual_const().data(),
                              d_positive_.data(), stream);
    const bool central = scheme_ == DifferenceScheme::CENTRAL;
    if (central) {
        coordinates_.enqueue_trial(d_base_.data(), d_direction,
                                   storage_.state_slab(), T(-1), stream,
                                   d_scale_.data());
        evaluate(stream);
        newton_detail::check_probe<<<1, 1, 0, stream>>>(
            &equilibrium_.control_device()->status, d_valid_.data());
        coordinates_.enqueue_pack(equilibrium_.residual_const().data(),
                                  d_negative_.data(), stream);
    }
    newton_detail::difference<<<(size() + 255) / 256, 256, 0, stream>>>(
        size(), d_positive_.data(),
        central ? d_negative_.data() : d_rhs_.data(), d_scale_.data(),
        d_valid_.data(), d_result, central);
    storage_.state_buffer().copy_from_async(d_base_, stream);
    cumes::check_cuda(cudaGetLastError(), "full Newton JVP");
}

template <class T>
void NewtonCorrection<T>::enqueue_solve(int iterations,
                                        T tolerance,
                                        T step,
                                        cudaStream_t stream) {
    auto apply = [&](const T* d_input, T* d_output, cudaStream_t work) {
        enqueue_jvp(d_input, d_output, step, work);
    };
    krylov_.enqueue(apply, d_rhs_.data(), d_delta_.data(), iterations,
                    tolerance, stream);
}

template <class T>
void NewtonCorrection<T>::enqueue_trial(T scale, cudaStream_t stream) {
    coordinates_.enqueue_trial(d_base_.data(), d_delta_.data(),
                               storage_.state_slab(), scale, stream);
    evaluate(stream);
}

template <class T>
void NewtonCorrection<T>::enqueue_restore(cudaStream_t stream) {
    storage_.state_buffer().copy_from_async(d_base_, stream);
    evaluate(stream);
}

template <class T>
void NewtonCorrection<T>::evaluate(cudaStream_t stream) {
    // Probes and line-search trials are not outer solver observations.
    const ScopedDumpEnvironment dump_environment(false);
    equilibrium_.enqueue(1000, 1000, schedule_, stream, norm_rz_, norm_l_);
}

}  // namespace cumes
#endif  // CUMES_SRC_KERNELS_NEWTON_IMPL_CUH_
