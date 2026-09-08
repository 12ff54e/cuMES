// Experimental restarted GMRES. Arnoldi, reorthogonalization, Givens QR and
// residual replacement stay on device; the host submits a fixed work budget.
#ifndef CUMES_BENCHMARKS_DEVICE_GMRES_CUH_
#define CUMES_BENCHMARKS_DEVICE_GMRES_CUH_

#include "cumes/core/checked_size.hpp"
#include "cumes/runtime/device_buffer.cuh"

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>

namespace block_bench {

template <class T>
struct GmresControl {
    using val_type = T;
    T residual_norm = T(0), norm_squared = T(0), target_squared = T(0);
    T rhs_norm = T(0), target_norm = T(0);
    int steps = 0, cycles = 0;
    int active = 0, converged = 0;
    // 0: no breakdown; 1: singular projected system; 2: nonfinite arithmetic.
    int breakdown = 0;
    int cycle_active = 0, cycle_steps = 0, cycle_started = 0;
    int update_ready = 0;
    int reserved =
        0;  // keep the double record free of uninitialized tail padding
};

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

template <class T>
class DeviceGmres {
   public:
    using val_type = T;

    explicit DeviceGmres(int size, int max_basis = 32)
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

    // apply(d_input,d_output,stream) is a linear map without a host fence.
    // b and x must be nonoverlapping size-element vectors. x starts at zero.
    // The fixed submission contains iterations Arnoldi maps and one true
    // residual map per restart cycle, including inactive maps after stopping.
    // Check converged/breakdown and residual_norm, not active alone. All maps
    // and this object's lifetime must use the same ordered compute stream.
    template <class Apply>
    void enqueue(Apply&& apply,
                 const T* d_b,
                 T* d_x,
                 int iterations,
                 T tolerance,
                 cudaStream_t stream) {
        if (!d_b || !d_x || d_b == d_x || iterations < 1 ||
            !(tolerance > T(0)) || !std::isfinite(tolerance) ||
            tolerance >= T(1))
            throw cumes::CumesError(
                "GMRES: invalid vectors, budget or tolerance");
        const int blocks = (size_ - 1) / 256 + 1;
        auto* d_control = d_control_.data();
        gmres_detail::initialize<<<1, 256, 0, stream>>>(
            size_, d_b, d_x, d_residual_.data(), d_control, tolerance);
        for (int completed = 0; completed < iterations;) {
            const int count = std::min(max_basis_, iterations - completed);
            // Future Arnoldi columns are valid zero inputs even when a
            // converged/broken recurrence no longer generates new vectors.
            d_basis_.zero_async(stream);
            gmres_detail::begin_cycle<<<1, 256, 0, stream>>>(
                size_, max_basis_, d_residual_.data(), d_basis_.data(),
                d_hessenberg_.data(), d_g_.data(), d_y_.data(), d_control);
            for (int j = 0; j < count; ++j) {
                apply(d_basis_.data() + static_cast<std::size_t>(j) * size_,
                      d_work_.data(), stream);
                for (int sweep = 0; sweep < 2; ++sweep) {
                    gmres_detail::project<<<j + 1, 256, 0, stream>>>(
                        size_, max_basis_, j, d_basis_.data(), d_work_.data(),
                        d_projection_.data(), d_hessenberg_.data(), d_control);
                    gmres_detail::
                        subtract_projection<<<blocks, 256, 0, stream>>>(
                            size_, j, d_basis_.data(), d_projection_.data(),
                            d_work_.data(), d_control);
                }
                gmres_detail::arnoldi<<<1, 256, 0, stream>>>(
                    size_, max_basis_, j, d_basis_.data(), d_work_.data(),
                    d_hessenberg_.data(), d_cosine_.data(), d_sine_.data(),
                    d_g_.data(), d_control);
            }
            gmres_detail::backsolve<<<1, 1, 0, stream>>>(
                max_basis_, d_hessenberg_.data(), d_g_.data(), d_y_.data(),
                d_control);
            gmres_detail::update<<<blocks, 256, 0, stream>>>(
                size_, d_basis_.data(), d_y_.data(), d_x, d_control);
            apply(d_x, d_work_.data(), stream);
            gmres_detail::residual<<<blocks, 256, 0, stream>>>(
                size_, d_b, d_work_.data(), d_residual_.data());
            gmres_detail::finish_cycle<<<1, 256, 0, stream>>>(
                size_, d_residual_.data(), d_control);
            completed += count;
        }
        cumes::check_cuda(cudaGetLastError(), "experimental GMRES");
    }

    const GmresControl<T>* control_device() const { return d_control_.data(); }

   private:
    static int check_size(int size) {
        if (size < 1 || size > INT_MAX - 255)
            throw cumes::CumesError(
                "GMRES: size exceeds positive int indexing");
        return size;
    }

    static int check_basis(int basis) {
        if (basis < 1 || basis > 256)
            throw cumes::CumesError("GMRES: basis must be in [1,256]");
        return basis;
    }

    static std::size_t basis_count(int size, int basis) {
        const auto count =
            cumes::checked_mul(static_cast<std::size_t>(size),
                               static_cast<std::size_t>(basis + 1));
        if (!count) throw cumes::CumesError("GMRES: basis size overflow");
        return *count;
    }

    int size_, max_basis_;
    cumes::DeviceBuffer<T> d_basis_, d_work_, d_residual_, d_hessenberg_;
    cumes::DeviceBuffer<T> d_projection_, d_cosine_, d_sine_, d_g_, d_y_;
    cumes::DeviceBuffer<GmresControl<T>> d_control_;
};

}  // namespace block_bench
#endif  // CUMES_BENCHMARKS_DEVICE_GMRES_CUH_
