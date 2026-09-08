// Experimental two-level FAS correction in physical coefficient coordinates.
#ifndef CUMES_BENCHMARKS_COARSE_CORRECTION_CUH_
#define CUMES_BENCHMARKS_COARSE_CORRECTION_CUH_

#include "correction_coordinates.cuh"
#include "cumes/physics/profiles.hpp"
#include "cumes/solver/equilibrium_operator.hpp"
#include "cumes/state/mode_table.cuh"
#include "cumes/transforms/axisymmetric_operator.hpp"

#include <memory>

namespace block_bench {
namespace coarse_detail {

template <class T>
class RealSpace {
   public:
    using val_type = T;
    explicit RealSpace(const DeviceParams<T>& p)
        : value(real_space_create<T>(p, std::nullopt)) {}
    ~RealSpace() { real_space_free(value); }
    cumes::RealSpaceStorage<T> value;
};

class ModeTable {
   public:
    template <class T>
    explicit ModeTable(const DeviceParams<T>& p)
        : value(cumes::mode_table_create<T>(p, std::nullopt)) {}
    ~ModeTable() { cumes::mode_table_free(value); }
    cumes::DeviceModeTable value;
};

// Unlike the mixed-coordinate mask, the physical fixed-m1 space permits
// equal Rss and Zcs increments. The coordinate map already enforces equality.
__device__ inline bool physical_active(int c, int m, int n, int j, int ns) {
    return correction_detail::active(c, m, n, j, ns, false);
}

template <class T>
__device__ T regular_value(const T* d_x, int j, int ns, bool odd) {
    if (!odd) return d_x[j];
    if (j == 0)
        return T(2) * d_x[1] * sqrt(T(ns - 1)) -
               d_x[2] * sqrt(T(ns - 1) / T(2));
    return d_x[j] * sqrt(T(ns - 1) / T(j));
}

// Radial interpolation in the same odd-m regularized space as production
// prolongation. State endpoints are copied; correction endpoints are masked.
template <class T, bool Correction>
__global__ void transfer_kernel(const T* d_input,
                                T* d_output,
                                int ns_old,
                                int ns_new,
                                int mnmax,
                                int ntor) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 6 * mnmax * ns_new) return;
    const int profile = i / ns_new, j = i % ns_new;
    const int c = profile / mnmax, mode = profile % mnmax;
    const int m = mode / (ntor + 1), n = mode % (ntor + 1);
    if constexpr (Correction) {
        if (!physical_active(c, m, n, j, ns_new)) {
            d_output[i] = T(0);
            return;
        }
    }
    const T* d_x = d_input + profile * ns_old;
    if (j == ns_new - 1) {
        d_output[i] = d_x[ns_old - 1];
        return;
    }
    if ((m & 1) && j == 0) {
        d_output[i] = T(0);
        return;
    }
    const int lo = j * (ns_old - 1) / (ns_new - 1);
    const int hi = min(lo + 1, ns_old - 1);
    const T weight = T(j) * T(ns_old - 1) / T(ns_new - 1) - T(lo);
    const bool odd = (m & 1) != 0;
    const T left = regular_value(d_x, lo, ns_old, odd);
    const T right = regular_value(d_x, hi, ns_old, odd);
    T value = (T(1) - weight) * left + weight * right;
    if (odd) value *= sqrt(T(j) / T(ns_new - 1));
    d_output[i] = value;
}

template <class T>
__global__ void status_kernel(const cumes::DeviceControlRecord<T>* d_record,
                              int* d_failed,
                              int pass) {
    if (!*d_failed && (!d_record->status.jacobian_valid ||
                       d_record->status.invariant_nonfinite ||
                       !d_record->status.preconditioned_evaluated))
        *d_failed = pass + 1;
}

template <class T>
__global__ void tau_kernel(const T* d_g0,
                           const T* d_restricted,
                           T* d_tau,
                           int size) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) d_tau[i] = d_g0[i] - d_restricted[i];
}

template <class T>
__global__ void smooth_kernel(T* d_state,
                              T* d_velocity,
                              const T* d_g,
                              const T* d_g0,
                              const T* d_restricted,
                              const int* d_failed,
                              int size,
                              T delta) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size || *d_failed) return;
    // G(y)-tau = (G(y)-G(x0))+R G_h. This ordering makes a zero fine
    // residual an EXACT fixed point, despite the coarse discretization defect.
    const T shifted = (d_g[i] - d_g0[i]) + d_restricted[i];
    const T velocity = T(0.8) * (T(0.75) * d_velocity[i] + delta * shifted);
    d_velocity[i] = velocity;
    if (velocity != T(0)) d_state[i] += delta * velocity;
}

template <class T>
__global__ void difference_kernel(const T* d_state,
                                  const T* d_base,
                                  T* d_error,
                                  const int* d_failed,
                                  int size,
                                  int ns,
                                  int mnmax,
                                  int ntor) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    if (*d_failed) {
        d_error[i] = T(0);
        return;
    }
    const int count = ns * mnmax, component = i / count;
    const int within = i % count, m = (within / ns) / (ntor + 1);
    if (m == 1 && (component == 3 || component == 4)) {
        // Subtracting two large physical states can round the common m1
        // increment differently in Rss and Zcs. Project the ERROR back into
        // their common-increment space before radial prolongation.
        const T dr = d_state[3 * count + within] - d_base[3 * count + within];
        const T dz = d_state[4 * count + within] - d_base[4 * count + within];
        d_error[i] = T(0.5) * (dr + dz);
    } else
        d_error[i] = d_state[i] - d_base[i];
}

template <class T>
__global__ void trial_kernel(const T* d_base,
                             const T* d_error,
                             T* d_trial,
                             int size,
                             T scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        d_trial[i] = d_error[i] == T(0) || scale == T(0)
                         ? d_base[i]
                         : d_base[i] + scale * d_error[i];
}

}  // namespace coarse_detail

template <class T>
class CoarseCorrection {
   public:
    using val_type = T;

    CoarseCorrection(const DeviceParams<T>& fine,
                     const cumes::ValidatedProblem& vp,
                     int coarse_ns,
                     int steps,
                     T delta = T(0.5))
        : fine_(fine),
          coarse_(coarse_params(fine, coarse_ns, steps, delta)),
          steps_(steps),
          delta_(delta),
          storage_(coarse_.ns, coarse_.mnmax),
          profiles_(coarse_, vp, std::nullopt, false),
          rs_(coarse_),
          modes_(coarse_),
          transform_(coarse_, rs_.value, modes_.value),
          geometry_(coarse_, std::nullopt),
          fine_coords_(fine_, true),
          coarse_coords_(coarse_, true),
          d_fine_g_(fine_size()),
          d_direction_(fine_size()),
          d_base_(coarse_size()),
          d_g0_(coarse_size()),
          d_g_(coarse_size()),
          d_restricted_(coarse_size()),
          d_tau_(coarse_size()),
          d_error_(coarse_size()),
          d_failed_(1) {
        if (fine.ntor == 0 && fine.nzeta == 1)
            axisym_ = std::make_unique<cumes::AxisymmetricOperator<T>>(coarse_);
        const auto op =
            axisym_ ? std::optional<
                          std::reference_wrapper<cumes::SpectralOperator<T>>>(
                          std::ref(*axisym_))
                    : std::nullopt;
        equilibrium_ = std::make_unique<cumes::EquilibriumOperator<T>>(
            coarse_, storage_, profiles_, transform_, rs_.value, geometry_,
            std::nullopt, op);
    }

    int fine_size() const { return 6 * fine_.ns * fine_.mnmax; }
    int coarse_size() const { return 6 * coarse_.ns * coarse_.mnmax; }
    int coarse_ns() const { return coarse_.ns; }
    int coarse_passes() const { return steps_ + 1; }
    T* direction_device() { return d_direction_.data(); }
    int* failed_device() { return d_failed_.data(); }
    T* tau_device() { return d_tau_.data(); }

    // Precondition: fixed boundary, stable fixed-m1 gauge.
    // Inputs are the SAME valid fine predescent state and postpreconditioned
    // residual. All coarse caches are private and initialized once per cycle.
    void enqueue(const T* d_fine_state,
                 const T* d_fine_residual,
                 cudaStream_t stream) {
        transform_.bind_stream(stream);
        d_failed_.zero_async(stream);
        storage_.velocity_buffer().zero_async(stream);
        coarse_detail::transfer_kernel<T, false>
            <<<(coarse_size() + 255) / 256, 256, 0, stream>>>(
                d_fine_state, storage_.state_slab(), fine_.ns, coarse_.ns,
                fine_.mnmax, fine_.ntor);
        fine_coords_.enqueue_pack(d_fine_residual, d_fine_g_.data(), stream);
        fine_coords_.enqueue_physical(d_fine_g_.data(), d_fine_g_.data(),
                                      stream);
        coarse_detail::transfer_kernel<T, true>
            <<<(coarse_size() + 255) / 256, 256, 0, stream>>>(
                d_fine_g_.data(), d_restricted_.data(), fine_.ns, coarse_.ns,
                fine_.mnmax, fine_.ntor);
        cumes::EvaluationSchedule schedule;
        schedule.update_iota_chi = true;
        schedule.zero_z_force_m1 = true;
        schedule.refresh_preconditioner = true;
        schedule.reset_constraint_reference = true;
        evaluate(schedule, stream, 0);
        copy(d_base_.data(), storage_.state_slab(), stream);
        copy(d_g0_.data(), d_g_.data(), stream);
        coarse_detail::
            tau_kernel<<<(coarse_size() + 255) / 256, 256, 0, stream>>>(
                d_g0_.data(), d_restricted_.data(), d_tau_.data(),
                coarse_size());
        schedule.refresh_preconditioner = false;
        schedule.reset_constraint_reference = false;
        for (int i = 0; i < steps_; ++i) {
            coarse_detail::
                smooth_kernel<<<(coarse_size() + 255) / 256, 256, 0, stream>>>(
                    storage_.state_slab(), storage_.velocity_slab(),
                    d_g_.data(), d_g0_.data(), d_restricted_.data(),
                    d_failed_.data(), coarse_size(), delta_);
            evaluate(schedule, stream, i + 1);
        }
        coarse_detail::
            difference_kernel<<<(coarse_size() + 255) / 256, 256, 0, stream>>>(
                storage_.state_slab(), d_base_.data(), d_error_.data(),
                d_failed_.data(), coarse_size(), coarse_.ns, coarse_.mnmax,
                coarse_.ntor);
        coarse_detail::transfer_kernel<T, true>
            <<<(fine_size() + 255) / 256, 256, 0, stream>>>(
                d_error_.data(), d_direction_.data(), coarse_.ns, fine_.ns,
                fine_.mnmax, fine_.ntor);
        cumes::check_cuda(cudaGetLastError(), "coarse FAS cycle");
    }

    void enqueue_trial(const T* d_base,
                       T* d_trial,
                       T scale,
                       cudaStream_t stream) const {
        coarse_detail::
            trial_kernel<<<(fine_size() + 255) / 256, 256, 0, stream>>>(
                d_base, d_direction_.data(), d_trial, fine_size(), scale);
        cumes::check_cuda(cudaGetLastError(), "coarse physical trial");
    }

   private:
    static DeviceParams<T> coarse_params(DeviceParams<T> p,
                                         int ns,
                                         int steps,
                                         T delta) {
        if (ns < 3 || ns >= p.ns || steps < 1 || !(delta > T(0)) ||
            !std::isfinite(delta))
            throw cumes::CumesError(
                "coarse correction: invalid grid/budget/step");
        if (p.radius_reference != T(0))
            throw cumes::CumesError(
                "coarse correction: physical storage required");
        p.ns = ns;
        p.ftol = T(0);  // internal FAS equation, never a convergence verdict
        return p;
    }
    void copy(T* d_to, const T* d_from, cudaStream_t stream) {
        cumes::check_cuda(
            cudaMemcpyAsync(d_to, d_from,
                            std::size_t(coarse_size()) * sizeof(T),
                            cudaMemcpyDeviceToDevice, stream),
            "coarse vector copy");
    }
    void evaluate(const cumes::EvaluationSchedule& schedule,
                  cudaStream_t stream,
                  int pass) {
        equilibrium_->enqueue(1000, 1000, schedule, stream);
        coarse_detail::status_kernel<<<1, 1, 0, stream>>>(
            equilibrium_->control_device(), d_failed_.data(), pass);
        coarse_coords_.enqueue_pack(equilibrium_->residual().data(),
                                    d_g_.data(), stream);
        coarse_coords_.enqueue_physical(d_g_.data(), d_g_.data(), stream);
    }

    DeviceParams<T> fine_, coarse_;
    int steps_;
    T delta_;
    cumes::SpectralStorage<T> storage_;
    cumes::Profiles<T> profiles_;
    coarse_detail::RealSpace<T> rs_;
    coarse_detail::ModeTable modes_;
    cumes::ToroidalFftOperator<T> transform_;
    cumes::GeometryOperator<T> geometry_;
    std::unique_ptr<cumes::AxisymmetricOperator<T>> axisym_;
    std::unique_ptr<cumes::EquilibriumOperator<T>> equilibrium_;
    CorrectionCoordinates<T> fine_coords_, coarse_coords_;
    cumes::DeviceBuffer<T> d_fine_g_, d_direction_, d_base_, d_g0_, d_g_,
        d_restricted_, d_tau_, d_error_;
    cumes::DeviceBuffer<int> d_failed_;
};

}  // namespace block_bench
#endif
