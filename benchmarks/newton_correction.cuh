// Experimental full-state Newton correction of a frozen production epoch.
#ifndef CUMES_BENCHMARKS_NEWTON_CORRECTION_CUH_
#define CUMES_BENCHMARKS_NEWTON_CORRECTION_CUH_

#include "correction_coordinates.cuh"
#include "cumes/solver/equilibrium_operator.hpp"
#include "device_gmres.cuh"

namespace block_bench {

enum class DifferenceScheme { CENTRAL, FORWARD };
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

__global__ void check_probe(const cumes::ControlStatus* d_status,
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
class NewtonCorrection {
   public:
    using val_type = T;

    NewtonCorrection(const DeviceParams<T>& p,
                     cumes::EquilibriumOperator<T>& equilibrium,
                     cumes::SpectralStorage<T>& storage,
                     int capacity,
                     DifferenceScheme scheme = DifferenceScheme::CENTRAL)
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

    int size() const { return 6 * p_.ns * p_.mnmax; }

    // The caller has already evaluated this state. Freeze the actual cached
    // preconditioner, constraint multiplier, reference and gauge; do not
    // differentiate a freshly rebuilt surrogate operator.
    void prepare(const cumes::EvaluationSchedule& schedule,
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

    // Finite differences in physical state coordinates. The entire active
    // R/Z/lambda direction is perturbed together. Scaling is computed on
    // device to bound its largest physical coefficient perturbation.
    void enqueue_jvp(const T* d_direction,
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

    void enqueue_solve(int iterations,
                       T tolerance,
                       T step,
                       cudaStream_t stream) {
        auto apply = [&](const T* d_input, T* d_output, cudaStream_t work) {
            enqueue_jvp(d_input, d_output, step, work);
        };
        krylov_.enqueue(apply, d_rhs_.data(), d_delta_.data(), iterations,
                        tolerance, stream);
    }

    void enqueue_trial(T scale, cudaStream_t stream) {
        coordinates_.enqueue_trial(d_base_.data(), d_delta_.data(),
                                   storage_.state_slab(), scale, stream);
        evaluate(stream);
    }

    void enqueue_restore(cudaStream_t stream) {
        storage_.state_buffer().copy_from_async(d_base_, stream);
        evaluate(stream);
    }

    int evaluations_per_jvp() const {
        return scheme_ == DifferenceScheme::CENTRAL ? 2 : 1;
    }
    void set_difference_scheme(DifferenceScheme scheme) { scheme_ = scheme; }

    const T* rhs_device() const { return d_rhs_.data(); }
    const T* correction_device() const { return d_delta_.data(); }
    const T* base_device() const { return d_base_.data(); }
    auto control_device() const { return krylov_.control_device(); }
    const CorrectionCoordinates<T>& coordinates() const { return coordinates_; }

   private:
    void evaluate(cudaStream_t stream) {
        equilibrium_.enqueue(1000, 1000, schedule_, stream, norm_rz_, norm_l_);
    }

    DeviceParams<T> p_;
    cumes::EquilibriumOperator<T>& equilibrium_;
    cumes::SpectralStorage<T>& storage_;
    CorrectionCoordinates<T> coordinates_;
    DifferenceScheme scheme_;
    DeviceGmres<T> krylov_;
    cumes::DeviceBuffer<T> d_base_, d_rhs_, d_delta_, d_physical_;
    cumes::DeviceBuffer<T> d_positive_, d_negative_, d_scale_;
    cumes::DeviceBuffer<int> d_valid_;
    cumes::EvaluationSchedule schedule_;
    double norm_rz_ = 1, norm_l_ = 1;
};

}  // namespace block_bench
#endif  // CUMES_BENCHMARKS_NEWTON_CORRECTION_CUH_
