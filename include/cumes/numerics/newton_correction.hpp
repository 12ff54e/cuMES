// Full-state Newton correction of a frozen production equilibrium epoch.
#ifndef CUMES_INCLUDE_CUMES_NUMERICS_NEWTON_CORRECTION_HPP_
#define CUMES_INCLUDE_CUMES_NUMERICS_NEWTON_CORRECTION_HPP_

#include "cumes/numerics/correction_coordinates.hpp"
#include "cumes/numerics/device_gmres.hpp"
#include "cumes/solver/equilibrium_operator.hpp"

namespace cumes {

enum class DifferenceScheme { CENTRAL, FORWARD };

template <class T>
class NewtonCorrection {
   public:
    using val_type = T;

    NewtonCorrection(const DeviceParams<T>& p,
                     EquilibriumOperator<T>& equilibrium,
                     SpectralStorage<T>& storage,
                     int capacity,
                     DifferenceScheme scheme = DifferenceScheme::CENTRAL);

    int size() const { return 6 * p_.ns * p_.mnmax; }

    // The caller has already evaluated this state. Freeze the actual cached
    // preconditioner, constraint multiplier, reference and gauge; do not
    // differentiate a freshly rebuilt surrogate operator. The caller owns
    // convergence checks, nonlinear trial acceptance, and rollback decisions.
    void prepare(const EvaluationSchedule& schedule,
                 double norm_rz,
                 double norm_l,
                 cudaStream_t stream);

    // The entire active R/Z/lambda direction is perturbed together. Device
    // scaling bounds its largest physical coefficient perturbation by step.
    void enqueue_jvp(const T* d_direction,
                     T* d_result,
                     T step,
                     cudaStream_t stream);
    void enqueue_solve(int iterations,
                       T tolerance,
                       T step,
                       cudaStream_t stream);
    void enqueue_trial(T scale, cudaStream_t stream);
    void enqueue_restore(cudaStream_t stream);

    int evaluations_per_jvp() const {
        return scheme_ == DifferenceScheme::CENTRAL ? 2 : 1;
    }
    void set_difference_scheme(DifferenceScheme scheme) { scheme_ = scheme; }

    const T* rhs_device() const { return d_rhs_.data(); }
    const T* correction_device() const { return d_delta_.data(); }
    const T* base_device() const { return d_base_.data(); }
    const GmresControl<T>* control_device() const {
        return krylov_.control_device();
    }
    const CorrectionCoordinates<T>& coordinates() const { return coordinates_; }

   private:
    void evaluate(cudaStream_t stream);

    DeviceParams<T> p_;
    EquilibriumOperator<T>& equilibrium_;
    SpectralStorage<T>& storage_;
    CorrectionCoordinates<T> coordinates_;
    DifferenceScheme scheme_;
    DeviceGmres<T> krylov_;
    DeviceBuffer<T> d_base_, d_rhs_, d_delta_, d_physical_;
    DeviceBuffer<T> d_positive_, d_negative_, d_scale_;
    DeviceBuffer<int> d_valid_;
    EvaluationSchedule schedule_;
    double norm_rz_ = 1, norm_l_ = 1;
};

}  // namespace cumes
#endif  // CUMES_INCLUDE_CUMES_NUMERICS_NEWTON_CORRECTION_HPP_
