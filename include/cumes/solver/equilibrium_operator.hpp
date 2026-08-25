// equilibrium_operator.hpp — the per-iteration equilibrium DAG (blueprint
// §6.11, §7).
//
// Composes the state-to-residual device pipeline the fixed-point loop runs once
// per pass: axis extrapolation → inverse transform → base geometry → oriented
// Jacobian stats → magnetic field → (optional) constraint-reference reset /
// preconditioner refresh → MHD + constraint force → forward transform → odd-m
// decomposition + m=1 gauge → invariant residual → in-place preconditioner →
// preconditioned residual, all reduced into one device ControlRecord. The
// ordered host-side decisions (descent, post-descent checkpoint
// capture/restore) stay with the solver/controller (blueprint §6.10/§6.11).
//
// The hot-loop dump windows (dump_windows.hpp) fire inside enqueue at the
// same observation points as the legacy machinery; the windows themselves
// live with the dump module, not the operator (all legacy workspace structs
// are gone — migration step 13).
#ifndef CUMES_INCLUDE_CUMES_SOLVER_EQUILIBRIUM_OPERATOR_HPP_
#define CUMES_INCLUDE_CUMES_SOLVER_EQUILIBRIUM_OPERATOR_HPP_

#include "cumes/numerics/preconditioner.hpp"
#include "cumes/numerics/residual_operator.hpp"
#include "cumes/physics/constraint_operator.hpp"
#include "cumes/physics/force_operator.hpp"
#include "cumes/physics/free_boundary_operator.hpp"
#include "cumes/physics/geometry_operator.hpp"
#include "cumes/physics/magnetic_field_operator.hpp"
#include "cumes/physics/profiles.hpp"
#include "cumes/solver/control_record.hpp"
#include "cumes/state/spectral_storage.hpp"
#include "cumes/transforms/toroidal_fft_operator.hpp"

#include <cuda_runtime.h>

#include <functional>
#include <optional>

namespace cumes {

// Per-iteration flags the controller derives before the DAG is enqueued.
struct EvaluationSchedule {
    bool update_iota_chi = false;             // (ncurr==1) || first pass
    bool reset_constraint_reference = false;  // iter2 == iter1
    bool refresh_preconditioner = false;      // (iter2 - iter1) % 25 == 0
    bool zero_z_force_m1 = false;             // iter2 < 2 || fsqz_prev < 1e-6
    bool include_edge_rz_invariant = true;
    bool run_vacuum_block = false;
    bool decay_rcon0_zcon0 = false;
    bool apply_vacuum_edge_force = false;
};

template <class T>
class EquilibriumOperator {
   public:
    using val_type = T;

    EquilibriumOperator(
        const DeviceParams<T>& p,
        SpectralStorage<T>& storage,
        const Profiles<T>& profiles,
        ToroidalFftOperator<T>& transform,
        RealSpaceStorage<T>& rs,
        GeometryOperator<T>& geometry,
        const std::optional<std::reference_wrapper<DeviceArena>>& arena,
        const std::optional<std::reference_wrapper<SpectralOperator<T>>>& op,
        const std::optional<std::reference_wrapper<FreeBoundaryOperator<T>>>&
            vacuum = std::nullopt);
    ~EquilibriumOperator();

    EquilibriumOperator(const EquilibriumOperator&) = delete;
    EquilibriumOperator& operator=(const EquilibriumOperator&) = delete;

    // Enqueue one pass's device DAG, reducing into the owned control record
    // (control_device()). The caller transfers it and fences after this
    // returns. f_norm_rz/f_norm_l are the host's cached force-norm factors
    // (from the last refresh pass): the device terminal predicate needs them to
    // classify the invariant residual against ftol BEFORE in-place
    // preconditioning on non-refresh passes. On refresh passes the DAG
    // finalizes the factors ON DEVICE from this pass's force norms
    // (final_f_norm_* in the record) before the predicate, which then
    // classifies convergence from those — the host consumes the same record
    // fields at the fence (completion-plan follow-up §2.3). The defaults keep
    // the benchmark harnesses (which drive enqueue directly) compilable.
    void enqueue(int iter,
                 int iter2,
                 const EvaluationSchedule& schedule,
                 cudaStream_t stream,
                 double f_norm_rz = 1.0,
                 double f_norm_l = 1.0);
    void enqueue_prefix(int iter,
                        int iter2,
                        const EvaluationSchedule& schedule,
                        cudaStream_t stream,
                        double f_norm_rz,
                        double f_norm_l);
    void enqueue_suffix(int iter,
                        int iter2,
                        const EvaluationSchedule& schedule,
                        cudaStream_t stream,
                        double f_norm_rz,
                        double f_norm_l);

    T* buco_bvco_device() { return d_buco_bvco_.data(); }
    T* repack_device() { return d_repack_.data(); }
    T* axis_device() { return d_axis_.data(); }
    T* delbsq_device() { return d_delbsq_.data(); }
    T* rbsq_device() { return d_rbsq_.data(); }
    Preconditioner<T>* preconditioner() { return &precon_; }

    // The reduced control record: Jacobian stats + status, invariant /
    // preconditioned raw sums, force-norm partials (completion plan step 1.3).
    // DOUBLE numeric slots in both builds (ADR-0001 follow-up): the double
    // accumulations reach the host controller unrounded; the double build is
    // identical by construction.
    ControlRecord* control_device() { return d_control_.data(); }

    // The decomposed residual slab (for the solver's descent + dump machinery).
    SpectralView<T, DecomposedResidualDomain> residual() {
        return residual_view_;
    }
    SpectralView<const T, DecomposedResidualDomain> residual_const() const {
        return residual_view_const_;
    }
    SpectralView<T, PhysicalStateDomain> state() { return state_view_; }
    SpectralView<T, DecomposedVelocityDomain> velocity() {
        return velocity_view_;
    }

    // The mode table (xm/xn) shared by scalxc / m1 gauge / descent.
    const int* xm() const { return transform_.xm(); }
    const int* xn() const { return transform_.xn(); }

    // Accumulated transform-timing (inverse/forward), sampled at each control
    // fence; read once per stage by the solver's timing report.
    void transform_timing_ms(float& inv_ms, float& fwd_ms) const {
        inv_ms = t_inv_ms_;
        fwd_ms = t_fwd_ms_;
    }

    // Sample the transform-timing events at the (already-reached) control
    // fence.
    void sample_transform_timing() {
        float ms;
        cudaEventElapsedTime(&ms, ev0_inv_, ev1_inv_);
        t_inv_ms_ += ms;
        cudaEventElapsedTime(&ms, ev0_fwd_, ev1_fwd_);
        t_fwd_ms_ += ms;
    }

   private:
    const DeviceParams<T>& p_;
    SpectralStorage<T>& storage_;
    const Profiles<T>& profiles_;
    ToroidalFftOperator<T>& transform_;
    RealSpaceStorage<T>& rs_;
    GeometryOperator<T>& geometry_;

    Preconditioner<T> precon_;
    ConstraintOperator<T> constraint_;
    BaseGeometryHalfViews<T> base_views_;
    MagneticFieldViews<T> field_views_;
    RadialProfileViews<T> rpv_;
    DeviceBuffer<T> d_f_spec_;
    DeviceBuffer<ControlRecord> d_control_;
    DeviceBuffer<T> d_psum_;

    SpectralView<T, PhysicalStateDomain> state_view_;
    SpectralView<const T, PhysicalStateDomain> state_view_const_;
    SpectralView<T, DecomposedVelocityDomain> velocity_view_;
    SpectralView<T, DecomposedResidualDomain> residual_view_;
    SpectralView<const T, DecomposedResidualDomain> residual_view_const_;

    SpectralOperator<T>* transform_op_ = nullptr;
    FreeBoundaryOperator<T>* vacuum_ = nullptr;
    GeometryParityViews<T> geom_views_;
    ForceParityViews<const T> force_views_;
    ConstraintForceViews<const T> conforce_views_;

    DeviceBuffer<T> d_buco_bvco_;
    DeviceBuffer<T> d_repack_;
    DeviceBuffer<T> d_axis_;
    DeviceBuffer<T> d_rbsq_;
    DeviceBuffer<T> d_delbsq_;

    // Transform-timing event pairs (recorded in enqueue, read at the fence).
    cudaEvent_t ev0_inv_{}, ev1_inv_{}, ev0_fwd_{}, ev1_fwd_{};
    float t_inv_ms_ = 0.0f, t_fwd_ms_ = 0.0f;
};

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_SOLVER_EQUILIBRIUM_OPERATOR_HPP_
