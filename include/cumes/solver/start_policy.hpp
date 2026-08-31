// start_policy.hpp — pure host policy for stage-initial solver controls.
#ifndef CUMES_INCLUDE_CUMES_SOLVER_START_POLICY_HPP_
#define CUMES_INCLUDE_CUMES_SOLVER_START_POLICY_HPP_

#include "cumes/solver/control_policy.hpp"

namespace cumes {

// Qualified double-precision cold starts tolerate a larger descent step than
// the general 3-D path. Coarse 3-D free-boundary grids use their measured
// stable step. For fixed-boundary axisymmetry, keep the first continuation
// grid conservative because it starts from an analytic cold guess; later grids
// start from a converged prolongation. A single fine grid has no continuation
// safety net and uses the intermediate factor. Float retains its configured
// step because the larger value amplifies its state-rounding residual floor.
template <typename T>
constexpr T initial_step_for_stage(T configured_step,
                                   int ntor,
                                   int nzeta,
                                   bool free_boundary,
                                   int radial_surfaces,
                                   int stage_count,
                                   int stage_index) noexcept {
    if constexpr (sizeof(T) < sizeof(double)) return configured_step;
    if (free_boundary) {
        if (ntor > 0 && radial_surfaces <=
                            control_policy::FREE_BOUNDARY_COARSE_MAX_SURFACES) {
            return configured_step *
                   T(control_policy::FREE_BOUNDARY_COARSE_STEP_NUMERATOR) /
                   T(control_policy::FREE_BOUNDARY_COARSE_STEP_DENOMINATOR);
        }
        return configured_step;
    }
    if (ntor != 0 || nzeta != 1) return configured_step;
    if (stage_count == 1) {
        return configured_step *
               T(control_policy::AXISYMMETRIC_SINGLE_GRID_STEP_NUMERATOR) /
               T(control_policy::AXISYMMETRIC_SINGLE_GRID_STEP_DENOMINATOR);
    }
    if (stage_index == 0) {
        return configured_step *
               T(control_policy::AXISYMMETRIC_COARSE_STEP_NUMERATOR) /
               T(control_policy::AXISYMMETRIC_COARSE_STEP_DENOMINATOR);
    }
    return configured_step *
           T(control_policy::AXISYMMETRIC_REFINED_STEP_NUMERATOR) /
           T(control_policy::AXISYMMETRIC_REFINED_STEP_DENOMINATOR);
}

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_SOLVER_START_POLICY_HPP_
