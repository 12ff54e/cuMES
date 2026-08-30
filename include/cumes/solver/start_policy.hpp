// start_policy.hpp — pure host policy for stage-initial solver controls.
#ifndef CUMES_INCLUDE_CUMES_SOLVER_START_POLICY_HPP_
#define CUMES_INCLUDE_CUMES_SOLVER_START_POLICY_HPP_

namespace cumes {

// Fixed-boundary axisymmetric stages tolerate a larger descent step than the
// general 3-D path.  Keep the first continuation grid conservative because it
// starts from an analytic cold guess; later grids start from a converged
// prolongation.  A single fine grid has no continuation safety net and uses
// the intermediate factor.
template <typename T>
constexpr T initial_step_for_stage(T configured_step,
                                   int ntor,
                                   int nzeta,
                                   bool free_boundary,
                                   int stage_count,
                                   int stage_index) noexcept {
    if (free_boundary || ntor != 0 || nzeta != 1) return configured_step;
    if (stage_count == 1) return configured_step * T(7) / T(6);
    if (stage_index == 0) return configured_step * T(17) / T(15);
    return configured_step * T(6) / T(5);
}

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_SOLVER_START_POLICY_HPP_
