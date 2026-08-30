// free_boundary_policy.hpp — pure host vacuum-controller constants.
#ifndef CUMES_INCLUDE_CUMES_PHYSICS_FREE_BOUNDARY_POLICY_HPP_
#define CUMES_INCLUDE_CUMES_PHYSICS_FREE_BOUNDARY_POLICY_HPP_

namespace cumes {

// Turn on the vacuum edge force once the fixed-boundary predictor has removed
// the large startup imbalance. Earlier activation avoids over-solving a force
// model whose boundary condition is about to change.
inline constexpr double DEFAULT_VACUUM_ACTIVATION_RESIDUAL = 3.0e-2;

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_PHYSICS_FREE_BOUNDARY_POLICY_HPP_
