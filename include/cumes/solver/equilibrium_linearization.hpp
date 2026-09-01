// equilibrium_linearization.hpp — retained host session for matrix-free
// fixed-boundary equilibrium residual derivatives.
#ifndef CUMES_INCLUDE_CUMES_SOLVER_EQUILIBRIUM_LINEARIZATION_HPP_
#define CUMES_INCLUDE_CUMES_SOLVER_EQUILIBRIUM_LINEARIZATION_HPP_

#include "cumes/config/validated_problem.hpp"
#include "cumes/io/equilibrium_snapshot.hpp"
#include "cumes/solver/equilibrium_tangent.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace cumes {

struct ResidualJvp {
    // Gauge-fixed physical spectral residual and its directional derivative,
    // both in component-major [6][mnmax][ns] layout.
    std::vector<double> residual;
    std::vector<double> tangent;
};

// Retains the final-grid CUDA workspaces for repeated JVPs around one primal
// equilibrium. Construction does not rerun the nonlinear solver. The class is
// host-facing: embedding applications compile as ordinary C++.
class EquilibriumLinearization {
   public:
    EquilibriumLinearization(const ValidatedProblem& problem,
                             const EquilibriumSnapshot& equilibrium);
    ~EquilibriumLinearization();

    EquilibriumLinearization(const EquilibriumLinearization&) = delete;
    EquilibriumLinearization& operator=(const EquilibriumLinearization&) =
        delete;
    EquilibriumLinearization(EquilibriumLinearization&&) noexcept;
    EquilibriumLinearization& operator=(EquilibriumLinearization&&) noexcept;

    std::size_t state_size() const;

    // Apply the state Jacobian F_u to a full spectral-state direction.
    ResidualJvp residual_jvp(std::span<const double> state_direction);

    // Apply F_x to a folded fixed-boundary direction. Interior and lambda
    // directions are zero; the four R/Z LCFS rows receive the supplied data.
    ResidualJvp boundary_residual_jvp(const BoundaryTangent& direction);

   private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_SOLVER_EQUILIBRIUM_LINEARIZATION_HPP_
