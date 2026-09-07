// Explicit mixed-float failure boundary for the precise-double-only
// equilibrium tangent API. Keeping linkable definitions makes the installed
// package honest: an unsupported call fails with context instead of producing
// unresolved symbols at final link.
#include "cumes/runtime/cuda_status.hpp"
#include "cumes/solver/equilibrium_linearization.hpp"

#include <memory>
#include <span>

namespace cumes {
namespace {

[[noreturn]] void throw_unsupported_tangent_precision() {
    throw CumesError(
        "equilibrium forward tangents require a precise-double cuMES build");
}

}  // namespace

class EquilibriumLinearization::Impl {};

EquilibriumLinearization::EquilibriumLinearization(const ValidatedProblem&,
                                                   const EquilibriumSnapshot&)
    : impl_(std::make_unique<Impl>()) {
    throw_unsupported_tangent_precision();
}

EquilibriumLinearization::~EquilibriumLinearization() = default;
EquilibriumLinearization::EquilibriumLinearization(
    EquilibriumLinearization&&) noexcept = default;
EquilibriumLinearization& EquilibriumLinearization::operator=(
    EquilibriumLinearization&&) noexcept = default;

std::size_t EquilibriumLinearization::state_size() const {
    throw_unsupported_tangent_precision();
}

ResidualJvp EquilibriumLinearization::residual_jvp(std::span<const double>) {
    throw_unsupported_tangent_precision();
}

ResidualJvp EquilibriumLinearization::boundary_residual_jvp(
    const BoundaryTangent&) {
    throw_unsupported_tangent_precision();
}

SpectralTangentSolve EquilibriumLinearization::solve_boundary_tangent(
    const BoundaryTangent&,
    const TangentLinearOptions&) {
    throw_unsupported_tangent_precision();
}

EquilibriumTangent EquilibriumLinearization::materialize_tangent(
    std::span<const double>,
    const EquilibriumSnapshot&,
    const EquilibriumProfiles&) {
    throw_unsupported_tangent_precision();
}

}  // namespace cumes
