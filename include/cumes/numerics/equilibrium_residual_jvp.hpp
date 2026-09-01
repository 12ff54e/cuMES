// equilibrium_residual_jvp.hpp — one-direction analytic JVP of the physical
// fixed-boundary equilibrium residual.
#ifndef CUMES_INCLUDE_CUMES_NUMERICS_EQUILIBRIUM_RESIDUAL_JVP_HPP_
#define CUMES_INCLUDE_CUMES_NUMERICS_EQUILIBRIUM_RESIDUAL_JVP_HPP_

#include "cumes/config/validated_problem.hpp"
#include "cumes/numerics/dual_spectral_operator.hpp"
#include "cumes/physics/force_operator.hpp"
#include "cumes/physics/geometry_operator.hpp"
#include "cumes/physics/magnetic_field_operator.hpp"
#include "cumes/physics/profiles.hpp"
#include "cumes/state/mode_table.cuh"
#include "cumes/state/spectral_storage.hpp"

#include <array>

namespace cumes {

// Evaluates the primal physical MHD residual and its exact directional
// derivative together. The state direction includes both interior unknowns
// and prescribed LCFS values; callers impose the fixed-boundary partition.
//
// The spectral-condensation force is deliberately absent. It is an iterative
// regularizer with a cadence-dependent preconditioner cache, rather than a
// physical equilibrium equation. The fixed-boundary tangent solve uses this
// gauge-fixed ideal-MHD residual and treats the production preconditioner only
// as a possible linear-solver preconditioner.
class EquilibriumResidualJvpOperator {
   public:
    using val_type = ForwardDualDouble;

    EquilibriumResidualJvpOperator(const DeviceParams<double>& primal_params,
                                   const ValidatedProblem& problem,
                                   const DeviceModeTable& mode_table);
    ~EquilibriumResidualJvpOperator();

    EquilibriumResidualJvpOperator(const EquilibriumResidualJvpOperator&) =
        delete;
    EquilibriumResidualJvpOperator& operator=(
        const EquilibriumResidualJvpOperator&) = delete;

    void enqueue(
        SpectralView<const ForwardDualDouble, PhysicalStateDomain> state,
        SpectralView<ForwardDualDouble, DecomposedResidualDomain> residual,
        cudaStream_t stream);

    const DeviceParams<ForwardDualDouble>& params() const { return p_; }
    const RealSpaceStorage<ForwardDualDouble>& real_space() const {
        return rs_;
    }
    const GeometryOperator<ForwardDualDouble>& geometry() const {
        return geometry_;
    }
    const Profiles<ForwardDualDouble>& profiles() const { return profiles_; }

   private:
    DeviceParams<ForwardDualDouble> p_{};
    Profiles<ForwardDualDouble> profiles_;
    SpectralStorage<ForwardDualDouble> state_;
    RealSpaceStorage<ForwardDualDouble> rs_;
    DualSpectralOperator transform_;
    GeometryOperator<ForwardDualDouble> geometry_;
    DeviceBuffer<ForwardDualDouble> rcon_;
    DeviceBuffer<ForwardDualDouble> zcon_;
    std::array<DeviceBuffer<ForwardDualDouble>, 4> zero_constraint_;
};

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_NUMERICS_EQUILIBRIUM_RESIDUAL_JVP_HPP_
