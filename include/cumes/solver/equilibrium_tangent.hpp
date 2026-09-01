// equilibrium_tangent.hpp — public host contracts for fixed-boundary
// equilibrium forward sensitivities.
#ifndef CUMES_INCLUDE_CUMES_SOLVER_EQUILIBRIUM_TANGENT_HPP_
#define CUMES_INCLUDE_CUMES_SOLVER_EQUILIBRIUM_TANGENT_HPP_

#include "cumes/config/validated_problem.hpp"
#include "cumes/io/equilibrium_profiles.hpp"
#include "cumes/io/equilibrium_snapshot.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace cumes {

// A direction in the same folded product basis as ValidatedProblem::boundary.
// It is deliberately independent of any optimizer parameterization: an
// embedding application owns the map from its variables to these four
// boundary families.
struct BoundaryTangent {
    std::vector<double> rbcc;
    std::vector<double> rbss;
    std::vector<double> zbsc;
    std::vector<double> zbcs;

    static BoundaryTangent zero(const ValidatedProblem& problem) {
        BoundaryTangent result;
        const std::size_t count = problem.boundary().size();
        result.rbcc.assign(count, 0.0);
        result.rbss.assign(count, 0.0);
        result.zbsc.assign(count, 0.0);
        result.zbcs.assign(count, 0.0);
        return result;
    }

    bool matches(const ValidatedProblem& problem) const {
        const std::size_t count = problem.boundary().size();
        return rbcc.size() == count && rbss.size() == count &&
               zbsc.size() == count && zbcs.size() == count;
    }
};

// Directional derivative of the public equilibrium result. The nested values
// use the exact public primal layouts; every number is a derivative, not an
// independently writable equilibrium. Keeping one layout contract lets meow
// apply its target chain rule without seeing CUDA storage.
struct EquilibriumTangent {
    EquilibriumSnapshot equilibrium;
    EquilibriumProfiles profiles;

    static EquilibriumTangent zero_like(
        const EquilibriumSnapshot& primal_equilibrium,
        const EquilibriumProfiles& primal_profiles) {
        EquilibriumTangent result;
        result.equilibrium.ns = primal_equilibrium.ns;
        result.equilibrium.mnmax = primal_equilibrium.mnmax;
        result.equilibrium.ntheta = primal_equilibrium.ntheta;
        result.equilibrium.nzeta = primal_equilibrium.nzeta;
        for (std::size_t component = 0; component < EquilibriumSnapshot::COUNT;
             ++component) {
            result.equilibrium.families[component].assign(
                primal_equilibrium.families[component].size(), 0.0);
        }
        for (std::size_t field = 0;
             field < EquilibriumSnapshot::HALF_FIELD_COUNT; ++field) {
            result.equilibrium.half_fields[field].assign(
                primal_equilibrium.half_fields[field].size(), 0.0);
        }
        for (std::size_t field = 0;
             field < EquilibriumSnapshot::FULL_FIELD_COUNT; ++field) {
            result.equilibrium.full_fields[field].assign(
                primal_equilibrium.full_fields[field].size(), 0.0);
        }
        result.profiles.toroidal_flux_derivative.assign(
            primal_profiles.toroidal_flux_derivative.size(), 0.0);
        result.profiles.poloidal_flux_derivative.assign(
            primal_profiles.poloidal_flux_derivative.size(), 0.0);
        result.profiles.rotational_transform.assign(
            primal_profiles.rotational_transform.size(), 0.0);
        result.profiles.poloidal_covariant_field.assign(
            primal_profiles.poloidal_covariant_field.size(), 0.0);
        result.profiles.toroidal_covariant_field.assign(
            primal_profiles.toroidal_covariant_field.size(), 0.0);
        return result;
    }

    bool matches(const EquilibriumSnapshot& primal_equilibrium,
                 const EquilibriumProfiles& primal_profiles) const {
        if (equilibrium.ns != primal_equilibrium.ns ||
            equilibrium.mnmax != primal_equilibrium.mnmax ||
            equilibrium.ntheta != primal_equilibrium.ntheta ||
            equilibrium.nzeta != primal_equilibrium.nzeta) {
            return false;
        }
        for (std::size_t component = 0; component < EquilibriumSnapshot::COUNT;
             ++component) {
            if (equilibrium.families[component].size() !=
                primal_equilibrium.families[component].size()) {
                return false;
            }
        }
        for (std::size_t field = 0;
             field < EquilibriumSnapshot::HALF_FIELD_COUNT; ++field) {
            if (equilibrium.half_fields[field].size() !=
                primal_equilibrium.half_fields[field].size()) {
                return false;
            }
        }
        for (std::size_t field = 0;
             field < EquilibriumSnapshot::FULL_FIELD_COUNT; ++field) {
            if (equilibrium.full_fields[field].size() !=
                primal_equilibrium.full_fields[field].size()) {
                return false;
            }
        }
        return profiles.toroidal_flux_derivative.size() ==
                   primal_profiles.toroidal_flux_derivative.size() &&
               profiles.poloidal_flux_derivative.size() ==
                   primal_profiles.poloidal_flux_derivative.size() &&
               profiles.rotational_transform.size() ==
                   primal_profiles.rotational_transform.size() &&
               profiles.poloidal_covariant_field.size() ==
                   primal_profiles.poloidal_covariant_field.size() &&
               profiles.toroidal_covariant_field.size() ==
                   primal_profiles.toroidal_covariant_field.size();
    }
};

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_SOLVER_EQUILIBRIUM_TANGENT_HPP_
