#include "cumes/config/solver_options.hpp"
#include "cumes/config/validated_problem.hpp"
#include "cumes/solver/equilibrium_tangent.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void test_boundary_shape() {
    cumes::ProblemSpec spec;
    spec.mpol = 3;
    spec.ntor = 2;
    spec.angular.ntheta = 8;
    spec.angular.nzeta = 8;
    spec.stages = {{5, 1, 1.0}};
    spec.toroidal_flux.coefficients = {1.0};
    spec.iota.coefficients = {0.4};
    spec.rbc = {{0, 0, 1.0}, {1, 0, 0.1}};
    spec.zbs = {{1, 0, 0.1}};
    const auto validated = cumes::validate(std::move(spec), {});
    require(validated.has_value(), "test problem validates");

    auto direction = cumes::BoundaryTangent::zero(validated.value());
    require(direction.matches(validated.value()),
            "zero boundary tangent has the folded extent");
    require(direction.rbcc.size() == 9,
            "folded direction has mpol*(ntor+1) entries");
    direction.zbcs.pop_back();
    require(!direction.matches(validated.value()),
            "malformed boundary tangent is rejected");
}

void test_equilibrium_shape() {
    cumes::EquilibriumSnapshot equilibrium;
    equilibrium.ns = 3;
    equilibrium.mnmax = 2;
    equilibrium.ntheta = 4;
    equilibrium.nzeta = 2;
    for (auto& family : equilibrium.families) family.assign(6, 1.0);
    for (auto& field : equilibrium.half_fields) field.assign(16, 1.0);
    for (auto& field : equilibrium.full_fields) field.assign(24, 1.0);

    cumes::EquilibriumProfiles profiles;
    profiles.toroidal_flux_derivative.assign(2, 1.0);
    profiles.poloidal_flux_derivative.assign(2, 1.0);
    profiles.rotational_transform.assign(2, 1.0);
    profiles.poloidal_covariant_field.assign(2, 1.0);
    profiles.toroidal_covariant_field.assign(2, 1.0);

    auto tangent = cumes::EquilibriumTangent::zero_like(equilibrium, profiles);
    require(tangent.matches(equilibrium, profiles),
            "zero equilibrium tangent matches its primal");
    require(tangent.equilibrium.half_fields[0][3] == 0.0,
            "tangent storage is zero initialized");
    tangent.profiles.rotational_transform.pop_back();
    require(!tangent.matches(equilibrium, profiles),
            "malformed profile tangent is rejected");
}

}  // namespace

int main() {
    test_boundary_shape();
    test_equilibrium_shape();
    std::cout << "equilibrium tangent contract tests passed\n";
    return 0;
}
