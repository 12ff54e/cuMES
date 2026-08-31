#include "cumes_test.h"
#include "magnetic_gradient_target.hpp"

#include <cmath>
#include <cstddef>

int main() {
    using cumes::test::check;

    constexpr int ns = 4;
    constexpr int ntheta = 8;
    constexpr int nzeta = 1;
    constexpr double b0 = 2.0;
    constexpr double amplitude = 0.25;
    constexpr double sqrtg_value = -2.0;
    constexpr double chi_prime = 3.0;
    const double two_pi = 2.0 * std::acos(-1.0);

    cumes::EquilibriumSnapshot equilibrium;
    equilibrium.ns = ns;
    equilibrium.ntheta = ntheta;
    equilibrium.nzeta = nzeta;
    for (auto& field : equilibrium.half_fields) {
        field.assign(equilibrium.half_field_size(), 0.0);
    }
    for (auto& field : equilibrium.full_fields) {
        field.assign(equilibrium.full_field_size(), 0.0);
    }

    for (int surface = 0; surface < ns - 1; ++surface) {
        for (int l = 0; l < ntheta; ++l) {
            const double theta = two_pi * l / ntheta;
            const double b = b0 + amplitude * std::sin(theta);
            const std::size_t index =
                static_cast<std::size_t>(surface) * ntheta + l;
            equilibrium.half_fields[cumes::EquilibriumSnapshot::SQRTG][index] =
                sqrtg_value;
            equilibrium.half_fields[cumes::EquilibriumSnapshot::BSUPU][index] =
                1.0;
            equilibrium.half_fields[cumes::EquilibriumSnapshot::BSUPV][index] =
                1.0;
            equilibrium.half_fields[cumes::EquilibriumSnapshot::BSUBV][index] =
                b * b;
        }
    }

    cumes::EquilibriumProfiles profiles;
    profiles.toroidal_flux_derivative.assign(ns - 1, 1.0);
    profiles.poloidal_flux_derivative.assign(ns - 1, chi_prime);
    profiles.rotational_transform.assign(ns - 1, 0.5);
    profiles.poloidal_covariant_field.assign(ns - 1, 0.0);
    profiles.toroidal_covariant_field.assign(ns - 1, 0.0);

    const auto fields = cumes_meow_example::calculate_magnetic_gradient_fields(
        equilibrium, profiles, 1);
    for (int surface = 0; surface < ns - 1; ++surface) {
        for (int l = 0; l < ntheta; ++l) {
            const double theta = two_pi * l / ntheta;
            const double b = b0 + amplitude * std::sin(theta);
            const double derivative = amplitude * std::cos(theta);
            const std::size_t index =
                static_cast<std::size_t>(surface) * ntheta + l;
            check(std::abs(fields.field_strength[index] - b) < 1.0e-13,
                  "magnetic target: field strength from mixed components");
            check(std::abs(fields.b_dot_grad_b[index] - derivative) < 1.0e-12,
                  "magnetic target: B dot grad B");
            const double expected_cross_s = b * b * derivative / sqrtg_value;
            check(std::abs(fields.b_cross_grad_s_dot_grad_b[index] -
                           expected_cross_s) < 1.0e-12,
                  "magnetic target: B cross grad s dot grad B");
            check(std::abs(fields.b_cross_grad_toroidal_flux_dot_grad_b[index] -
                           expected_cross_s) < 1.0e-12,
                  "magnetic target: toroidal-flux gradient observable");
            const double expected_cross =
                chi_prime / sqrtg_value * b * b * derivative;
            check(std::abs(fields.b_cross_grad_psi_p_dot_grad_b[index] -
                           expected_cross) < 1.0e-12,
                  "magnetic target: B cross grad psi_p dot grad B");
        }
    }

    return cumes::test::summary();
}
