#include "cumes_test.h"
#include "quasisymmetry_target.hpp"

#include <cmath>
#include <cstddef>
#include <numbers>

int main() {
    using cumes::test::check;

    constexpr int ns = 4;
    constexpr int ntheta = 16;
    constexpr int nzeta = 1;
    constexpr double major_radius = 5.0;
    constexpr double minor_radius = 1.0;
    constexpr double b0 = 2.0;
    constexpr double b_amplitude = 0.25;
    constexpr double sqrtg_value = -2.0;
    constexpr double g_covariant = 2.0;
    constexpr double iota = 0.5;

    cumes::EquilibriumSnapshot equilibrium;
    equilibrium.ns = ns;
    equilibrium.mnmax = 2;
    equilibrium.ntheta = ntheta;
    equilibrium.nzeta = nzeta;
    for (auto& family : equilibrium.families) {
        family.assign(equilibrium.family_size(), 0.0);
    }
    for (int surface = 0; surface < ns; ++surface) {
        equilibrium.families[cumes::EquilibriumSnapshot::RMNCC]
                            [static_cast<std::size_t>(surface)] = major_radius;
        equilibrium.families[cumes::EquilibriumSnapshot::RMNCC]
                            [static_cast<std::size_t>(ns + surface)] =
            minor_radius;
        equilibrium.families[cumes::EquilibriumSnapshot::ZMNSC]
                            [static_cast<std::size_t>(ns + surface)] =
            minor_radius;
    }
    for (auto& field : equilibrium.half_fields) {
        field.assign(equilibrium.half_field_size(), 0.0);
    }
    for (auto& field : equilibrium.full_fields) {
        field.assign(equilibrium.full_field_size(), 0.0);
    }

    double i_covariant = 0.0;
    for (int surface = 0; surface < ns - 1; ++surface) {
        for (int theta_index = 0; theta_index < ntheta; ++theta_index) {
            const double theta = 2.0 * std::numbers::pi * theta_index / ntheta;
            const double b = b0 + b_amplitude * std::sin(theta);
            const std::size_t index =
                static_cast<std::size_t>(surface) * ntheta + theta_index;
            equilibrium.half_fields[cumes::EquilibriumSnapshot::SQRTG][index] =
                sqrtg_value;
            equilibrium.half_fields[cumes::EquilibriumSnapshot::BSUPU][index] =
                1.0;
            equilibrium.half_fields[cumes::EquilibriumSnapshot::BSUBU][index] =
                b * b;
            equilibrium.half_fields[cumes::EquilibriumSnapshot::BSUBV][index] =
                g_covariant;
            if (surface == 0) i_covariant += b * b / ntheta;
        }
    }

    cumes::EquilibriumProfiles profiles;
    profiles.toroidal_flux_derivative.assign(ns - 1, 2.0 * std::numbers::pi);
    // With the negative cuMES Jacobian orientation this gives
    // psi_p'=-4 after division by 2*pi.
    profiles.poloidal_flux_derivative.assign(ns - 1, 8.0 * std::numbers::pi);
    profiles.rotational_transform.assign(ns - 1, iota);
    profiles.poloidal_covariant_field.assign(ns - 1, i_covariant);
    profiles.toroidal_covariant_field.assign(ns - 1, g_covariant);

    cumes::InputParams input;
    input.mpol = 2;
    input.ntor = 0;
    input.nfp = 1;
    input.ntheta = ntheta;
    input.nzeta = nzeta;

    cumes_meow_example::QuasisymmetryTargetSpec qa_spec;
    qa_spec.helicity_m = 1;
    qa_spec.helicity_n = 0;
    qa_spec.half_surface_indices =
        cumes_meow_example::all_half_grid_surfaces(ns);
    const auto qs = cumes_meow_example::calculate_quasisymmetry_target(
        equilibrium, profiles, input.nfp, qa_spec);
    check(qs.residuals.size() ==
              static_cast<std::size_t>((ns - 1) * ntheta * nzeta),
          "QS target: one residual per selected angular point");
    check(std::abs(qs.value) < 1.0e-24,
          "QS target: manufactured QA field has zero residual");

    const auto qa = cumes_meow_example::calculate_qa_target(
        equilibrium, profiles, input, qa_spec, major_radius / minor_radius,
        iota);
    check(qa.residuals.size() == qs.residuals.size() + 2,
          "QA target: aspect and iota residuals appended");
    check(std::abs(qa.value) < 1.0e-24,
          "QA target: matched composite target is zero");

    const auto penalized_qa = cumes_meow_example::calculate_qa_target(
        equilibrium, profiles, input, qa_spec,
        major_radius / minor_radius - 0.25, iota - 0.1);
    check(std::abs(penalized_qa.value - (0.25 * 0.25 + 0.1 * 0.1)) < 1.0e-13,
          "QA target: scalar is squared norm of composite residuals");

    cumes_meow_example::QuasisymmetryTargetSpec qh_spec = qa_spec;
    qh_spec.helicity_n = 1;
    const auto qh = cumes_meow_example::calculate_qh_target(
        equilibrium, profiles, input, qh_spec, major_radius / minor_radius);
    check(qh.residuals.size() == qh.qs.residuals.size() + 1,
          "QH target: aspect residual appended");
    check(std::abs(qh.value - qh.qs.value) < 1.0e-13,
          "QH target: matched aspect leaves QS value");

    return cumes::test::summary();
}
