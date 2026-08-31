// Optimizer-side evaluation of magnetic-gradient observables from a cuMES
// equilibrium. The returned pointwise fields are primitives; an optimizer
// application owns their surface selection, reduction, normalization, and
// residual weighting.
#ifndef CUMES_EXAMPLES_MAGNETIC_GRADIENT_TARGET_HPP_
#define CUMES_EXAMPLES_MAGNETIC_GRADIENT_TARGET_HPP_

#include "cumes/io/equilibrium_profiles.hpp"
#include "cumes/io/equilibrium_snapshot.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace cumes_meow_example {
namespace magnetic_target_detail {

inline std::vector<double> periodic_derivative(
    const std::vector<double>& values,
    int surfaces,
    int ntheta,
    int nzeta,
    int nfp,
    bool direction_theta) {
    const int length = direction_theta ? ntheta : nzeta;
    std::vector<double> derivative(values.size(), 0.0);
    if (length == 1) return derivative;

    const double pi = std::acos(-1.0);
    const double direction_scale = direction_theta ? 1.0 : nfp;
    const int lines_per_surface = direction_theta ? nzeta : ntheta;
    for (int surface = 0; surface < surfaces; ++surface) {
        for (int line = 0; line < lines_per_surface; ++line) {
            for (int target = 0; target < length; ++target) {
                double sum = 0.0;
                for (int source = 0; source < length; ++source) {
                    if (source == target) continue;
                    const int difference = target - source;
                    const double angle =
                        pi * static_cast<double>(difference) / length;
                    const double parity =
                        (std::abs(difference) % 2 == 0) ? 1.0 : -1.0;
                    const double weight = (length % 2 == 0)
                                              ? 0.5 * parity / std::tan(angle)
                                              : 0.5 * parity / std::sin(angle);
                    const std::size_t index =
                        direction_theta
                            ? static_cast<std::size_t>(surface) * ntheta *
                                      nzeta +
                                  static_cast<std::size_t>(line) * ntheta +
                                  source
                            : static_cast<std::size_t>(surface) * ntheta *
                                      nzeta +
                                  static_cast<std::size_t>(source) * ntheta +
                                  line;
                    sum += weight * values[index];
                }
                const std::size_t index =
                    direction_theta
                        ? static_cast<std::size_t>(surface) * ntheta * nzeta +
                              static_cast<std::size_t>(line) * ntheta + target
                        : static_cast<std::size_t>(surface) * ntheta * nzeta +
                              static_cast<std::size_t>(target) * ntheta + line;
                derivative[index] = direction_scale * sum;
            }
        }
    }
    return derivative;
}

inline std::vector<double> radial_derivative(const std::vector<double>& values,
                                             int surfaces,
                                             std::size_t points,
                                             double delta_s) {
    std::vector<double> derivative(values.size(), 0.0);
    if (surfaces == 1) return derivative;
    for (std::size_t point = 0; point < points; ++point) {
        if (surfaces == 2) {
            const double value =
                (values[points + point] - values[point]) / delta_s;
            derivative[point] = value;
            derivative[points + point] = value;
            continue;
        }
        derivative[point] =
            (-3.0 * values[point] + 4.0 * values[points + point] -
             values[2 * points + point]) /
            (2.0 * delta_s);
        for (int surface = 1; surface < surfaces - 1; ++surface) {
            const std::size_t index =
                static_cast<std::size_t>(surface) * points + point;
            derivative[index] =
                (values[index + points] - values[index - points]) /
                (2.0 * delta_s);
        }
        const std::size_t last =
            static_cast<std::size_t>(surfaces - 1) * points + point;
        derivative[last] = (3.0 * values[last] - 4.0 * values[last - points] +
                            values[last - 2 * points]) /
                           (2.0 * delta_s);
    }
    return derivative;
}

}  // namespace magnetic_target_detail

struct MagneticGradientFields {
    // Native half-grid layout: point + half_surface*(ntheta*nzeta).
    std::vector<double> field_strength;
    std::vector<double> b_dot_grad_b;
    std::vector<double> b_cross_grad_s_dot_grad_b;
    std::vector<double> b_cross_grad_toroidal_flux_dot_grad_b;
    std::vector<double> b_cross_grad_psi_p_dot_grad_b;
};

// Calculate pointwise half-grid observables. The captured profile uses VMEC's
// public physical-flux convention (webers). `poloidal_flux_scale` is an
// optional optimizer normalization; 1 uses physical poloidal flux.
inline MagneticGradientFields calculate_magnetic_gradient_fields(
    const cumes::EquilibriumSnapshot& equilibrium,
    const cumes::EquilibriumProfiles& profiles,
    int nfp,
    double poloidal_flux_scale = 1.0) {
    if (!equilibrium.has_derived_fields() ||
        !profiles.has_half_grid_profiles(equilibrium.ns) || nfp < 1 ||
        !std::isfinite(poloidal_flux_scale)) {
        throw std::invalid_argument(
            "magnetic-gradient target requires complete equilibrium fields");
    }

    const int half_surfaces = equilibrium.ns - 1;
    const std::size_t points = equilibrium.points_per_surface();
    const std::size_t count = equilibrium.half_field_size();
    const auto& bsups =
        equilibrium.half_fields[cumes::EquilibriumSnapshot::BSUPS];
    const auto& bsupu =
        equilibrium.half_fields[cumes::EquilibriumSnapshot::BSUPU];
    const auto& bsupv =
        equilibrium.half_fields[cumes::EquilibriumSnapshot::BSUPV];
    const auto& bsubs =
        equilibrium.half_fields[cumes::EquilibriumSnapshot::BSUBS];
    const auto& bsubu =
        equilibrium.half_fields[cumes::EquilibriumSnapshot::BSUBU];
    const auto& bsubv =
        equilibrium.half_fields[cumes::EquilibriumSnapshot::BSUBV];
    const auto& sqrtg =
        equilibrium.half_fields[cumes::EquilibriumSnapshot::SQRTG];

    MagneticGradientFields result;
    result.field_strength.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
        const double b_squared =
            bsups[i] * bsubs[i] + bsupu[i] * bsubu[i] + bsupv[i] * bsubv[i];
        if (!(b_squared > 0.0) || !std::isfinite(b_squared)) {
            throw std::runtime_error(
                "magnetic-gradient target encountered invalid B squared");
        }
        result.field_strength[i] = std::sqrt(b_squared);
    }

    const auto dtheta_b = magnetic_target_detail::periodic_derivative(
        result.field_strength, half_surfaces, equilibrium.ntheta,
        equilibrium.nzeta, nfp, true);
    const auto dzeta_b = magnetic_target_detail::periodic_derivative(
        result.field_strength, half_surfaces, equilibrium.ntheta,
        equilibrium.nzeta, nfp, false);
    const auto ds_b = magnetic_target_detail::radial_derivative(
        result.field_strength, half_surfaces, points,
        1.0 / static_cast<double>(equilibrium.ns - 1));

    result.b_dot_grad_b.resize(count);
    result.b_cross_grad_s_dot_grad_b.resize(count);
    result.b_cross_grad_toroidal_flux_dot_grad_b.resize(count);
    result.b_cross_grad_psi_p_dot_grad_b.resize(count);
    for (int surface = 0; surface < half_surfaces; ++surface) {
        const double toroidal_flux_prime =
            profiles
                .toroidal_flux_derivative[static_cast<std::size_t>(surface)];
        const double psi_p_prime =
            poloidal_flux_scale *
            profiles
                .poloidal_flux_derivative[static_cast<std::size_t>(surface)];
        for (std::size_t point = 0; point < points; ++point) {
            const std::size_t index =
                static_cast<std::size_t>(surface) * points + point;
            if (!std::isfinite(sqrtg[index]) ||
                std::abs(sqrtg[index]) <= 1.0e-30) {
                throw std::runtime_error(
                    "magnetic-gradient target encountered singular sqrt(g)");
            }
            result.b_dot_grad_b[index] = bsups[index] * ds_b[index] +
                                         bsupu[index] * dtheta_b[index] +
                                         bsupv[index] * dzeta_b[index];
            result.b_cross_grad_s_dot_grad_b[index] =
                1.0 / sqrtg[index] *
                (bsubv[index] * dtheta_b[index] -
                 bsubu[index] * dzeta_b[index]);
            result.b_cross_grad_toroidal_flux_dot_grad_b[index] =
                toroidal_flux_prime * result.b_cross_grad_s_dot_grad_b[index];
            result.b_cross_grad_psi_p_dot_grad_b[index] =
                psi_p_prime * result.b_cross_grad_s_dot_grad_b[index];
            if (!std::isfinite(result.b_dot_grad_b[index]) ||
                !std::isfinite(result.b_cross_grad_s_dot_grad_b[index]) ||
                !std::isfinite(
                    result.b_cross_grad_toroidal_flux_dot_grad_b[index]) ||
                !std::isfinite(result.b_cross_grad_psi_p_dot_grad_b[index])) {
                throw std::runtime_error(
                    "magnetic-gradient target produced a non-finite value");
            }
        }
    }
    return result;
}

}  // namespace cumes_meow_example

#endif  // CUMES_EXAMPLES_MAGNETIC_GRADIENT_TARGET_HPP_
