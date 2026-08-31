// Optimizer-side QS/QH/QA least-squares residuals from a cuMES equilibrium.
// cuMES supplies equilibrium fields and radial flux functions; this helper
// owns the target definition and never feeds target policy back into solver
// convergence.
#ifndef CUMES_EXAMPLES_QUASISYMMETRY_TARGET_HPP_
#define CUMES_EXAMPLES_QUASISYMMETRY_TARGET_HPP_

#include "cumes/io/equilibrium_profiles.hpp"
#include "cumes/io/equilibrium_snapshot.hpp"
#include "cumes/io/input_params.hpp"
#include "magnetic_gradient_target.hpp"
#include "plasma_size_target.hpp"

#include <cmath>
#include <cstddef>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace cumes_meow_example {

enum class QsFluxGradient {
    // psi = signed toroidal flux/(2*pi), the conventional QS definition.
    NORMALIZED_TOROIDAL,
    // psi_p = signed poloidal flux/(2*pi), the requested alternative.
    NORMALIZED_POLOIDAL,
};

struct QuasisymmetryTargetSpec {
    int helicity_m = 1;
    // Physical toroidal helicity N. A per-field-period mode must first be
    // multiplied by nfp.
    int helicity_n = 0;
    std::vector<int> half_surface_indices;
    // Empty means unit weight on every selected surface.
    std::vector<double> surface_weights;
    QsFluxGradient flux_gradient = QsFluxGradient::NORMALIZED_POLOIDAL;
};

struct QuasisymmetryTarget {
    // Pointwise weighted residuals, surface-major in the order requested by
    // half_surface_indices. sum(residuals^2) == value.
    std::vector<double> residuals;
    std::vector<double> surface_values;
    double value = 0.0;
};

struct CompositeQuasisymmetryTarget {
    QuasisymmetryTarget qs;
    PlasmaSize plasma_size;
    double iota_integral = 0.0;
    std::vector<double> residuals;
    double value = 0.0;
};

inline std::vector<int> all_half_grid_surfaces(int ns) {
    if (ns < 2) { throw std::invalid_argument("QS target requires ns >= 2"); }
    std::vector<int> surfaces(static_cast<std::size_t>(ns - 1));
    std::iota(surfaces.begin(), surfaces.end(), 0);
    return surfaces;
}

namespace qs_target_detail {

inline double orientation_from_jacobian(const std::vector<double>& sqrtg,
                                        std::size_t offset,
                                        std::size_t points) {
    double signed_sum = 0.0;
    double magnitude_sum = 0.0;
    for (std::size_t point = 0; point < points; ++point) {
        const double value = sqrtg[offset + point];
        if (!std::isfinite(value) || value == 0.0) {
            throw std::runtime_error(
                "QS target encountered an invalid Jacobian");
        }
        signed_sum += value;
        magnitude_sum += std::abs(value);
    }
    if (!(magnitude_sum > 0.0) || std::abs(signed_sum) < 0.5 * magnitude_sum) {
        throw std::runtime_error(
            "QS target encountered inconsistent Jacobian orientation");
    }
    return std::copysign(1.0, signed_sum);
}

inline double residual_square_sum(const std::vector<double>& residuals) {
    double value = 0.0;
    for (double residual : residuals) value += residual * residual;
    return value;
}

}  // namespace qs_target_detail

// Evaluate
//   f_QS = sum_j w_j <q_QS^2>_j,
// where
//   q_QS = [(N-iota*M)(B x grad(psi)) dot grad(B)
//           +(M*G+N*I) B dot grad(B)] / B^3.
//
// The flux-surface average is discretized with |sqrt(g)| on the native
// uniform angular grid. I=<B_theta> and G=<B_zeta> are the half-grid radial
// flux functions published by the solver.
inline QuasisymmetryTarget calculate_quasisymmetry_target(
    const cumes::EquilibriumSnapshot& equilibrium,
    const cumes::EquilibriumProfiles& profiles,
    int nfp,
    const QuasisymmetryTargetSpec& spec) {
    if (!equilibrium.has_derived_fields() ||
        !profiles.has_half_grid_profiles(equilibrium.ns) || nfp < 1 ||
        spec.half_surface_indices.empty() ||
        (spec.helicity_m == 0 && spec.helicity_n == 0)) {
        throw std::invalid_argument(
            "QS target requires complete fields, surfaces, and helicity");
    }
    if (!spec.surface_weights.empty() &&
        spec.surface_weights.size() != spec.half_surface_indices.size()) {
        throw std::invalid_argument(
            "QS target surface weights have the wrong extent");
    }

    const auto magnetic =
        calculate_magnetic_gradient_fields(equilibrium, profiles, nfp);
    const auto& sqrtg =
        equilibrium.half_fields[cumes::EquilibriumSnapshot::SQRTG];
    const std::size_t points = equilibrium.points_per_surface();
    const double two_pi = 2.0 * std::numbers::pi;

    QuasisymmetryTarget result;
    result.residuals.reserve(spec.half_surface_indices.size() * points);
    result.surface_values.reserve(spec.half_surface_indices.size());

    for (std::size_t requested = 0;
         requested < spec.half_surface_indices.size(); ++requested) {
        const int surface = spec.half_surface_indices[requested];
        if (surface < 0 || surface >= equilibrium.ns - 1) {
            throw std::out_of_range("QS target surface is outside half grid");
        }
        const double surface_weight = spec.surface_weights.empty()
                                          ? 1.0
                                          : spec.surface_weights[requested];
        if (!(surface_weight >= 0.0) || !std::isfinite(surface_weight)) {
            throw std::invalid_argument(
                "QS target surface weights must be finite and nonnegative");
        }

        const std::size_t radial = static_cast<std::size_t>(surface);
        const std::size_t offset = radial * points;
        const double orientation =
            qs_target_detail::orientation_from_jacobian(sqrtg, offset, points);
        const double public_flux_prime =
            spec.flux_gradient == QsFluxGradient::NORMALIZED_TOROIDAL
                ? profiles.toroidal_flux_derivative[radial]
                : profiles.poloidal_flux_derivative[radial];
        const double flux_prime = orientation * public_flux_prime / two_pi;
        const double iota = profiles.rotational_transform[radial];
        const double i_covariant = profiles.poloidal_covariant_field[radial];
        const double g_covariant = profiles.toroidal_covariant_field[radial];
        if (!std::isfinite(flux_prime) || !std::isfinite(iota) ||
            !std::isfinite(i_covariant) || !std::isfinite(g_covariant)) {
            throw std::runtime_error(
                "QS target encountered a non-finite radial flux function");
        }

        double jacobian_sum = 0.0;
        for (std::size_t point = 0; point < points; ++point) {
            jacobian_sum += std::abs(sqrtg[offset + point]);
        }
        double surface_value = 0.0;
        for (std::size_t point = 0; point < points; ++point) {
            const std::size_t index = offset + point;
            const double b = magnetic.field_strength[index];
            const double cross_flux =
                flux_prime * magnetic.b_cross_grad_s_dot_grad_b[index];
            const double numerator =
                (static_cast<double>(spec.helicity_n) -
                 iota * static_cast<double>(spec.helicity_m)) *
                    cross_flux +
                (static_cast<double>(spec.helicity_m) * g_covariant +
                 static_cast<double>(spec.helicity_n) * i_covariant) *
                    magnetic.b_dot_grad_b[index];
            const double q_qs = numerator / (b * b * b);
            const double quadrature_weight =
                surface_weight * std::abs(sqrtg[index]) / jacobian_sum;
            const double residual = std::sqrt(quadrature_weight) * q_qs;
            if (!std::isfinite(residual)) {
                throw std::runtime_error(
                    "QS target produced a non-finite residual");
            }
            result.residuals.push_back(residual);
            surface_value += residual * residual;
        }
        result.surface_values.push_back(surface_value);
        result.value += surface_value;
    }
    return result;
}

inline double integrate_iota(const cumes::EquilibriumProfiles& profiles) {
    if (profiles.rotational_transform.empty()) {
        throw std::invalid_argument("iota integral requires a half grid");
    }
    double sum = 0.0;
    for (double value : profiles.rotational_transform) {
        if (!std::isfinite(value)) {
            throw std::runtime_error(
                "iota integral encountered a non-finite value");
        }
        sum += value;
    }
    return sum / static_cast<double>(profiles.rotational_transform.size());
}

inline CompositeQuasisymmetryTarget calculate_qh_target(
    const cumes::EquilibriumSnapshot& equilibrium,
    const cumes::EquilibriumProfiles& profiles,
    const cumes::InputParams& input,
    const QuasisymmetryTargetSpec& spec,
    double target_aspect_ratio) {
    if (spec.helicity_n == 0 || !std::isfinite(target_aspect_ratio) ||
        !(target_aspect_ratio > 0.0)) {
        throw std::invalid_argument(
            "QH target requires nonzero N and a positive target aspect");
    }
    CompositeQuasisymmetryTarget result;
    result.qs =
        calculate_quasisymmetry_target(equilibrium, profiles, input.nfp, spec);
    result.plasma_size = calculate_plasma_size(equilibrium, input);
    result.iota_integral = integrate_iota(profiles);
    result.residuals = result.qs.residuals;
    result.residuals.push_back(result.plasma_size.aspect_ratio -
                               target_aspect_ratio);
    result.value = qs_target_detail::residual_square_sum(result.residuals);
    return result;
}

inline CompositeQuasisymmetryTarget calculate_qa_target(
    const cumes::EquilibriumSnapshot& equilibrium,
    const cumes::EquilibriumProfiles& profiles,
    const cumes::InputParams& input,
    const QuasisymmetryTargetSpec& spec,
    double target_aspect_ratio,
    double target_iota_integral) {
    if (spec.helicity_n != 0 || !std::isfinite(target_aspect_ratio) ||
        !(target_aspect_ratio > 0.0) || !std::isfinite(target_iota_integral)) {
        throw std::invalid_argument(
            "QA target requires N=0 and finite aspect/iota targets");
    }
    CompositeQuasisymmetryTarget result;
    result.qs =
        calculate_quasisymmetry_target(equilibrium, profiles, input.nfp, spec);
    result.plasma_size = calculate_plasma_size(equilibrium, input);
    result.iota_integral = integrate_iota(profiles);
    result.residuals = result.qs.residuals;
    result.residuals.push_back(result.plasma_size.aspect_ratio -
                               target_aspect_ratio);
    result.residuals.push_back(result.iota_integral - target_iota_integral);
    result.value = qs_target_detail::residual_square_sum(result.residuals);
    return result;
}

}  // namespace cumes_meow_example

#endif  // CUMES_EXAMPLES_QUASISYMMETRY_TARGET_HPP_
