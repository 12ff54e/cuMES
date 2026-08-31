// Optimizer-side target helper for VMEC-compatible plasma size diagnostics.
//
// This deliberately does not belong to the cuMES solver library: cuMES maps a
// problem to an equilibrium, while the optimization application decides which
// scalar observables to derive from that equilibrium.
#ifndef CUMES_EXAMPLES_PLASMA_SIZE_TARGET_HPP_
#define CUMES_EXAMPLES_PLASMA_SIZE_TARGET_HPP_

#include "cumes/io/equilibrium_snapshot.hpp"
#include "cumes/io/input_params.hpp"

#include <cmath>
#include <cstddef>
#include <numbers>
#include <stdexcept>

namespace cumes_meow_example {

struct PlasmaSize {
    double cross_section_area = 0.0;
    double volume = 0.0;
    double major_radius = 0.0;
    double minor_radius = 0.0;
    double aspect_ratio = 0.0;
};

// Reconstruct the final LCFS on the snapshot's uniform angular grid and apply
// the VMEC definitions
//
//   cross_area = 2*pi*abs(<R * dZ/dtheta>)
//   volume     = 2*pi^2*abs(<R^2 * dZ/dtheta>)
//   Rmajor     = volume / (2*pi*cross_area)
//   Aminor     = sqrt(cross_area/pi).
//
// The average is over one field period and the full poloidal angle. Periodic
// uniform quadrature is spectrally exact for the resolved Fourier products.
inline PlasmaSize calculate_plasma_size(
    const cumes::EquilibriumSnapshot& equilibrium,
    const cumes::InputParams& input) {
    if (equilibrium.ns < 1 || equilibrium.ntheta < 2 || equilibrium.nzeta < 1 ||
        input.mpol < 1 || input.ntor < 0) {
        throw std::invalid_argument(
            "plasma-size target requires a nonempty equilibrium grid");
    }
    if (equilibrium.ntheta != input.ntheta ||
        equilibrium.nzeta != input.nzeta ||
        equilibrium.mnmax != input.mpol * (input.ntor + 1)) {
        throw std::invalid_argument(
            "plasma-size target received inconsistent equilibrium metadata");
    }
    const std::size_t family_size = equilibrium.family_size();
    for (const auto& family : equilibrium.families) {
        if (family.size() != family_size) {
            throw std::invalid_argument(
                "plasma-size target received an incomplete spectral state");
        }
    }

    constexpr double TWO_PI = 2.0 * std::numbers::pi;
    const int boundary = equilibrium.ns - 1;
    double cross_section_sum = 0.0;
    double volume_sum = 0.0;

    for (int k = 0; k < equilibrium.nzeta; ++k) {
        const double zeta = TWO_PI * static_cast<double>(k) /
                            static_cast<double>(equilibrium.nzeta);
        for (int l = 0; l < equilibrium.ntheta; ++l) {
            const double theta = TWO_PI * static_cast<double>(l) /
                                 static_cast<double>(equilibrium.ntheta);
            double r = 0.0;
            double z_theta = 0.0;

            for (int m = 0; m < input.mpol; ++m) {
                const double cos_m = std::cos(static_cast<double>(m) * theta);
                const double sin_m = std::sin(static_cast<double>(m) * theta);
                for (int n = 0; n <= input.ntor; ++n) {
                    const double cos_n =
                        std::cos(static_cast<double>(n) * zeta);
                    const double sin_n =
                        std::sin(static_cast<double>(n) * zeta);
                    const int mode = m * (input.ntor + 1) + n;
                    const std::size_t index =
                        static_cast<std::size_t>(mode) * equilibrium.ns +
                        boundary;

                    r += equilibrium.families[cumes::EquilibriumSnapshot::RMNCC]
                                             [index] *
                             cos_m * cos_n +
                         equilibrium.families[cumes::EquilibriumSnapshot::RMNSS]
                                             [index] *
                             sin_m * sin_n;
                    z_theta +=
                        static_cast<double>(m) *
                        (equilibrium.families[cumes::EquilibriumSnapshot::ZMNSC]
                                             [index] *
                             cos_m * cos_n -
                         equilibrium.families[cumes::EquilibriumSnapshot::ZMNCS]
                                             [index] *
                             sin_m * sin_n);
                }
            }

            cross_section_sum += r * z_theta;
            volume_sum += r * r * z_theta;
        }
    }

    const double point_count =
        static_cast<double>(equilibrium.ntheta) * equilibrium.nzeta;
    PlasmaSize result;
    result.cross_section_area =
        TWO_PI * std::abs(cross_section_sum / point_count);
    result.volume = 2.0 * std::numbers::pi * std::numbers::pi *
                    std::abs(volume_sum / point_count);
    if (!(result.cross_section_area > 0.0) ||
        !std::isfinite(result.cross_section_area) ||
        !std::isfinite(result.volume)) {
        throw std::runtime_error(
            "plasma-size target produced degenerate boundary geometry");
    }
    result.major_radius = result.volume / (TWO_PI * result.cross_section_area);
    result.minor_radius =
        std::sqrt(result.cross_section_area / std::numbers::pi);
    result.aspect_ratio = result.major_radius / result.minor_radius;
    return result;
}

}  // namespace cumes_meow_example

#endif  // CUMES_EXAMPLES_PLASMA_SIZE_TARGET_HPP_
