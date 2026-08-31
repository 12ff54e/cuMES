#include "cumes_test.h"
#include "plasma_size_target.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>

int main() {
    using cumes::test::check;

    constexpr double major_radius = 4.0;
    constexpr double minor_radius = 1.25;

    cumes::InputParams input;
    input.mpol = 2;
    input.ntor = 0;
    input.nfp = 1;
    input.ntheta = 18;
    input.nzeta = 1;

    cumes::EquilibriumSnapshot equilibrium;
    equilibrium.ns = 2;
    equilibrium.mnmax = 2;
    equilibrium.ntheta = input.ntheta;
    equilibrium.nzeta = input.nzeta;
    for (auto& family : equilibrium.families) {
        family.assign(equilibrium.family_size(), 0.0);
    }
    const auto index = [&](int m) {
        return static_cast<std::size_t>(m) * equilibrium.ns +
               (equilibrium.ns - 1);
    };
    equilibrium.families[cumes::EquilibriumSnapshot::RMNCC][index(0)] =
        major_radius;
    equilibrium.families[cumes::EquilibriumSnapshot::RMNCC][index(1)] =
        minor_radius;
    equilibrium.families[cumes::EquilibriumSnapshot::ZMNSC][index(1)] =
        minor_radius;

    const auto size =
        cumes_meow_example::calculate_plasma_size(equilibrium, input);
    const double expected_area = std::numbers::pi * minor_radius * minor_radius;
    const double expected_volume =
        2.0 * std::numbers::pi * major_radius * expected_area;
    check(std::abs(size.cross_section_area - expected_area) < 1.0e-12,
          "plasma-size target: circular cross-section area");
    check(std::abs(size.volume - expected_volume) < 1.0e-12,
          "plasma-size target: circular torus volume");
    check(std::abs(size.major_radius - major_radius) < 1.0e-12,
          "plasma-size target: major radius");
    check(std::abs(size.minor_radius - minor_radius) < 1.0e-12,
          "plasma-size target: minor radius");
    check(std::abs(size.aspect_ratio - major_radius / minor_radius) < 1.0e-12,
          "plasma-size target: aspect ratio");

    bool rejected = false;
    try {
        auto invalid = input;
        invalid.ntor = 1;
        static_cast<void>(
            cumes_meow_example::calculate_plasma_size(equilibrium, invalid));
    } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "plasma-size target: inconsistent metadata is rejected");

    return cumes::test::summary();
}
