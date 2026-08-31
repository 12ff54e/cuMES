// test_seed_envelope.cpp — host checks for the shaped cold-start envelope.
#include "cumes/state/seed_envelope.hpp"
#include "cumes_test.h"

#include <cmath>

using namespace cumes::test;

static void check_near(double actual, double expected, const char* message) {
    check(std::abs(actual - expected) <= 1e-14, message);
}

int main() {
    check_near(cumes::default_seed_envelope(12, false, 99, 3), 0.12,
               "multigrid fixed-boundary 3-D selects the shaped seed");
    check_near(cumes::default_seed_envelope(12, false, 99, 1), 0.129,
               "single-grid fixed-boundary 3-D selects its fitted seed");
    check_near(cumes::default_seed_envelope(0, false, 5, 3), -0.07,
               "coarse axisymmetric starts select the fitted seed");
    check_near(cumes::default_seed_envelope(0, false, 55, 1), 0.0,
               "fine axisymmetric starts retain the reference seed");
    check_near(cumes::default_seed_envelope(4, true, 15, 2), 0.12,
               "coarse 3-D free-boundary uses the shaped envelope");
    check_near(cumes::default_seed_envelope(6, true, 51, 1), 0.03,
               "fine 3-D free-boundary uses the conservative envelope");
    check_near(cumes::default_seed_envelope(0, true, 16, 1), 0.0,
               "axisymmetric free-boundary retains the reference envelope");

    check_near(cumes::seed_radial_weight(1, 0.25, 0.0), 0.5,
               "zero correction reproduces the m=1 regular weight");
    check_near(cumes::seed_radial_weight(2, 0.25, 0.0), 0.25,
               "zero correction reproduces the m=2 regular weight");
    check_near(cumes::seed_radial_weight(2, 0.25, 0.12), 0.2725,
               "interior correction follows the documented envelope");
    check(std::abs(cumes::seed_radial_weight<float>(2, 0.25F, 0.12F) -
                   0.2725F) <= 1e-6F,
          "float uses the same shaped envelope");

    for (int m = 1; m <= 8; ++m) {
        check_near(cumes::seed_radial_weight(m, 1.0, 0.12), 1.0,
                   "the shaped seed preserves the exact LCFS");
        check_near(cumes::seed_radial_weight(m, 0.0, 0.12), 0.0,
                   "the shaped seed preserves axis regularity");
    }

    if (failures()) {
        std::cout << format("test_seed_envelope: {} failure(s)\n", failures());
        return 1;
    }
    std::cout << "test_seed_envelope: all checks passed\n";
    return 0;
}
