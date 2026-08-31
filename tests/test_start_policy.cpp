// test_start_policy.cpp — host checks for stage-initial solver controls.
#include "cumes/physics/free_boundary_policy.hpp"
#include "cumes/solver/start_policy.hpp"
#include "cumes_test.h"

#include <cmath>

using namespace cumes::test;

static void check_near(double actual, double expected, const char* message) {
    check(std::abs(actual - expected) <= 1e-14, message);
}

int main() {
    check_near(cumes::control_policy::VACUUM_ACTIVATION_RESIDUAL, 3.0e-2,
               "vacuum activates at the qualified residual gate");
    check_near(cumes::initial_step_for_stage(0.9, 12, 12, false, 99, 3, 0), 0.9,
               "3-D stages retain the configured step");
    check_near(cumes::initial_step_for_stage(0.7, 4, 36, true, 15, 2, 0), 0.85,
               "coarse 3-D free-boundary uses the tuned step");
    check_near(cumes::initial_step_for_stage(1.0, 6, 36, true, 51, 1, 0), 1.0,
               "fine 3-D free-boundary retains the configured step");
    check_near(cumes::initial_step_for_stage(0.9, 0, 1, true, 16, 2, 0), 0.9,
               "axisymmetric free-boundary retains the configured step");
    check_near(cumes::initial_step_for_stage(0.9, 0, 1, false, 55, 1, 0), 1.05,
               "single-grid axisym uses the intermediate step");
    check_near(cumes::initial_step_for_stage(0.9, 0, 1, false, 5, 3, 0), 1.02,
               "cold coarse axisym uses the conservative step");
    check_near(cumes::initial_step_for_stage(0.9, 0, 1, false, 11, 3, 1), 1.08,
               "prolonged axisym grids use the larger step");
    check(std::abs(cumes::initial_step_for_stage(0.9F, 0, 1, false, 55, 3, 2) -
                   0.9F) <= 1e-6F,
          "float retains its configured stable step");
    check(std::abs(cumes::initial_step_for_stage(0.7F, 4, 36, true, 15, 2, 0) -
                   0.7F) <= 1e-6F,
          "free-boundary float retains its configured step");

    if (failures()) {
        std::cout << format("test_start_policy: {} failure(s)\n", failures());
        return 1;
    }
    std::cout << "test_start_policy: all checks passed\n";
    return 0;
}
