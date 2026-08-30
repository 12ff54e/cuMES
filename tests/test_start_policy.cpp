// test_start_policy.cpp — host checks for stage-initial solver controls.
#include "cumes/solver/start_policy.hpp"
#include "cumes_test.h"

#include <cmath>

using namespace cumes::test;

static void check_near(double actual, double expected, const char* message) {
    check(std::abs(actual - expected) <= 1e-14, message);
}

int main() {
    check_near(cumes::initial_step_for_stage(0.9, 12, 12, false, 3, 0), 0.9,
               "3-D stages retain the configured step");
    check_near(cumes::initial_step_for_stage(0.9, 0, 1, true, 3, 2), 0.9,
               "free-boundary stages retain the configured step");
    check_near(cumes::initial_step_for_stage(0.9, 0, 1, false, 1, 0), 1.05,
               "single-grid axisym uses the intermediate step");
    check_near(cumes::initial_step_for_stage(0.9, 0, 1, false, 3, 0), 1.02,
               "cold coarse axisym uses the conservative step");
    check_near(cumes::initial_step_for_stage(0.9, 0, 1, false, 3, 1), 1.08,
               "prolonged axisym grids use the larger step");
    check(std::abs(cumes::initial_step_for_stage(0.9F, 0, 1, false, 3, 2) -
                   1.08F) <= 1e-6F,
          "float uses the same axisymmetric step policy");

    if (failures()) {
        std::cout << format("test_start_policy: {} failure(s)\n", failures());
        return 1;
    }
    std::cout << "test_start_policy: all checks passed\n";
    return 0;
}
