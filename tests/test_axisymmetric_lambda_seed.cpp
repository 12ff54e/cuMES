// test_axisymmetric_lambda_seed.cpp — host checks for the geometry predictor.
#include "cumes/state/axisymmetric_lambda_seed.hpp"
#include "cumes/state/seed_envelope.hpp"
#include "cumes_test.h"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace cumes::test;

int main() {
    check(cumes::default_axisymmetric_lambda_seed(0, false) == 0.65,
          "fixed-boundary axisym selects the lambda predictor");
    check(cumes::default_axisymmetric_lambda_seed(1, false) == 0.0,
          "3-D starts retain zero lambda");
    check(cumes::default_axisymmetric_lambda_seed(0, true) == 1.0,
          "axisymmetric free-boundary uses the full geometry predictor");

    constexpr int ns = 5;
    constexpr int mpol = 6;
    constexpr double major_radius = 4.0;
    constexpr double minor_r = 1.0;
    constexpr double minor_z = 1.5;
    std::vector<double> rmncc(ns * mpol);
    std::vector<double> zmnsc(ns * mpol);
    std::vector<double> lmnsc(ns * mpol);
    std::vector<double> boundary_r(mpol);
    std::vector<double> boundary_z(mpol);
    boundary_r[0] = major_radius;
    boundary_r[1] = minor_r;
    boundary_z[1] = minor_z;
    for (int j = 0; j < ns; ++j) {
        const double s = static_cast<double>(j) / (ns - 1);
        rmncc[j] = major_radius;
        rmncc[ns + j] = minor_r * std::sqrt(s);
        zmnsc[ns + j] = minor_z * std::sqrt(s);
    }

    check(cumes::seed_axisymmetric_lambda<double>(ns, mpol, rmncc, zmnsc, lmnsc,
                                                  boundary_r, boundary_z,
                                                  major_radius, 0.0, 0.65),
          "regular elliptical geometry accepts the lambda predictor");
    for (int m = 1; m < mpol; ++m) {
        check(lmnsc[m * ns] == 0.0, "lambda remains regular at the axis");
        for (int j = 1; j < ns; ++j) {
            const double s = static_cast<double>(j) / (ns - 1);
            const double a = minor_r * std::sqrt(s);
            const double rho =
                (major_radius -
                 std::sqrt(major_radius * major_radius - a * a)) /
                a;
            const double sign = (m % 2 == 1) ? 1.0 : -1.0;
            const double expected = 0.65 * 2.0 * sign * std::pow(rho, m) / m;
            check(std::abs(lmnsc[m * ns + j] - expected) <= 2.0e-13,
                  "elliptical predictor matches the analytic Fourier series");
        }
    }

    std::vector<double> invalid_r(ns * mpol);
    std::vector<double> untouched(ns * mpol, 0.25);
    check(!cumes::seed_axisymmetric_lambda<double>(
              ns, mpol, invalid_r, zmnsc, untouched, boundary_r, boundary_z,
              major_radius, 0.0, 0.65),
          "zero-radius geometry is rejected");
    check(std::all_of(untouched.begin(), untouched.end(),
                      [](double x) { return x == 0.25; }),
          "rejected geometry leaves lambda untouched");

    check(std::abs(cumes::seed_radial_weight_derivative(1, 0.25, -0.07) -
                   0.9825) <= 1.0e-14,
          "radial-envelope derivative includes the shaping correction");

    if (failures()) {
        std::cout << format("test_axisymmetric_lambda_seed: {} failure(s)\n",
                            failures());
        return 1;
    }
    std::cout << "test_axisymmetric_lambda_seed: all checks passed\n";
    return 0;
}
