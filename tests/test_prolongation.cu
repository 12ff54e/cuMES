// test_prolongation.cu — direct CPU/GPU checks for linear and cubic radial
// multigrid transfer, including odd-m decomposition and endpoint contracts.
#include "cumes/numerics/prolongation.hpp"
#include "cumes_test_cuda_helper.cuh"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using namespace cumes::test;

template <typename T>
static T scalxc_cpu(int j, int ns) {
    return T(1) /
           std::max(std::sqrt(T(j) / T(ns - 1)), std::sqrt(T(1) / T(ns - 1)));
}

template <typename T>
static T expected_value(const std::vector<T>& old_state,
                        int component,
                        int mode,
                        int j_new,
                        int ns_old,
                        int ns_new,
                        int mnmax,
                        bool cubic) {
    const bool odd = mode % 2 == 1;
    const T s = T(j_new) / T(ns_new - 1);
    const int j0 = (j_new * (ns_old - 1)) / (ns_new - 1);
    const int j1 = std::min(j0 + 1, ns_old - 1);
    const T t = std::clamp(s * T(ns_old - 1) - T(j0), T(0), T(1));
    const std::size_t family =
        static_cast<std::size_t>(component) * mnmax * ns_old;
    const auto xs = [&](int j) {
        const T value = old_state[family + mode * ns_old + j];
        return odd ? value * scalxc_cpu<T>(j, ns_old) : value;
    };
    const auto axis_regular = [&](int j) {
        return odd && j == 0 ? T(2) * xs(1) - xs(2) : xs(j);
    };

    const T y0 = axis_regular(j0);
    const T y1 = axis_regular(j1);
    T value = (T(1) - t) * y0 + t * y1;
    if (cubic && j0 != j1) {
        const T ym1 = j0 > 0 ? axis_regular(j0 - 1) : T(2) * y0 - y1;
        const T yp2 = j1 + 1 < ns_old ? axis_regular(j1 + 1) : T(2) * y1 - y0;
        value = y0 + T(0.5) * t *
                         (y1 - ym1 +
                          t * (T(2) * ym1 - T(5) * y0 + T(4) * y1 - yp2 +
                               t * (T(3) * (y0 - y1) + yp2 - ym1)));
    }
    value *=
        odd ? std::max(std::sqrt(s), std::sqrt(T(1) / T(ns_new - 1))) : T(1);
    return odd && j_new == 0 ? T(0) : value;
}

template <typename T>
static void run_case(bool cubic) {
    constexpr int ns_old = 5;
    constexpr int ns_new = 8;  // deliberately not an integer refinement
    constexpr int mnmax = 2;   // m=0 even, m=1 odd (ntor=0)
    constexpr int families = cumes::SPECTRAL_COMPONENT_COUNT;

    DeviceParams<T> p_old{};
    p_old.ns = ns_old;
    p_old.mnmax = mnmax;
    p_old.ntor = 0;
    DeviceParams<T> p_new = p_old;
    p_new.ns = ns_new;

    std::vector<T> input(families * mnmax * ns_old);
    for (int f = 0; f < families; ++f) {
        for (int mode = 0; mode < mnmax; ++mode) {
            for (int j = 0; j < ns_old; ++j) {
                const T base = T(0.4 * (f + 1) + 0.2 * mode);
                T value = base + T(0.08) * T(j) + T(0.013) * T(j * j);
                if (mode == 1) {
                    value *= T(1) / scalxc_cpu<T>(j, ns_old);
                    if (j == 0) value = T(0);
                }
                input[(f * mnmax + mode) * ns_old + j] = value;
            }
        }
    }

    cumes::SpectralStorage<T> old_state(ns_old, mnmax);
    cc(cudaMemcpy(old_state.state_slab(), input.data(),
                  input.size() * sizeof(T), cudaMemcpyHostToDevice),
       "prolongation input upload");
    cumes::SpectralStorage<T> new_state =
        cumes::Prolongation<T>{}.enqueue(p_new, old_state, p_old, 0, cubic);

    std::vector<T> actual(families * mnmax * ns_new);
    std::vector<T> velocity(actual.size(), T(1));
    cc(cudaMemcpy(actual.data(), new_state.state_slab(),
                  actual.size() * sizeof(T), cudaMemcpyDeviceToHost),
       "prolongation output download");
    cc(cudaMemcpy(velocity.data(), new_state.velocity_slab(),
                  velocity.size() * sizeof(T), cudaMemcpyDeviceToHost),
       "prolongation velocity download");

    double error = 0.0;
    for (int f = 0; f < families; ++f) {
        for (int mode = 0; mode < mnmax; ++mode) {
            for (int j = 0; j < ns_new; ++j) {
                const T expected = expected_value(input, f, mode, j, ns_old,
                                                  ns_new, mnmax, cubic);
                error = std::max(
                    error,
                    std::abs(static_cast<double>(
                        actual[(f * mnmax + mode) * ns_new + j] - expected)));
            }
        }
    }
    const double tolerance = sizeof(T) == sizeof(double) ? 2e-13 : 3e-6;
    check(error <= tolerance,
          std::string(cubic ? "cubic" : "linear") +
              (sizeof(T) == sizeof(double) ? " double" : " float") +
              " CPU/GPU agreement");
    check(std::all_of(velocity.begin(), velocity.end(),
                      [](T value) { return value == T(0); }),
          "prolongation resets every velocity family");

    for (int f = 0; f < families; ++f) {
        for (int mode = 0; mode < mnmax; ++mode) {
            expect_near(actual[(f * mnmax + mode) * ns_new + ns_new - 1],
                        input[(f * mnmax + mode) * ns_old + ns_old - 1],
                        tolerance, "prolongation preserves the exact LCFS");
        }
        expect_near(actual[(f * mnmax + 1) * ns_new], 0.0, tolerance,
                    "prolongation zeroes the odd-m axis");
    }
}

int main() {
    run_case<double>(false);
    run_case<double>(true);
    run_case<float>(false);
    run_case<float>(true);
    return summary();
}
