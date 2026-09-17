// test_prolongation.cu — direct CPU/GPU checks for linear and cubic radial
// multigrid transfer, including odd-m decomposition and endpoint contracts.
#include "cumes/numerics/prolongation.hpp"
#include "cumes/runtime/stream.hpp"
#ifdef CUMES_HAVE_BSPLINE_PROLONGATION
#include "cumes/numerics/bspline_matrix.hpp"
#endif
#include "cumes_test_cuda_helper.cuh"

#include <algorithm>
#include <cmath>
#include <span>
#include <string>
#include <vector>

using namespace cumes::test;

template <typename T>
static T scalxc_cpu(int j, int ns) {
    return T(1) /
           std::max(std::sqrt(T(j) / T(ns - 1)), std::sqrt(T(1) / T(ns - 1)));
}

template <typename T>
static void run_case(cumes::RadialInterpolation interpolation,
                     bool precompute_bspline = false) {
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
    std::vector<double> bspline_matrix;
#ifdef CUMES_HAVE_BSPLINE_PROLONGATION
    if (interpolation == cumes::RadialInterpolation::BSPLINE &&
        precompute_bspline) {
        bspline_matrix =
            cumes::cubic_bspline_interpolation_matrix(ns_old, ns_new);
    }
#endif
    cumes::SpectralStorage<T> new_state = cumes::Prolongation<T>{}.enqueue(
        p_new, old_state, p_old, 0, interpolation, bspline_matrix);

#ifdef CUMES_HAVE_BSPLINE_PROLONGATION
    if (interpolation == cumes::RadialInterpolation::BSPLINE &&
        bspline_matrix.empty()) {
        bspline_matrix =
            cumes::cubic_bspline_interpolation_matrix(ns_old, ns_new);
    }
#endif

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
            const auto values = std::span<const T>(input).subspan(
                (f * mnmax + mode) * ns_old, ns_old);
            for (int j = 0; j < ns_new; ++j) {
                const auto weights =
                    bspline_matrix.empty()
                        ? std::span<const double>{}
                        : std::span<const double>(bspline_matrix)
                              .subspan(j * ns_old, ns_old);
                const T expected = cumes::interpolate_radial_value(
                    values, ns_new, j, mode % 2 == 1, interpolation, weights);
                error = std::max(
                    error,
                    std::abs(static_cast<double>(
                        actual[(f * mnmax + mode) * ns_new + j] - expected)));
            }
        }
    }
    const double tolerance = sizeof(T) == sizeof(double) ? 2e-13 : 3e-6;
    check(error <= tolerance,
          std::string(interpolation == cumes::RadialInterpolation::LINEAR
                          ? "linear"
                      : interpolation == cumes::RadialInterpolation::CATMULL_ROM
                          ? "Catmull-Rom"
                      : precompute_bspline ? "precomputed B-spline"
                                           : "fallback B-spline") +
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

#ifdef CUMES_HAVE_BSPLINE_PROLONGATION
template <typename T>
static void run_large_bspline_case(bool lasym) {
    // The heliotron's 128 -> 256 stage transfers a 256 KiB double matrix.
    // Exercise the production nonblocking stream with independently known
    // constant/linear profiles in every symmetric and asymmetric family.
    constexpr int ns_old = 128;
    constexpr int ns_new = 256;
    constexpr int ntor = 3;
    constexpr int mnmax = 2 * (ntor + 1);
    DeviceParams<T> old_params{};
    old_params.ns = ns_old;
    old_params.ntor = ntor;
    old_params.mnmax = mnmax;
    old_params.lasym = lasym;
    auto new_params = old_params;
    new_params.ns = ns_new;
    cumes::SpectralStorage<T> old_state(ns_old, mnmax, {}, lasym);
    std::vector<T> input(old_state.components() * mnmax * ns_old);
    for (int profile = 0; profile < old_state.components() * mnmax; ++profile) {
        const bool odd = profile % mnmax >= ntor + 1;
        for (int j = 0; j < ns_old; ++j) {
            const T s = T(j) / T(ns_old - 1);
            input[profile * ns_old + j] =
                (T(1) + T(profile) / T(10) + s) * (odd ? std::sqrt(s) : T(1));
        }
    }
    cc(cudaMemcpy(old_state.state_slab(), input.data(),
                  input.size() * sizeof(T), cudaMemcpyHostToDevice),
       "large prolongation input");
    cumes::Stream stream;
    auto output = cumes::Prolongation<T>{}.enqueue(
        new_params, old_state, old_params, stream.get(),
        cumes::RadialInterpolation::BSPLINE);
    std::vector<T> actual(output.components() * mnmax * ns_new);
    cc(cudaMemcpy(actual.data(), output.state_slab(), actual.size() * sizeof(T),
                  cudaMemcpyDeviceToHost),
       "large prolongation result");
    double error = 0;
    for (int profile = 0; profile < output.components() * mnmax; ++profile) {
        const bool odd = profile % mnmax >= ntor + 1;
        for (int j = 0; j < ns_new; ++j) {
            const double s = double(j) / (ns_new - 1);
            const double expected =
                (1 + double(profile) / 10 + s) * (odd ? std::sqrt(s) : 1);
            error = std::max(error,
                             std::abs(actual[profile * ns_new + j] - expected));
        }
    }
    check(error < (sizeof(T) == sizeof(double) ? 2e-11 : 2e-5),
          "large nonblocking B-spline transfer preserves analytic profiles");
}
#endif

int main() {
    run_case<double>(cumes::RadialInterpolation::LINEAR);
    run_case<double>(cumes::RadialInterpolation::CATMULL_ROM);
#ifdef CUMES_HAVE_BSPLINE_PROLONGATION
    run_case<double>(cumes::RadialInterpolation::BSPLINE, true);
    run_case<double>(cumes::RadialInterpolation::BSPLINE, false);
    check(cumes::cubic_bspline_interpolation_matrix(3, 5).size() == 15,
          "B-spline matrix supports the minimum coarse grid");
    for (bool lasym : {false, true}) {
        run_large_bspline_case<double>(lasym);
        run_large_bspline_case<float>(lasym);
    }
#endif
    run_case<float>(cumes::RadialInterpolation::LINEAR);
    run_case<float>(cumes::RadialInterpolation::CATMULL_ROM);
    return summary();
}
