#include "cumes/runtime/stream.hpp"
#include "device_bicgstab.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

int failures = 0;

void check(bool passed, std::string_view message) {
    if (!passed) {
        std::fprintf(stderr, "FAIL: %.*s\n", static_cast<int>(message.size()),
                     message.data());
        ++failures;
    }
}

template <typename T>
__global__ void tridiagonal_apply(int size,
                                  const T* d_lower,
                                  const T* d_diagonal,
                                  const T* d_upper,
                                  const T* d_input,
                                  T* d_output) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    T value = d_diagonal[i] * d_input[i];
    if (i > 0) value += d_lower[i] * d_input[i - 1];
    if (i + 1 < size) value += d_upper[i] * d_input[i + 1];
    d_output[i] = value;
}

template <typename T>
void exercise_system(int size, bool exact_diagonal) {
    constexpr bool IS_FLOAT = std::is_same_v<T, float>;
    const T tolerance = IS_FLOAT ? T(1e-5) : T(1e-12);
    const long double error_limit = IS_FLOAT ? 5e-5L : 5e-12L;
    const int budget = exact_diagonal ? 7 : 128;
    std::vector<T> lower(size), diagonal(size), upper(size), solution(size);
    std::vector<T> rhs(size), result(size, T(17));
    for (int i = 0; i < size; ++i) {
        lower[i] = exact_diagonal ? T(0) : T(-0.6 + 0.01 * (i % 7));
        diagonal[i] = exact_diagonal ? T(2) : T(3.0 + 0.02 * (i % 17));
        upper[i] = exact_diagonal ? T(0) : T(0.9 - 0.02 * (i % 11));
        solution[i] = exact_diagonal
                          ? T(0.125 * (i % 9 - 4))
                          : T(std::sin(0.013 * i) + 0.1 * (i % 7 - 3) + 0.5);
    }
    for (int i = 0; i < size; ++i) {
        long double value = static_cast<long double>(diagonal[i]) * solution[i];
        if (i > 0)
            value += static_cast<long double>(lower[i]) * solution[i - 1];
        if (i + 1 < size)
            value += static_cast<long double>(upper[i]) * solution[i + 1];
        rhs[i] = static_cast<T>(value);
    }

    cumes::DeviceBuffer<T> d_lower(size), d_diagonal(size), d_upper(size);
    cumes::DeviceBuffer<T> d_rhs(size), d_result(size);
    cumes::Stream stream;
    const std::size_t bytes = static_cast<std::size_t>(size) * sizeof(T);
    auto upload = [&](const std::vector<T>& values,
                      cumes::DeviceBuffer<T>& buffer) {
        cumes::check_cuda(cudaMemcpyAsync(buffer.data(), values.data(), bytes,
                                          cudaMemcpyHostToDevice, stream.get()),
                          "BiCGStab test upload");
    };
    upload(lower, d_lower);
    upload(diagonal, d_diagonal);
    upload(upper, d_upper);

    auto apply = [&](const T* d_input, T* d_output, cudaStream_t compute) {
        tridiagonal_apply<<<(size + 255) / 256, 256, 0, compute>>>(
            size, d_lower.data(), d_diagonal.data(), d_upper.data(), d_input,
            d_output);
    };
    block_bench::DeviceBicgstab<T> solver(size);
    block_bench::BicgControl<T> control;
    auto solve = [&]() {
        upload(rhs, d_rhs);
        upload(result, d_result);  // the solver must initialize its own x
        solver.enqueue(apply, d_rhs.data(), d_result.data(), budget, tolerance,
                       stream.get());
        cumes::check_cuda(cudaMemcpyAsync(result.data(), d_result.data(), bytes,
                                          cudaMemcpyDeviceToHost, stream.get()),
                          "BiCGStab test solution copy");
        cumes::check_cuda(
            cudaMemcpyAsync(&control, solver.control_device(), sizeof(control),
                            cudaMemcpyDeviceToHost, stream.get()),
            "BiCGStab test control copy");
        cumes::check_cuda(cudaStreamSynchronize(stream.get()),
                          "BiCGStab test completion");
    };
    solve();

    long double residual_squared = 0, rhs_squared = 0;
    long double error_squared = 0, solution_squared = 0;
    for (int i = 0; i < size; ++i) {
        long double value = static_cast<long double>(diagonal[i]) * result[i];
        if (i > 0) value += static_cast<long double>(lower[i]) * result[i - 1];
        if (i + 1 < size)
            value += static_cast<long double>(upper[i]) * result[i + 1];
        const long double residual = value - rhs[i];
        const long double error =
            static_cast<long double>(result[i]) - solution[i];
        residual_squared += residual * residual;
        rhs_squared += static_cast<long double>(rhs[i]) * rhs[i];
        error_squared += error * error;
        solution_squared += static_cast<long double>(solution[i]) * solution[i];
    }
    const long double relative_residual =
        std::sqrt(residual_squared / rhs_squared);
    const long double relative_error =
        std::sqrt(error_squared / solution_squared);
    check(std::isfinite(relative_residual) && relative_residual <= error_limit,
          "manufactured system true residual");
    check(std::isfinite(relative_error) && relative_error <= error_limit,
          "manufactured system known-solution error");
    check(control.active == 0 && control.iterations > 0 &&
              control.iterations < budget,
          "manufactured system converges before the fixed budget");
    check(std::isfinite(control.norm_squared) &&
              control.norm_squared <=
                  control.target_squared *
                      (T(1) + T(8) * std::numeric_limits<T>::epsilon()),
          "recursive residual meets the requested tolerance");
    if (exact_diagonal) {
        check(result == solution && control.iterations == 1 &&
                  control.norm_squared == T(0),
              "zero intermediate residual preserves the exact alpha update");
    }
    std::printf("%s %s n=%d: iterations=%d true_residual=%.3Le error=%.3Le\n",
                IS_FLOAT ? "float" : "double",
                exact_diagonal ? "2I" : "nonsymmetric tridiagonal", size,
                control.iterations, relative_residual, relative_error);

    // Reuse the populated workspace for a zero RHS. No scalar recurrence or
    // iterate from the previous solve may survive initialization.
    std::fill(rhs.begin(), rhs.end(), T(0));
    std::fill(result.begin(), result.end(), T(17));
    solve();
    check(std::all_of(result.begin(), result.end(),
                      [](T value) { return value == T(0); }) &&
              control.active == 0 && control.iterations == 0 &&
              control.norm_squared == T(0) && control.target_squared == T(0),
          "reused workspace solves zero RHS without an iteration");
}

}  // namespace

int main() {
    try {
        exercise_system<double>(4099, false);
        exercise_system<float>(4099, false);
        exercise_system<double>(513, true);
        exercise_system<float>(513, true);
        exercise_system<double>(1, true);
        exercise_system<float>(1, true);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::printf("device BiCGStab: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
