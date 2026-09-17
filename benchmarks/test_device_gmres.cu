#include "cumes/runtime/stream.hpp"
#include "device_gmres.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

enum class System {
    TRIDIAGONAL,
    DIAGONAL,
    NEAR_BREAKDOWN,
    SINGULAR,
    NONFINITE,
    ROTATION,
    CYCLIC
};
int failures = 0;

void check(bool passed, std::string_view message) {
    if (!passed) {
        std::fprintf(stderr, "FAIL: %.*s\n", static_cast<int>(message.size()),
                     message.data());
        ++failures;
    }
}

template <class T>
__global__ void apply_tridiagonal(int size,
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

template <class T>
__global__ void apply_cyclic(int size, const T* d_input, T* d_output) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) d_output[i] = d_input[(i + size - 1) % size];
}

template <class T>
void exercise_system(System system,
                     int size,
                     int basis,
                     int budget,
                     T amplitude = T(1),
                     bool incremental = false) {
    constexpr bool IS_FLOAT = std::is_same_v<T, float>;
    const T tolerance = IS_FLOAT ? T(1e-5) : T(1e-12);
    const long double error_limit = IS_FLOAT ? 2e-5L : 2e-12L;
    std::vector<T> lower(size, T(0)), diagonal(size, T(2)), upper(size, T(0));
    std::vector<T> solution(size), rhs(size), result(size, T(17));
    for (int i = 0; i < size; ++i) {
        if (system == System::TRIDIAGONAL) {
            lower[i] = T(-0.6 + 0.01 * (i % 7));
            diagonal[i] = T(3.0 + 0.02 * (i % 17));
            upper[i] = T(0.9 - 0.02 * (i % 11));
        } else if (system == System::NEAR_BREAKDOWN) {
            diagonal[i] = T(1);
            upper[i] = IS_FLOAT ? T(1e-3) : T(1e-7);
        } else if (system == System::SINGULAR) {
            diagonal[i] = T(0);
        } else if (system == System::NONFINITE) {
            diagonal[i] = std::numeric_limits<T>::quiet_NaN();
        } else if (system == System::ROTATION) {
            lower[i] = T(-1);
            diagonal[i] = T(0);
            upper[i] = T(1);
        }
        solution[i] = amplitude *
                      (system == System::DIAGONAL
                           ? T(0.125 * (i % 9 - 4))
                           : T(std::sin(0.013 * i) + 0.1 * (i % 7 - 3) + 0.5));
        if (system == System::CYCLIC) solution[i] = i == 0 ? amplitude : T(0);
    }
    for (int i = 0; i < size; ++i) {
        long double value = static_cast<long double>(diagonal[i]) * solution[i];
        if (i > 0)
            value += static_cast<long double>(lower[i]) * solution[i - 1];
        if (i + 1 < size)
            value += static_cast<long double>(upper[i]) * solution[i + 1];
        if (system == System::CYCLIC) value = solution[(i + size - 1) % size];
        rhs[i] = system == System::SINGULAR || system == System::NONFINITE
                     ? T(1)
                     : static_cast<T>(value);
    }

    cumes::DeviceBuffer<T> d_lower(size), d_diagonal(size), d_upper(size);
    cumes::DeviceBuffer<T> d_rhs(size), d_result(size);
    cumes::Stream stream;
    const std::size_t bytes = static_cast<std::size_t>(size) * sizeof(T);
    auto upload = [&](const std::vector<T>& values,
                      cumes::DeviceBuffer<T>& buffer) {
        cumes::check_cuda(cudaMemcpyAsync(buffer.data(), values.data(), bytes,
                                          cudaMemcpyHostToDevice, stream.get()),
                          "GMRES test upload");
    };
    upload(lower, d_lower);
    upload(diagonal, d_diagonal);
    upload(upper, d_upper);
    auto apply = [&](const T* d_input, T* d_output, cudaStream_t compute) {
        if (system == System::CYCLIC) {
            apply_cyclic<<<(size + 255) / 256, 256, 0, compute>>>(size, d_input,
                                                                  d_output);
            return;
        }
        apply_tridiagonal<<<(size + 255) / 256, 256, 0, compute>>>(
            size, d_lower.data(), d_diagonal.data(), d_upper.data(), d_input,
            d_output);
    };
    block_bench::DeviceGmres<T> solver(size, basis);
    block_bench::GmresControl<T> control;
    auto solve = [&]() {
        upload(rhs, d_rhs);
        upload(result, d_result);
        if (incremental) {
            auto read_control = [&]() {
                cumes::check_cuda(
                    cudaMemcpyAsync(&control, solver.control_device(),
                                    sizeof(control), cudaMemcpyDeviceToHost,
                                    stream.get()),
                    "incremental GMRES control");
                stream.synchronize();
            };
            solver.enqueue_start(d_rhs.data(), d_result.data(), tolerance, T(0),
                                 stream.get());
            read_control();
            while (control.active && control.steps < budget) {
                const int count = std::min(basis, budget - control.steps);
                for (int first = 0; first < count;) {
                    const int chunk = std::min(3, count - first);
                    solver.enqueue_steps(apply, first, chunk, stream.get());
                    first += chunk;
                    read_control();
                    if (!control.cycle_active) break;
                }
                solver.enqueue_finish(apply, d_rhs.data(), d_result.data(),
                                      stream.get());
                read_control();
            }
        } else {
            solver.enqueue(apply, d_rhs.data(), d_result.data(), budget,
                           tolerance, stream.get());
        }
        cumes::check_cuda(cudaMemcpyAsync(result.data(), d_result.data(), bytes,
                                          cudaMemcpyDeviceToHost, stream.get()),
                          "GMRES test result");
        cumes::check_cuda(
            cudaMemcpyAsync(&control, solver.control_device(), sizeof(control),
                            cudaMemcpyDeviceToHost, stream.get()),
            "GMRES test control");
        stream.synchronize();
    };
    solve();
    if (system == System::SINGULAR || system == System::NONFINITE) {
        check(!control.converged && !control.active &&
                  control.breakdown == (system == System::SINGULAR ? 1 : 2),
              "singular/nonfinite map reports breakdown, not convergence");
        check(std::all_of(result.begin(), result.end(),
                          [](T x) { return x == T(0); }),
              "failed first Arnoldi step retains zero initial solution");
    } else {
        long double residual_squared = 0, rhs_squared = 0;
        long double error_squared = 0, solution_squared = 0;
        for (int i = 0; i < size; ++i) {
            long double value =
                static_cast<long double>(diagonal[i]) * result[i];
            if (i > 0)
                value += static_cast<long double>(lower[i]) * result[i - 1];
            if (i + 1 < size)
                value += static_cast<long double>(upper[i]) * result[i + 1];
            if (system == System::CYCLIC) value = result[(i + size - 1) % size];
            const long double residual = value - rhs[i];
            const long double error =
                static_cast<long double>(result[i]) - solution[i];
            residual_squared += residual * residual;
            rhs_squared += static_cast<long double>(rhs[i]) * rhs[i];
            error_squared += error * error;
            solution_squared +=
                static_cast<long double>(solution[i]) * solution[i];
        }
        const long double residual = std::sqrt(residual_squared / rhs_squared);
        const long double error = std::sqrt(error_squared / solution_squared);
        if (budget == 1) {
            check(
                !control.converged && control.active &&
                    control.breakdown == 0 && control.steps == 1 &&
                    residual > tolerance,
                "budget exhaustion is distinct from convergence and breakdown");
        } else {
            check(std::isfinite(residual) && residual <= error_limit,
                  "independent true relative residual");
            check(std::isfinite(error) && error <= error_limit,
                  "manufactured known-solution error");
            check(control.converged && !control.active &&
                      control.breakdown == 0 && control.steps > 0 &&
                      control.steps <= budget,
                  "converged control record with bounded Arnoldi steps");
            check(std::isfinite(control.residual_norm) &&
                      control.residual_norm <= control.target_norm,
                  "stopping uses freshly evaluated b-Ax norm");
        }
        if (system == System::DIAGONAL)
            check(control.steps == 1 && control.cycles == 1,
                  "identity multiple terminates after one Arnoldi step");
        if (system == System::NEAR_BREAKDOWN)
            check(control.steps <= 3, "near-invariant Krylov space converges");
        if (system == System::TRIDIAGONAL && !IS_FLOAT && basis == 8 &&
            budget > 1)
            check(control.cycles > 1, "manufactured solve exercises restart");
        if (system == System::ROTATION)
            check(control.steps == 2,
                  "skew rotation with zero first Arnoldi diagonal converges");
        if (system == System::CYCLIC)
            check(control.steps == size && control.cycles == 1,
                  "cyclic permutation exercises every restart basis column");
        std::printf(
            "%s system=%d n=%d basis=%d scale=%.1e: steps=%d cycles=%d "
            "residual=%.3Le error=%.3Le\n",
            IS_FLOAT ? "float" : "double", static_cast<int>(system), size,
            basis, static_cast<double>(amplitude), control.steps,
            control.cycles, residual, error);
    }
    if (system == System::NONFINITE) return;
    // Reuse every previously populated workspace; initialization must discard
    // its old Arnoldi basis, QR coefficients and stopped/breakdown flags.
    std::fill(rhs.begin(), rhs.end(), T(0));
    std::fill(result.begin(), result.end(), T(17));
    solve();
    check(std::all_of(result.begin(), result.end(),
                      [](T x) { return x == T(0); }) &&
              control.converged && !control.active && control.breakdown == 0 &&
              control.steps == 0 && control.cycles == 0 &&
              control.residual_norm == T(0),
          "zero RHS reinitializes reused workspace and needs zero steps");
}

template <class T>
void run_precision() {
    exercise_system<T>(System::TRIDIAGONAL, 4099, 8, 64);
    exercise_system<T>(System::TRIDIAGONAL, 513, 64, 64);
    exercise_system<T>(System::TRIDIAGONAL, 257, 8, 1);
    exercise_system<T>(System::DIAGONAL, 513, 8, 7);
    exercise_system<T>(System::DIAGONAL, 1, 1, 3);
    exercise_system<T>(System::NEAR_BREAKDOWN, 257, 32, 16);
    exercise_system<T>(System::ROTATION, 2, 8, 8);
    exercise_system<T>(System::SINGULAR, 17, 8, 8);
    exercise_system<T>(System::NONFINITE, 17, 8, 8);
    const T tiny = sizeof(T) == sizeof(float) ? T(1e-25) : T(1e-180);
    const T huge = sizeof(T) == sizeof(float) ? T(1e25) : T(1e200);
    exercise_system<T>(System::DIAGONAL, 129, 8, 7, tiny);
    exercise_system<T>(System::DIAGONAL, 129, 8, 7, huge);
    exercise_system<T>(System::TRIDIAGONAL, 4099, 8, 64, T(1), true);
    exercise_system<T>(System::TRIDIAGONAL, 513, 300, 600, T(1), true);
    exercise_system<T>(System::TRIDIAGONAL, 257, 8, 1, T(1), true);
    exercise_system<T>(System::DIAGONAL, 513, 8, 7, T(1), true);
    exercise_system<T>(System::SINGULAR, 17, 8, 8, T(1), true);
    exercise_system<T>(System::NONFINITE, 17, 8, 8, T(1), true);
    exercise_system<T>(System::CYCLIC, 300, 300, 300, T(1), true);
}

}  // namespace

int main() {
    try {
        run_precision<double>();
        run_precision<float>();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::printf("device GMRES: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
