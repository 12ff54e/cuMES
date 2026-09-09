#include "bench_common.cuh"
#include "cumes/io/checkpoint.hpp"
#include "cumes/runtime/stream.hpp"
#include "cumes/state/seed_state.hpp"
#include "device_bicgstab.cuh"
#include "newton_correction.cuh"

#include <array>
#include <charconv>
#include <iomanip>
#include <iostream>

namespace {

__global__ void direction_kernel(int size, double* d_direction) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) d_direction[i] = sin(0.137 * (i + 1));
}

__global__ void stats_kernel(int size,
                             const double* d_a,
                             const double* d_b,
                             double* d_stats) {
    double sum = 0;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        const double a = d_a[i], b = d_b[i];
        if (blockIdx.x == 0) sum += a * a;
        if (blockIdx.x == 1) sum += b * b;
        if (blockIdx.x == 2) sum += (a - b) * (a - b);
    }
    sum = block_bench::block_sum(sum);
    if (threadIdx.x == 0) d_stats[blockIdx.x] = sum;
}

void number(double value) {
    if (std::isfinite(value))
        std::cout << value;
    else
        std::cout << "null";
}

void triple(const cumes::ControlRecord& rec) {
    std::cout << '[';
    for (int i = 0; i < 3; ++i) {
        if (i) std::cout << ',';
        number(rec.invariant_scaled[i]);
    }
    std::cout << ']';
}

double merit(const cumes::ControlRecord& rec) {
    if (!rec.status.jacobian_valid || rec.status.invariant_nonfinite)
        return INFINITY;
    return rec.invariant_scaled[0] + rec.invariant_scaled[1] +
           rec.invariant_scaled[2];
}

int integer(std::string_view text) {
    int value = 0;
    const auto [last, error] =
        std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || last != text.data() + text.size())
        throw cumes::CumesError("invalid integer option");
    return value;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string input = "inputs/w7x.json", restart;
        int iterations = 32, basis = 32;
        double epsilon = 1e-8;
        auto scheme = block_bench::DifferenceScheme::CENTRAL;
        bench_common::ArgParser args(argc, argv, "newton-correction");
        for (int i = 1; i < argc; ++i) {
            if (const char* v = args.need(i, "input"))
                input = v;
            else if (const char* v = args.need(i, "restart"))
                restart = v;
            else if (const char* v = args.need(i, "iterations"))
                iterations = integer(v);
            else if (const char* v = args.need(i, "basis"))
                basis = integer(v);
            else if (const char* v = args.need(i, "difference")) {
                if (std::string_view(v) == "central")
                    scheme = block_bench::DifferenceScheme::CENTRAL;
                else if (std::string_view(v) == "forward")
                    scheme = block_bench::DifferenceScheme::FORWARD;
                else
                    throw cumes::CumesError("invalid difference scheme");
            } else if (const char* v = args.need(i, "epsilon")) {
                char* end = nullptr;
                epsilon = std::strtod(v, &end);
                if (!end || *end != '\0')
                    throw cumes::CumesError("invalid epsilon");
            } else
                throw cumes::CumesError("unknown newton-correction option");
        }
        if (restart.empty() || iterations < 1 || iterations > 256 ||
            basis < 1 || basis > 128 || !(epsilon > 0) ||
            !std::isfinite(epsilon))
            throw cumes::CumesError("invalid options or missing --restart");
        const auto validated = bench_common::load_validated(input, "newton");
        if (!validated.has_value()) return 2;
        const auto& vp = validated.value();
        if (vp.spec().free_boundary.lfreeb)
            throw cumes::CumesError("Newton probe requires fixed boundary");
        const auto checkpoint = cumes::read_checkpoint(restart);
        if (!checkpoint.has_value())
            throw cumes::CumesError(checkpoint.error());
        auto p = cumes::init_params<double>(vp, false);
        p.ns = checkpoint.value().ns;
        p.ftol = 0;
        if (p.mnmax != checkpoint.value().mnmax ||
            !std::any_of(vp.spec().stages.begin(), vp.spec().stages.end(),
                         [&](const auto& stage) {
                             return stage.radial_surfaces == std::size_t(p.ns);
                         }))
            throw cumes::CumesError("checkpoint shape not configured");
        auto storage =
            cumes::restart_state<double>(p, vp, checkpoint.value(), false);
        cumes::Stream stream;
        const int size = 6 * p.ns * p.mnmax;
        cumes::DeviceBuffer<double> d_direction(size), d_first(size),
            d_second(size), d_stats(3);
        std::cout << std::setprecision(17);
        return cumes::run_in_stage_arena<double>(p, [&](cumes::DeviceArena&
                                                            arena) {
            bench_common::OperatorStack<double> stack(p, vp, arena);
            stack.transform.bind_stream(stream.get());
            const auto transform = stack.axisym
                                       ? std::optional<std::reference_wrapper<
                                             cumes::SpectralOperator<double>>>(
                                             std::ref(*stack.axisym))
                                       : std::nullopt;
            cumes::EquilibriumOperator<double> eq(
                p, storage, stack.profiles, stack.transform, stack.rs,
                stack.geometry, std::ref(arena), transform);
            block_bench::NewtonCorrection<double> newton(p, eq, storage, basis,
                                                         scheme);
            cumes::EvaluationSchedule schedule;
            schedule.update_iota_chi = true;
            schedule.zero_z_force_m1 = true;
            schedule.refresh_preconditioner = true;
            schedule.reset_constraint_reference = true;
            auto record = [&]() {
                cumes::ControlRecord rec;
                cumes::check_cuda(
                    cudaMemcpyAsync(&rec, eq.control_device(), sizeof(rec),
                                    cudaMemcpyDeviceToHost, stream.get()),
                    "Newton control");
                stream.synchronize();
                return rec;
            };
            auto relative = [&](const double* d_a, const double* d_b) {
                stats_kernel<<<3, 256, 0, stream.get()>>>(size, d_a, d_b,
                                                          d_stats.data());
                std::array<double, 3> stats;
                cumes::check_cuda(
                    cudaMemcpyAsync(stats.data(), d_stats.data(), sizeof(stats),
                                    cudaMemcpyDeviceToHost, stream.get()),
                    "Newton diagnostics");
                stream.synchronize();
                return std::sqrt(stats[2] / std::max(stats[0], 1e-300));
            };
            eq.enqueue(1000, 1000, schedule, stream.get());
            const auto base = record();
            if (!std::isfinite(merit(base)) || base.invariant_scaled[1] >= 1e-6)
                throw cumes::CumesError("requires a valid stable-gauge state");
            newton.prepare(schedule, base.final_f_norm_rz, base.final_f_norm_l,
                           stream.get());
            direction_kernel<<<(size + 255) / 256, 256, 0, stream.get()>>>(
                size, d_direction.data());
            newton.enqueue_jvp(d_direction.data(), d_first.data(), epsilon,
                               stream.get());
            newton.enqueue_jvp(d_direction.data(), d_second.data(), epsilon / 2,
                               stream.get());
            const double fd_error = relative(d_first.data(), d_second.data());
            newton.enqueue_restore(stream.get());
            const auto before = record();
            bool restored = true;
            for (int i = 0; i < 3; ++i)
                restored &=
                    before.invariant_scaled[i] == base.invariant_scaled[i];
            std::cout << "{\"difference\":\""
                      << (scheme == block_bench::DifferenceScheme::CENTRAL
                              ? "central"
                              : "forward")
                      << "\",\"ns\":" << p.ns
                      << ",\"iterations\":" << iterations
                      << ",\"basis\":" << basis << ",\"epsilon\":" << epsilon
                      << ",\"jvp_halving_error\":";
            number(fd_error);
            std::cout << ",\"base\":";
            triple(base);
            if (!(fd_error < 1e-4) || !restored) {
                std::cout << ",\"operator_gate\":false}\n";
                return 1;
            }
            cumes::stage_detail::ScopedDeviceTimer timer;
            timer.start(stream.get());
            newton.enqueue_solve(iterations, 1e-3, epsilon, stream.get());
            const double solve_ms = timer.stop(stream.get());
            block_bench::GmresControl<double> control;
            cumes::check_cuda(
                cudaMemcpy(&control, newton.control_device(), sizeof(control),
                           cudaMemcpyDeviceToHost),
                "Newton GMRES result");
            newton.enqueue_jvp(newton.correction_device(), d_first.data(),
                               epsilon, stream.get());
            const double linear_error =
                relative(newton.rhs_device(), d_first.data());
            newton.set_difference_scheme(
                block_bench::DifferenceScheme::CENTRAL);
            newton.enqueue_jvp(newton.correction_device(), d_second.data(),
                               epsilon, stream.get());
            const double scheme_error =
                relative(d_second.data(), d_first.data());
            newton.set_difference_scheme(scheme);
            std::cout << ",\"operator_gate\":true,\"solve_ms\":" << solve_ms
                      << ",\"krylov_steps\":" << control.steps
                      << ",\"krylov_cycles\":" << control.cycles
                      << ",\"krylov_converged\":" << control.converged
                      << ",\"krylov_breakdown\":" << control.breakdown
                      << ",\"linear_relative_residual\":";
            number(linear_error);
            std::cout << ",\"correction_central_error\":";
            number(scheme_error);
            std::cout << ",\"trials\":[";
            bool first = true;
            for (double scale : {1.0, 0.5, 0.25, 0.125}) {
                newton.enqueue_trial(scale, stream.get());
                const auto trial = record();
                if (!first) std::cout << ',';
                first = false;
                std::cout << "{\"scale\":" << scale << ",\"residual\":";
                triple(trial);
                std::cout << ",\"merit_ratio\":";
                number(merit(trial) / merit(base));
                std::cout << '}';
            }
            newton.enqueue_restore(stream.get());
            const auto after = record();
            for (int i = 0; i < 3; ++i)
                restored &=
                    after.invariant_scaled[i] == base.invariant_scaled[i];
            std::cout << "],\"restored_exact\":"
                      << (restored ? "true" : "false") << "}\n";
            return restored ? 0 : 1;
        });
    } catch (const std::exception& error) {
        std::cerr << "newton-correction: " << error.what() << '\n';
        return 2;
    }
}
