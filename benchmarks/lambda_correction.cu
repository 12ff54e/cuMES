// Frozen-geometry lambda correction experiment; no production solver policy.
#include "bench_common.cuh"
#include "cumes/io/checkpoint.hpp"
#include "cumes/runtime/stream.hpp"
#include "cumes/solver/equilibrium_operator.hpp"
#include "cumes/state/seed_state.hpp"
#include "device_bicgstab.cuh"
#include "frozen_lambda.cuh"

#include <array>
#include <iomanip>
#include <iostream>

namespace {

using cumes::SpectralComponent;

__global__ void extract_lambda(const double* d_state,
                               double* d_lambda,
                               int family_size) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 2 * family_size)
        d_lambda[i] = d_state[(i < family_size ? 2 : 4) * family_size + i];
}

__global__ void physical_residual(const double* d_residual,
                                  double* d_lambda,
                                  int ns,
                                  int mnmax,
                                  int ntor) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = ns * mnmax;
    if (i >= 2 * size) return;
    const int j = i % ns;
    const int mode = (i / ns) % mnmax;
    const int m = mode / (ntor + 1), n = mode % (ntor + 1);
    const bool active = j > 0 && (i < size ? m > 0 : n > 0);
    const double factor = (m ? sqrt(2.0) : 1.0) * (n ? sqrt(2.0) : 1.0);
    d_lambda[i] =
        active ? d_residual[(i < size ? 2 : 4) * size + i] * factor : 0.0;
}

__global__ void add_vector(int size,
                           const double* d_base,
                           const double* d_direction,
                           double scale,
                           double* d_result) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) d_result[i] = d_base[i] + scale * d_direction[i];
}

__global__ void negate_vector(int size, double* d_values) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) d_values[i] = -d_values[i];
}

__global__ void update_lambda(double* d_state,
                              const double* d_base,
                              const double* d_direction,
                              double scale,
                              int family_size) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 2 * family_size)
        d_state[(i < family_size ? 2 : 4) * family_size + i] =
            d_base[i] + scale * d_direction[i];
}

__global__ void make_direction(int ns,
                               int mnmax,
                               int ntor,
                               double seed,
                               double* d_direction) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = ns * mnmax;
    if (i >= 2 * size) return;
    const int j = i % ns;
    const int mode = (i / ns) % mnmax;
    const int m = mode / (ntor + 1), n = mode % (ntor + 1);
    const bool active = j > 0 && (i < size ? m > 0 : n > 0);
    d_direction[i] = active ? 1e-5 * sin(seed * (i + 1)) / (1 + m + n) : 0.0;
}

// Each block computes one diagnostic scalar; all vector arithmetic is CUDA.
__global__ void compare_vectors(int size,
                                const double* d_a,
                                const double* d_b,
                                double* d_stats) {
    double sum = 0.0;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        const double a = d_a[i], b = d_b[i];
        if (blockIdx.x == 0) sum += a * a;
        if (blockIdx.x == 1) sum += b * b;
        if (blockIdx.x == 2) sum += (a - b) * (a - b);
        if (blockIdx.x == 3) sum += a * b;
    }
    sum = block_bench::block_sum(sum);
    if (threadIdx.x == 0) d_stats[blockIdx.x] = sum;
}

double merit(const cumes::ControlRecord& rec) {
    if (!rec.status.jacobian_valid || rec.status.invariant_nonfinite)
        return INFINITY;
    return rec.invariant_scaled[0] + rec.invariant_scaled[1] +
           rec.invariant_scaled[2];
}

void print_record(const cumes::ControlRecord& rec) {
    std::cout << '[' << rec.invariant_scaled[0] << ','
              << rec.invariant_scaled[1] << ',' << rec.invariant_scaled[2]
              << ']';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string input = "inputs/w7x.json", restart;
        int iterations = 16;
        bench_common::ArgParser args(argc, argv, "lambda-correction");
        for (int i = 1; i < argc; ++i) {
            if (const char* v = args.need(i, "input"))
                input = v;
            else if (const char* v = args.need(i, "restart"))
                restart = v;
            else if (const char* v = args.need(i, "iterations"))
                iterations = std::atoi(v);
            else
                throw cumes::CumesError("unknown lambda-correction option");
        }
        if (restart.empty() || iterations < 1 || iterations > 256)
            throw cumes::CumesError("requires --restart and iterations 1..256");
        const auto validated = bench_common::load_validated(input, "lambda");
        if (!validated.has_value()) return 2;
        const auto& vp = validated.value();
        if (vp.spec().free_boundary.lfreeb)
            throw cumes::CumesError("lambda probe requires fixed boundary");
        const auto checkpoint = cumes::read_checkpoint(restart);
        if (!checkpoint.has_value())
            throw cumes::CumesError(checkpoint.error());
        auto p = cumes::init_params<double>(vp);
        p.ns = checkpoint.value().ns;
        p.ftol = 0.0;
        if (p.mnmax != checkpoint.value().mnmax)
            throw cumes::CumesError("checkpoint mode count mismatch");
        auto storage = cumes::restart_state<double>(p, vp, checkpoint.value());
        cumes::Stream stream;
        const int family_size = p.ns * p.mnmax;
        const int size = 2 * family_size, blocks = (size + 255) / 256;
        cumes::DeviceBuffer<double> d_base(size), d_rhs(size), d_full(size),
            d_delta(size), d_trial(size), d_result(size), d_direction(size),
            d_jvp(size), d_second(size), d_stats(4);
        block_bench::DeviceBicgstab<double> krylov(size);
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
            lambda_bench::FrozenLambdaOperator<double> frozen(p, vp);
            cumes::EvaluationSchedule schedule;
            schedule.update_iota_chi = true;
            schedule.zero_z_force_m1 = true;
            schedule.refresh_preconditioner = true;
            schedule.reset_constraint_reference = true;
            double norm_rz = 1.0, norm_l = 1.0;
            auto evaluate = [&]() {
                eq.enqueue(1000, 1000, schedule, stream.get(), norm_rz, norm_l);
                cumes::ControlRecord rec;
                cumes::check_cuda(
                    cudaMemcpyAsync(&rec, eq.control_device(), sizeof(rec),
                                    cudaMemcpyDeviceToHost, stream.get()),
                    "control download");
                stream.synchronize();
                return rec;
            };
            auto stats = [&](const double* d_a, const double* d_b) {
                compare_vectors<<<4, 256, 0, stream.get()>>>(size, d_a, d_b,
                                                             d_stats.data());
                std::array<double, 4> result;
                cumes::check_cuda(
                    cudaMemcpyAsync(result.data(), d_stats.data(),
                                    sizeof(result), cudaMemcpyDeviceToHost,
                                    stream.get()),
                    "diagnostics download");
                stream.synchronize();
                return result;
            };
            const auto base = evaluate();
            if (!std::isfinite(merit(base)))
                throw cumes::CumesError("invalid frozen base geometry");
            norm_rz = base.final_f_norm_rz;
            norm_l = base.final_f_norm_l;
            schedule.refresh_preconditioner = false;
            schedule.reset_constraint_reference = false;
            extract_lambda<<<blocks, 256, 0, stream.get()>>>(
                storage.state_slab(), d_base.data(), family_size);
            physical_residual<<<blocks, 256, 0, stream.get()>>>(
                eq.residual_const().data(), d_full.data(), p.ns, p.mnmax,
                p.ntor);
            frozen.set_base(storage.physical_const(), stream.get());
            frozen.enqueue_residual(d_base.data(), d_rhs.data(), stream.get());
            const auto agreement = stats(d_full.data(), d_rhs.data());
            const double relative_error =
                std::sqrt(agreement[2] / std::max(agreement[0], 1e-300));
            // Affinity: F(lambda+d) - F(lambda) = J*d, including ncurr=1.
            make_direction<<<blocks, 256, 0, stream.get()>>>(
                p.ns, p.mnmax, p.ntor, 0.731, d_direction.data());
            add_vector<<<blocks, 256, 0, stream.get()>>>(
                size, d_base.data(), d_direction.data(), 1.0, d_trial.data());
            frozen.enqueue_residual(d_trial.data(), d_result.data(),
                                    stream.get());
            add_vector<<<blocks, 256, 0, stream.get()>>>(
                size, d_result.data(), d_rhs.data(), -1.0, d_result.data());
            frozen.enqueue_jvp(d_direction.data(), d_jvp.data(), stream.get());
            const auto affinity = stats(d_result.data(), d_jvp.data());
            const double affine_error =
                std::sqrt(affinity[2] / std::max(affinity[0], 1e-300));
            make_direction<<<blocks, 256, 0, stream.get()>>>(
                p.ns, p.mnmax, p.ntor, 0.419, d_second.data());
            const double v_ju = stats(d_second.data(), d_jvp.data())[3];
            frozen.enqueue_jvp(d_second.data(), d_jvp.data(), stream.get());
            const double u_jv = stats(d_direction.data(), d_jvp.data())[3];
            std::cout << "{\"ns\":" << p.ns << ",\"iterations\":" << iterations
                      << ",\"map_relative_error\":" << relative_error
                      << ",\"affine_relative_error\":" << affine_error
                      << ",\"symmetry_relative_error\":"
                      << std::abs(v_ju - u_jv) /
                             std::max({std::abs(v_ju), std::abs(u_jv), 1e-300})
                      << ",\"base\":";
            print_record(base);
            if (!(relative_error < 1e-5 && affine_error < 1e-7)) {
                std::cout << ",\"operator_gate\":false}\n";
                return 1;
            }
            auto apply = [&](const double* d_input, double* d_output,
                             cudaStream_t work_stream) {
                frozen.enqueue_jvp(d_input, d_output, work_stream);
                negate_vector<<<blocks, 256, 0, work_stream>>>(size, d_output);
            };
            cumes::stage_detail::ScopedDeviceTimer timer;
            timer.start(stream.get());
            krylov.enqueue(apply, d_rhs.data(), d_delta.data(), iterations,
                           1e-5, stream.get());
            const double correction_ms = timer.stop(stream.get());
            apply(d_delta.data(), d_result.data(), stream.get());
            const auto linear = stats(d_rhs.data(), d_result.data());
            std::cout << ",\"operator_gate\":true,\"correction_ms\":"
                      << correction_ms << ",\"linear_relative_residual\":"
                      << std::sqrt(linear[2] / std::max(linear[0], 1e-300))
                      << ",\"trials\":[";
            bool first = true;
            for (const double scale : {1.0, 0.5, 0.25, 0.125}) {
                update_lambda<<<blocks, 256, 0, stream.get()>>>(
                    storage.state_slab(), d_base.data(), d_delta.data(), scale,
                    family_size);
                timer.start(stream.get());
                const auto trial = evaluate();
                const double evaluation_ms = timer.stop(stream.get());
                if (!first) std::cout << ',';
                first = false;
                std::cout << "{\"scale\":" << scale << ",\"residual\":";
                print_record(trial);
                std::cout << ",\"merit_ratio\":" << merit(trial) / merit(base)
                          << ",\"evaluation_ms\":" << evaluation_ms << '}';
            }
            update_lambda<<<blocks, 256, 0, stream.get()>>>(
                storage.state_slab(), d_base.data(), d_delta.data(), 0.0,
                family_size);
            const auto restored = evaluate();
            std::cout << "],\"restored_merit_ratio\":"
                      << merit(restored) / merit(base) << "}\n";
            return 0;
        });
    } catch (const std::exception& e) {
        std::cerr << "lambda-correction: " << e.what() << '\n';
        return 2;
    }
}
