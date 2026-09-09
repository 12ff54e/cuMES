#include "bench_common.cuh"
#include "coarse_correction.cuh"
#include "cumes/io/checkpoint.hpp"
#include "cumes/runtime/stream.hpp"
#include "cumes/solver/start_policy.hpp"
#include "cumes/state/seed_state.hpp"

#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

namespace {
double merit(const cumes::ControlRecord& r) {
    if (!r.status.jacobian_valid || r.status.invariant_nonfinite)
        return std::numeric_limits<double>::infinity();
    return r.invariant_scaled[0] + r.invariant_scaled[1] +
           r.invariant_scaled[2];
}
void emit(std::ostream& out, const cumes::ControlRecord& r) {
    out << "{\"invariant\":[" << r.invariant_scaled[0] << ','
        << r.invariant_scaled[1] << ',' << r.invariant_scaled[2]
        << "],\"jacobian_valid\":" << r.status.jacobian_valid << '}';
}
}  // namespace

int main(int argc, char** argv) {
    try {
        std::string input = "inputs/solovev.json", restart, output;
        int coarse_ns = 0, steps = 8, ordinary_passes = 16;
        double delta = 0.5;
        bench_common::ArgParser args(argc, argv, "coarse-correction");
        for (int i = 1; i < argc; ++i) {
            if (const char* v = args.need(i, "input"))
                input = v;
            else if (const char* v = args.need(i, "restart"))
                restart = v;
            else if (const char* v = args.need(i, "out"))
                output = v;
            else if (const char* v = args.need(i, "coarse-ns"))
                coarse_ns = atoi(v);
            else if (const char* v = args.need(i, "steps"))
                steps = atoi(v);
            else if (const char* v = args.need(i, "delta"))
                delta = atof(v);
            else if (const char* v = args.need(i, "ordinary-passes"))
                ordinary_passes = atoi(v);
            else
                throw cumes::CumesError("coarse-correction: unknown option");
        }
        if (restart.empty() || !(delta > 0) || !std::isfinite(delta) ||
            steps < 1)
            throw cumes::CumesError("coarse-correction: invalid options");
        auto validated =
            bench_common::load_validated(input, "coarse-correction");
        if (!validated.has_value()) return 2;
        const auto& vp = validated.value();
        if (vp.spec().free_boundary.lfreeb)
            throw cumes::CumesError("coarse-correction: fixed boundary only");
        auto checkpoint = cumes::read_checkpoint(restart);
        if (!checkpoint.has_value())
            throw cumes::CumesError(checkpoint.error());
        auto p = cumes::init_params<double>(vp, false);
        p.ns = checkpoint.value().ns;
        if (!coarse_ns) coarse_ns = (p.ns + 1) / 2;
        int stage = -1;
        for (std::size_t g = 0; g < vp.spec().stages.size(); ++g)
            if (vp.spec().stages[g].radial_surfaces == std::size_t(p.ns))
                stage = int(g);
        if (stage < 0)
            throw cumes::CumesError("checkpoint grid not configured");
        p.delt =
            cumes::initial_step_for_stage(p.delt, p.ntor, p.nzeta, false, p.ns,
                                          int(vp.spec().stages.size()), stage);
        p.ftol = 0;
        auto storage = cumes::restart_state<double>(p, vp, checkpoint.value());
        auto ordinary = cumes::restart_state<double>(p, vp, checkpoint.value());
        cumes::Stream stream;
        block_bench::CoarseCorrection<double> correction(p, vp, coarse_ns,
                                                         steps, delta);
        const int size = 6 * p.ns * p.mnmax;
        cumes::DeviceBuffer<double> d_base(size), d_f0(size), d_zero(size);
        d_zero.zero();
        std::ostringstream json;
        json << std::setprecision(17) << "{\"ns\":" << p.ns
             << ",\"coarse_ns\":" << coarse_ns << ",\"steps\":" << steps
             << ",\"delta\":" << delta;
        cumes::run_in_stage_arena<double>(p, [&](cumes::DeviceArena& arena) {
            bench_common::OperatorStack<double> stack(p, vp, arena);
            stack.transform.bind_stream(stream.get());
            const auto op = stack.axisym
                                ? std::optional<std::reference_wrapper<
                                      cumes::SpectralOperator<double>>>(
                                      std::ref(*stack.axisym))
                                : std::nullopt;
            cumes::EquilibriumOperator<double> eq(
                p, storage, stack.profiles, stack.transform, stack.rs,
                stack.geometry, std::ref(arena), op);
            cumes::EvaluationSchedule schedule;
            schedule.update_iota_chi = true;
            schedule.zero_z_force_m1 = true;
            schedule.refresh_preconditioner = true;
            schedule.reset_constraint_reference = true;
            double norm_rz = 1, norm_l = 1;
            auto evaluate = [&]() {
                eq.enqueue(1000, 1000, schedule, stream.get(), norm_rz, norm_l);
                cumes::ControlRecord rec;
                cumes::check_cuda(
                    cudaMemcpyAsync(&rec, eq.control_device(), sizeof(rec),
                                    cudaMemcpyDeviceToHost, stream.get()),
                    "fine record");
                stream.synchronize();
                return rec;
            };
            const auto base = evaluate();
            if (!std::isfinite(merit(base)) || base.invariant_scaled[1] >= 1e-6)
                throw cumes::CumesError("requires valid stable m1 gauge");
            norm_rz = base.final_f_norm_rz;
            norm_l = base.final_f_norm_l;
            schedule.refresh_preconditioner = false;
            schedule.reset_constraint_reference = false;
            auto copy = [&](double* d_to, const double* d_from) {
                cumes::check_cuda(
                    cudaMemcpyAsync(d_to, d_from,
                                    std::size_t(size) * sizeof(double),
                                    cudaMemcpyDeviceToDevice, stream.get()),
                    "fine vector copy");
            };
            copy(d_base.data(), storage.state_slab());
            copy(d_f0.data(), eq.residual().data());
            correction.enqueue(d_base.data(), d_zero.data(), stream.get());
            std::vector<double> direction(size);
            cumes::check_cuda(
                cudaMemcpyAsync(direction.data(), correction.direction_device(),
                                std::size_t(size) * sizeof(double),
                                cudaMemcpyDeviceToHost, stream.get()),
                "zero FAS check");
            stream.synchronize();
            int zero_failed = 0;
            cumes::check_cuda(
                cudaMemcpy(&zero_failed, correction.failed_device(),
                           sizeof(int), cudaMemcpyDeviceToHost),
                "zero FAS status");
            if (zero_failed)
                throw cumes::CumesError("zero-forcing coarse state invalid");
            for (double value : direction)
                if (value != 0)
                    throw cumes::CumesError("zero forcing moved FAS state");
            cumes::stage_detail::ScopedDeviceTimer timer;
            timer.start(stream.get());
            correction.enqueue(d_base.data(), d_f0.data(), stream.get());
            const double cycle_ms = timer.stop(stream.get());
            int failed = 0;
            cumes::check_cuda(cudaMemcpy(&failed, correction.failed_device(),
                                         sizeof(int), cudaMemcpyDeviceToHost),
                              "coarse status");
            cumes::check_cuda(
                cudaMemcpy(direction.data(), correction.direction_device(),
                           std::size_t(size) * sizeof(double),
                           cudaMemcpyDeviceToHost),
                "coarse direction");
            for (int mode = 0; mode < p.mnmax; ++mode) {
                const int m = mode / (p.ntor + 1), n = mode % (p.ntor + 1);
                for (int j = 0; j < p.ns; ++j) {
                    const int i = mode * p.ns + j;
                    for (int c = 0; c < 6; ++c)
                        if (!block_bench::correction_detail::active(
                                c, m, n, j, p.ns, false) &&
                            direction[c * p.ns * p.mnmax + i] != 0)
                            throw cumes::CumesError(
                                "coarse correction changed inactive entry");
                    if (m == 1 && direction[3 * p.ns * p.mnmax + i] !=
                                      direction[4 * p.ns * p.mnmax + i])
                        throw cumes::CumesError(
                            "coarse correction changed m1 gauge");
                }
            }
            auto best = base;
            double best_alpha = 0;
            json << ",\"base\":";
            emit(json, base);
            json << ",\"trials\":[";
            timer.start(stream.get());
            int trial = 0;
            for (double alpha : {1.0, 0.5, 0.25}) {
                correction.enqueue_trial(d_base.data(), storage.state_slab(),
                                         alpha, stream.get());
                const auto rec = evaluate();
                if (trial++) json << ',';
                json << "{\"alpha\":" << alpha << ",\"record\":";
                emit(json, rec);
                json << '}';
                if (merit(rec) < merit(best)) {
                    best = rec;
                    best_alpha = alpha;
                }
            }
            const double trials_ms = timer.stop(stream.get());
            copy(storage.state_slab(), d_base.data());
            const auto replay = evaluate();
            for (int c = 0; c < 3; ++c)
                if (base.invariant_scaled[c] != replay.invariant_scaled[c])
                    throw cumes::CumesError("frozen base replay changed");
            json << "],\"cycle_ms\":" << cycle_ms
                 << ",\"trials_ms\":" << trials_ms
                 << ",\"coarse_failed\":" << failed
                 << ",\"coarse_passes\":" << correction.coarse_passes()
                 << ",\"best_alpha\":" << best_alpha << ",\"best\":";
            emit(json, best);
            auto ordinary_p = p;
            ordinary_p.max_iter = ordinary_passes;
            cumes::SolverBench bench;
            bench.enabled = true;
            double ordinary_ms = 0;
            const auto result = cumes::StageSolver<double>::run(
                ordinary_p, vp, ordinary, stream.get(), std::ref(bench),
                std::nullopt, nullptr, nullptr, true, std::ref(ordinary_ms),
                false, false);
            copy(storage.state_slab(), ordinary.state_slab());
            const auto ordinary_rec = evaluate();
            json << ",\"ordinary_passes\":" << bench.pass_wall_us.size()
                 << ",\"ordinary_iterations\":" << result.iterations
                 << ",\"ordinary_ms\":" << ordinary_ms << ",\"ordinary\":";
            emit(json, ordinary_rec);
            json
                << ",\"zero_forcing_exact\":true,\"frozen_replay_exact\":true}";
            return 0;
        });
        if (output.empty())
            std::printf("%s\n", json.str().c_str());
        else {
            std::ofstream out(output);
            if (!out)
                throw cumes::CumesError("coarse-correction: cannot open --out");
            out << json.str() << '\n';
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "coarse-correction: %s\n", error.what());
        return 2;
    }
}
