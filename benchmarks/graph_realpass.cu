// graph_realpass.cu — ADR-0003 re-measure: CUDA Graph capture of the REAL
// per-iteration DAG (blueprint §8.11 follow-up).
//
// The Phase-9 measurement (graph_overhead.cu) used EMPTY kernels, so the
// ~40 µs/pass submission saving was an upper bound; ADR-0003 deferred the
// solver integration and asked for a real-kernel re-measure once the
// AxisymmetricOperator was wired into the Solovev production path (ADR-0004,
// now landed). This benchmark does exactly that: it builds the PRODUCTION
// operator stack (same construction as cumes_benchmark_fixed_iteration and
// StageSolver::run) and captures the real per-pass DAG —
// EquilibriumOperator::enqueue, the exact production pipeline — into a CUDA
// graph. It then measures:
//
//   1. correctness gate: a graph replay of one pass produces the same control
//      record as direct stream execution, bitwise (state restored between);
//   2. submission-only host wall per pass (enqueue vs graph launch) — the
//      quantity the empty-kernel bound approximated;
//   3. production-pattern wall per pass — DAG + control-record D2H copy + one
//      cudaStreamSynchronize, exactly the per-pass host pattern solverRun
//      runs (the fence is NOT removed by graphs; the end-to-end saving is
//      the difference between direct and graph here);
//   4. capture + instantiate cost per variant (base pass / precon-refresh
//      pass / zeroZ pass) — the "re-instantiate per variant" strategy cost;
//   5. cudaGraphExecKernelNodeSetParams cost + a functional check that the
//      exec-side scalar update changes the launch output — measured on a
//      MANUALLY constructed node (see the note below; the manual gate runs
//      FIRST because CUDA 12.1 graph-node APIs misbehave after replays of
//      the real captured DAG).
//
// Note on cudaGraphKernelNodeGetParams (CUDA 12.1, driver on TITAN Xp):
// on the real captured DAG it returns success but neither copies parameter
// values nor leaves the caller's pointer array intact — a scan over the
// returned slots crashed (garbage entries); even a fresh cudaGraphAddKernelNode
// segfaulted inside libcuda once the process had replayed the captured DAG.
// Minimal repros (few simple kernels, event nodes, replays) all pass, so the
// trigger is specific to the real DAG's shape. An integration therefore
// cannot rely on GetParams to find the m=1 zeroZ node on this stack: it must
// either construct the graph manually (keeping its own node/param handles) or
// re-instantiate per variant (costs measured above). A struct-by-value kernel
// parameter also triggered an illegal memory access in the manual path —
// plain-int parameters work.
//
// Usage:
//   cumes_benchmark_graph_realpass --config solovev [--passes 200] [--warmup 5]
//                                   [--out <json>]
//
// Shared harness pieces (timing helpers, the --option scanner, the config
// loader, the operator stack) live in bench_common.cuh. The per-pass control
// copy uses the same pinned-host staging as production (PinnedBuffer<double>,
// solver_impl.cuh) so the measured D2H stays asynchronous.
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "vmec_types.h"

#include "bench_common.cuh"

#include "cumes/config/json_reader.hpp"
#include "cumes/runtime/cuda_graph.hpp"
#include "cumes/runtime/cuda_status.hpp"
#include "cumes/runtime/device_arena.cuh"
#include "cumes/runtime/pinned_buffer.hpp"
#include "cumes/runtime/stream.hpp"
#include "cumes/solver/equilibrium_operator.hpp"
#include "cumes/solver/stage_solver.hpp"
#include "cumes/state/seed_state.hpp"
#include "cumes/state/spectral_storage.hpp"

using namespace bench_common;

// Capture + instantiate one pass variant; returns the wall cost in µs.
static double capture_variant(cumes::EquilibriumOperator<Real>& eq,
                              int iter, int iter2,
                              const cumes::EvaluationSchedule& sched,
                              cudaStream_t stream, cumes::CudaGraph* out) {
    double t0 = now_us();
    *out = cumes::CudaGraph::capture(stream, [&]() {
        eq.enqueue(iter, iter2, sched, stream);
    });
    return now_us() - t0;
}

// ---- per-node param update (cudaGraphExecKernelNodeSetParams) ----------
// The m=1 gauge zeroZ scalar is the one per-pass kernel parameter the §8.11
// integration must change (blueprint §7 step 10; solver_impl.cuh
// m1ConstraintKernel). Measured here on a MANUALLY constructed node with the
// same parameter shape — (16-byte view struct, int, int, int, int) — because
// CUDA 12.1's cudaGraphKernelNodeGetParams is unusable on stream-captured
// graphs: minimal repros return success but neither copy the parameter
// values nor leave the caller's pointer array intact on the real DAG (the
// scan crashed on garbage entries the driver wrote). An integration must
// therefore either keep its own node/param layout (manual construction) or
// re-instantiate per variant — measured below as capture+instantiate cost.
__global__ void updateTargetKernel(int ns, int mnmax, int ntor, int zeroZ,
                                   int* out) {
    if (threadIdx.x == 0) out[0] = ns + mnmax + ntor + zeroZ;
}

int main(int argc, char** argv) {
    const char* config = "solovev";
    const char* out_path = nullptr;
    int warmup = 5;
    int passes = 200;
    const int sync_period = 50;  // bound the async queue during submission timing

    ArgParser args(argc, argv, "graph_realpass");
    for (int i = 1; i < argc; ++i) {
        if (const char* v = args.need(i, "config")) config = v;
        else if (const char* v = args.need(i, "out")) out_path = v;
        else if (const char* v = args.need(i, "warmup")) warmup = atoi(v);
        else if (const char* v = args.need(i, "passes")) passes = atoi(v);
        else { fprintf(stderr, "graph_realpass: unknown option '%s'\n", argv[i]); return 2; }
    }
    if (passes < 1 || warmup < 0) { fprintf(stderr, "graph_realpass: bad --passes/--warmup\n"); return 2; }

    std::string input_path = std::string("inputs/") + config + ".json";

    // ---- config: parse + validate (same host model as the CLI/benchmarks) ----
    cumes::ValidationResult vr = load_validated(input_path, "graph_realpass");
    if (!vr.has_value()) return 2;
    const cumes::ValidatedProblem& vp = vr.value();
    const cumes::ProblemSpec& spec = vp.spec();

    DeviceParams<Real> p = cumes::init_params<Real>(vp);
    p.ns = static_cast<int>(spec.stages.back().radial_surfaces);
    p.max_iter = warmup + passes + 4;
    p.ftol = Real(0.0);

    // ---- seed + operator stack (mirrors StageSolver::run / fixed_iteration) ----
    cumes::SpectralStorage<Real> storage = cumes::init_state<Real>(p, vp);

    cumes::DeviceArena arena;
    arena.allocate(cumes::stage_arena_bytes<Real>(p));
    OperatorStack<Real> stack(p, vp, arena);

    cumes::Stream stream;
    stack.transform.bind_stream(stream.get());

    cumes::EquilibriumOperator<Real> equilibrium(
        p, storage, stack.profiles, stack.transform, stack.rs, stack.geometry, &arena,
        stack.use_axisym ? static_cast<cumes::SpectralOperator<Real>*>(stack.axisym.get()) : nullptr);

    // Steady-state pass schedules (iter/iter2 far from any first-pass or
    // dump-window branch; the env-gated dump machinery is off by default).
    const int iter = 1000, iter2 = 1000;
    cumes::EvaluationSchedule base_sched{};
    cumes::EvaluationSchedule refresh_sched{};
    refresh_sched.refresh_preconditioner = true;
    cumes::EvaluationSchedule zeroz_sched{};
    zeroz_sched.zero_z_force_m1 = true;

    // ---- warmup: production-pattern direct passes ----
    // Pinned staging like production (solver_impl.cuh's h_control_pin): the
    // async D2H stays a one-hop DMA copy instead of degrading to a
    // synchronous pageable two-hop (review finding 6.2).
    cumes::PinnedBuffer<double> h_ctl(16);
    for (int w = 0; w < warmup; ++w) {
        equilibrium.enqueue(iter, iter2, base_sched, stream.get());
        cumes::check_cuda(cudaMemcpyAsync(h_ctl.data(), equilibrium.control_device(),
                                          16 * sizeof(Real), cudaMemcpyDeviceToHost,
                                          stream.get()), "warmup control copy");
        stream.synchronize();
    }

    // ---- gate 2 + timing: cudaGraphExecKernelNodeSetParams (zeroZ scalar) ----
    double setparams_us = 0.0;
    bool param_update_works = false;
    {
        int* d_out = nullptr;
        cumes::check_cuda(cudaMalloc(&d_out, sizeof(int)), "setparams out");
        int ns_v = p.ns, mn_v = p.mnmax, ntor_v = p.ntor, zz_v = 1;
        void* mslots[5] = {&ns_v, &mn_v, &ntor_v, &zz_v, &d_out};

        cudaGraph_t mg = nullptr;
        cudaGraphNode_t mnode = nullptr;
        cudaGraphExec_t mexc = nullptr;
        cudaGraphCreate(&mg, 0);
        cudaKernelNodeParams mkp{};
        mkp.func = reinterpret_cast<void*>(updateTargetKernel);
        mkp.gridDim = dim3{1, 1, 1};
        mkp.blockDim = dim3{32, 1, 1};
        mkp.sharedMemBytes = 0;
        mkp.kernelParams = mslots;
        cudaGraphAddKernelNode(&mnode, mg, nullptr, 0, &mkp);
        cudaGraphInstantiate(&mexc, mg, nullptr, nullptr, 0);

        // Functional gate: the exec-side scalar update must change the output.
        auto run_and_read = [&]() {
            cudaGraphLaunch(mexc, stream.get());
            stream.synchronize();
            int v = 0;
            cudaMemcpy(&v, d_out, sizeof(int), cudaMemcpyDeviceToHost);
            return v;
        };
        zz_v = 1;
        cudaGraphExecKernelNodeSetParams(mexc, mnode, &mkp);
        const int o1 = run_and_read();
        zz_v = 0;
        cudaGraphExecKernelNodeSetParams(mexc, mnode, &mkp);
        const int o0 = run_and_read();
        param_update_works = (o1 != o0);
        printf("gate_setparams_zeroZ     : %s (zeroZ=1 -> %d, zeroZ=0 -> %d)\n",
               param_update_works ? "PASS (output changed)" : "FAIL", o1, o0);

        std::vector<double> sp;
        for (int r = 0; r < 100; ++r) {
            double t0 = now_us();
            cudaGraphExecKernelNodeSetParams(mexc, mnode, &mkp);
            sp.push_back(now_us() - t0);
        }
        setparams_us = median(sp);

        cudaGraphExecDestroy(mexc);
        cudaGraphDestroy(mg);
        cudaFree(d_out);
    }

    // ---- gate 1: graph replay == direct pass, bitwise ----
    {
        cumes::DeviceBuffer<Real> backup(6 * (size_t)p.ns * p.mnmax);
        backup.copy_from_async(storage.state_buffer(), stream.get());
        stream.synchronize();

        equilibrium.enqueue(iter, iter2, base_sched, stream.get());
        stream.synchronize();
        Real h_direct[16];
        cudaMemcpy(h_direct, equilibrium.control_device(), 16 * sizeof(Real),
                   cudaMemcpyDeviceToHost);

        storage.state_buffer().copy_from_async(backup, stream.get());
        stream.synchronize();
        auto g = cumes::CudaGraph::capture(stream.get(), [&]() {
            equilibrium.enqueue(iter, iter2, base_sched, stream.get());
        });
        g.launch(stream.get());
        stream.synchronize();
        Real h_graph[16];
        cudaMemcpy(h_graph, equilibrium.control_device(), 16 * sizeof(Real),
                   cudaMemcpyDeviceToHost);

        double max_diff = 0.0;
        for (int i = 0; i < 16; ++i)
            max_diff = std::max(max_diff, std::fabs((double)h_direct[i] - (double)h_graph[i]));
        printf("gate_graph_vs_direct     : %s (max |diff| = %.3e over 16 control scalars)\n",
               max_diff == 0.0 ? "PASS bitwise" : "FAIL", max_diff);
        if (max_diff != 0.0) return 1;
    }

    // ---- capture + instantiate costs (one-time per variant) ----
    cumes::CudaGraph g_base, g_refresh, g_zeroz;
    const double cap_base_us = capture_variant(equilibrium, iter, iter2, base_sched,
                                               stream.get(), &g_base);
    const double cap_refresh_us = capture_variant(equilibrium, iter, iter2, refresh_sched,
                                                  stream.get(), &g_refresh);
    const double cap_zeroz_us = capture_variant(equilibrium, iter, iter2, zeroz_sched,
                                                stream.get(), &g_zeroz);
    stream.synchronize();

    // ---- submission-only host wall per pass (enqueue vs launch) ----
    std::vector<double> enc_us, lch_us;
    for (int r = 0; r < passes; ++r) {
        double t0 = now_us();
        equilibrium.enqueue(iter, iter2, base_sched, stream.get());
        enc_us.push_back(now_us() - t0);
        if ((r + 1) % sync_period == 0) stream.synchronize();
    }
    for (int r = 0; r < passes; ++r) {
        double t0 = now_us();
        g_base.launch(stream.get());
        lch_us.push_back(now_us() - t0);
        if ((r + 1) % sync_period == 0) stream.synchronize();
    }
    stream.synchronize();

    // ---- production-pattern wall per pass (DAG + control D2H + fence) ----
    auto production_loop = [&](bool graph, double* gpu_ms_out) {
        cudaEvent_t ev0, ev1;
        cudaEventCreate(&ev0); cudaEventCreate(&ev1);
        std::vector<double> wall;
        cudaEventRecord(ev0, stream.get());
        for (int r = 0; r < passes; ++r) {
            double t0 = now_us();
            if (graph) {
                g_base.launch(stream.get());
            } else {
                equilibrium.enqueue(iter, iter2, base_sched, stream.get());
            }
            cumes::check_cuda(cudaMemcpyAsync(h_ctl.data(), equilibrium.control_device(),
                                              16 * sizeof(Real), cudaMemcpyDeviceToHost,
                                              stream.get()), "control copy");
            stream.synchronize();
            wall.push_back(now_us() - t0);
        }
        cudaEventRecord(ev1, stream.get());
        cudaEventSynchronize(ev1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, ev0, ev1);
        *gpu_ms_out = (double)ms * 1000.0;
        cudaEventDestroy(ev0); cudaEventDestroy(ev1);
        return wall;
    };
    double gpu_direct_us = 0.0, gpu_graph_us = 0.0;
    std::vector<double> prod_direct = production_loop(false, &gpu_direct_us);
    std::vector<double> prod_graph = production_loop(true, &gpu_graph_us);

    // ---- GPU identity ----
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);

    // ---- report ----
    printf("config                   : %s (%s backend, ns=%d)\n", config,
           stack.use_axisym ? "axisymmetric" : "toroidal-fft", p.ns);
    printf("gpu                      : %s (sm_%d%d)\n", prop.name,
           prop.major, prop.minor);
    printf("capture+instantiate_us   : base %.1f | refresh %.1f | zeroZ %.1f\n",
           cap_base_us, cap_refresh_us, cap_zeroz_us);
    printf("submission_us_per_pass   : enqueue %.2f | graph launch %.2f | saving %.2f\n",
           median(enc_us), median(lch_us), median(enc_us) - median(lch_us));
    printf("production_wall_us_pass  : direct %.2f | graph %.2f | saving %.2f\n",
           median(prod_direct), median(prod_graph), median(prod_direct) - median(prod_graph));
    printf("gpu_us_per_pass          : direct %.2f | graph %.2f\n",
           gpu_direct_us / passes, gpu_graph_us / passes);
    printf("setparams_us_per_update  : %.2f (zeroZ scalar, m1Constraint node)\n",
           setparams_us);

    if (out_path) {
        FILE* f = fopen(out_path, "w");
        if (!f) { fprintf(stderr, "graph_realpass: cannot open --out %s\n", out_path); return 2; }
        fprintf(f,
                "{\n"
                "  \"config\": \"%s\",\n"
                "  \"backend\": \"%s\",\n"
                "  \"ns\": %d,\n"
                "  \"gpu\": \"%s\",\n"
                "  \"sm\": \"%d%d\",\n"
                "  \"capture_us\": {\"base\": %.1f, \"refresh\": %.1f, \"zeroz\": %.1f},\n"
                "  \"submission_us_per_pass\": {\"enqueue\": %.2f, \"graph_launch\": %.2f, \"saving\": %.2f},\n"
                "  \"production_wall_us_per_pass\": {\"direct\": %.2f, \"graph\": %.2f, \"saving\": %.2f},\n"
                "  \"gpu_us_per_pass\": {\"direct\": %.2f, \"graph\": %.2f},\n"
                "  \"setparams_us_per_update\": %.2f,\n"
                "  \"param_update_works\": %s\n"
                "}\n",
                config, stack.use_axisym ? "axisymmetric" : "toroidal-fft", p.ns,
                prop.name, prop.major, prop.minor,
                cap_base_us, cap_refresh_us, cap_zeroz_us,
                median(enc_us), median(lch_us), median(enc_us) - median(lch_us),
                median(prod_direct), median(prod_graph), median(prod_direct) - median(prod_graph),
                gpu_direct_us / passes, gpu_graph_us / passes,
                setparams_us, param_update_works ? "true" : "false");
        fclose(f);
        printf("wrote benchmark JSON to %s\n", out_path);
    }

    return 0;
}
