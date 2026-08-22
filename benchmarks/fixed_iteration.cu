// fixed_iteration.cu — the §8.1 fixed-iteration benchmark harness.
//
// Runs ONE radial stage at a config's FINAL shape (Solovev ns=55, W7-X ns=99)
// for a fixed number of passes with ftol=0 (so the solver never converges), and
// reports the hot-loop cost. The first `--warmup` passes are discarded and the
// remainder summarized by median/p95 wall microseconds per evaluated pass
// (sampled at the single control fence via cumes::SolverBench — see
// solver_bench.hpp). The controller's refresh/restart/gauge decisions are NOT
// disabled: the fixed window replays the real trajectory's first N passes, and
// a `--restart` checkpoint lets a steady-state window be measured instead.
//
// Emits one JSON object (stdout, or `--out <path>`) with the §8.1 fields:
//   gpu{name,compute_cap,memory_mib,driver,runtime,toolkit}, build{scalar,
//   fast_math}, shape{ns,mnmax,mpol,ntor,nfp,ntheta,nzeta,nZnT,ncurr,backend},
//   memory{arena_bytes,cufft_work_bytes}, timing{setup_us,solve_wall_us,
//   solve_gpu_us,passes,median_us_per_iter,p95_us_per_iter},
//   result{state_hash,total_effective_iterations}.
//
// Usage:
//   cumes_benchmark_fixed_iteration --config solovev --passes 100 [--warmup 10]
//                                   [--restart <ckpt>] [--out <json>]
//
// Shared harness pieces (timing helpers, the --option scanner, the config
// loader, the operator stack) live in bench_common.cuh.
#include "bench_common.cuh"
#include "cumes/config/json_reader.hpp"
#include "cumes/io/checkpoint.hpp"
#include "cumes/io/equilibrium_snapshot.hpp"
#include "cumes/io/snapshot_bridge.cuh"
#include "cumes/runtime/device_arena.cuh"
#include "cumes/runtime/stream.hpp"
#include "cumes/solver/solver_bench.hpp"
#include "cumes/solver/stage_solver.hpp"
#include "cumes/state/seed_state.hpp"
#include "cumes/state/spectral_storage.hpp"
#include "solver.cuh"
#include "vmec_types.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace bench_common;

// FNV-1a 64-bit hash over the raw little-endian double bytes of all six
// families (component order Rcc Zsc Lsc Rss Zcs Lcs), hex-encoded. A different
// final state (fast-but-wrong run) produces a different hash.
static std::string hash_snapshot(const cumes::EquilibriumSnapshot& snap) {
    std::uint64_t h = 1469598103934665603ULL;
    for (int c = 0; c < cumes::EquilibriumSnapshot::kCount; ++c) {
        for (double v : snap.families[c]) {
            unsigned char b[8];
            std::memcpy(b, &v, 8);
            for (unsigned char byte : b) {
                h ^= byte;
                h *= 1099511628211ULL;
            }
        }
    }
    char buf[17];
    snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(h));
    return buf;
}

int main(int argc, char** argv) {
    const char* config = "solovev";
    const char* restart_path = nullptr;
    const char* out_path = nullptr;
    int warmup = 10;
    int passes = 100;

    ArgParser args(argc, argv, "bench");
    for (int i = 1; i < argc; ++i) {
        if (const char* v = args.need(i, "config"))
            config = v;
        else if (const char* v = args.need(i, "restart"))
            restart_path = v;
        else if (const char* v = args.need(i, "out"))
            out_path = v;
        else if (const char* v = args.need(i, "warmup"))
            warmup = atoi(v);
        else if (const char* v = args.need(i, "passes"))
            passes = atoi(v);
        else {
            fprintf(stderr, "bench: unknown option '%s'\n", argv[i]);
            return 2;
        }
    }
    if (passes < 1) {
        fprintf(stderr, "bench: --passes must be >= 1\n");
        return 2;
    }
    if (warmup < 0) {
        fprintf(stderr, "bench: --warmup must be >= 0\n");
        return 2;
    }

    std::string input_path = std::string("inputs/") + config + ".json";

    // ---- config: parse + validate (same host model as the CLI) ----
    cumes::ValidationResult vr = load_validated(input_path, "bench");
    if (!vr.has_value()) return 2;
    const cumes::ValidatedProblem& vp = vr.value();
    const cumes::ProblemSpec& spec = vp.spec();

    // Single stage at the config's FINAL radial grid.
    DeviceParams<Real> p = cumes::init_params<Real>(vp);
    p.ns = static_cast<int>(spec.stages.back().radial_surfaces);
    p.max_iter = warmup + passes;
    p.ftol = Real(0.0);  // never converge: run exactly warmup+passes passes

    // ---- seed ----
    cumes::SpectralStorage<Real> storage;
    if (restart_path) {
        auto ck = cumes::read_checkpoint(restart_path);
        if (!ck.has_value()) {
            fprintf(stderr, "bench: %s\n", ck.error().c_str());
            return 2;
        }
        if (ck.value().ns != p.ns || ck.value().mnmax != p.mnmax) {
            fprintf(stderr,
                    "bench: checkpoint (ns=%d,mnmax=%d) does not match "
                    "shape (ns=%d,mnmax=%d)\n",
                    ck.value().ns, ck.value().mnmax, p.ns, p.mnmax);
            return 2;
        }
        storage = cumes::restart_state<Real>(p, vp, ck.value());
    } else {
        storage = cumes::init_state<Real>(p, vp);
    }

    // ---- drive one stage directly (mirrors StageSolver::run) with timing ----
    cumes::SolverBench bench;
    bench.enabled = true;
    cumes::Stream stream;

    double setup_us = 0.0, solve_wall_us = 0.0, solve_gpu_us = 0.0;
    std::size_t arena_bytes = 0;
    std::size_t cufft_work_bytes = 0;
    SolverResult<Real> result{false,     0,         Real(1.0), Real(1.0),
                              Real(1.0), Real(0.9), {}};
    cumes::EquilibriumSnapshot snap;

    // One arena allocation, one construction of every module, one solve
    // (completion plan step 3.2 — the same growth-retry helper the production
    // StageSolver uses; no temporary measuring arena). The production
    // operator stack (bench_common::OperatorStack) carves profiles/
    // real-space/mode-table/transform/geometry operators, with the
    // axisymmetric direct-poloidal backend for ntor=0/nzeta=1 shapes
    // (blueprint §8.5) unless CUMES_FORCE_GENERIC=1 restores the generic
    // cuFFT backend for A/B comparison.
    bool backend_axisym = false;
    cumes::run_in_stage_arena<Real>(p, [&](cumes::DeviceArena& arena) {
        double t0 = now_us();
        OperatorStack<Real> stack(p, vp, arena);
        backend_axisym = stack.use_axisym;
        setup_us = now_us() - t0;

        cudaEvent_t ev0, ev1;
        cudaEventCreate(&ev0);
        cudaEventCreate(&ev1);
        cudaEventRecord(ev0, stream.get());
        double w0 = now_us();
        result = solverRun<Real>(storage, p, stack.profiles, stack.transform,
                                 stack.rs, stack.geometry, &arena, stream.get(),
                                 &bench, stack.axisym.get());
        double w1 = now_us();
        cudaEventRecord(ev1, stream.get());
        cudaEventSynchronize(ev1);
        float gpu_ms = 0.0f;
        cudaEventElapsedTime(&gpu_ms, ev0, ev1);
        solve_wall_us = w1 - w0;
        solve_gpu_us = gpu_ms * 1000.0;

        arena_bytes = arena.peak_bytes();
        cufft_work_bytes = stack.transform.cufft_work_bytes();

        // profiles/transform/geometry are RAII (operator destructors).
        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);

        // ---- residual/state hash (a fast-but-different run is obvious) ----
        snap = cumes::snapshot_from_device(storage);
        return result;
    });
    std::string state_hash = hash_snapshot(snap);

    // ---- per-pass statistics (discard warmup) ----
    std::vector<double> timed;
    if (bench.pass_wall_us.size() > static_cast<std::size_t>(warmup)) {
        timed.assign(bench.pass_wall_us.begin() + warmup,
                     bench.pass_wall_us.end());
    }
    const double med = median(timed);
    const double p95us = p95(timed);

    // ---- GPU identity ----
    int driver = 0, runtime = 0;
    cudaDriverGetVersion(&driver);
    cudaRuntimeGetVersion(&runtime);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);

    // ---- emit JSON ----
    std::string json;
    auto q = [](const std::string& s) { return "\"" + s + "\""; };
    auto kv = [&](const std::string& k, const std::string& v) {
        return q(k) + ": " + v;
    };
    char num[64];
    json += "{\n";
    json += "  " + kv("config", q(config)) + ",\n";
    snprintf(num, sizeof num, "%d", prop.major * 10 + prop.minor);
    json += "  " + kv("gpu_name", q(prop.name)) + ",\n";
    json += "  " + kv("gpu_compute_cap", num) + ",\n";
    snprintf(num, sizeof num, "%zu", prop.totalGlobalMem / (1024 * 1024));
    json += "  " + kv("gpu_memory_mib", num) + ",\n";
    snprintf(num, sizeof num, "%d", driver);
    json += "  " + kv("driver", num) + ",\n";
    snprintf(num, sizeof num, "%d", runtime);
    json += "  " + kv("runtime", num) + ",\n";
    snprintf(num, sizeof num, "%d", CUDART_VERSION);
    json += "  " + kv("toolkit", num) + ",\n";
    json += "  " +
            kv("scalar_type",
               q(sizeof(Real) == sizeof(double) ? "double" : "float")) +
            ",\n";
    // Precision-policy provenance (completion plan step 3.1): the named
    // policy and its effective flags, from the CMake-provided defines.
    json +=
        "  " + kv("precision_policy", q(CUMES_PRECISION_POLICY_NAME)) + ",\n";
    json += "  " + kv("precision_flags", q(CUMES_PRECISION_FLAGS)) + ",\n";
    snprintf(num, sizeof num, "%d", p.ns);
    json += "  " + kv("ns", num) + ",\n";
    snprintf(num, sizeof num, "%d", p.mnmax);
    json += "  " + kv("mnmax", num) + ",\n";
    snprintf(num, sizeof num, "%d", p.mpol);
    json += "  " + kv("mpol", num) + ",\n";
    snprintf(num, sizeof num, "%d", p.ntor);
    json += "  " + kv("ntor", num) + ",\n";
    snprintf(num, sizeof num, "%d", p.nfp);
    json += "  " + kv("nfp", num) + ",\n";
    snprintf(num, sizeof num, "%d", p.ntheta);
    json += "  " + kv("ntheta", num) + ",\n";
    snprintf(num, sizeof num, "%d", p.nzeta);
    json += "  " + kv("nzeta", num) + ",\n";
    snprintf(num, sizeof num, "%d", p.ncurr);
    json += "  " + kv("ncurr", num) + ",\n";
    json += "  " +
            kv("transform_backend",
               q(backend_axisym ? "axisymmetric" : "toroidal-fft")) +
            ",\n";
    snprintf(num, sizeof num, "%zu", arena_bytes);
    json += "  " + kv("arena_bytes", num) + ",\n";
    snprintf(num, sizeof num, "%zu", cufft_work_bytes);
    json += "  " + kv("cufft_work_bytes", num) + ",\n";
    snprintf(num, sizeof num, "%.1f", setup_us);
    json += "  " + kv("setup_us", num) + ",\n";
    snprintf(num, sizeof num, "%.1f", solve_wall_us);
    json += "  " + kv("solve_wall_us", num) + ",\n";
    snprintf(num, sizeof num, "%.1f", solve_gpu_us);
    json += "  " + kv("solve_gpu_us", num) + ",\n";
    snprintf(num, sizeof num, "%zu", timed.size());
    json += "  " + kv("timed_passes", num) + ",\n";
    snprintf(num, sizeof num, "%.2f", med);
    json += "  " + kv("median_us_per_iter", num) + ",\n";
    snprintf(num, sizeof num, "%.2f", p95us);
    json += "  " + kv("p95_us_per_iter", num) + ",\n";
    json += "  " + kv("state_hash", q(state_hash)) + ",\n";
    snprintf(num, sizeof num, "%d", result.iterations);
    json += "  " + kv("total_effective_iterations", num) + "\n";
    json += "}\n";

    if (out_path) {
        FILE* f = fopen(out_path, "w");
        if (!f) {
            fprintf(stderr, "bench: cannot open --out %s\n", out_path);
            return 2;
        }
        fputs(json.c_str(), f);
        fclose(f);
        printf("wrote benchmark JSON to %s\n", out_path);
    } else {
        fputs(json.c_str(), stdout);
    }
    return 0;
}
