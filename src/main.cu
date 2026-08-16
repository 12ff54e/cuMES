// main.cu — entry point: parse/validate → init → solve → output.
//
// Input: argv[1] is the JSON input file (vmecpp indata schema), or
// --input <path>; without either, inputs/solovev.json is used. Config is
// parsed + validated by the Phase 2 host model (read_and_validate, see
// cumes/config/*.hpp) into an immutable ValidatedProblem the solver consumes
// directly.
//
// Output: argv[2] (or --output <path>) selects the destination; binary output
// goes through the versioned writers (legacy-v0 by default — byte-identical to
// the pre-overhaul outputSaveBinary — or v1 via --output-schema v1), while
// .nc/.h5 keep the legacy device-reading backends.
//
// Restart: --restart <checkpoint> (v1 checkpoint) or --restart-legacy <init>
// (legacy six-family vmecpp_init payload) replace the removed CUMES_LOAD_INIT
// environment path.
//
// Precision: `Real` (vmec_types.h) is the compile-time switch between double
// and float — configure with -DCUMES_USE_FLOAT=ON.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <string>

#include "vmec_types.h"
#include "solver.cuh"
#include "output.cuh"
#include "refine.cuh"

#include "cumes/config/json_reader.hpp"
#include "cumes/config/precision_policy.hpp"
#include "cumes/config/solver_options.hpp"
#include "cumes/config/validated_problem.hpp"
#include "cumes/io/checkpoint.hpp"
#include "cumes/io/output_spec.hpp"
#include "cumes/io/run_report.hpp"
#include "cumes/io/snapshot_bridge.cuh"
#include "cumes/io/writer.hpp"
#include "cumes/runtime/cuda_status.hpp"
#include "cumes/runtime/stream.hpp"
#include "cumes/state/seed_state.hpp"
#include "cumes/state/spectral_storage.hpp"
#include "cumes/solver/multigrid_solver.hpp"

// Build provenance injected at configure time (see CMakeLists.txt).
#ifndef CUMES_GIT_REVISION
#define CUMES_GIT_REVISION ""
#endif
#ifndef CUMES_GIT_DIRTY
#define CUMES_GIT_DIRTY 0
#endif
#ifndef CUMES_BUILD_TYPE
#define CUMES_BUILD_TYPE ""
#endif

// FNV-1a 64-bit hash of a file's bytes, hex-encoded ("" on open failure). Cheap
// input provenance for the v1 schema; not a cryptographic digest.
static std::string hashFile(const char* path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return "";
    std::uint64_t h = 1469598103934665603ULL;  // FNV-1a offset basis
    char c;
    while (in.get(c)) {
        h ^= static_cast<unsigned char>(c);
        h *= 1099511628211ULL;  // FNV-1a prime
    }
    char buf[17];
    snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(h));
    return buf;
}

// Fill the build/input/runtime provenance of `report` (consumed only by the v1
// writer; the legacy-v0 writer ignores it). Called after the solve so a CUDA
// context is live for the device/driver queries.
static void fill_provenance(cumes::RunReport& report, const char* inputPath) {
    report.build.revision = CUMES_GIT_REVISION;
    report.build.dirty = (CUMES_GIT_DIRTY != 0);
    report.build.build_type = CUMES_BUILD_TYPE;
    report.build.scalar_type = sizeof(Real) == sizeof(double) ? "double" : "float";
    report.input.source_path = inputPath;
    report.input.source_hash = hashFile(inputPath);

    int driver = 0, runtime = 0;
    cudaDriverGetVersion(&driver);
    cudaRuntimeGetVersion(&runtime);
    report.runtime.driver = std::to_string(driver);
    report.runtime.runtime = std::to_string(runtime);
    report.runtime.toolkit = std::to_string(CUDART_VERSION);

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        report.runtime.gpu_name = prop.name;
    }
}

static const char* severityName(cumes::Severity s) {
    return s == cumes::Severity::kError ? "error" : "warning";
}

int main(int argc, char** argv) {
    // ---- CLI ----------------------------------------------------------------
    const char* inputPath = "inputs/solovev.json";
    const char* outputPath = "cumes_state.bin";
    cumes::OutputSchema schema = cumes::OutputSchema::kLegacyV0;
    std::string restartPath;        // --restart (v1 checkpoint)
    std::string restartLegacyPath;  // --restart-legacy (six-family payload)
    std::string checkpointPath;     // --checkpoint (write a v1 checkpoint after solve)
    bool haveOutput = false;
    int positional = 0;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto match = [&](const char* name, std::string& out) -> bool {
            const std::string prefix = std::string("--") + name;
            if (a == prefix) {
                if (i + 1 >= argc) return false;
                out = argv[++i];
                return true;
            }
            if (a.compare(0, prefix.size() + 1, prefix + "=") == 0) {
                out = a.substr(prefix.size() + 1);
                return true;
            }
            return false;
        };
        std::string v;
        if (match("output-schema", v)) {
            if (v == "legacy-v0") {
                schema = cumes::OutputSchema::kLegacyV0;
            } else if (v == "v1") {
                schema = cumes::OutputSchema::kV1;
            } else {
                fprintf(stderr, "cuMES: unknown --output-schema '%s' "
                                "(expected legacy-v0|v1)\n", v.c_str());
                return EXIT_FAILURE;
            }
        } else if (match("restart", v)) {
            restartPath = v;
        } else if (match("restart-legacy", v)) {
            restartLegacyPath = v;
        } else if (match("checkpoint", v)) {
            checkpointPath = v;
        } else if (!a.empty() && a[0] == '-') {
            fprintf(stderr, "cuMES: unknown option '%s'\n", a.c_str());
            return EXIT_FAILURE;
        } else if (positional == 0) {
            inputPath = argv[i];
            ++positional;
        } else if (positional == 1) {
            outputPath = argv[i];
            haveOutput = true;
            ++positional;
        } else {
            fprintf(stderr, "cuMES: unexpected extra argument '%s'\n", a.c_str());
            return EXIT_FAILURE;
        }
    }
    if (!restartPath.empty() && !restartLegacyPath.empty()) {
        fprintf(stderr, "cuMES: --restart and --restart-legacy are mutually "
                        "exclusive\n");
        return EXIT_FAILURE;
    }
    if (!haveOutput)
        fprintf(stderr, "WARNING: no output path given - "
                        "writing binary cumes_state.bin\n");

    // ---- output preflight (before any CUDA work) ----------------------------
    // A requested-but-unlinked format (e.g. .nc on a binary-only build) and an
    // unknown suffix are rejected HERE, before the CUDA context is created and
    // before any grid stage runs.
    auto resolved = cumes::resolve_output_spec(outputPath, /*compatibility=*/false);
    if (!resolved.has_value()) {
        fprintf(stderr, "cuMES: %s\n", resolved.error().c_str());
        return EXIT_FAILURE;
    }
    const cumes::OutputSpec outSpec = resolved.value();
    if (!cumes::output_format_available(outSpec.format)) {
        fprintf(stderr, "cuMES: output format '%s' is not available in this "
                        "build; no output will be written\n",
                cumes::output_suffix(outSpec.format));
        return EXIT_FAILURE;
    }

    // ---- config: parse + validate -------------------------------------------
    cumes::SolverOptions opts;
#ifdef CUMES_USE_FLOAT
    // Float runs stall at ~1e-7 (the float rounding floor) and can never meet
    // the double-tuned stage tolerances. Declaring the mixed-float policy lets
    // validation reject impossible tolerances instead of failing every stage at
    // the end of the run.
    opts.precision = cumes::PrecisionPolicy::kMixedFloat;
#endif
    cumes::ValidationResult vr = cumes::ValidationResult(cumes::ValidationReport{});
    try {
        vr = cumes::read_and_validate(inputPath, opts);
    } catch (const std::exception& e) {
        fprintf(stderr, "cuMES: error loading input file: %s\n", e.what());
        return EXIT_FAILURE;
    }
    if (!vr.has_value()) {
        fprintf(stderr, "cuMES: input validation failed:\n");
        for (const auto& issue : vr.error().issues()) {
            fprintf(stderr, "  [%s] %s: %s\n", severityName(issue.severity),
                    issue.key.c_str(), issue.message.c_str());
        }
        return EXIT_FAILURE;
    }
    const cumes::ValidatedProblem& vp = vr.value();
    const cumes::ProblemSpec& spec = vp.spec();
    const int n_grids = static_cast<int>(spec.stages.size());

    DeviceParams<Real> p = cumes::init_params<Real>(vp);
    printf("=== cuMES — CUDA Magnetic Equilibrium Solver ===\n");
    fflush(stdout);
    printf("input: %s\n", inputPath);
    printf("precision: %s\n", sizeof(Real) == sizeof(double) ? "double" : "float");
    printf("mpol=%d ntor=%d nfp=%d ntheta=%d nzeta=%d nZnT=%d ncurr=%d\n",
           p.mpol,p.ntor,p.nfp,p.ntheta,p.nzeta,p.nZnT,p.ncurr);
    // Multi-radial-grid stage sequence (vmecpp ns_array/niter_array/ftol_array)
    printf("grids=%d: ns", n_grids);
    for (int g = 0; g < n_grids; ++g)
        printf("%s%zu", g == 0 ? "" : "->", spec.stages[g].radial_surfaces);
    printf(" (niter");
    for (int g = 0; g < n_grids; ++g)
        printf(" %zu", spec.stages[g].max_iterations);
    printf(", ftol");
    for (int g = 0; g < n_grids; ++g) printf(" %.0e", spec.stages[g].tolerance);
    printf(")\n");

    // ---- Multi-radial-grid stage loop (delegated to MultigridSolver) ----
    cumes::SpectralStorage<Real> storage;
    SolverResult<Real> result{false, 0, Real(1.0), Real(1.0), Real(1.0), Real(0.9)};
    int total_iter = 0;

    try {
        // Stage-0 seed on ns_array[0]: a checkpoint/legacy-init restart, or the
        // interpFromBoundaryAndAxis cold start.
        p.ns = static_cast<int>(spec.stages[0].radial_surfaces);
        p.max_iter = static_cast<int>(spec.stages[0].max_iterations);
        p.ftol = Real(spec.stages[0].tolerance);
        cumes::SpectralStorage<Real> seed;
        if (!restartPath.empty()) {
            auto ck = cumes::read_checkpoint(restartPath);
            if (!ck.has_value()) {
                fprintf(stderr, "cuMES: %s\n", ck.error().c_str());
                return EXIT_FAILURE;
            }
            const auto& snap = ck.value();
            if (snap.ns != static_cast<int>(spec.stages[0].radial_surfaces) ||
                snap.mnmax != p.mnmax) {
                fprintf(stderr, "cuMES: restart checkpoint (ns=%d, mnmax=%d) "
                                "does not match stage-0 grid (ns=%zu, mnmax=%d)\n",
                        snap.ns, snap.mnmax, spec.stages[0].radial_surfaces, p.mnmax);
                return EXIT_FAILURE;
            }
            seed = cumes::restart_state<Real>(p, vp, snap);
        } else if (!restartLegacyPath.empty()) {
            auto ck = cumes::convert_legacy_init(restartLegacyPath,
                static_cast<int>(spec.stages[0].radial_surfaces), p.mnmax);
            if (!ck.has_value()) {
                fprintf(stderr, "cuMES: %s\n", ck.error().c_str());
                return EXIT_FAILURE;
            }
            seed = cumes::restart_state<Real>(p, vp, ck.value());
        } else {
            seed = cumes::init_state<Real>(p, vp);
        }

        // One explicit nonblocking compute stream owns the whole run: the
        // stage setup (synchronous default-stream copies) completes before the
        // solve, and every hot-loop kernel / cuFFT transform / D2H transfer is
        // enqueued on this stream (Phase 6A explicit-stream scheduling).
        cumes::Stream compute_stream;
        auto outcome = cumes::MultigridSolver<Real>::run(p, vp, std::move(seed),
                                                         compute_stream.get());
        storage = std::move(outcome.state);
        result = outcome.result;
        total_iter = outcome.total_iterations;

        // vmecpp semantics (vmec.cc:367-392): a stage that exhausts its
        // iteration cap without meeting ftol fails the whole run. Single-grid
        // runs keep the lenient report-and-return path below.
        if (outcome.failed_stage >= 0) {
            int g = outcome.failed_stage;
            fprintf(stderr, "FATAL: grid stage %d/%d (ns=%d) completed %d/%d "
                            "iterations without meeting ftol=%.0e; final "
                            "residuals fsqr=%.3e fsqz=%.3e fsql=%.3e\n",
                    g + 1, n_grids, p.ns, result.iterations, p.max_iter,
                    (double)p.ftol, (double)result.fsqr, (double)result.fsqz,
                    (double)result.fsql);
            return EXIT_FAILURE;
        }

        // Output success is part of the run result: a converged solve whose
        // state file could not be written must NOT exit 0.
        bool output_ok = true;
        // One host snapshot serves the binary writer and/or the optional
        // checkpoint (built once, only when either needs it).
        cumes::EquilibriumSnapshot snapshot;
        if (outSpec.format == cumes::OutputFormat::kBinary || !checkpointPath.empty()) {
            snapshot = cumes::snapshot_from_device(storage);
        }
        if (outSpec.format == cumes::OutputFormat::kBinary) {
            // Versioned binary writer: legacy-v0 (default) is byte-identical to
            // the pre-overhaul outputSaveBinary (proved by test_io_golden); v1
            // adds the provenance trailer.
            cumes::OutputSpec spec = outSpec;
            spec.schema = schema;
            auto writer = cumes::make_writer(spec.format, spec.schema);
            if (!writer) {
                fprintf(stderr, "cuMES: no writer for binary schema\n");
                output_ok = false;
            } else {
                fill_provenance(outcome.report, inputPath);
                const cumes::Status status =
                    writer->write_atomic(snapshot, outcome.report, spec);
                if (status.has_value()) {
                    printf("Saved binary state to %s\n", spec.path.c_str());
                } else {
                    fprintf(stderr, "cuMES: %s\n", status.error().c_str());
                    output_ok = false;
                }
            }
        } else {
            // NetCDF/HDF5: the legacy device-reading backends (host adapters
            // deferred). outputSave is never called for .bin from here.
            output_ok = outputSave<Real>(storage.legacy_view(), p, vp, result,
                                         outputPath, inputPath);
        }

        // Optional v1 restart checkpoint (blueprint §6.13): written after the
        // solve so a run that stops at its iteration cap can be resumed via
        // --restart.
        if (!checkpointPath.empty()) {
            const cumes::Status status =
                cumes::write_checkpoint(snapshot, checkpointPath);
            if (status.has_value()) {
                printf("Saved checkpoint to %s\n", checkpointPath.c_str());
            } else {
                fprintf(stderr, "cuMES: %s\n", status.error().c_str());
                output_ok = false;
            }
        }

        outputPrint<Real>(storage.legacy_view(), p, result.iterations,
                          result.converged, result.fsqr, result.fsqz, result.fsql);
        if (n_grids > 1)
            printf("multigrid: total effective iterations over %d grids = %d\n",
                   n_grids, total_iter);
        printf("\nDone.\n");
        if (!output_ok) {
            fprintf(stderr, "cuMES: FAILED to write output state (%s)\n", outputPath);
            return EXIT_FAILURE;
        }
        return result.converged ? 0 : 1;
    } catch (const cumes::CumesError& e) {
        fprintf(stderr, "cuMES: %s\n", e.what());
        return EXIT_FAILURE;
    }
}
