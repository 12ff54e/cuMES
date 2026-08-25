// main.cu — entry point: parse/validate → init → solve → output.
//
// CLI: cuMES [<input> <output>] [options]. The input is a vmecpp-indata JSON
// file. Positional <input>/<output> and the --input <path>/--output <path>
// flags fill the same two slots; each positional fills the first free slot
// (input, then output), and a --input/--output flag overrides the positional
// value for its slot (so `cuMES --input x.json y.bin` writes y.bin, and
// `cuMES a.json b.json --input x.json` reads x.json, discarding a.json).
// Without an input, inputs/solovev.json is used; an output path is REQUIRED
// (no default). Config is parsed + validated by the Phase 2 host model
// (read_and_validate, see cumes/config/*.hpp) into an immutable
// ValidatedProblem the solver consumes directly.
//
// Output: every backend writes the schema-v1 container (binary/NetCDF/HDF5
// through the Writer interface, dispatched by the output suffix).
//
// Restart: --restart <checkpoint> (versioned checkpoint) replaces the removed
// CUMES_LOAD_INIT environment path.
//
// Precision: `Real` (vmec_types.h) is the compile-time switch between double
// and float — configure with -DCUMES_USE_FLOAT=ON.
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
#include "cumes/solver/multigrid_solver.hpp"
#include "cumes/state/seed_state.hpp"
#include "cumes/state/spectral_storage.hpp"
#include "output.cuh"
#include "solver.cuh"
#include "vmec_types.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iterator>
#include <string>

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

// Read a whole file into a string ("" on open failure).
static std::string readFileBytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return "";
    return std::string(std::istreambuf_iterator<char>(in),
                       std::istreambuf_iterator<char>());
}

// FNV-1a 64-bit hash of raw bytes, hex-encoded. Cheap input provenance for the
// v1 schema; not a cryptographic digest. Identical algorithm to the removed
// hashFile(), so recorded source_hash values are unchanged for unchanged files.
static std::string hashBytes(const std::string& bytes) {
    std::uint64_t h = 1469598103934665603ULL;  // FNV-1a offset basis
    for (unsigned char c : bytes) {
        h ^= c;
        h *= 1099511628211ULL;  // FNV-1a prime
    }
    char buf[17];
    snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(h));
    return buf;
}

// Fill the build/input/runtime provenance of `report` (consumed by the v1
// writers). Called after the solve so a CUDA
// context is live for the device/driver queries. sourceHash is the hash of the
// input bytes captured at read time — the input file is NOT re-opened here
// (a mid-solve replacement must not change the recorded provenance).
static void fill_provenance(cumes::RunReport& report,
                            const std::string& inputPath,
                            const std::string& sourceHash) {
    report.build.revision = CUMES_GIT_REVISION;
    report.build.dirty = (CUMES_GIT_DIRTY != 0);
    report.build.build_type = CUMES_BUILD_TYPE;
    report.build.scalar_type =
        sizeof(Real) == sizeof(double) ? "double" : "float";
    report.build.precision_policy = CUMES_PRECISION_POLICY_NAME;
    report.build.compile_flags = CUMES_PRECISION_FLAGS;
    report.input.source_path = inputPath;
    report.input.source_hash = sourceHash;

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
    std::string inputPath = "inputs/solovev.json";
    std::string outputPath;
    std::string restartPath;  // --restart (versioned checkpoint)
    std::string
        checkpointPath;  // --checkpoint (write a checkpoint after solve)
    // Slot occupancy: a --input/--output flag pins its slot (flags override
    // positionals); each positional fills the first free slot (input, output).
    bool inputGiven = false;
    bool outputGiven = false;
    // --compatibility (completion plan step 2.1): vmecpp-style warn-and-ignore
    // for unknown input keys. Strict schema-v1 input parsing is the default;
    // the output policy (explicit path, known suffix) is always strict.
    bool compatibility = false;

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
        if (match("restart", v)) {
            restartPath = v;
        } else if (match("checkpoint", v)) {
            checkpointPath = v;
        } else if (match("input", v)) {
            inputPath = v;
            inputGiven = true;
        } else if (match("output", v)) {
            outputPath = v;
            outputGiven = true;
        } else if (a == "--compatibility") {
            compatibility = true;
        } else if (!a.empty() && a[0] == '-') {
            fprintf(stderr, "cuMES: unknown option '%s'\n", a.c_str());
            return EXIT_FAILURE;
        } else if (!inputGiven) {
            inputPath = argv[i];
            inputGiven = true;
        } else if (!outputGiven) {
            outputPath = argv[i];
            outputGiven = true;
        } else {
            fprintf(stderr, "cuMES: unexpected extra argument '%s'\n",
                    a.c_str());
            return EXIT_FAILURE;
        }
    }
    if (!outputGiven) {
        fprintf(stderr, "cuMES: no output path given; pass --output <path>\n");
        return EXIT_FAILURE;
    }

    // ---- output preflight (before any CUDA work) ----------------------------
    // A requested-but-unlinked format (e.g. .nc on a binary-only build) and an
    // unknown suffix are rejected HERE, before the CUDA context is created and
    // before any grid stage runs.
    auto resolved = cumes::resolve_output_spec(outputPath);
    if (!resolved.has_value()) {
        fprintf(stderr, "cuMES: %s\n", resolved.error().c_str());
        return EXIT_FAILURE;
    }
    const cumes::OutputSpec outSpec = resolved.value();
    if (!cumes::output_format_available(outSpec.format)) {
        fprintf(stderr,
                "cuMES: output format '%s' is not available in this "
                "build; no output will be written\n",
                cumes::output_suffix(outSpec.format));
        return EXIT_FAILURE;
    }

    // ---- config: parse + validate -------------------------------------------
    cumes::SolverOptions opts;
    opts.strict_schema = !compatibility;  // strict schema-v1 is the default
#ifdef CUMES_USE_FLOAT
    // Float runs stall at ~1e-7 (the float rounding floor) and can never meet
    // the double-tuned stage tolerances. Declaring the mixed-float policy lets
    // validation reject impossible tolerances instead of failing every stage at
    // the end of the run.
    opts.precision = cumes::PrecisionPolicy::kMixedFloat;
#endif
    // Capture the raw input bytes once, at read time: the v1 provenance
    // source_hash must be the hash of the bytes the solver actually consumed,
    // not of whatever the path holds after the solve (TOCTOU — the pre-fix
    // fill_provenance re-opened and re-hashed the file post-solve).
    const std::string inputBytes = readFileBytes(inputPath);
    const std::string sourceHash =
        inputBytes.empty() ? "" : hashBytes(inputBytes);
    cumes::ValidationResult vr =
        cumes::ValidationResult(cumes::ValidationReport{});
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
    // Non-fatal validation warnings (an unknown input key, a skipped
    // out-of-range boundary harmonic, ...). The legacy parser printed these
    // to stderr; reproduce its wording ("cuMES: WARNING: ...").
    for (const auto& issue : vp.warnings().issues()) {
        if (issue.severity != cumes::Severity::kWarning) continue;
        if (issue.message.compare(0, 17, "unknown input key") == 0) {
            fprintf(stderr, "cuMES: WARNING: unknown input key '%s' ignored\n",
                    issue.key.c_str());
        } else {
            fprintf(stderr, "cuMES: WARNING: %s\n", issue.message.c_str());
        }
    }
    const cumes::ProblemSpec& spec = vp.spec();
    const int n_grids = static_cast<int>(spec.stages.size());

    DeviceParams<Real> p = cumes::init_params<Real>(vp);
    printf("=== cuMES — CUDA Magnetic Equilibrium Solver ===\n");
    fflush(stdout);
    printf("input: %s\n", inputPath.c_str());
    printf("precision: %s\n",
           sizeof(Real) == sizeof(double) ? "double" : "float");
    printf("mpol=%d ntor=%d nfp=%d ntheta=%d nzeta=%d nZnT=%d ncurr=%d\n",
           p.mpol, p.ntor, p.nfp, p.ntheta, p.nzeta, p.nZnT, p.ncurr);
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
    SolverResult<Real> result{false,     0,         Real(1.0), Real(1.0),
                              Real(1.0), Real(0.9), {}};
    int total_iter = 0;

    try {
        // Stage-0 seed on ns_array[0]: a checkpoint restart, or the
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
                fprintf(stderr,
                        "cuMES: restart checkpoint (ns=%d, mnmax=%d) "
                        "does not match stage-0 grid (ns=%zu, mnmax=%d)\n",
                        snap.ns, snap.mnmax, spec.stages[0].radial_surfaces,
                        p.mnmax);
                return EXIT_FAILURE;
            }
            seed = cumes::restart_state<Real>(p, vp, snap);
        } else {
            seed = cumes::init_state<Real>(p, vp);
        }

        // One explicit nonblocking compute stream owns the whole run: the
        // stage setup (synchronous default-stream copies) completes before the
        // solve, and every hot-loop kernel / cuFFT transform / D2H transfer is
        // enqueued on this stream (Phase 6A explicit-stream scheduling).
        cumes::Stream compute_stream;
        // A checkpoint restart is a free-boundary HOT restart (vmecpp: the
        // vacuum state starts INITIALIZED so the first pass runs the vacuum
        // block).
        auto outcome = cumes::MultigridSolver<Real>::run(
            p, vp, std::move(seed), compute_stream.get(),
            /*hot_start=*/!restartPath.empty());
        storage = std::move(outcome.state);
        result = outcome.result;
        total_iter = outcome.total_iterations;

        // vmecpp semantics (vmec.cc:367-392): a stage that exhausts its
        // iteration cap without meeting ftol fails the whole run. Single-grid
        // runs keep the lenient report-and-return path below.
        if (outcome.failed_stage >= 0) {
            int g = outcome.failed_stage;
            fprintf(stderr,
                    "FATAL: grid stage %d/%d (ns=%d) completed %d/%d "
                    "iterations without meeting ftol=%.0e; final "
                    "residuals fsqr=%.3e fsqz=%.3e fsql=%.3e\n",
                    g + 1, n_grids, p.ns, result.iterations, p.max_iter,
                    (double)p.ftol, (double)result.fsqr, (double)result.fsqz,
                    (double)result.fsql);
            return EXIT_FAILURE;
        }

        // Output success is part of the run result: a converged solve whose
        // state file could not be written must NOT exit 0.
        //
        // ONE host snapshot serves every backend and the optional checkpoint
        // (completion plan step 2.2): the binary, NetCDF and HDF5 writers all
        // consume the same host EquilibriumSnapshot + RunReport through the
        // Writer interface — no backend reads device state or performs its own
        // D2H copies.
        bool output_ok = true;
        cumes::EquilibriumSnapshot snapshot =
            cumes::snapshot_from_device(storage);
        // The embedded normalized-input record: every output container
        // (including the checkpoint below) carries it, so consumers can
        // reconstruct the equilibrium without the input JSON.
        outcome.report.input_params = cumes::make_input_params(vp);
        const cumes::OutputSpec spec = outSpec;
        auto writer = cumes::make_writer(spec.format);
        if (!writer) {
            fprintf(stderr, "cuMES: no writer for suffix '%s'\n",
                    cumes::output_suffix(spec.format));
            output_ok = false;
        } else {
            fill_provenance(outcome.report, inputPath, sourceHash);
            const cumes::Status status =
                writer->write_atomic(snapshot, outcome.report, spec, vp);
            if (status.has_value()) {
                printf("Saved %s state to %s\n",
                       cumes::output_suffix(spec.format), spec.path.c_str());
            } else {
                fprintf(stderr, "cuMES: %s\n", status.error().c_str());
                output_ok = false;
            }
        }

        // Optional versioned restart checkpoint (blueprint §6.13): written
        // after the solve so a run that stops at its iteration cap can be
        // resumed via --restart.
        if (!checkpointPath.empty()) {
            const cumes::Status status = cumes::write_checkpoint(
                snapshot, outcome.report.input_params, checkpointPath);
            if (status.has_value()) {
                printf("Saved checkpoint to %s\n", checkpointPath.c_str());
            } else {
                fprintf(stderr, "cuMES: %s\n", status.error().c_str());
                output_ok = false;
            }
        }

        outputPrint<Real>(storage, p, result.iterations, result.converged,
                          result.fsqr, result.fsqz, result.fsql);
        if (n_grids > 1)
            printf("multigrid: total effective iterations over %d grids = %d\n",
                   n_grids, total_iter);
        printf("\nDone.\n");
        if (!output_ok) {
            fprintf(stderr, "cuMES: FAILED to write output state (%s)\n",
                    outputPath.c_str());
            return EXIT_FAILURE;
        }
        return result.converged ? 0 : 1;
    } catch (const cumes::CumesError& e) {
        fprintf(stderr, "cuMES: %s\n", e.what());
        return EXIT_FAILURE;
    }
}
