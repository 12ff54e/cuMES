// main.cu — entry point: parse/validate → init → solve → output.
//
// CLI: cuMES [OPTION]... INPUT_FILE. The mandatory positional input is a
// vmecpp-indata JSON file. Output, restart, checkpoint, and compatibility are
// options; output defaults to $PWD/cumes-output.bin. Config is parsed +
// validated by the Phase 2 host model (read_and_validate, see
// cumes/config/*.hpp) into an immutable ValidatedProblem the solver consumes
// directly.
//
// Output: every backend writes the schema-v1 container (binary/NetCDF/HDF5
// through the Writer interface, dispatched by the output suffix).
//
// Restart: --restart <checkpoint> (v1 checkpoint) replaces the removed
// CUMES_LOAD_INIT environment path.
//
// Precision: `Real` (vmec_types.h) is the compile-time switch between double
// and float — configure with -DCUMES_USE_FLOAT=ON.
#include "clap.h"
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

#ifdef CUMES_HAVE_MAGNETIC_COORDINATE
#include "cumes/io/magnetic_coordinate_bridge.hpp"

#include <magnetic_coordinate/boozer_output.hpp>
#include <magnetic_coordinate/transform.hpp>
#endif

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
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
static std::string read_file_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return "";
    return std::string(std::istreambuf_iterator<char>(in),
                       std::istreambuf_iterator<char>());
}

// FNV-1a 64-bit hash of raw bytes, hex-encoded. Cheap input provenance for the
// v1 schema; not a cryptographic digest. Identical algorithm to the removed
// hashFile(), so recorded source_hash values are unchanged for unchanged files.
static std::string hash_bytes(const std::string& bytes) {
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
// context is live for the device/driver queries. source_hash is the hash of the
// input bytes captured at read time — the input file is NOT re-opened here
// (a mid-solve replacement must not change the recorded provenance).
static void fill_provenance(cumes::RunReport& report,
                            const std::string& input_path,
                            const std::string& source_hash) {
    report.build.revision = CUMES_GIT_REVISION;
    report.build.dirty = (CUMES_GIT_DIRTY != 0);
    report.build.build_type = CUMES_BUILD_TYPE;
    report.build.scalar_type =
        sizeof(Real) == sizeof(double) ? "double" : "float";
    report.build.precision_policy = CUMES_PRECISION_POLICY_NAME;
    report.build.compile_flags = CUMES_PRECISION_FLAGS;
    report.input.source_path = input_path;
    report.input.source_hash = source_hash;

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

static std::string_view severity_name(cumes::Severity s) {
    return s == cumes::Severity::ERROR ? "error" : "warning";
}

int main(int argc, char** argv) {
    // ---- CLI ----------------------------------------------------------------
    struct CliInput {
        std::string input_path;
        std::string output_path;
        std::string restart_path;
        std::string checkpoint_path;
        std::string boozer_output_path;
        int boozer_ntheta;
        int boozer_nzeta;
        int boozer_mmax;
        int boozer_nmax;
        int boozer_radial_order;
        double boozer_resonance_tolerance;
        bool compatibility;
    };

    CLAP_BEGIN(CliInput)
    CLAP_ADD_USAGE("[OPTION]... INPUT_FILE")
    CLAP_ADD_DESCRIPTION("Solve a magnetic equilibrium on a CUDA GPU.")
    CLAP_REGISTER_ARG(input_path)
    CLAP_REGISTER_OPTION_WITH_DESCRIPTION(
        output_path, "--output", "-o",
        "write the result to PATH (default: $PWD/cumes-output.bin)")
    CLAP_REGISTER_OPTION_WITH_DESCRIPTION(restart_path, "--restart", "-r",
                                          "initialize from checkpoint PATH")
    CLAP_REGISTER_OPTION_WITH_DESCRIPTION(
        checkpoint_path, "--checkpoint", "-c",
        "write a restart checkpoint to PATH after solving")
    CLAP_REGISTER_OPTION_WITH_DESCRIPTION(
        boozer_output_path, "--boozer-output",
        "write Boozer output to PATH instead of the native result")
    CLAP_REGISTER_OPTION_WITH_DESCRIPTION(
        boozer_ntheta, "--boozer-ntheta",
        "set the Boozer output poloidal grid size")
    CLAP_REGISTER_OPTION_WITH_DESCRIPTION(
        boozer_nzeta, "--boozer-nzeta",
        "set the unchanged-toroidal-angle grid size")
    CLAP_REGISTER_OPTION_WITH_DESCRIPTION(boozer_mmax, "--boozer-mmax",
                                          "set the Boozer poloidal mode limit")
    CLAP_REGISTER_OPTION_WITH_DESCRIPTION(boozer_nmax, "--boozer-nmax",
                                          "set the Boozer toroidal mode limit")
    CLAP_REGISTER_OPTION_WITH_DESCRIPTION(
        boozer_radial_order, "--boozer-radial-order",
        "set half-to-full radial interpolation order (2 or 4)")
    CLAP_REGISTER_OPTION_WITH_DESCRIPTION(
        boozer_resonance_tolerance, "--boozer-resonance-tolerance",
        "set the resonant-denominator tolerance")
    CLAP_REGISTER_OPTION_WITH_DESCRIPTION(
        compatibility, "--compatibility",
        "warn and ignore unknown input keys instead of rejecting them")
    CLAP_END(CliInput)

    CliInput cli{};
    cli.boozer_mmax = -1;
    cli.boozer_nmax = -1;
    cli.boozer_radial_order = 4;
    cli.boozer_resonance_tolerance = 1.0e-12;
    try {
        CLAP<CliInput>::parse_input(cli, argc, argv);
    } catch (const std::exception& e) {
        std::cerr << e.what();
        return EINVAL;
    }

    if (!cli.output_path.empty() && !cli.boozer_output_path.empty()) {
        fprintf(stderr,
                "cuMES: --output and --boozer-output are mutually exclusive\n");
        return EINVAL;
    }
    const bool boozer_output_requested = !cli.boozer_output_path.empty();
    if (!boozer_output_requested && cli.output_path.empty()) {
        cli.output_path =
            (std::filesystem::current_path() / "cumes-output.bin").string();
    }

    const std::string& input_path = cli.input_path;
    const std::string& output_path = cli.output_path;
    const std::string& restart_path = cli.restart_path;
    const std::string& checkpoint_path = cli.checkpoint_path;
    const bool compatibility = cli.compatibility;

#ifndef CUMES_HAVE_MAGNETIC_COORDINATE
    if (boozer_output_requested) {
        fprintf(stderr,
                "cuMES: Boozer conversion is not available in this build; "
                "configure with -DCUMES_BUILD_MAGNETIC_COORDINATE=ON\n");
        return EXIT_FAILURE;
    }
#else
    magnetic_coordinate::TransformSettings boozer_settings;
    magnetic_coordinate::BoozerOutputSpec boozer_output_spec;
    if (boozer_output_requested) {
        if (cli.boozer_ntheta < 0 || cli.boozer_nzeta < 0 ||
            cli.boozer_mmax < -1 || cli.boozer_nmax < -1 ||
            !(cli.boozer_resonance_tolerance > 0.0) ||
            !std::isfinite(cli.boozer_resonance_tolerance) ||
            (cli.boozer_radial_order != 2 && cli.boozer_radial_order != 4)) {
            fprintf(stderr, "cuMES: invalid Boozer transform options\n");
            return EINVAL;
        }
        boozer_settings.output_ntheta = cli.boozer_ntheta;
        boozer_settings.output_nzeta = cli.boozer_nzeta;
        boozer_settings.mmax = cli.boozer_mmax;
        boozer_settings.nmax = cli.boozer_nmax;
        boozer_settings.resonance_tolerance = cli.boozer_resonance_tolerance;
        boozer_settings.radial_order =
            cli.boozer_radial_order == 2
                ? magnetic_coordinate::RadialInterpolationOrder::TWO_POINT
                : magnetic_coordinate::RadialInterpolationOrder::FOUR_POINT;
        try {
            boozer_output_spec =
                magnetic_coordinate::resolve_boozer_output_spec(
                    cli.boozer_output_path);
        } catch (const std::exception& error) {
            fprintf(stderr, "cuMES: %s\n", error.what());
            return EINVAL;
        }
        if (!magnetic_coordinate::boozer_output_format_available(
                boozer_output_spec.format)) {
            fprintf(stderr,
                    "cuMES: Boozer output format '%s' is not available in "
                    "this build\n",
                    magnetic_coordinate::boozer_output_suffix(
                        boozer_output_spec.format)
                        .data());
            return EXIT_FAILURE;
        }
    }
#endif

    // ---- output preflight (before any CUDA work) ----------------------------
    // A requested-but-unlinked format (e.g. .nc on a binary-only build) and an
    // unknown suffix are rejected HERE, before the CUDA context is created and
    // before any grid stage runs.
    cumes::OutputSpec output_spec;
    if (!boozer_output_requested) {
        auto resolved = cumes::resolve_output_spec(output_path);
        if (!resolved.has_value()) {
            fprintf(stderr, "cuMES: %s\n", resolved.error().c_str());
            return EXIT_FAILURE;
        }
        output_spec = resolved.value();
        if (!cumes::output_format_available(output_spec.format)) {
            fprintf(stderr,
                    "cuMES: output format '%s' is not available in this "
                    "build; no output will be written\n",
                    cumes::output_suffix(output_spec.format).data());
            return EXIT_FAILURE;
        }
    }

    // ---- config: parse + validate -------------------------------------------
    cumes::SolverOptions opts;
    opts.strict_schema = !compatibility;  // strict schema-v1 is the default
#ifdef CUMES_USE_FLOAT
    // Float runs stall at ~1e-7 (the float rounding floor) and can never meet
    // the double-tuned stage tolerances. Declaring the mixed-float policy lets
    // validation reject impossible tolerances instead of failing every stage at
    // the end of the run.
    opts.precision = cumes::PrecisionPolicy::MIXED_FLOAT;
#endif
    // Capture the raw input bytes once, at read time: the v1 provenance
    // source_hash must be the hash of the bytes the solver actually consumed,
    // not of whatever the path holds after the solve (TOCTOU — the pre-fix
    // fill_provenance re-opened and re-hashed the file post-solve).
    const std::string inputBytes = read_file_bytes(input_path);
    const std::string source_hash =
        inputBytes.empty() ? "" : hash_bytes(inputBytes);
    cumes::ValidationResult vr =
        cumes::ValidationResult(cumes::ValidationReport{});
    try {
        vr = cumes::read_and_validate(input_path, opts);
    } catch (const std::exception& e) {
        fprintf(stderr, "cuMES: error loading input file: %s\n", e.what());
        return EXIT_FAILURE;
    }
    if (!vr.has_value()) {
        fprintf(stderr, "cuMES: input validation failed:\n");
        for (const auto& issue : vr.error().issues()) {
            fprintf(stderr, "  [%s] %s: %s\n",
                    severity_name(issue.severity).data(), issue.key.c_str(),
                    issue.message.c_str());
        }
        return EXIT_FAILURE;
    }
    const cumes::ValidatedProblem& vp = vr.value();
    // Non-fatal validation warnings (an unknown input key, a skipped
    // out-of-range boundary harmonic, ...). The legacy parser printed these
    // to stderr; reproduce its wording ("cuMES: WARNING: ...").
    for (const auto& issue : vp.warnings().issues()) {
        if (issue.severity != cumes::Severity::WARNING) continue;
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
    printf("input: %s\n", input_path.c_str());
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
        if (!restart_path.empty()) {
            auto ck = cumes::read_checkpoint(restart_path);
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
        auto outcome = cumes::MultigridSolver<Real>::run(
            p, vp, std::move(seed), compute_stream.get(),
            /*hot_start=*/!restart_path.empty());
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
        bool native_output_failed = false;
        bool checkpoint_failed = false;
        bool boozer_output_failed = false;
        cumes::EquilibriumSnapshot snapshot = std::move(outcome.snapshot);
        cumes::populate_snapshot_state_from_device(storage, snapshot);
        // The embedded normalized-input record: every output container
        // (including the checkpoint below) carries it, so consumers can
        // reconstruct the equilibrium without the input JSON.
        outcome.report.input_params = cumes::make_input_params(vp);
        if (!boozer_output_requested) {
            auto writer = cumes::make_writer(output_spec.format);
            if (!writer) {
                fprintf(stderr, "cuMES: no writer for suffix '%s'\n",
                        cumes::output_suffix(output_spec.format).data());
                output_ok = false;
                native_output_failed = true;
            } else {
                fill_provenance(outcome.report, input_path, source_hash);
                const cumes::Status status = writer->write_atomic(
                    snapshot, outcome.report, output_spec, vp);
                if (status.has_value()) {
                    printf("Saved %s state to %s\n",
                           cumes::output_suffix(output_spec.format).data(),
                           output_spec.path.c_str());
                } else {
                    fprintf(stderr, "cuMES: %s\n", status.error().c_str());
                    output_ok = false;
                    native_output_failed = true;
                }
            }
        }

        // Optional v2 restart checkpoint (blueprint §6.13): written after the
        // solve so a run that stops at its iteration cap can be resumed via
        // --restart.
        if (!checkpoint_path.empty()) {
            const cumes::Status status = cumes::write_checkpoint(
                snapshot, outcome.report.input_params, checkpoint_path);
            if (status.has_value()) {
                printf("Saved checkpoint to %s\n", checkpoint_path.c_str());
            } else {
                fprintf(stderr, "cuMES: %s\n", status.error().c_str());
                output_ok = false;
                checkpoint_failed = true;
            }
        }

#ifdef CUMES_HAVE_MAGNETIC_COORDINATE
        if (boozer_output_requested) {
            try {
                const auto boozer_result =
                    magnetic_coordinate::transform_to_boozer(
                        cumes::make_magnetic_coordinate_view(
                            snapshot, outcome.report.input_params),
                        boozer_settings);
                const auto real_modes =
                    static_cast<std::size_t>(boozer_result.spectrum.mmax + 1) *
                    static_cast<std::size_t>(boozer_result.spectrum.nmax + 1);
                printf(
                    "Calculated Boozer representation: %zu non-axis "
                    "surfaces, %dx%d mixed-grid points per surface, "
                    "%zu real modes per surface\n",
                    boozer_result.s.size(), boozer_result.grid.ntheta,
                    boozer_result.grid.nzeta, real_modes);
                magnetic_coordinate::write_boozer_output(
                    boozer_output_spec, boozer_result, input_path);
                printf("Saved Boozer representation to %s\n",
                       cli.boozer_output_path.c_str());
            } catch (const std::exception& error) {
                fprintf(stderr, "cuMES: Boozer transform failed: %s\n",
                        error.what());
                output_ok = false;
                boozer_output_failed = true;
            }
        }
#endif

        output_print<Real>(storage, p, result.iterations, result.converged,
                           result.fsqr, result.fsqz, result.fsql);
        if (n_grids > 1)
            printf("multigrid: total effective iterations over %d grids = %d\n",
                   n_grids, total_iter);
        printf("\nDone.\n");
        if (!output_ok) {
            if (native_output_failed) {
                fprintf(stderr, "cuMES: FAILED to write output state (%s)\n",
                        output_path.c_str());
            }
            if (checkpoint_failed) {
                fprintf(stderr, "cuMES: FAILED to write checkpoint (%s)\n",
                        checkpoint_path.c_str());
            }
            if (boozer_output_failed) {
                fprintf(stderr, "cuMES: FAILED to write Boozer output (%s)\n",
                        cli.boozer_output_path.c_str());
            }
            return EXIT_FAILURE;
        }
        return result.converged ? 0 : 1;
    } catch (const cumes::CumesError& e) {
        fprintf(stderr, "cuMES: %s\n", e.what());
        return EXIT_FAILURE;
    }
}
