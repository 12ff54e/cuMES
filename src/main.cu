// main.cu — entry point: parse/validate → init → solve → output.
//
// Input: argv[1] is the JSON input file (vmecpp indata schema), or
// --input <path>; without either, inputs/solovev.json is used. Config is
// parsed + validated by the Phase 2 host model (read_and_validate →
// to_input_params, see cumes/config/*.hpp), which reproduces the legacy
// parser's defaults/folding field-for-field (proved by test_host_config's
// testAdapterParity).
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

#include "input.h"  // InputParams (via the to_input_params bridge)
#include "vmec_types.h"
#include "fourier.cuh"
#include "geometry.cuh"
#include "forces.cuh"
#include "solver.cuh"
#include "output.cuh"
#include "profiles.cuh"
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

static GridParams<Real> initParams(const InputParams& ip) {
    GridParams<Real> p;
    p.ns=ip.ns; p.mpol=ip.mpol; p.ntor=ip.ntor;
    p.ntheta=ip.ntheta; p.nzeta=ip.nzeta; p.nfp=ip.nfp;
    p.nZnT=p.ntheta*p.nzeta;
    p.mnmax=p.mpol*(p.ntor+1);   // folded basis: mode = m*(ntor+1)+n
    p.ncurr=ip.ncurr;
    p.delt=ip.delt; p.ftol=ip.ftol; p.max_iter=ip.max_iter;
    p.tcon0=Real(ip.tcon0);      // constraint-force multiplier
    p.lamscale=Real(0.0);        // set by profilesCreate
    return p;
}

// Initial state from vmecpp's interpFromBoundaryAndAxis (fourier_geometry.cc):
//   m=0: linear interpolation in s between the magnetic axis (raxis_c /
//        zaxis_s) and the boundary; zmnsc/rmnss have no m=0 content.
//   m>0: s^(m/2) radial envelope so higher modes vanish faster near the axis.
// cuMES stores the plain physical coefficients (vmecpp's internal state
// divides by mscale*nscale, but its mscale'd basis makes the real-space
// reconstruction identical).
template <typename T>
static cumes::SpectralStorage<T> initState(const GridParams<T>& p, const InputParams& ip) {
    size_t nb = (size_t)p.ns * p.mnmax * sizeof(T);
    cumes::SpectralStorage<T> storage(p.ns, p.mnmax);
    SpectralState<T> st = storage.legacy_view();

    auto* c=new T[p.ns*p.mnmax](), *s=new T[p.ns*p.mnmax]();
    auto* zsc=new T[p.ns*p.mnmax](), *zcs=new T[p.ns*p.mnmax]();
    auto* lsc=new T[p.ns*p.mnmax](), *lcs=new T[p.ns*p.mnmax]();

    for(int j=0;j<p.ns;++j){
        T sFlux = T(j)/T(p.ns-1);          // normalized flux s
        T sqrtS  = std::sqrt(sFlux);        // sqrt(s)
        for(int m=0;m<p.mpol;++m){
            for(int n=0;n<p.ntor+1;++n){
                int mn = m*(p.ntor+1)+n;
                if(m==0){
                    // m=0: linear in s between axis and boundary
                    c[j+mn*p.ns]   = sFlux*T(ip.rbcc[0][n]) + (T(1.0)-sFlux)*T(ip.raxis_c[n]);
                    zcs[j+mn*p.ns] = sFlux*T(ip.zbcs[0][n]) - (T(1.0)-sFlux)*T(ip.zaxis_s[n]);
                    // rmnss/zmnsc: no m=0 content; lambda: zero initially
                } else if(m==1){
                    // m=1: s^(1/2) radial envelope (s^(m/2)), matching
                    // vmecpp's physical state (interpFromBoundaryAndAxis).
                    // NOTE: the real-space odd-parity values then carry the
                    // 1/max(sqrt(s),sqrt(1/(ns-1))) decomposition factor
                    // (applied in the inverse DFT), so the real-space m=1
                    // contribution is constant across the interior — matching
                    // vmecpp's decomposed real space (its real-space odd =
                    // physical/max).
                    T w = sqrtS;  // s^(1/2)
                    c[j+mn*p.ns]   = w * T(ip.rbcc[m][n]);
                    s[j+mn*p.ns]   = w * T(ip.rbss[m][n]);
                    zsc[j+mn*p.ns] = w * T(ip.zbsc[m][n]);
                    zcs[j+mn*p.ns] = w * T(ip.zbcs[m][n]);
                } else {
                    // m>=2: s^(m/2) radial envelope, vanishing at axis
                    T w = std::pow(sqrtS, m);  // s^(m/2)
                    c[j+mn*p.ns]   = w * T(ip.rbcc[m][n]);
                    s[j+mn*p.ns]   = w * T(ip.rbss[m][n]);
                    zsc[j+mn*p.ns] = w * T(ip.zbsc[m][n]);
                    zcs[j+mn*p.ns] = w * T(ip.zbcs[m][n]);
                }
                // lmnsc/lmncs: zero initially (lambda is a free gauge)
            }
        }
    }
    printf("  initState: vmecpp interpFromBoundaryAndAxis (m>0 s^(m/2))\n");

    cumes::check_cuda(cudaMemcpy(st.d_rmncc,c,nb,cudaMemcpyHostToDevice),"cpy cc");
    cumes::check_cuda(cudaMemcpy(st.d_rmnss,s,nb,cudaMemcpyHostToDevice),"cpy ss");
    cumes::check_cuda(cudaMemcpy(st.d_zmnsc,zsc,nb,cudaMemcpyHostToDevice),"cpy zsc");
    cumes::check_cuda(cudaMemcpy(st.d_zmncs,zcs,nb,cudaMemcpyHostToDevice),"cpy zcs");
    cumes::check_cuda(cudaMemcpy(st.d_lmnsc,lsc,nb,cudaMemcpyHostToDevice),"cpy lsc");
    cumes::check_cuda(cudaMemcpy(st.d_lmncs,lcs,nb,cudaMemcpyHostToDevice),"cpy lcs");
    delete[] c; delete[] s; delete[] zsc; delete[] zcs; delete[] lsc; delete[] lcs;
    return storage;
}

// Restart state from a host snapshot (read_checkpoint / convert_legacy_init),
// uploading the six families and applying the same LCFS-boundary + axis-
// regularity patch the legacy CUMES_LOAD_INIT path did. The checkpoint stores
// doubles regardless of T; the conversion mirrors outputSaveBinary's T->double
// in reverse.
template <typename T>
static cumes::SpectralStorage<T> restartState(const GridParams<T>& p, const InputParams& ip,
                                              const cumes::EquilibriumSnapshot& snap) {
    const size_t one = (size_t)p.ns * p.mnmax;
    const size_t nb = one * sizeof(T);
    cumes::SpectralStorage<T> storage(p.ns, p.mnmax);
    SpectralState<T> st = storage.legacy_view();

    auto* c = new T[one];   // rmncc
    auto* zsc = new T[one]; // zmnsc
    auto* lsc = new T[one]; // lmnsc
    auto* s = new T[one];   // rmnss
    auto* zcs = new T[one]; // zmncs
    auto* lcs = new T[one]; // lmncs
    for (size_t i = 0; i < one; ++i) {
        c[i]   = T(snap.families[cumes::EquilibriumSnapshot::kRmncc][i]);
        zsc[i] = T(snap.families[cumes::EquilibriumSnapshot::kZmnsc][i]);
        lsc[i] = T(snap.families[cumes::EquilibriumSnapshot::kLmnsc][i]);
        s[i]   = T(snap.families[cumes::EquilibriumSnapshot::kRmnss][i]);
        zcs[i] = T(snap.families[cumes::EquilibriumSnapshot::kZmncs][i]);
        lcs[i] = T(snap.families[cumes::EquilibriumSnapshot::kLmncs][i]);
    }

    // vmecpp stores boundary values separately (not in the spectral state);
    // cuMES embeds the boundary in the spectral coefficients at j=ns-1. Patch
    // the LCFS values to match the folded boundary; also zero m>0 modes at the
    // magnetic axis (j=0) — vmecpp does this via extrapolateTowardsAxis().
    {
        int jB = p.ns - 1;  // LCFS index
        for (int m = 0; m < p.mpol; ++m) {
            for (int n = 0; n < p.ntor + 1; ++n) {
                int mn = m * (p.ntor + 1) + n;
                c[jB + mn * p.ns] = T(ip.rbcc[m][n]);
                s[jB + mn * p.ns] = T(ip.rbss[m][n]);
                zsc[jB + mn * p.ns] = T(ip.zbsc[m][n]);
                zcs[jB + mn * p.ns] = T(ip.zbcs[m][n]);
                if (m > 0) {
                    c[0 + mn * p.ns] = T(0.0);
                    s[0 + mn * p.ns] = T(0.0);
                    zsc[0 + mn * p.ns] = T(0.0);
                    zcs[0 + mn * p.ns] = T(0.0);
                    lsc[0 + mn * p.ns] = T(0.0);
                    lcs[0 + mn * p.ns] = T(0.0);
                }
            }
        }
    }

    cumes::check_cuda(cudaMemcpy(st.d_rmncc, c, nb, cudaMemcpyHostToDevice), "restart cc");
    cumes::check_cuda(cudaMemcpy(st.d_zmnsc, zsc, nb, cudaMemcpyHostToDevice), "restart zsc");
    cumes::check_cuda(cudaMemcpy(st.d_lmnsc, lsc, nb, cudaMemcpyHostToDevice), "restart lsc");
    cumes::check_cuda(cudaMemcpy(st.d_rmnss, s, nb, cudaMemcpyHostToDevice), "restart ss");
    cumes::check_cuda(cudaMemcpy(st.d_zmncs, zcs, nb, cudaMemcpyHostToDevice), "restart zcs");
    cumes::check_cuda(cudaMemcpy(st.d_lmncs, lcs, nb, cudaMemcpyHostToDevice), "restart lcs");
    delete[] c; delete[] s; delete[] zsc; delete[] zcs; delete[] lsc; delete[] lcs;
    printf("  restartState: uploaded checkpoint + LCFS/axis patch\n");
    return storage;
}

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
    const auto to_ip = vr.value().to_input_params();
    if (!to_ip.has_value()) {
        fprintf(stderr, "cuMES: %s\n", to_ip.error().c_str());
        return EXIT_FAILURE;
    }
    InputParams ip = to_ip.value();

    GridParams<Real> p = initParams(ip);
    printf("=== cuMES — CUDA Magnetic Equilibrium Solver ===\n");
    fflush(stdout);
    printf("input: %s\n", inputPath);
    printf("precision: %s\n", sizeof(Real) == sizeof(double) ? "double" : "float");
    printf("mpol=%d ntor=%d nfp=%d ntheta=%d nzeta=%d nZnT=%d ncurr=%d\n",
           p.mpol,p.ntor,p.nfp,p.ntheta,p.nzeta,p.nZnT,p.ncurr);
    // Multi-radial-grid stage sequence (vmecpp ns_array/niter_array/ftol_array)
    printf("grids=%d: ns", ip.n_grids);
    for (int g = 0; g < ip.n_grids; ++g) printf("%s%d", g == 0 ? "" : "->", ip.ns_array[g]);
    printf(" (niter");
    for (int g = 0; g < ip.n_grids; ++g) printf(" %d", ip.niter_array[g]);
    printf(", ftol");
    for (int g = 0; g < ip.n_grids; ++g) printf(" %.0e", ip.ftol_array[g]);
    printf(")\n");

    // ---- Multi-radial-grid stage loop (delegated to MultigridSolver) ----
    cumes::SpectralStorage<Real> storage;
    SolverResult<Real> result{false, 0, Real(1.0), Real(1.0), Real(1.0), Real(0.9)};
    int total_iter = 0;

    try {
        // Stage-0 seed on ns_array[0]: a checkpoint/legacy-init restart, or the
        // interpFromBoundaryAndAxis cold start.
        p.ns = ip.ns_array[0];
        p.max_iter = ip.niter_array[0];
        p.ftol = ip.ftol_array[0];
        cumes::SpectralStorage<Real> seed;
        if (!restartPath.empty()) {
            auto ck = cumes::read_checkpoint(restartPath);
            if (!ck.has_value()) {
                fprintf(stderr, "cuMES: %s\n", ck.error().c_str());
                return EXIT_FAILURE;
            }
            const auto& snap = ck.value();
            if (snap.ns != ip.ns_array[0] || snap.mnmax != p.mnmax) {
                fprintf(stderr, "cuMES: restart checkpoint (ns=%d, mnmax=%d) "
                                "does not match stage-0 grid (ns=%d, mnmax=%d)\n",
                        snap.ns, snap.mnmax, ip.ns_array[0], p.mnmax);
                return EXIT_FAILURE;
            }
            seed = restartState<Real>(p, ip, snap);
        } else if (!restartLegacyPath.empty()) {
            auto ck = cumes::convert_legacy_init(restartLegacyPath, ip.ns_array[0], p.mnmax);
            if (!ck.has_value()) {
                fprintf(stderr, "cuMES: %s\n", ck.error().c_str());
                return EXIT_FAILURE;
            }
            seed = restartState<Real>(p, ip, ck.value());
        } else {
            seed = initState<Real>(p, ip);
        }

        auto outcome = cumes::MultigridSolver<Real>::run(p, ip, std::move(seed));
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
                    g + 1, ip.n_grids, p.ns, result.iterations, p.max_iter,
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
            output_ok = outputSave<Real>(storage.legacy_view(), p, ip, result,
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
        if (ip.n_grids > 1)
            printf("multigrid: total effective iterations over %d grids = %d\n",
                   ip.n_grids, total_iter);
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
