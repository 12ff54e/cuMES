// main.cu — entry point: init → solve → output.
// Input selection: argv[1] is the JSON input file (vmecpp indata schema);
// without an argument the default inputs/solovev.json is used. Parsing and
// validation live in src/input_json.cu (see include/input_json.h).
//
// Precision: `Real` (vmec_types.h) is the compile-time switch between double
// and float — configure with -DCUMES_USE_FLOAT=ON. All modules are templated
// on T; this file instantiates them with Real.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <exception>

#include "input_json.h"
#include "vmec_types.h"
#include "fourier.cuh"
#include "geometry.cuh"
#include "forces.cuh"
#include "solver.cuh"
#include "output.cuh"
#include "profiles.cuh"
#include "refine.cuh"

static void checkCuda(cudaError_t err, const char* tag) {
    if(err!=cudaSuccess){fprintf(stderr,"CUDA error [%s]: %s\n",tag,cudaGetErrorString(err));exit(1);}
}

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
static void initState(SpectralState<T>& st, const GridParams<T>& p, const InputParams& ip) {
    bool loadInit = false;
    if (const char* e = getenv("CUMES_LOAD_INIT")) loadInit = atoi(e) != 0;
    size_t nb = (size_t)p.ns * p.mnmax * sizeof(T);
    if (loadInit) {
        FILE* fp = fopen("vmecpp_init.bin", "rb");
    if (fp) {
        int ns_file, mnmax_file;
        bool hdrRead = fread(&ns_file, sizeof(int), 1, fp) == 1 &&
                       fread(&mnmax_file, sizeof(int), 1, fp) == 1;
        bool headerOk = hdrRead && ns_file == p.ns && mnmax_file == p.mnmax;
        if (headerOk) {
            printf("Loading initial state from vmecpp_init.bin (ns=%d, mnmax=%d)\n", ns_file, mnmax_file);
            // The file stores doubles regardless of T: read into double
            // staging, convert, then upload.
            auto* h_rmncc = new double[p.ns * p.mnmax];
            auto* h_zmnsc = new double[p.ns * p.mnmax];
            auto* h_lmnsc = new double[p.ns * p.mnmax];
            auto* h_rmnss = new double[p.ns * p.mnmax];
            auto* h_zmncs = new double[p.ns * p.mnmax];
            auto* h_lmncs = new double[p.ns * p.mnmax];
            if (fread(h_rmncc, sizeof(double), p.ns * p.mnmax, fp) != (size_t)(p.ns * p.mnmax) ||
                fread(h_zmnsc, sizeof(double), p.ns * p.mnmax, fp) != (size_t)(p.ns * p.mnmax) ||
                fread(h_lmnsc, sizeof(double), p.ns * p.mnmax, fp) != (size_t)(p.ns * p.mnmax) ||
                fread(h_rmnss, sizeof(double), p.ns * p.mnmax, fp) != (size_t)(p.ns * p.mnmax) ||
                fread(h_zmncs, sizeof(double), p.ns * p.mnmax, fp) != (size_t)(p.ns * p.mnmax) ||
                fread(h_lmncs, sizeof(double), p.ns * p.mnmax, fp) != (size_t)(p.ns * p.mnmax)) {
                fprintf(stderr, "vmecpp_init.bin: truncated state data\n");
                fclose(fp);
                exit(1);
            }
            fclose(fp);

            auto* c = new T[p.ns * p.mnmax];
            auto* s = new T[p.ns * p.mnmax];
            auto* zsc = new T[p.ns * p.mnmax];
            auto* zcs = new T[p.ns * p.mnmax];
            auto* lsc = new T[p.ns * p.mnmax];
            auto* lcs = new T[p.ns * p.mnmax];
            for (size_t i = 0; i < (size_t)p.ns * p.mnmax; ++i) {
                c[i] = T(h_rmncc[i]); s[i] = T(h_rmnss[i]);
                zsc[i] = T(h_zmnsc[i]); zcs[i] = T(h_zmncs[i]);
                lsc[i] = T(h_lmnsc[i]); lcs[i] = T(h_lmncs[i]);
            }
            delete[] h_rmncc; delete[] h_zmnsc; delete[] h_lmnsc;
            delete[] h_rmnss; delete[] h_zmncs; delete[] h_lmncs;

            checkCuda(cudaMalloc(&st.d_rmncc,nb),"cc"); checkCuda(cudaMalloc(&st.d_rmnss,nb),"ss");
            checkCuda(cudaMalloc(&st.d_zmnsc,nb),"zsc"); checkCuda(cudaMalloc(&st.d_zmncs,nb),"zcs");
            checkCuda(cudaMalloc(&st.d_lmnsc,nb),"lsc"); checkCuda(cudaMalloc(&st.d_lmncs,nb),"lcs");
            checkCuda(cudaMalloc(&st.d_v_rmncc,nb),"vcc"); checkCuda(cudaMalloc(&st.d_v_rmnss,nb),"vss");
            checkCuda(cudaMalloc(&st.d_v_zmnsc,nb),"vzsc"); checkCuda(cudaMalloc(&st.d_v_zmncs,nb),"vzcs");
            checkCuda(cudaMalloc(&st.d_v_lmnsc,nb),"vlsc"); checkCuda(cudaMalloc(&st.d_v_lmncs,nb),"vlcs");

            checkCuda(cudaMemcpy(st.d_rmncc, c, nb, cudaMemcpyHostToDevice), "cpy cc");
            checkCuda(cudaMemcpy(st.d_zmnsc, zsc, nb, cudaMemcpyHostToDevice), "cpy zsc");
            checkCuda(cudaMemcpy(st.d_lmnsc, lsc, nb, cudaMemcpyHostToDevice), "cpy lsc");
            checkCuda(cudaMemcpy(st.d_rmnss, s, nb, cudaMemcpyHostToDevice), "cpy ss");
            checkCuda(cudaMemcpy(st.d_zmncs, zcs, nb, cudaMemcpyHostToDevice), "cpy zcs");
            checkCuda(cudaMemcpy(st.d_lmncs, lcs, nb, cudaMemcpyHostToDevice), "cpy lcs");

            // vmecpp stores boundary values separately (not in the spectral
            // state). cuMES embeds the boundary in the spectral coefficients
            // at j=ns-1. Patch the LCFS values to match the folded boundary;
            // also zero out m>0 modes at the magnetic axis (j=0) — vmecpp
            // does this via extrapolateTowardsAxis().
            {
                int jB = p.ns - 1;  // LCFS index
                for (int m = 0; m < p.mpol; ++m) {
                    for (int n = 0; n < p.ntor + 1; ++n) {
                        int mn = m * (p.ntor + 1) + n;
                        c[jB + mn * p.ns] = T(ip.rbcc[m][n]);
                        s[jB + mn * p.ns] = T(ip.rbss[m][n]);
                        zsc[jB + mn * p.ns] = T(ip.zbsc[m][n]);
                        zcs[jB + mn * p.ns] = T(ip.zbcs[m][n]);
                        // Fix axis: zero all m>0 modes at j=0 (axis regularity)
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
                checkCuda(cudaMemcpy(st.d_rmncc, c, nb, cudaMemcpyHostToDevice), "set cc");
                checkCuda(cudaMemcpy(st.d_rmnss, s, nb, cudaMemcpyHostToDevice), "set ss");
                checkCuda(cudaMemcpy(st.d_zmnsc, zsc, nb, cudaMemcpyHostToDevice), "set zsc");
                checkCuda(cudaMemcpy(st.d_zmncs, zcs, nb, cudaMemcpyHostToDevice), "set zcs");
                checkCuda(cudaMemcpy(st.d_lmnsc, lsc, nb, cudaMemcpyHostToDevice), "set lsc");
                checkCuda(cudaMemcpy(st.d_lmncs, lcs, nb, cudaMemcpyHostToDevice), "set lcs");
            }
            delete[] c; delete[] s; delete[] zsc; delete[] zcs; delete[] lsc; delete[] lcs;
            printf("  Fixed LCFS boundary and axis regularity\n");
            checkCuda(cudaMemset(st.d_v_rmncc,0,nb),"vcc"); checkCuda(cudaMemset(st.d_v_rmnss,0,nb),"vss");
            checkCuda(cudaMemset(st.d_v_zmnsc,0,nb),"vzsc"); checkCuda(cudaMemset(st.d_v_zmncs,0,nb),"vzcs");
            checkCuda(cudaMemset(st.d_v_lmnsc,0,nb),"vlsc"); checkCuda(cudaMemset(st.d_v_lmncs,0,nb),"vlcs");

            return;
        }
        if (hdrRead) {
            // File exists but the resolution doesn't match the (stage-0)
            // grid — e.g. an init file from another ns. Established behavior
            // is a silent cold-start fallback; vmecpp hard-errors instead.
            fprintf(stderr, "WARNING: vmecpp_init.bin header (ns=%d, mnmax=%d) "
                            "does not match ns=%d, mnmax=%d — falling back to "
                            "the cold start\n",
                    ns_file, mnmax_file, p.ns, p.mnmax);
        }
            fclose(fp);
        }
    }

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

    checkCuda(cudaMalloc(&st.d_rmncc,nb),"cc"); checkCuda(cudaMalloc(&st.d_rmnss,nb),"ss");
    checkCuda(cudaMalloc(&st.d_zmnsc,nb),"zsc"); checkCuda(cudaMalloc(&st.d_zmncs,nb),"zcs");
    checkCuda(cudaMalloc(&st.d_lmnsc,nb),"lsc"); checkCuda(cudaMalloc(&st.d_lmncs,nb),"lcs");
    checkCuda(cudaMalloc(&st.d_v_rmncc,nb),"vcc"); checkCuda(cudaMalloc(&st.d_v_rmnss,nb),"vss");
    checkCuda(cudaMalloc(&st.d_v_zmnsc,nb),"vzsc"); checkCuda(cudaMalloc(&st.d_v_zmncs,nb),"vzcs");
    checkCuda(cudaMalloc(&st.d_v_lmnsc,nb),"vlsc"); checkCuda(cudaMalloc(&st.d_v_lmncs,nb),"vlcs");

    checkCuda(cudaMemcpy(st.d_rmncc,c,nb,cudaMemcpyHostToDevice),"cpy cc");
    checkCuda(cudaMemcpy(st.d_rmnss,s,nb,cudaMemcpyHostToDevice),"cpy ss");
    checkCuda(cudaMemcpy(st.d_zmnsc,zsc,nb,cudaMemcpyHostToDevice),"cpy zsc");
    checkCuda(cudaMemcpy(st.d_zmncs,zcs,nb,cudaMemcpyHostToDevice),"cpy zcs");
    checkCuda(cudaMemcpy(st.d_lmnsc,lsc,nb,cudaMemcpyHostToDevice),"cpy lsc");
    checkCuda(cudaMemcpy(st.d_lmncs,lcs,nb,cudaMemcpyHostToDevice),"cpy lcs");
    checkCuda(cudaMemset(st.d_v_rmncc,0,nb),"vcc"); checkCuda(cudaMemset(st.d_v_rmnss,0,nb),"vss");
    checkCuda(cudaMemset(st.d_v_zmnsc,0,nb),"vzsc"); checkCuda(cudaMemset(st.d_v_zmncs,0,nb),"vzcs");
    checkCuda(cudaMemset(st.d_v_lmnsc,0,nb),"vlsc"); checkCuda(cudaMemset(st.d_v_lmncs,0,nb),"vlcs");

    delete[] c; delete[] s; delete[] zsc; delete[] zcs; delete[] lsc; delete[] lcs;
}

template <typename T>
static void freeState(SpectralState<T>& st) {
    cudaFree(st.d_rmncc); cudaFree(st.d_rmnss); cudaFree(st.d_zmnsc);
    cudaFree(st.d_zmncs); cudaFree(st.d_lmnsc); cudaFree(st.d_lmncs);
    cudaFree(st.d_v_rmncc); cudaFree(st.d_v_rmnss); cudaFree(st.d_v_zmnsc);
    cudaFree(st.d_v_zmncs); cudaFree(st.d_v_lmnsc); cudaFree(st.d_v_lmncs);
}

int main(int argc, char** argv) {
    const char* inputPath = (argc > 1) ? argv[1] : "inputs/solovev.json";
    // Output format is selected by the argv[2] suffix: .nc -> NetCDF,
    // .h5/.hdf5 -> HDF5, .bin -> binary at that path; missing or
    // unrecognized falls back to the legacy binary cumes_state.bin.
    const char* outputPath = (argc > 2) ? argv[2] : "cumes_state.bin";
    if (argc <= 2)
        fprintf(stderr, "WARNING: no output path given (argv[2]) - "
                        "writing binary cumes_state.bin\n");
    InputParams ip;
    try {
        ip = initInputParams(inputPath);
    } catch (const std::exception& e) {
        fprintf(stderr, "cuMES: error loading input file: %s\n", e.what());
        return EXIT_FAILURE;
    }

    // Output-backend preflight: a requested-but-unlinked format (e.g. .nc on
    // a build without NetCDF) is rejected HERE, before the CUDA context is
    // created and before any grid stage runs — not after thousands of
    // solver iterations (and without the deep exit() that used to bypass
    // cleanup).
    if (!outputFormatAvailable(outputPath)) {
        fprintf(stderr, "cuMES: no output will be written; bailing out\n");
        return EXIT_FAILURE;
    }

#ifdef CUMES_USE_FLOAT
    // Float runs stall at ~1e-7 (the float rounding floor) and can never
    // meet the double-tuned stage tolerances — reject impossible values
    // instead of failing every stage at the end of the run.
    for (int g = 0; g < ip.n_grids; ++g) {
        if (ip.ftol_array[g] < 1.0e-6) {
            fprintf(stderr,
                    "cuMES: float build cannot meet ftol_array[%d]=%.0e "
                    "(float residual floor is ~1e-7); relax the ftol_array "
                    "entries to >= 1e-6 for float experiments\n",
                    g, ip.ftol_array[g]);
            return EXIT_FAILURE;
        }
    }
#endif

    GridParams<Real> p=initParams(ip);
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

    // ---- Multi-radial-grid stage loop ----
    // Each stage runs the solver on its own radial grid (with its own
    // iteration cap and ftol), seeded by the previous stage's converged
    // state interpolated onto the new grid (vmecpp grid sequencing). All
    // ns-dependent objects are re-created per stage; profiles re-evaluate
    // analytically on the new grid.
    SpectralState<Real> st{};
    GridParams<Real> p_prev;
    SolverResult<Real> result{false, 0, Real(1.0), Real(1.0), Real(1.0), Real(0.9)};
    int total_iter = 0;
    for (int g = 0; g < ip.n_grids; ++g) {
        p_prev = p;                    // previous stage's params (ns, ...)
        p.ns = ip.ns_array[g];
        p.max_iter = ip.niter_array[g];
        p.ftol = ip.ftol_array[g];
        printf("\n=== grid stage %d/%d: ns=%d mnmax=%d max_iter=%d ftol=%.0e ===\n",
               g + 1, ip.n_grids, p.ns, p.mnmax, p.max_iter, (double)p.ftol);
        if (g == 0) {
            initState<Real>(st, p, ip);  // cold start (interpFromBoundaryAndAxis)
        } else {
            SpectralState<Real> st_new{};
            interpolateState<Real>(st_new, p, st, p_prev);
            freeState(st);
            st = st_new;
        }
        RadialProfiles<Real> rp = profilesCreate<Real>(p, ip);  // sets p.lamscale
        FourierPlan<Real> fp = fourierCreate<Real>(p);
        MetricWorkspace<Real> mw = metricCreate<Real>(p);

        result = solverRun<Real>(st, p, rp, fp, mw);

        fourierFree(fp); metricFree(mw); profilesFree(rp);
        total_iter += result.iterations;

        // vmecpp semantics (vmec.cc:367-392): a stage that exhausts its
        // iteration cap without meeting ftol fails the whole run. Single-
        // grid runs keep the lenient report-and-return path below.
        if (!result.converged && ip.n_grids > 1) {
            fprintf(stderr, "FATAL: grid stage %d/%d (ns=%d) completed %d/%d "
                            "iterations without meeting ftol=%.0e; final "
                            "residuals fsqr=%.3e fsqz=%.3e fsql=%.3e\n",
                    g + 1, ip.n_grids, p.ns, result.iterations, p.max_iter,
                    (double)p.ftol, (double)result.fsqr, (double)result.fsqz,
                    (double)result.fsql);
            freeState(st);
            return EXIT_FAILURE;
        }
    }

    // Output success is part of the run result: a converged solve whose
    // state file could not be written must NOT exit 0 (the writers return
    // false on open/write/close failure and clean up partial files).
    const bool output_ok = outputSave<Real>(st, p, ip, result, outputPath, inputPath);
    outputPrint<Real>(st, p, result.iterations, result.converged,
                      result.fsqr, result.fsqz, result.fsql);
    if (ip.n_grids > 1)
        printf("multigrid: total effective iterations over %d grids = %d\n",
               ip.n_grids, total_iter);
    freeState(st);
    printf("\nDone.\n");
    if (!output_ok) {
        fprintf(stderr, "cuMES: FAILED to write output state (%s)\n", outputPath);
        return EXIT_FAILURE;
    }
    return result.converged ? 0 : 1;
}
