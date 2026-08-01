// solver.cu — fixed-point iteration with Garabedian acceleration.
// The dump/debug machinery below (DUMP_CUMES_VERIFY blocks) is compiled in
// but RUNTIME-GATED: nothing is written and no debug output is produced
// unless the CUMES_DUMP=1 environment variable is set (see dumpEnabled()).
#define DUMP_CUMES_VERIFY
#include "solver.cuh"
#include "precon.cuh"
#include "constraint.cuh"
#include "input.h"
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <algorithm>

static constexpr int kPreconInterval = 25;  // update preconditioner every N iters

static void checkCuda(cudaError_t err, const char* tag) {
    if (err != cudaSuccess) { fprintf(stderr, "CUDA error [%s]: %s\n", tag, cudaGetErrorString(err)); exit(EXIT_FAILURE); }
}

// extrapolateTowardsAxis: copy m=1 coefficients from first interior
// surface (j=1) to the magnetic axis (j=0), matching vmecpp's
// extrapolateTowardsAxis(). Only m=1 has a finite value at the axis
// for stellarator-symmetric, axisymmetric equilibria.
__global__ void extrapolateAxisKernel(
    double* __restrict__ rmncc, double* __restrict__ zmnsc,
    double* __restrict__ lmnsc, double* __restrict__ rmnss,
    double* __restrict__ zmncs,
    int ns, int mnmax, int ntor) {
    int mode = blockIdx.x;
    if (mode >= mnmax) return;
    int m = mode / ntor;  // poloidal mode number
    if (m != 1) return;   // only m=1 needs extrapolation
    // Copy from j=1 to j=0
    rmncc[0 + mode*ns] = rmncc[1 + mode*ns];
    zmnsc[0 + mode*ns] = zmnsc[1 + mode*ns];
    lmnsc[0 + mode*ns] = lmnsc[1 + mode*ns];
    rmnss[0 + mode*ns] = rmnss[1 + mode*ns];
    zmncs[0 + mode*ns] = zmncs[1 + mode*ns];
}

// Apply vmecpp's even/odd-m decomposition scaling (decomposeInto) to the
// spectral forces. vmecpp stores and evolves the "decomposed" representation:
// odd-m coefficients carry an extra 1/max(sqrt(s), sqrt(1/(ns-1))) factor
// (Eqn. 8c in Hirshman, Schwenn & Nuehrenberg 1990); the axis is clamped to
// the innermost surface's sqrt(s) (constant extrapolation towards the axis).
// cuMES's geometry coefficients already follow this convention (initState),
// so the forces must be scaled the same way before the residuals, the
// preconditioned solve, and the descent — otherwise the per-mode dynamics
// differ from vmecpp. Even-m modes: scalxc = 1 (no change).
__global__ void scalxcApplyKernel(
    double* __restrict__ f_spec, const double* __restrict__ sqrtS_F,
    int ns, int mnmax, double sqrtS1)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x, total = mnmax * ns;
    if (i >= total) return;
    int m = i / ns;
    if (m % 2 == 0) return;  // even-m: scalxc = 1
    int j = i % ns;
    double scal = 1.0 / fmax(sqrtS_F[j], sqrtS1);
    for (int c = 0; c < 5; ++c) f_spec[c * mnmax * ns + i] *= scal;
}

__global__ void computeResidualsKernel(const double* __restrict__ f_spec, int ns, int mnmax,
                                        double* __restrict__ sq_out) {
    int comp = blockIdx.x; if(comp>=3)return;
    double sum=0; int offset=comp*mnmax*ns, total=mnmax*ns;
    for(int i=threadIdx.x; i<total; i+=blockDim.x){double v=f_spec[i+offset]; sum+=v*v;}
    __shared__ double s_sum[256]; int tid=threadIdx.x; s_sum[tid]=sum; __syncthreads();
    for(int s=blockDim.x/2; s>0; s>>=1){if(tid<s)s_sum[tid]+=s_sum[tid+s]; __syncthreads();}
    if(tid==0) sq_out[comp]=s_sum[0]/(mnmax*ns);
}

__global__ void descentStepKernel(
    double* __restrict__ x_cc, double* __restrict__ x_ss,
    double* __restrict__ x_zsc, double* __restrict__ x_zcs,
    double* __restrict__ x_lsc,
    double* __restrict__ v_cc, double* __restrict__ v_ss,
    double* __restrict__ v_zsc, double* __restrict__ v_zcs,
    double* __restrict__ v_lsc,
    const double* __restrict__ f_spec,
    const int* __restrict__ xm,
    int ns, int mnmax, double delt, double b1, double fac)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x, total = mnmax*ns;
    if(i>=total)return;
    int m=i/ns, j=i%ns, mm=xm[m];
    // Skip m>0 modes at axis: coordinate singularity at s=0. The forces
    // there are zeroed by the preconditioners (jMin identity rows for R/Z,
    // sqrt(s)^pwr for lambda), so vmecpp never moves these coefficients.
    if (j == 0 && mm > 0) return;

    // R/Z components (0,1,3,4): LCFS is fixed — the force was zeroed by
    // fixBoundaryKernel and the coefficient must not move. Lambda (comp 2)
    // is free on all surfaces including the LCFS, matching vmecpp.
    if (j < ns-1) {
        double vr=v_cc[i]; vr=fac*(b1*vr+ delt*f_spec[i+0*mnmax*ns]); v_cc[i]=vr; x_cc[i]+=delt*vr;
        double vz=v_zsc[i];vz=fac*(b1*vz+ delt*f_spec[i+1*mnmax*ns]);v_zsc[i]=vz;x_zsc[i]+=delt*vz;
        double vs=v_ss[i]; vs=fac*(b1*vs+ delt*f_spec[i+3*mnmax*ns]); v_ss[i]=vs; x_ss[i]+=delt*vs;
        double vzc=v_zcs[i];vzc=fac*(b1*vzc+ delt*f_spec[i+4*mnmax*ns]);v_zcs[i]=vzc;x_zcs[i]+=delt*vzc;
    }
    double vl=v_lsc[i];vl=fac*(b1*vl+ delt*f_spec[i+2*mnmax*ns]);v_lsc[i]=vl;x_lsc[i]+=delt*vl;
}

#ifdef DUMP_CUMES_VERIFY
// Master switch for the dump/debug machinery: off unless CUMES_DUMP=1.
// All dump output routes through dumpEnsureDir/dumpDeviceArray, which no-op
// when disabled, so default runs write nothing to dump/ and print no debug
// noise. (The CUMES_DUMP_ITER knob below still selects WHICH iterations the
// windowed dumps fire on; CUMES_DUMP is the master enable.)
static bool dumpEnabled() {
    const char* e = getenv("CUMES_DUMP");
    return e != nullptr && atoi(e) != 0;
}

static void dumpEnsureDir() {
    if (!dumpEnabled()) return;
    (void)system("mkdir -p dump/cuMES");
}

static void dumpDeviceArray(const char* filename, const double* d_data, size_t nelem) {
    if (!dumpEnabled()) return;
    double* h_tmp = new double[nelem];
    cudaError_t err = cudaMemcpy(h_tmp, d_data, nelem * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "dumpDeviceArray cudaMemcpy failed for %s: %s\n", filename, cudaGetErrorString(err));
    }
    FILE* fp = fopen(filename, "wb");
    if (fp) {
        uint64_t n = nelem;
        fwrite(&n, sizeof(uint64_t), 1, fp);
        fwrite(h_tmp, sizeof(double), nelem, fp);
        fclose(fp);
    }
    delete[] h_tmp;
}
#endif

SolverResult solverRun(SpectralState& st, const GridParams& p,
                       const RadialProfiles& rp, FourierPlan& fp, MetricWorkspace& mw) {
    SolverResult res{false,0,1.0,1.0,1.0,kDelt0};
    size_t nb = 5*p.ns*p.mnmax*sizeof(double);
    double *d_f_spec, *d_sq;
    checkCuda(cudaMalloc(&d_f_spec,nb),"malloc f");
    checkCuda(cudaMalloc(&d_sq,3*sizeof(double)),"malloc sq");
    PreconWorkspace pw = preconCreate(p);
    ConstraintWorkspace cw = constraintCreate(p);

    // ---- env-gated knobs for convergence experiments (defaults = hardcoded
    // input values; set via CUMES_MAX_ITER, CUMES_DELT0, CUMES_DTAU_FLOOR,
    // CUMES_DUMP_ITER, CUMES_E2_START) ----
    int kMaxIterEff = kMaxIter;
    double kDelt0Eff = kDelt0;
    double kDtauFloor = 0.0;
    int kDumpIter = 150, kE2Start = 560;
    if (const char* e = getenv("CUMES_MAX_ITER"))   kMaxIterEff = atoi(e);
    if (const char* e = getenv("CUMES_DELT0"))      kDelt0Eff = atof(e);
    if (const char* e = getenv("CUMES_DTAU_FLOOR")) kDtauFloor = atof(e);
    if (const char* e = getenv("CUMES_DUMP_ITER"))  kDumpIter = atoi(e);
    if (const char* e = getenv("CUMES_E2_START"))   kE2Start = atoi(e);
    printf("knobs: max_iter=%d delt0=%.3f dtau_floor=%.3e dump_iter=%d e2_start=%d\n",
           kMaxIterEff, kDelt0Eff, kDtauFloor, kDumpIter, kE2Start);

    // ---- vmecpp VMEC_8_52 time-step control state ----
    // iter2: effective iteration counter — does NOT advance on restart
    // passes (vmecpp's bad_resets mechanism). iter1: branch-point marker,
    // set to the current iteration on every restart; grace periods and the
    // invTau reinitialization key off (iter2 - iter1).
    // res0: running minimum of the preconditioned residual sum fsq.
    int iter2 = 1, iter1 = 1;
    double res0 = -1.0;
    double fsq_prev = 1.0;   // vmecpp: fc_.fsq = 1.0 at stage start
    int ijacob = 0;
    double inv_tau_hist[10];
    for (int ii = 0; ii < 10; ++ii) inv_tau_hist[ii] = 0.15 / kDelt0Eff;
    double delt=kDelt0Eff;

    // State rollback: backup arrays for restoring spectral state on restart.
    size_t nb_one = (size_t)p.ns * (size_t)p.mnmax * sizeof(double);
    double *d_bk_rmncc, *d_bk_zmnsc, *d_bk_lmnsc, *d_bk_rmnss, *d_bk_zmncs;
    checkCuda(cudaMalloc(&d_bk_rmncc, nb_one), "bk cc");
    checkCuda(cudaMalloc(&d_bk_zmnsc, nb_one), "bk zsc");
    checkCuda(cudaMalloc(&d_bk_lmnsc, nb_one), "bk lsc");
    checkCuda(cudaMalloc(&d_bk_rmnss, nb_one), "bk ss");
    checkCuda(cudaMalloc(&d_bk_zmncs, nb_one), "bk zcs");

    // Helper: copy current spectral state to backup (GPU device-to-device)
    auto backupState = [&]() {
        cudaMemcpy(d_bk_rmncc, st.d_rmncc, nb_one, cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_bk_zmnsc, st.d_zmnsc, nb_one, cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_bk_lmnsc, st.d_lmnsc, nb_one, cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_bk_rmnss, st.d_rmnss, nb_one, cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_bk_zmncs, st.d_zmncs, nb_one, cudaMemcpyDeviceToDevice);
    };

    // Helper: restore spectral state from backup + zero velocities
    auto restoreState = [&]() {
        cudaMemcpy(st.d_rmncc, d_bk_rmncc, nb_one, cudaMemcpyDeviceToDevice);
        cudaMemcpy(st.d_zmnsc, d_bk_zmnsc, nb_one, cudaMemcpyDeviceToDevice);
        cudaMemcpy(st.d_lmnsc, d_bk_lmnsc, nb_one, cudaMemcpyDeviceToDevice);
        cudaMemcpy(st.d_rmnss, d_bk_rmnss, nb_one, cudaMemcpyDeviceToDevice);
        cudaMemcpy(st.d_zmncs, d_bk_zmncs, nb_one, cudaMemcpyDeviceToDevice);
        cudaMemset(st.d_v_rmncc, 0, nb_one);
        cudaMemset(st.d_v_zmnsc, 0, nb_one);
        cudaMemset(st.d_v_lmnsc, 0, nb_one);
        cudaMemset(st.d_v_rmnss, 0, nb_one);
        cudaMemset(st.d_v_zmncs, 0, nb_one);
    };

    // Take initial backup
    backupState();

#ifdef DUMP_CUMES_VERIFY
    // Per-pass record for convergence analysis (mirrors vmecpp's
    // per_iter_residuals.bin + control scalars). 15 columns:
    // fsqr_i fsqz_i fsql_i fsqr fsqz fsql delt otav dtau b1 fac
    // iter2 iter1 reason rax(axis R_00, pre-descent of the pass).
    double (*per_iter)[15] = new double[kMaxIter][15];
    int n_passes = 0;
    auto recordPass = [&](int reason, double fRi, double fZi, double fLi,
                          double fR, double fZ, double fL, double d,
                          double o, double dt, double b1v, double fcv) {
        if (!dumpEnabled()) return;
        if (n_passes < kMaxIter) {
            double* r = per_iter[n_passes++];
            r[0]=fRi; r[1]=fZi; r[2]=fLi; r[3]=fR; r[4]=fZ; r[5]=fL;
            r[6]=d; r[7]=o; r[8]=dt; r[9]=b1v; r[10]=fcv;
            r[11]=(double)iter2; r[12]=(double)iter1; r[13]=(double)reason;
            double h_rax;
            cudaMemcpy(&h_rax, st.d_rmncc, sizeof(double), cudaMemcpyDeviceToHost);
            r[14]=h_rax;
        }
    };
#endif

    // Diagnostic: test inverse DFT at specified surface (CUMES_DUMP=1 only).
    if (dumpEnabled()) {
        inverseDFT(fp, st, p);
        auto* h_re = new double[p.nZnT * p.ns];
        auto* h_ro = new double[p.nZnT * p.ns];
        checkCuda(cudaMemcpy(h_re, fp.d_r_e, p.nZnT*p.ns*sizeof(double), cudaMemcpyDeviceToHost), "diag re");
        checkCuda(cudaMemcpy(h_ro, fp.d_r_o, p.nZnT*p.ns*sizeof(double), cudaMemcpyDeviceToHost), "diag ro");
        // Check surface j=ns-1 (LCFS): r_e should be rbc[0]*cos(0)=3.999, r_o should be sum of odd m
        int jB = p.ns - 1;
        double re_lcfs = h_re[0 + jB * p.nZnT];  // theta=0
        double ro_lcfs = h_ro[0 + jB * p.nZnT];
        printf("  [diag] LCFS theta=0: r_e=%.4f r_o=%.4f r_total=%.4f (expect ~3.93 + ~1.03 = ~4.96)\n",
               re_lcfs, ro_lcfs, re_lcfs + ro_lcfs);
        delete[] h_re;
        delete[] h_ro;
    }

    printf("\n ITER |    FSQR        FSQZ        FSQL    |   DELT\n");
    printf("------+------------------------------------+----------\n");

#ifdef DUMP_CUMES_VERIFY
    {
        dumpEnsureDir();
        size_t n_spec = (size_t)p.ns * (size_t)p.mnmax;
        dumpDeviceArray("dump/cuMES/step_0_rmncc.bin",      st.d_rmncc, n_spec);
        dumpDeviceArray("dump/cuMES/step_0_zmnsc.bin",      st.d_zmnsc, n_spec);
        dumpDeviceArray("dump/cuMES/step_0_lmnsc.bin",      st.d_lmnsc, n_spec);
        dumpDeviceArray("dump/cuMES/step_0_rmnss.bin",      st.d_rmnss, n_spec);
        dumpDeviceArray("dump/cuMES/step_0_zmncs.bin",      st.d_zmncs, n_spec);
    }
#endif

    for(int iter=0; iter<kMaxIterEff; ++iter){
        // Extrapolate m=1 coefficients to the magnetic axis (j=0)
        // before inverse DFT, matching vmecpp's extrapolateTowardsAxis().
        // Must be done each iteration since the descent step updates j=1
        // but skips j=0 for m>0 (axis regularity).
        extrapolateAxisKernel<<<p.mnmax, 1>>>(
            st.d_rmncc, st.d_zmnsc, st.d_lmnsc, st.d_rmnss, st.d_zmncs,
            p.ns, p.mnmax, p.ntor);
        checkCuda(cudaGetLastError(), "extrapAxis");

        inverseDFT(fp,st,p);

        if (iter == 0 && dumpEnabled()) {
            auto* h_test = new double[p.nZnT * p.ns];
            checkCuda(cudaMemcpy(h_test, fp.d_r_e, p.nZnT*p.ns*sizeof(double), cudaMemcpyDeviceToHost), "loop test");
            int jB = p.ns - 1;
            printf("  [loop diag] LCFS theta=0: r_e=%.4f (expect ~3.93)\n", h_test[0 + jB * p.nZnT]);
            // Also write to file for comparison
            FILE* dbg = fopen("dump/cuMES/debug_r_e.bin", "wb");
            if (dbg) {
                uint64_t n = p.nZnT * p.ns;
                fwrite(&n, sizeof(uint64_t), 1, dbg);
                fwrite(h_test, sizeof(double), n, dbg);
                fclose(dbg);
            }
            delete[] h_test;
        }

#ifdef DUMP_CUMES_VERIFY
        if (iter == 0 || iter2 == 2) {
            // iter 2 = first pass with lambda != 0: dump the real-space
            // lambda derivatives for the basis-convention check.
            size_t n_real = (size_t)p.ns * (size_t)p.nZnT;
            char fn[128];
            snprintf(fn, sizeof fn, "dump/cuMES/step_A_lu_e_iter_%d.bin", iter == 0 ? 1 : 2);
            dumpDeviceArray(fn, fp.d_lu_e, n_real);
            snprintf(fn, sizeof fn, "dump/cuMES/step_A_lu_o_iter_%d.bin", iter == 0 ? 1 : 2);
            dumpDeviceArray(fn, fp.d_lu_o, n_real);
            snprintf(fn, sizeof fn, "dump/cuMES/step_A_l_real_iter_%d.bin", iter == 0 ? 1 : 2);
            dumpDeviceArray(fn, fp.d_l_real, n_real);
        }
        if (iter == 0) {
            size_t n_real = (size_t)p.ns * (size_t)p.nZnT;
            // Full R, Z, lambda (even+odd)
            dumpDeviceArray("dump/cuMES/step_A_r_real_iter_1.bin", fp.d_r_real, n_real);
            dumpDeviceArray("dump/cuMES/step_A_z_real_iter_1.bin", fp.d_z_real, n_real);
            // Even-m parity
            dumpDeviceArray("dump/cuMES/step_A_r_e_iter_1.bin", fp.d_r_e, n_real);
            dumpDeviceArray("dump/cuMES/step_A_z_e_iter_1.bin", fp.d_z_e, n_real);
            dumpDeviceArray("dump/cuMES/step_A_l_e_iter_1.bin", fp.d_l_e, n_real);
            // Odd-m parity
            dumpDeviceArray("dump/cuMES/step_A_r_o_iter_1.bin", fp.d_r_o, n_real);
            dumpDeviceArray("dump/cuMES/step_A_z_o_iter_1.bin", fp.d_z_o, n_real);
            dumpDeviceArray("dump/cuMES/step_A_l_o_iter_1.bin", fp.d_l_o, n_real);
            // Poloidal derivatives
            dumpDeviceArray("dump/cuMES/step_A_ru_real_iter_1.bin", fp.d_ru_real, n_real);
            dumpDeviceArray("dump/cuMES/step_A_zu_real_iter_1.bin", fp.d_zu_real, n_real);
            dumpDeviceArray("dump/cuMES/step_A_lu_real_iter_1.bin", fp.d_lu_real, n_real);
            // Toroidal derivatives
            dumpDeviceArray("dump/cuMES/step_A_rv_real_iter_1.bin", fp.d_rv_real, n_real);
            dumpDeviceArray("dump/cuMES/step_A_zv_real_iter_1.bin", fp.d_zv_real, n_real);
            dumpDeviceArray("dump/cuMES/step_A_lv_real_iter_1.bin", fp.d_lv_real, n_real);
            // Even-m poloidal derivatives
            dumpDeviceArray("dump/cuMES/step_A_ru_e_iter_1.bin", fp.d_ru_e, n_real);
            dumpDeviceArray("dump/cuMES/step_A_zu_e_iter_1.bin", fp.d_zu_e, n_real);
            dumpDeviceArray("dump/cuMES/step_A_lu_e_iter_1.bin", fp.d_lu_e, n_real);
            // Odd-m poloidal derivatives
            dumpDeviceArray("dump/cuMES/step_A_ru_o_iter_1.bin", fp.d_ru_o, n_real);
            dumpDeviceArray("dump/cuMES/step_A_zu_o_iter_1.bin", fp.d_zu_o, n_real);
            dumpDeviceArray("dump/cuMES/step_A_lu_o_iter_1.bin", fp.d_lu_o, n_real);
        }
#endif

        computeGeometry(fp,p,rp,mw);

#ifdef DUMP_CUMES_VERIFY
        if (iter == 0 || iter2 == 2) {
            // iter 2 = first pass with lambda != 0 (E3-D bsupv check)
            size_t n_half = (size_t)(p.ns - 1) * (size_t)p.nZnT;
            char fn[128];
            snprintf(fn, sizeof fn, "dump/cuMES/step_D_bsupu_iter_%d.bin", iter == 0 ? 1 : 2);
            dumpDeviceArray(fn, mw.d_bsupu, n_half);
            snprintf(fn, sizeof fn, "dump/cuMES/step_D_bsupv_iter_%d.bin", iter == 0 ? 1 : 2);
            dumpDeviceArray(fn, mw.d_bsupv, n_half);
        }
        if (iter == 0) {
            size_t n_half = (size_t)(p.ns - 1) * (size_t)p.nZnT;
            dumpDeviceArray("dump/cuMES/step_B_gsqrt_iter_1.bin", mw.d_gsqrt, n_half);
            dumpDeviceArray("dump/cuMES/step_C_guu_iter_1.bin",   mw.d_guu, n_half);
            dumpDeviceArray("dump/cuMES/step_C_guv_iter_1.bin",   mw.d_guv, n_half);
            dumpDeviceArray("dump/cuMES/step_C_gvv_iter_1.bin",   mw.d_gvv, n_half);
        }
#endif

        // Compute the xmpq-weighted real-space combination rCon/zCon from
        // the current spectral state (vmecpp's rCon/zCon in the inverse
        // DFT), used by the spectral-condensation constraint.
        constraintRzConCompute(p, fp, st, cw, rp.d_sqrtS_F);

        // Reset the constraint-force reference (rCon0/zCon0) to the
        // LCFS-extrapolated profile on the first iteration and after every
        // restart (iter2 == iter1), matching vmecpp's rzConIntoVolume
        // ("initialization/soft reset").
        if (iter2 == iter1) {
            constraintResetRzCon0(p, cw, rp.d_sqrtS_F);
        }

        // Update the radial tridiagonal + lambda preconditioners BEFORE the
        // forces, so the constraint-force multiplier (tcon) can use the
        // current iteration's preconditioner elements, matching vmecpp
        // (updateRadialPreconditioner + constraintForceMultiplier at the
        // start of update()). Cadence matches vmecpp:
        // (iter2 - iter1) % 25 == 0 — also refreshed on the first pass
        // after a restart.
        bool precon_updated = ((iter2 - iter1) % kPreconInterval) == 0;
        if (precon_updated) {
            preconCompute(fp, p, rp, mw, pw);

#ifdef DUMP_CUMES_VERIFY
            if (iter == 0) {
                // Dump tridiagonal preconditioner matrices for comparison
                // with vmecpp. cuMES layout: ar[mode * ns + jF] (mode-major).
                size_t n_tri = (size_t)p.mnmax * (size_t)p.ns;
                size_t n_half_2 = (size_t)2 * (size_t)(p.ns - 1);
                size_t n_full_2 = (size_t)2 * (size_t)p.ns;
                size_t n_half_1 = (size_t)(p.ns - 1);
                size_t n_full_1 = (size_t)p.ns;

                // Tridiagonal matrix elements (mode-major: [mode, jF])
                dumpDeviceArray("dump/cuMES/step_precon_ar_iter_1.bin", pw.d_ar, n_tri);
                dumpDeviceArray("dump/cuMES/step_precon_dr_iter_1.bin", pw.d_dr, n_tri);
                dumpDeviceArray("dump/cuMES/step_precon_br_iter_1.bin", pw.d_br, n_tri);
                dumpDeviceArray("dump/cuMES/step_precon_az_iter_1.bin", pw.d_az, n_tri);
                dumpDeviceArray("dump/cuMES/step_precon_dz_iter_1.bin", pw.d_dz, n_tri);
                dumpDeviceArray("dump/cuMES/step_precon_bz_iter_1.bin", pw.d_bz, n_tri);

                // jMin per mode (stored as int, convert to double for dump)
                {
                    int* h_jMin = new int[p.mnmax];
                    cudaMemcpy(h_jMin, pw.d_jMin, p.mnmax * sizeof(int), cudaMemcpyDeviceToHost);
                    double* h_jMin_dbl = new double[p.mnmax];
                    for (int i = 0; i < p.mnmax; ++i) h_jMin_dbl[i] = (double)h_jMin[i];
                    FILE* fj = fopen("dump/cuMES/step_precon_jMin_iter_1.bin", "wb");
                    if (fj) {
                        uint64_t n = p.mnmax;
                        fwrite(&n, sizeof(uint64_t), 1, fj);
                        fwrite(h_jMin_dbl, sizeof(double), p.mnmax, fj);
                        fclose(fj);
                    }
                    delete[] h_jMin;
                    delete[] h_jMin_dbl;
                }

                // Intermediate arrays
                dumpDeviceArray("dump/cuMES/step_precon_arm_iter_1.bin", pw.d_arm, n_half_2);
                dumpDeviceArray("dump/cuMES/step_precon_ard_iter_1.bin", pw.d_ard, n_full_2);
                dumpDeviceArray("dump/cuMES/step_precon_brm_iter_1.bin", pw.d_brm, n_half_2);
                dumpDeviceArray("dump/cuMES/step_precon_brd_iter_1.bin", pw.d_brd, n_full_2);
                dumpDeviceArray("dump/cuMES/step_precon_azm_iter_1.bin", pw.d_azm, n_half_2);
                dumpDeviceArray("dump/cuMES/step_precon_azd_iter_1.bin", pw.d_azd, n_full_2);
                dumpDeviceArray("dump/cuMES/step_precon_bzm_iter_1.bin", pw.d_bzm, n_half_2);
                dumpDeviceArray("dump/cuMES/step_precon_bzd_iter_1.bin", pw.d_bzd, n_full_2);
                dumpDeviceArray("dump/cuMES/step_precon_cxd_iter_1.bin", pw.d_cxd, n_full_1);

                // Sizes for comparison script
                double sizes_dbl[4] = {(double)p.ns, (double)(p.ns-1), (double)p.mpol, 1.0};
                FILE* fs = fopen("dump/cuMES/step_precon_sizes_iter_1.bin", "wb");
                if (fs) {
                    uint64_t n = 4;
                    fwrite(&n, sizeof(uint64_t), 1, fs);
                    fwrite(sizes_dbl, sizeof(double), 4, fs);
                    fclose(fs);
                }
            }
#endif  // DUMP_CUMES_VERIFY
        }

        computeForces(fp,p,rp,mw);

#ifdef DUMP_CUMES_VERIFY
        if (iter == 0 || iter2 == 2) {
            // iter 2 = first pass with lambda != 0 (E3-B blmn blending check)
            size_t n_half = (size_t)(p.ns - 1) * (size_t)p.nZnT;
            if (iter == 0) {
                dumpDeviceArray("dump/cuMES/step_half_r12_iter_1.bin", mw.d_r12, n_half);
                dumpDeviceArray("dump/cuMES/step_half_zu12_iter_1.bin", mw.d_zu12, n_half);
                dumpDeviceArray("dump/cuMES/step_half_tau_iter_1.bin", mw.d_tau, n_half);
                dumpDeviceArray("dump/cuMES/step_half_gsqrt_iter_1.bin", mw.d_gsqrt, n_half);
                dumpDeviceArray("dump/cuMES/step_half_totalP_iter_1.bin", mw.d_totalPressure, n_half);
                dumpDeviceArray("dump/cuMES/step_half_bsupu_iter_1.bin", mw.d_bsupu, n_half);
                dumpDeviceArray("dump/cuMES/step_half_bsupv_iter_1.bin", mw.d_bsupv, n_half);
                dumpDeviceArray("dump/cuMES/step_half_bsubu_iter_1.bin", mw.d_bsubu, n_half);
                dumpDeviceArray("dump/cuMES/step_half_bsubv_iter_1.bin", mw.d_bsubv, n_half);
                dumpDeviceArray("dump/cuMES/step_half_rs_iter_1.bin", mw.d_rs, n_half);
                dumpDeviceArray("dump/cuMES/step_half_zs_iter_1.bin", mw.d_zs, n_half);
            }

            size_t n_real = (size_t)p.ns * (size_t)p.nZnT;
            char fn[128];
            int itag = (iter == 0) ? 1 : iter2;
            snprintf(fn, sizeof fn, "dump/cuMES/step_F_brmn_e_iter_%d.bin", itag);
            dumpDeviceArray(fn, fp.d_brmn_e, n_real);
            snprintf(fn, sizeof fn, "dump/cuMES/step_F_brmn_o_iter_%d.bin", itag);
            dumpDeviceArray(fn, fp.d_brmn_o, n_real);
            snprintf(fn, sizeof fn, "dump/cuMES/step_F_bzmn_e_iter_%d.bin", itag);
            dumpDeviceArray(fn, fp.d_bzmn_e, n_real);
            snprintf(fn, sizeof fn, "dump/cuMES/step_F_bzmn_o_iter_%d.bin", itag);
            dumpDeviceArray(fn, fp.d_bzmn_o, n_real);
            snprintf(fn, sizeof fn, "dump/cuMES/step_F_blmn_e_iter_%d.bin", itag);
            dumpDeviceArray(fn, fp.d_blmn_e, n_real);
            snprintf(fn, sizeof fn, "dump/cuMES/step_F_blmn_o_iter_%d.bin", itag);
            dumpDeviceArray(fn, fp.d_blmn_o, n_real);
            if (iter == 0) {
                dumpDeviceArray("dump/cuMES/step_E_armn_e_iter_1.bin", fp.d_armn_e, n_real);
                dumpDeviceArray("dump/cuMES/step_E_armn_o_iter_1.bin", fp.d_armn_o, n_real);
                dumpDeviceArray("dump/cuMES/step_E_azmn_e_iter_1.bin", fp.d_azmn_e, n_real);
                dumpDeviceArray("dump/cuMES/step_E_azmn_o_iter_1.bin", fp.d_azmn_o, n_real);
                dumpDeviceArray("dump/cuMES/step_F_fr_real_iter_1.bin", fp.d_fr_real, n_real);
                dumpDeviceArray("dump/cuMES/step_F_fz_real_iter_1.bin", fp.d_fz_real, n_real);
                dumpDeviceArray("dump/cuMES/step_F_fl_real_iter_1.bin", fp.d_fl_real, n_real);
            }
        }
#endif

        // Add spectral condensation constraint force to brmn/bzmn.
        // Uses the current-iteration tcon (refreshed above when the
        // preconditioner was updated), matching vmecpp.
        constraintCompute(p, fp, pw, cw, rp.d_sqrtS_F, precon_updated);

#ifdef DUMP_CUMES_VERIFY
        if (iter == 0) {
            size_t n_real = (size_t)p.ns * (size_t)p.nZnT;
            dumpDeviceArray("dump/cuMES/step_G_brmn_e_iter_1.bin", fp.d_brmn_e, n_real);
            dumpDeviceArray("dump/cuMES/step_G_brmn_o_iter_1.bin", fp.d_brmn_o, n_real);
            dumpDeviceArray("dump/cuMES/step_G_bzmn_e_iter_1.bin", fp.d_bzmn_e, n_real);
            dumpDeviceArray("dump/cuMES/step_G_bzmn_o_iter_1.bin", fp.d_bzmn_o, n_real);
        }
#endif

        forwardDFT(fp,d_f_spec,p,cw);

        // Apply the odd-m decomposition scaling (vmecpp decomposeInto).
        // The forward DFT already zeroed the LCFS R/Z entries and the axis
        // m>0 entries, so the boundary stays rigid and only the lambda
        // force is present at the LCFS (free gauge, evolved by descent).
        { dim3 bs(256), gs((p.ns*p.mnmax+255)/256);
          scalxcApplyKernel<<<gs,bs>>>(d_f_spec, rp.d_sqrtS_F, p.ns, p.mnmax,
                                       sqrt(1.0 / (p.ns - 1.0)));
          checkCuda(cudaGetLastError(), "scalxc"); }

#ifdef DUMP_CUMES_VERIFY
        if (iter == 0 || iter2 == kDumpIter) {
            // Dump AFTER the decomposition scaling, matching vmecpp's dump
            // of m_decomposed_f (post-decomposeInto). Keyed on iter2 so a
            // handoff/plateau comparison can use the same effective counter
            // as vmecpp's dump blocks.
            size_t n_fspec = (size_t)5 * (size_t)p.mnmax * (size_t)p.ns;
            char fn[128];
            snprintf(fn, sizeof fn, "dump/cuMES/step_H_f_spec_iter_%d.bin",
                     iter == 0 ? 1 : iter2);
            dumpDeviceArray(fn, d_f_spec, n_fspec);
        }
        if (iter2 >= kE2Start && iter2 < kE2Start + 40) {
            char fn[128];
            snprintf(fn, sizeof fn, "dump/cuMES/fspec_invariant_iter_%d.bin", iter2);
            dumpDeviceArray(fn, d_f_spec, (size_t)5 * p.mnmax * p.ns);
        }
#endif

        // ---- Invariant (unpreconditioned) residuals ----
        // Used for the stopping criterion and the BAD_PROGRESS threshold,
        // matching vmecpp's fsqr/fsqz/fsql (evalFResInvar). Computed before
        // preconditioning so a false minimum (tiny preconditioned forces on
        // garbage geometry) can never be mistaken for convergence.
#ifdef DUMP_CUMES_VERIFY
        if (iter == kMaxIterEff - 1) {
            dumpDeviceArray("dump/cuMES/step_final_f_spec.bin", d_f_spec,
                            (size_t)5 * p.mnmax * p.ns);
        }
#endif
        { dim3 b3(256),g3(3); computeResidualsKernel<<<g3,b3>>>(d_f_spec,p.ns,p.mnmax,d_sq); }
        double h_sq_i[3]; checkCuda(cudaMemcpy(h_sq_i,d_sq,3*sizeof(double),cudaMemcpyDeviceToHost),"cpy sqi");
        double fsqr_i=h_sq_i[0], fsqz_i=h_sq_i[1], fsql_i=h_sq_i[2];

        // ---- Stopping criterion (vmecpp Evolve) ----
        if (!(std::isfinite(fsqr_i) && std::isfinite(fsqz_i) && std::isfinite(fsql_i))) {
            // vmecpp hard-fails on non-finite residuals (status BAD_JACOBIAN);
            // we recover instead: restore the last good state and shrink delt.
#ifdef DUMP_CUMES_VERIFY
            recordPass(1, fsqr_i, fsqz_i, fsql_i, 0,0,0, delt,0,0,0,0);
#endif
            restoreState();
            delt *= 0.9;
            iter1 = iter2;
            printf("  -> BAD JACOBIAN (non-finite residuals) delt=%.3e\n",delt);
            continue;
        }
        if (fsqr_i <= kFtol && fsqz_i <= kFtol && fsql_i <= kFtol) {
#ifdef DUMP_CUMES_VERIFY
            recordPass(0, fsqr_i, fsqz_i, fsql_i, 0,0,0, delt,0,0,0,0);
#endif
            res.converged=true; res.iterations=iter+1;
            res.fsqr=fsqr_i;res.fsqz=fsqz_i;res.fsql=fsql_i;res.delt=delt;
            printf("  -> CONVERGED at iter %d\n",iter+1); break;
        }

        // Apply the radial tridiagonal + lambda preconditioners to the
        // (decomposed) spectral forces.
        preconApply(d_f_spec, p, pw, fp.basis.d_xm, fp.basis.d_xn);

#ifdef DUMP_CUMES_VERIFY
        if (iter == 0 || (iter2 >= kDumpIter && iter2 <= kDumpIter + 2)) {
            size_t n_fspec = (size_t)5 * (size_t)p.mnmax * (size_t)p.ns;
            size_t n_spec = (size_t)p.mnmax * (size_t)p.ns;
            char fn[128];
            snprintf(fn, sizeof fn, "dump/cuMES/step_I_f_spec_iter_%d.bin",
                     iter == 0 ? 1 : iter2);
            dumpDeviceArray(fn, d_f_spec, n_fspec);
            // State + velocities at the handoff window (pre-descent of the
            // pass, matching vmecpp's dump phase at vmec.cc).
            if (iter2 >= kDumpIter) {
                snprintf(fn, sizeof fn, "dump/cuMES/state_rmncc_iter_%d.bin", iter2);
                dumpDeviceArray(fn, st.d_rmncc, n_spec);
                snprintf(fn, sizeof fn, "dump/cuMES/state_zmnsc_iter_%d.bin", iter2);
                dumpDeviceArray(fn, st.d_zmnsc, n_spec);
                snprintf(fn, sizeof fn, "dump/cuMES/state_lmnsc_iter_%d.bin", iter2);
                dumpDeviceArray(fn, st.d_lmnsc, n_spec);
                snprintf(fn, sizeof fn, "dump/cuMES/state_rmnss_iter_%d.bin", iter2);
                dumpDeviceArray(fn, st.d_rmnss, n_spec);
                snprintf(fn, sizeof fn, "dump/cuMES/state_zmncs_iter_%d.bin", iter2);
                dumpDeviceArray(fn, st.d_zmncs, n_spec);
                snprintf(fn, sizeof fn, "dump/cuMES/vel_vrmncc_iter_%d.bin", iter2);
                dumpDeviceArray(fn, st.d_v_rmncc, n_spec);
                snprintf(fn, sizeof fn, "dump/cuMES/vel_vzmnsc_iter_%d.bin", iter2);
                dumpDeviceArray(fn, st.d_v_zmnsc, n_spec);
                snprintf(fn, sizeof fn, "dump/cuMES/vel_vlmnsc_iter_%d.bin", iter2);
                dumpDeviceArray(fn, st.d_v_lmnsc, n_spec);
                snprintf(fn, sizeof fn, "dump/cuMES/vel_vrmnss_iter_%d.bin", iter2);
                dumpDeviceArray(fn, st.d_v_rmnss, n_spec);
                snprintf(fn, sizeof fn, "dump/cuMES/vel_vzmncs_iter_%d.bin", iter2);
                dumpDeviceArray(fn, st.d_v_zmncs, n_spec);
            }
        }
        if (iter2 >= kE2Start && iter2 < kE2Start + 40) {
            char fn[128];
            snprintf(fn, sizeof fn, "dump/cuMES/fspec_precon_iter_%d.bin", iter2);
            dumpDeviceArray(fn, d_f_spec, (size_t)5 * p.mnmax * p.ns);
        }
#endif

        // ---- Preconditioned residuals (vmecpp fsqr1/fsqz1/fsql1) ----
        { dim3 b3(256),g3(3); computeResidualsKernel<<<g3,b3>>>(d_f_spec,p.ns,p.mnmax,d_sq); }
        double h_sq[3]; checkCuda(cudaMemcpy(h_sq,d_sq,3*sizeof(double),cudaMemcpyDeviceToHost),"cpy sq");
        double fsqr=h_sq[0], fsqz=h_sq[1], fsql=h_sq[2];
        double fsq = fsqr + fsqz + fsql;  // vmecpp fsq1: drives damping/restart control

        // ---- Damping parameter (vmecpp Evolve) ----
        // 1/tau tracks the RATE of decrease of fsq (log-ratio), capped at
        // 0.15/delt, averaged over a 10-iteration window. On the first pass
        // after a restart (iter2 == iter1) the history is reinitialized.
        if (iter2 == iter1) {
            for (int ii = 0; ii < 10; ++ii) inv_tau_hist[ii] = 0.15 / delt;
        }
        for (int ii = 0; ii < 9; ++ii) inv_tau_hist[ii] = inv_tau_hist[ii + 1];
        if (iter2 > iter1) {
            double invtau_num = 0.0;
            if (fsq != 0.0) {
                invtau_num = std::min(std::abs(std::log(fsq / fsq_prev)), 0.15);
            }
            inv_tau_hist[9] = invtau_num / delt;
        }
        fsq_prev = fsq;

        double otav = 0.0;
        for (double v : inv_tau_hist) otav += v;
        otav /= 10.0;
        double dtau = delt * otav / 2.0;
        if (kDtauFloor > 0.0) dtau = fmax(dtau, kDtauFloor);  // E4-A experiment
        double b1 = 1.0 - dtau, fac = 1.0 / (1.0 + dtau);

        // ---- Time-step control (vmecpp VMEC_8_52) ----
        // res0 is the running minimum of fsq. The state is backed up
        // whenever fsq hits a new minimum after 10 consistent iterations;
        // BAD_JACOBIAN restores it when fsq blows up 100x past the minimum;
        // BAD_PROGRESS restores it when the solver stalls at large invariant
        // forces for too long.
        if (iter2 == iter1 || res0 == -1.0) res0 = fsq;
        res0 = std::min(res0, fsq);

        enum RestartReason { kNoRestart, kBadJacobian, kBadProgress };
        RestartReason reason = kNoRestart;
        if (fsq <= res0 && (iter2 - iter1) > 10) {
            backupState();  // consistent progress: refresh rollback target
        } else if (fsq > 100.0 * res0 && iter2 > iter1) {
            reason = kBadJacobian;
        } else if ((iter2 - iter1) > 12 && iter2 > 50 &&
                   (fsqr_i + fsqz_i) > 1.0e-2) {
            reason = kBadProgress;
        }

#ifdef DUMP_CUMES_VERIFY
        recordPass((int)reason, fsqr_i, fsqz_i, fsql_i,
                   fsqr, fsqz, fsql, delt, otav, dtau, b1, fac);
#endif

        if (reason != kNoRestart) {
            restoreState();
            if (reason == kBadJacobian) { delt *= 0.9; ++ijacob; }
            else { delt /= 1.03; }
            iter1 = iter2;
            printf("  -> %s (iter2=%d) delt=%.3e\n",
                   reason == kBadJacobian ? "BAD JACOBIAN" : "BAD PROGRESS",
                   iter2, delt);
            continue;  // no descent this pass; next pass reinitializes damping
        }

        // ---- Output (every 50 iters, first 5, or late iterations) ----
        if(iter%50==0||iter<5||(iter>1950&&iter%10==0)) {
            printf("%5d | %11.3e %11.3e %11.3e | %8.2e",iter2,fsqr_i,fsqz_i,fsql_i,delt);
            // Print R_00 at axis and boundary
            double h_rmncc_axis, h_rmncc_bnd;
            checkCuda(cudaMemcpy(&h_rmncc_axis, st.d_rmncc, sizeof(double), cudaMemcpyDeviceToHost), "cpy Rax");
            checkCuda(cudaMemcpy(&h_rmncc_bnd, st.d_rmncc + (p.ns-1), sizeof(double), cudaMemcpyDeviceToHost), "cpy Rbnd");
            printf(" | Rax=%.4f Rbnd=%.4f\n", h_rmncc_axis, h_rmncc_bnd);
        }

        // ---- Descent step (Garabedian second-order Richardson) ----------
        { dim3 bd(256), gd((p.ns*p.mnmax+255)/256);
          descentStepKernel<<<gd,bd>>>(
              st.d_rmncc,st.d_rmnss,st.d_zmnsc,st.d_zmncs,st.d_lmnsc,
              st.d_v_rmncc,st.d_v_rmnss,st.d_v_zmnsc,st.d_v_zmncs,st.d_v_lmnsc,
              d_f_spec, fp.basis.d_xm,
              p.ns,p.mnmax,delt,b1,fac);
          checkCuda(cudaGetLastError(),"descent"); }

        iter2++;  // effective iteration counter advances on good passes only

        if(iter==kMaxIterEff-1){ res.iterations=kMaxIterEff;
            res.fsqr=fsqr_i;res.fsqz=fsqz_i;res.fsql=fsql_i;res.delt=delt; }
    }

#ifdef DUMP_CUMES_VERIFY
    if (dumpEnabled()) {
        // Per-pass record, column-major: 15 blocks of n_passes doubles.
        FILE* fpr = fopen("dump/cuMES/per_iter_residuals_cumes.bin", "wb");
        if (fpr) {
            uint64_t n = (uint64_t)n_passes;
            fwrite(&n, sizeof(uint64_t), 1, fpr);
            for (int c = 0; c < 15; ++c)
                for (int i = 0; i < n_passes; ++i)
                    fwrite(&per_iter[i][c], sizeof(double), 1, fpr);
            fclose(fpr);
        }
        delete[] per_iter;
    } else {
        delete[] per_iter;
    }
#endif

    checkCuda(cudaFree(d_f_spec),"free f");
    checkCuda(cudaFree(d_sq),"free sq");
    checkCuda(cudaFree(d_bk_rmncc),"free bk cc");
    checkCuda(cudaFree(d_bk_zmnsc),"free bk zsc");
    checkCuda(cudaFree(d_bk_lmnsc),"free bk lsc");
    checkCuda(cudaFree(d_bk_rmnss),"free bk ss");
    checkCuda(cudaFree(d_bk_zmncs),"free bk zcs");
    preconFree(pw); constraintFree(cw);
    return res;
}
