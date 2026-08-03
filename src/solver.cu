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
#include <vector>

static constexpr int kPreconInterval = 25;  // update preconditioner every N iters

static void checkCuda(cudaError_t err, const char* tag) {
    if (err != cudaSuccess) { fprintf(stderr, "CUDA error [%s]: %s\n", tag, cudaGetErrorString(err)); exit(EXIT_FAILURE); }
}

#ifdef DUMP_CUMES_VERIFY
static bool dumpEnabled();  // defined below with the dump machinery
static void dumpDeviceArray(const char* filename, const double* d_data,
                            size_t nelem);  // defined below
#endif

__global__ void rzNormKernel(  // defined below (before computeResidualsKernel)
    const double* __restrict__ rmncc, const double* __restrict__ zmnsc,
    const double* __restrict__ rmnss, const double* __restrict__ zmncs,
    const int* __restrict__ xm, const int* __restrict__ xn,
    int ns, int mnmax, double* __restrict__ out);

// ---- vmecpp force-norm assembly (ideal_mhd_model.cc computeForceNorms) ---
// Combines the per-surface partial sums (computeForceNormPartials) with the
// decomposed R/Z coefficient norm (rzNorm) into the residual normalization
// factors used by the control:
//   E_mag  = |Σ gsqrt·|B|²/2·wInt|·deltaS      (magnetic energy)
//   E_therm = Σ presH·dVdsH·deltaS             (thermal energy; gamma=0 →
//              presH = mass profile, which is what cuMES stores in d_pres_H)
//   V      = Σ dVdsH·deltaS                    (plasma volume)
//   energyDensity = max(E_mag, E_therm)/V
//   fNormRZ = 1/(Σ guu·r12²·wInt · energyDensity²)
//   fNormL = 1/(Σ (bsubu²+bsubv²)·wInt · lamscale²)
//   fNorm1 = 1/rzNorm
// with the wInt trapezoid over the reduced poloidal grid [0, pi].
// Called on the same cadence as the preconditioner update (vmecpp:
// computeForceNorms inside shouldUpdateRadialPreconditioner).
static void computeForceNorms(SpectralState& st, const FourierPlan& fp,
                              const GridParams& p, const RadialProfiles& rp,
                              const MetricWorkspace& mw,
                              double* d_psum, double* d_rzsum, int iter2,
                              double& fNormRZ, double& fNormL, double& fNorm1) {
    computeForceNormPartials(p, mw, rp.d_dVds_H, d_psum);

    { dim3 b1(256), g1(1);
      rzNormKernel<<<g1, b1>>>(st.d_rmncc, st.d_zmnsc, st.d_rmnss, st.d_zmncs,
                               fp.basis.d_xm, fp.basis.d_xn,
                               p.ns, p.mnmax, d_rzsum);
      checkCuda(cudaGetLastError(), "rzNorm");
      checkCuda(cudaDeviceSynchronize(), "rzNorm sync"); }

    int nH = p.ns - 1;
    double* h_psum = new double[4 * nH];
    double* h_dVds = new double[nH];
    double* h_pres = new double[nH];
    double h_rz = 0.0;
    checkCuda(cudaMemcpy(h_psum, d_psum, 4 * nH * sizeof(double),
                         cudaMemcpyDeviceToHost), "psum cpy");
    checkCuda(cudaMemcpy(h_dVds, rp.d_dVds_H, nH * sizeof(double),
                         cudaMemcpyDeviceToHost), "dVds cpy");
    checkCuda(cudaMemcpy(h_pres, rp.d_pres_H, nH * sizeof(double),
                         cudaMemcpyDeviceToHost), "pres cpy");
    checkCuda(cudaMemcpy(&h_rz, d_rzsum, sizeof(double),
                         cudaMemcpyDeviceToHost), "rz cpy");

    double sRZ = 0.0, sL = 0.0, sMag = 0.0, eTherm = 0.0, vol = 0.0;
    for (int j = 0; j < nH; ++j) {
        sRZ += h_psum[4 * j + 0];
        sL  += h_psum[4 * j + 1];
        sMag += h_psum[4 * j + 2];
        eTherm += h_pres[j] * h_dVds[j];
        vol += h_dVds[j];
    }
    double deltaS = rp.delta_s;
    double eMag = fabs(sMag) * deltaS;   // vmecpp: fabs(localMagneticEnergy)*deltaS
    eTherm *= deltaS;
    vol *= deltaS;
    double energyDensity = std::max(eMag, eTherm) / vol;
    fNormRZ = 1.0 / (sRZ * energyDensity * energyDensity);
    fNormL = 1.0 / (sL * p.lamscale * p.lamscale);
    fNorm1 = 1.0 / h_rz;

#ifdef DUMP_CUMES_VERIFY
    if (dumpEnabled()) {
        // Same format as vmecpp's dump/vmecpp/force_norms_iter_<iter2>.txt
        char fn[128];
        snprintf(fn, sizeof fn, "dump/cuMES/force_norms_iter_%d.txt", iter2);
        FILE* fp2 = fopen(fn, "w");
        if (fp2) {
            fprintf(fp2,
                    "magneticEnergy %.17e\n"
                    "thermalEnergy %.17e\n"
                    "plasmaVolume %.17e\n"
                    "energyDensity %.17e\n"
                    "forceNormSumRZ %.17e\n"
                    "forceNormSumL %.17e\n"
                    "rzNorm %.17e\n"
                    "fNormRZ %.17e\n"
                    "fNormL %.17e\n"
                    "fNorm1 %.17e\n",
                    eMag, eTherm, vol, energyDensity, sRZ, sL, h_rz,
                    fNormRZ, fNormL, fNorm1);
            fclose(fp2);
        }
    }
#endif

    delete[] h_psum;
    delete[] h_dVds;
    delete[] h_pres;
}

// extrapolateTowardsAxis: copy m=1 coefficients from first interior
// surface (j=1) to the magnetic axis (j=0), matching vmecpp's
// extrapolateTowardsAxis(). Only m=1 has a finite value at the axis
// for stellarator-symmetric equilibria. Mode table: mode = m*(ntor+1)+n.
__global__ void extrapolateAxisKernel(
    double* __restrict__ rmncc, double* __restrict__ zmnsc,
    double* __restrict__ lmnsc, double* __restrict__ rmnss,
    double* __restrict__ zmncs, double* __restrict__ lmncs,
    int ns, int mnmax, int ntorp1) {
    int mode = blockIdx.x * blockDim.x + threadIdx.x;
    if (mode >= mnmax) return;
    int m = mode / ntorp1;  // poloidal mode number
    if (m == 0) {
        // m=0: only the lambda sin(n zeta) component is extrapolated —
        // vmecpp's extrapolateTowardsAxis ("m=0 component of lambda
        // leftover from chi-force"). This makes the even-parity lambda_zeta
        // at the axis match vmecpp, which feeds the axis-adjacent B^theta
        // through the half-grid average (FIXED 2026-08-02: without it the
        // jH=0 bsupu was off by ~30% on the first lambda != 0 pass, seeding
        // a 1e-4-level drift of the lambda channel).
        lmncs[0 + mode*ns] = lmncs[1 + mode*ns];
        return;
    }
    if (m != 1) return;     // only m=1 needs extrapolation
    // Copy from j=1 to j=0
    rmncc[0 + mode*ns] = rmncc[1 + mode*ns];
    zmnsc[0 + mode*ns] = zmnsc[1 + mode*ns];
    lmnsc[0 + mode*ns] = lmnsc[1 + mode*ns];
    rmnss[0 + mode*ns] = rmnss[1 + mode*ns];
    zmncs[0 + mode*ns] = zmncs[1 + mode*ns];
    lmncs[0 + mode*ns] = lmncs[1 + mode*ns];
}

// Apply vmecpp's even/odd-m decomposition scaling (decomposeInto) to the
// spectral forces. vmecpp decomposes its FORCES: odd-m force coefficients
// carry an extra 1/max(sqrt(s_F), sqrt(1/(ns-1))) factor (Eqn. 8c in
// Hirshman, Schwenn & Nuehrenberg 1990), which it then evolves through the
// residual, preconditioned-solve, and descent pipeline (m_decomposed_f).
// max(..., sqrt(s1)) clamps the factor at the axis to the innermost
// surface's sqrt(s), keeping it finite at s=0 (constant extrapolation,
// the same treatment as extrapolateAxisKernel above).
//
// IMPORTANT: only the FORCES are transformed here. The spectral state
// (d_rmncc/...) stays in plain physical coefficients throughout — initState
// builds s^(m/2) profiles, and the dumped state matches vmecpp's physical
// wout coefficients exactly. cuMES's real-space odd arrays likewise carry
// the decomposed form only transiently (inverseDFT divides by max(...) to
// feed the vmecpp-formula kernels). Consequently the odd-m factor must
// NEVER be re-applied when post-processing the state (e.g. flux-surface
// plots): doing so double-scales and flattens the m=1 shift profile into
// a linear-in-s one. Even-m modes: scalxc = 1 (no change).
__global__ void scalxcApplyKernel(
    double* __restrict__ f_spec, const double* __restrict__ sqrtS_F,
    const int* __restrict__ xm, int ns, int mnmax, double sqrtS1)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x, total = mnmax * ns;
    if (i >= total) return;
    int mode = i / ns;
    // Parity test on the POLOIDAL m (xm[mode]), NOT the mode index
    // (m*(ntor+1)+n): for odd n the mode index parity is flipped.
    // (FIXED 2026-08-02: `int m = i/ns` tested the mode index, so all
    // n=1,3,5... entries got scalxc applied to the wrong parity --
    // even-m scaled by 1/sqrt(s), odd-m left unscaled.)
    int m = xm[mode];
    if (m % 2 == 0) return;  // even-m: scalxc = 1
    int j = i % ns;
    double scal = 1.0 / fmax(sqrtS_F[j], sqrtS1);
    for (int c = 0; c < 6; ++c) f_spec[c * mnmax * ns + i] *= scal;
}

// vmecpp's m1 gauge on the decomposed forces, applied every iteration after
// the forward DFT:
//   1. m1Constraint: rss = (rss+zcs)/sqrt(2), zcs = (rss-zcs)/sqrt(2)
//   2. zeroZForceForM1 (conditional): fzcs = 0  (m=1 Z force constrained
//      to zero)
// The zeroing mirrors vmecpp's `fix_m1_gauge` flag in IdealMhdModel::update:
// with always_fix_m1_gauge = false (the vmec_standalone default) the m=1
// fzcs is zeroed ONLY on the first pass (iter2 < 2) or once the invariant
// Z-residual dropped below 1e-6. In between, the mixed fzcs stays nonzero
// and the m=1 mixed zcs' velocity evolves freely -- cuMES previously zeroed
// it unconditionally, which froze the mixed gauge and drifted the physical
// m=1 R/Z coefficients ~1e-3 off the reference.
// Applied to components 3 (frss) and 4 (fzcs) of the 6-component layout.
__global__ void m1ConstraintKernel(double* __restrict__ f_spec, int ns,
                                   int mnmax, int ntor, int zeroZ) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ns) return;
    const double s = 1.0 / std::sqrt(2.0);
    int m1base = ntor + 1;  // mode index of (m=1, n=0)
    for (int n = 0; n < ntor + 1; ++n) {
        int mn = m1base + n;
        double* rss = &f_spec[j + mn * ns + 3 * mnmax * ns];
        double* zcs = &f_spec[j + mn * ns + 4 * mnmax * ns];
        double old_rss = *rss;
        double old_zcs = *zcs;
        *rss = (old_rss + old_zcs) * s;
        if (zeroZ) {
            *zcs = 0.0;  // zeroZForceForM1
        } else {
            *zcs = (old_rss - old_zcs) * s;  // keep the mixed zcs
        }
    }
}

// vmecpp's applyM1Preconditioner (FourierForces): scales the m=1 frss by
// (ard+brd)/denom and fzcs by (azd+bzd)/denom using the odd-parity diagonal
// precon elements. The fzcs scale matters only when the mixed fzcs is
// nonzero (fix_m1_gauge = false), i.e. for iter2 >= 2 before convergence.
// Applied right before the RZ preconditioner (after the invariant residuals).
__global__ void m1PreconScaleKernel(double* __restrict__ f_spec,
                                    const double* __restrict__ ard,
                                    const double* __restrict__ brd,
                                    const double* __restrict__ azd,
                                    const double* __restrict__ bzd,
                                    int ns, int mnmax, int ntor) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ns) return;
    int m1base = ntor + 1;
    double denom = ard[j * 2 + 1] + brd[j * 2 + 1] +
                   azd[j * 2 + 1] + bzd[j * 2 + 1];
    double scaleR = (ard[j * 2 + 1] + brd[j * 2 + 1]) / denom;
    double scaleZ = (azd[j * 2 + 1] + bzd[j * 2 + 1]) / denom;
    for (int n = 0; n < ntor + 1; ++n) {
        int mn = m1base + n;
        f_spec[j + mn * ns + 3 * mnmax * ns] *= scaleR;
        f_spec[j + mn * ns + 4 * mnmax * ns] *= scaleZ;
    }
}

// ---- rzNorm (vmecpp FourierCoeffs::rzNorm) -------------------------------
// Sum of squared decomposed R/Z coefficients over all ns rows (axis through
// LCFS), excluding the rcc (m=0,n=0) offset. cuMES stores the state as plain
// PHYSICAL coefficients (state = vmecpp-decomposed * ms*ns), so each term is
// scaled by 1/(ms*ns)^2. For m=1 the decomposed (rmnss, zmncs) pair is
// stored M(1/2)-mixed in vmecpp, giving
//   rss_d^2 + zcs_d^2 = (rss_p^2 + zcs_p^2) / (2 * (ms*ns)^2)
// (the 0.5 factor below); all other modes use the unmixed square.
__global__ void rzNormKernel(
    const double* __restrict__ rmncc, const double* __restrict__ zmnsc,
    const double* __restrict__ rmnss, const double* __restrict__ zmncs,
    const int* __restrict__ xm, const int* __restrict__ xn,
    int ns, int mnmax, double* __restrict__ out)
{
    double sum = 0.0;
    int total = mnmax * ns;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        int m = i / ns, j = i % ns, mm = xm[m], nn = xn[m];
        if (j == 0 && mm > 0) continue;  // vmecpp keeps the stored axis m>0
                                         // at 0 (extrapolated only in real
                                         // space); the state-file axis row
                                         // therefore contributes nothing to
                                         // rzNorm
        double mfac = (mm == 0) ? 1.0 : std::sqrt(2.0);
        double nfac = (nn == 0) ? 1.0 : std::sqrt(2.0);
        // decomposed = physical/(ms*ns): the squared term picks up 1/(ms*ns)^2
        double inv2 = 1.0 / (mfac * nfac * mfac * nfac);
        double rcc = rmncc[i], zsc = zmnsc[i];
        double rss = rmnss[i], zcs = zmncs[i];
        if (mm > 0 || nn > 0) sum += rcc * rcc * inv2;
        sum += zsc * zsc * inv2;
        if (mm == 1) {
            // decomposed pair is mixed: (rss_d^2 + zcs_d^2) = (rss_p^2 +
            // zcs_p^2) / (2 * (ms*ns)^2)
            sum += 0.5 * (rss * rss + zcs * zcs) * inv2;
        } else {
            sum += (rss * rss + zcs * zcs) * inv2;
        }
    }
    __shared__ double s_sum[256];
    int tid = threadIdx.x;
    s_sum[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_sum[tid] += s_sum[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[0] = s_sum[0];
}

// Residual groups match vmecpp's FourierForces::residuals (folded basis):
//   fsqr = Σ frcc² + frss²,  fsqz = Σ fzsc² + fzcs²,  fsql = Σ flsc² + flcs²
// (components 0..5 of f_spec: frcc fzsc flsc frss fzcs flcs).
__global__ void computeResidualsKernel(const double* __restrict__ f_spec, int ns, int mnmax,
                                        double* __restrict__ sq_out) {
    int comp = blockIdx.x; if(comp>=3)return;
    double sum=0; int total=mnmax*ns;
    for(int i=threadIdx.x; i<total; i+=blockDim.x){
        double a=f_spec[i+comp*total];
        double b=f_spec[i+(comp+3)*total];
        sum+=a*a+b*b;
    }
    __shared__ double s_sum[256]; int tid=threadIdx.x; s_sum[tid]=sum; __syncthreads();
    for(int s=blockDim.x/2; s>0; s>>=1){if(tid<s)s_sum[tid]+=s_sum[tid+s]; __syncthreads();}
    if(tid==0) sq_out[comp]=s_sum[0]/(mnmax*ns);
}

__global__ void descentStepKernel(
    double* __restrict__ x_cc, double* __restrict__ x_ss,
    double* __restrict__ x_zsc, double* __restrict__ x_zcs,
    double* __restrict__ x_lsc, double* __restrict__ x_lcs,
    double* __restrict__ v_cc, double* __restrict__ v_ss,
    double* __restrict__ v_zsc, double* __restrict__ v_zcs,
    double* __restrict__ v_lsc, double* __restrict__ v_lcs,
    const double* __restrict__ f_spec,
    const int* __restrict__ xm, const int* __restrict__ xn,
    int ns, int mnmax, double delt, double b1, double fac)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x, total = mnmax*ns;
    if(i>=total)return;
    int m=i/ns, j=i%ns, mm=xm[m];
    // Skip m>0 modes at axis: coordinate singularity at s=0. The forces
    // there are zeroed by the preconditioners (jMin identity rows for R/Z,
    // sqrt(s)^pwr for lambda), so vmecpp never moves these coefficients.
    if (j == 0 && mm > 0) return;

    // State/velocity representations: the spectral STATE carries the
    // mscale*nscale basis factors (state = vmecpp-decomposed * ms*ns) while
    // the FORCES and VELOCITIES are vmecpp-decomposed. The state increment
    // must therefore be scaled by ms*ns per mode. (FIXED 2026-08-02: the
    // odd-n/odd-m increments were ms*ns too small, e.g. sqrt(2) for m=1n=0,
    // making the iter-2+ states drift ~0.7%/step.)
    double mfac = (mm == 0) ? 1.0 : std::sqrt(2.0);
    double nfac = (xn[m] == 0) ? 1.0 : std::sqrt(2.0);
    double f = mfac * nfac;

    // R/Z components (0,1,3,4): LCFS is fixed — the force was zeroed by
    // fixBoundaryKernel and the coefficient must not move. Lambda (comps
    // 2,5) is free on all surfaces including the LCFS, matching vmecpp.
    if (j < ns-1) {
        double vr=v_cc[i]; vr=fac*(b1*vr+ delt*f_spec[i+0*mnmax*ns]); v_cc[i]=vr; x_cc[i]+=delt*vr*f;
        double vz=v_zsc[i];vz=fac*(b1*vz+ delt*f_spec[i+1*mnmax*ns]);v_zsc[i]=vz;x_zsc[i]+=delt*vz*f;
        double vs=v_ss[i]; vs=fac*(b1*vs+ delt*f_spec[i+3*mnmax*ns]); v_ss[i]=vs;
        double vzc=v_zcs[i];vzc=fac*(b1*vzc+ delt*f_spec[i+4*mnmax*ns]);v_zcs[i]=vzc;
        if (mm == 1) {
            // m1 gauge: the state is stored in the UNDONE gauge while the
            // velocities/forces are vmecpp-decomposed (mixed gauge). vmecpp's
            // state evolves in the mixed gauge and is undone each update, so
            // the undone state must increment with the UNDONE velocity:
            //   rmnss += (vrss+vzcs), zmncs += (vrss-vzcs)
            // (FIXED 2026-08-02: without the mixing the iter-2+ m=1 states
            // drifted from vmecpp by ~0.07, corrupting the real-space.)
            x_ss[i]  += delt * (vs + vzc) * f;
            x_zcs[i] += delt * (vs - vzc) * f;
        } else {
            x_ss[i]  += delt * vs * f;
            x_zcs[i] += delt * vzc * f;
        }
    }
    double vl=v_lsc[i];vl=fac*(b1*vl+ delt*f_spec[i+2*mnmax*ns]);v_lsc[i]=vl;x_lsc[i]+=delt*vl*f;
    double vlc=v_lcs[i];vlc=fac*(b1*vlc+ delt*f_spec[i+5*mnmax*ns]);v_lcs[i]=vlc;x_lcs[i]+=delt*vlc*f;
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
    SolverResult res{false,0,1.0,1.0,1.0,p.delt};
    size_t nb = 6*p.ns*p.mnmax*sizeof(double);
    double *d_f_spec, *d_sq;
    checkCuda(cudaMalloc(&d_f_spec,nb),"malloc f");
    checkCuda(cudaMalloc(&d_sq,3*sizeof(double)),"malloc sq");
    PreconWorkspace pw = preconCreate(p);
    ConstraintWorkspace cw = constraintCreate(p);

    // ---- transform timing (cudaEvent pairs around inverseDFT/forwardDFT) ----
    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);
    float t_inv_ms = 0.0f, t_fwd_ms = 0.0f;

    // ---- env-gated knobs for convergence experiments (defaults = input
    // values; set via CUMES_MAX_ITER, CUMES_DELT0, CUMES_DTAU_FLOOR,
    // CUMES_DUMP_ITER, CUMES_E2_START) ----
    int kMaxIterEff = p.max_iter;
    double kDelt0Eff = p.delt;
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
    double fsqz_prev = 0.0;  // vmecpp: fc_.fsqz = 0.0 at stage start; feeds
                             // the fix_m1_gauge condition (zeroZForceForM1)
    int ijacob = 0;
    double inv_tau_hist[10];
    for (int ii = 0; ii < 10; ++ii) inv_tau_hist[ii] = 0.15 / kDelt0Eff;
    double delt=kDelt0Eff;

    // vmecpp residual normalization factors (computeForceNorms), refreshed on
    // the same cadence as the preconditioner (every kPreconInterval passes).
    double fNormRZ = 0.0, fNormL = 0.0, fNorm1 = 0.0;
    double *d_psum, *d_rzsum;
    checkCuda(cudaMalloc(&d_psum, 4 * (size_t)(p.ns - 1) * sizeof(double)), "malloc psum");
    checkCuda(cudaMalloc(&d_rzsum, sizeof(double)), "malloc rzsum");

    // State rollback: backup arrays for restoring spectral state on restart.
    size_t nb_one = (size_t)p.ns * (size_t)p.mnmax * sizeof(double);
    double *d_bk_rmncc, *d_bk_zmnsc, *d_bk_lmnsc, *d_bk_rmnss, *d_bk_zmncs, *d_bk_lmncs;
    checkCuda(cudaMalloc(&d_bk_rmncc, nb_one), "bk cc");
    checkCuda(cudaMalloc(&d_bk_zmnsc, nb_one), "bk zsc");
    checkCuda(cudaMalloc(&d_bk_lmnsc, nb_one), "bk lsc");
    checkCuda(cudaMalloc(&d_bk_rmnss, nb_one), "bk ss");
    checkCuda(cudaMalloc(&d_bk_zmncs, nb_one), "bk zcs");
    checkCuda(cudaMalloc(&d_bk_lmncs, nb_one), "bk lcs");

    // Helper: copy current spectral state to backup (GPU device-to-device)
    auto backupState = [&]() {
        cudaMemcpy(d_bk_rmncc, st.d_rmncc, nb_one, cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_bk_zmnsc, st.d_zmnsc, nb_one, cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_bk_lmnsc, st.d_lmnsc, nb_one, cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_bk_rmnss, st.d_rmnss, nb_one, cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_bk_zmncs, st.d_zmncs, nb_one, cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_bk_lmncs, st.d_lmncs, nb_one, cudaMemcpyDeviceToDevice);
    };

    // Helper: restore spectral state from backup + zero velocities
    auto restoreState = [&]() {
        cudaMemcpy(st.d_rmncc, d_bk_rmncc, nb_one, cudaMemcpyDeviceToDevice);
        cudaMemcpy(st.d_zmnsc, d_bk_zmnsc, nb_one, cudaMemcpyDeviceToDevice);
        cudaMemcpy(st.d_lmnsc, d_bk_lmnsc, nb_one, cudaMemcpyDeviceToDevice);
        cudaMemcpy(st.d_rmnss, d_bk_rmnss, nb_one, cudaMemcpyDeviceToDevice);
        cudaMemcpy(st.d_zmncs, d_bk_zmncs, nb_one, cudaMemcpyDeviceToDevice);
        cudaMemcpy(st.d_lmncs, d_bk_lmncs, nb_one, cudaMemcpyDeviceToDevice);
        cudaMemset(st.d_v_rmncc, 0, nb_one);
        cudaMemset(st.d_v_zmnsc, 0, nb_one);
        cudaMemset(st.d_v_lmnsc, 0, nb_one);
        cudaMemset(st.d_v_rmnss, 0, nb_one);
        cudaMemset(st.d_v_zmncs, 0, nb_one);
        cudaMemset(st.d_v_lmncs, 0, nb_one);
    };

    // Take initial backup
    backupState();

#ifdef DUMP_CUMES_VERIFY
    // Per-pass record for convergence analysis (mirrors vmecpp's
    // per_iter_residuals.bin + control scalars). 15 columns:
    // fsqr_i fsqz_i fsql_i fsqr fsqz fsql delt otav dtau b1 fac
    // iter2 iter1 reason rax(axis R at zeta=0, pre-descent of the pass).
    double (*per_iter)[15] = new double[kMaxIterEff][15];
    int n_passes = 0;
    // Radial location of the magnetic axis at zeta=0, matching vmecpp's r00
    // (Printout: geometric_offset.r_00 = r1_e[0] — the real-space even-m R
    // at the axis at theta=0, zeta=0). With the axis coefficients this is
    // the sum of the m=0 row: sum_n rmncc(0,n)@axis * cos(n*nfp*0) — the
    // plain R_00 coefficient alone misses the axis R wobble with zeta
    // (~+0.36 for W7-X, dominated by rmnc(0,1)@axis = +0.35).
    auto axisRAtZeta0 = [&]() {
        // mode-major layout [mode*ns + j]: the m=0 modes (mode = n) at the
        // axis row (j=0) sit at indices n*ns — not contiguous.
        double h = 0.0;
        for (int n = 0; n <= p.ntor; ++n) {
            double v;
            checkCuda(cudaMemcpy(&v, st.d_rmncc + (size_t)n * p.ns, sizeof(double),
                                 cudaMemcpyDeviceToHost), "cpy Rax");
            h += v;
        }
        return h;
    };
    auto recordPass = [&](int reason, double fRi, double fZi, double fLi,
                          double fR, double fZ, double fL, double d,
                          double o, double dt, double b1v, double fcv) {
        if (!dumpEnabled()) return;
        if (n_passes < kMaxIterEff) {
            double* r = per_iter[n_passes++];
            r[0]=fRi; r[1]=fZi; r[2]=fLi; r[3]=fR; r[4]=fZ; r[5]=fL;
            r[6]=d; r[7]=o; r[8]=dt; r[9]=b1v; r[10]=fcv;
            r[11]=(double)iter2; r[12]=(double)iter1; r[13]=(double)reason;
            r[14]=axisRAtZeta0();
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
        dumpDeviceArray("dump/cuMES/step_0_lmncs.bin",      st.d_lmncs, n_spec);
        dumpDeviceArray("dump/cuMES/step_0_currH.bin",      rp.d_curr_H, p.ns - 1);
        dumpDeviceArray("dump/cuMES/step_0_chipH.bin",      rp.d_chip_H, p.ns - 1);
        dumpDeviceArray("dump/cuMES/step_0_iotaH.bin",      rp.d_iota_H, p.ns - 1);
        dumpDeviceArray("dump/cuMES/step_0_iotaF.bin",      rp.d_iota_F, p.ns);
    }
#endif

    for(int iter=0; iter<kMaxIterEff; ++iter){
        // vmecpp: after 25/50 bad Jacobians, restore the state and reset the
        // time step to 0.98/0.96 of the INITIAL delt (vmec.cc, "HAVING A
        // CONVERGENCE PROBLEM: RESETTING DELT"). The restoreState() below
        // mirrors vmecpp's RestartIteration(BAD_JACOBIAN) in that branch;
        // ++ijacob matches the increment inside RestartIteration (so the
        // 0.98/0.96 scale keys off the post-increment value, as in vmecpp).
        if (ijacob == 25 || ijacob == 50) {
            restoreState();
            ++ijacob;
            delt = (ijacob < 50 ? 0.98 : 0.96) * kDelt0Eff;
            iter1 = iter2;
            printf("  -> CONVERGENCE PROBLEM: RESETTING DELT to %.3e (ijacob=%d)\n",
                   delt, ijacob);
            continue;
        }

        // Extrapolate m=1 coefficients to the magnetic axis (j=0)
        // before inverse DFT, matching vmecpp's extrapolateTowardsAxis().
        // Must be done each iteration since the descent step updates j=1
        // but skips j=0 for m>0 (axis regularity).
        extrapolateAxisKernel<<<(p.mnmax + 31) / 32, 32>>>(
            st.d_rmncc, st.d_zmnsc, st.d_lmnsc, st.d_rmnss, st.d_zmncs,
            st.d_lmncs, p.ns, p.mnmax, p.ntor + 1);
        checkCuda(cudaGetLastError(), "extrapAxis");

        cudaEventRecord(ev0);
        inverseDFT(fp,st,p,false);
        cudaEventRecord(ev1);
        cudaEventSynchronize(ev1);
        { float ms; cudaEventElapsedTime(&ms, ev0, ev1); t_inv_ms += ms; }

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
        if (iter == 0 || iter2 == 2) {
            size_t n_real = (size_t)p.ns * (size_t)p.nZnT;
            char fn[128];
            snprintf(fn, sizeof fn, "dump/cuMES/step_A_lv_e_iter_%d.bin", iter == 0 ? 1 : 2);
            dumpDeviceArray(fn, fp.d_lv_e, n_real);
            snprintf(fn, sizeof fn, "dump/cuMES/step_A_lv_o_iter_%d.bin", iter == 0 ? 1 : 2);
            dumpDeviceArray(fn, fp.d_lv_o, n_real);
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
        // start of update()). Cadence matches vmecpp's
        // shouldUpdateRadialPreconditioner: (iter2 - iter1) % 25 == 0, plus
        // the dump-mode refresh at the handoff window (iter2 == kDumpIter)
        // so dump-mode runs stay same-state with the vmecpp reference.
        bool precon_updated = ((iter2 - iter1) % kPreconInterval) == 0 ||
                              (dumpEnabled() && iter2 == kDumpIter);
        if (precon_updated) {
            preconCompute(fp, p, rp, mw, pw);

            // vmecpp computeForceNorms (same cadence): residual normalization
            // factors feeding fsqr/fsqz/fsql and fsqr1/fsqz1/fsql1.
            computeForceNorms(st, fp, p, rp, mw, d_psum, d_rzsum, iter2,
                              fNormRZ, fNormL, fNorm1);

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
                dumpDeviceArray("dump/cuMES/step_E_brmn_e_iter_1.bin", fp.d_brmn_e, n_real);
                dumpDeviceArray("dump/cuMES/step_E_brmn_o_iter_1.bin", fp.d_brmn_o, n_real);
                dumpDeviceArray("dump/cuMES/step_E_bzmn_e_iter_1.bin", fp.d_bzmn_e, n_real);
                dumpDeviceArray("dump/cuMES/step_E_bzmn_o_iter_1.bin", fp.d_bzmn_o, n_real);
                dumpDeviceArray("dump/cuMES/step_E_crmn_e_iter_1.bin", fp.d_crmn_e, n_real);
                dumpDeviceArray("dump/cuMES/step_E_crmn_o_iter_1.bin", fp.d_crmn_o, n_real);
                dumpDeviceArray("dump/cuMES/step_E_czmn_e_iter_1.bin", fp.d_czmn_e, n_real);
                dumpDeviceArray("dump/cuMES/step_E_czmn_o_iter_1.bin", fp.d_czmn_o, n_real);
                dumpDeviceArray("dump/cuMES/step_E_blmn_e_iter_1.bin", fp.d_blmn_e, n_real);
                dumpDeviceArray("dump/cuMES/step_E_blmn_o_iter_1.bin", fp.d_blmn_o, n_real);
                dumpDeviceArray("dump/cuMES/step_E_clmn_e_iter_1.bin", fp.d_clmn_e, n_real);
                dumpDeviceArray("dump/cuMES/step_E_clmn_o_iter_1.bin", fp.d_clmn_o, n_real);
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
            // Constraint-chain intermediates (stage-by-stage vs vmecpp)
            // State as consumed by the constraint chain at iter 1 (post-descent)
            size_t n_spec2 = (size_t)p.ns * (size_t)p.mnmax;
            dumpDeviceArray("dump/cuMES/step_GC_rmncc_iter_1.bin", st.d_rmncc, n_spec2);
            dumpDeviceArray("dump/cuMES/step_GC_rmnss_iter_1.bin", st.d_rmnss, n_spec2);
            dumpDeviceArray("dump/cuMES/step_GC_zmnsc_iter_1.bin", st.d_zmnsc, n_spec2);
            dumpDeviceArray("dump/cuMES/step_GC_zmncs_iter_1.bin", st.d_zmncs, n_spec2);
            dumpDeviceArray("dump/cuMES/step_GC_rCon_iter_1.bin", cw.d_rCon, n_real);
            dumpDeviceArray("dump/cuMES/step_GC_zCon_iter_1.bin", cw.d_zCon, n_real);
            dumpDeviceArray("dump/cuMES/step_GC_gConEff_iter_1.bin", cw.d_gConEff, n_real);
            dumpDeviceArray("dump/cuMES/step_GC_gCon_iter_1.bin", cw.d_gCon, n_real);
            dumpDeviceArray("dump/cuMES/step_GC_frcon_e_iter_1.bin", cw.d_frcon_e, n_real);
            dumpDeviceArray("dump/cuMES/step_GC_frcon_o_iter_1.bin", cw.d_frcon_o, n_real);
            dumpDeviceArray("dump/cuMES/step_GC_fzcon_e_iter_1.bin", cw.d_fzcon_e, n_real);
            dumpDeviceArray("dump/cuMES/step_GC_fzcon_o_iter_1.bin", cw.d_fzcon_o, n_real);
            // tcon/faccon profiles (device arrays; h_tcon is stale -- the
            // kernel writes d_tcon directly)
            dumpDeviceArray("dump/cuMES/step_GC_tcon_iter_1.bin", cw.d_tcon, p.ns);
            dumpDeviceArray("dump/cuMES/step_GC_faccon_iter_1.bin", cw.d_faccon, p.mpol);
        }
#endif

        cudaEventRecord(ev0);
        forwardDFT(fp,d_f_spec,p,cw);
        cudaEventRecord(ev1);
        cudaEventSynchronize(ev1);
        { float ms; cudaEventElapsedTime(&ms, ev0, ev1); t_fwd_ms += ms; }

        // Apply the odd-m decomposition scaling (vmecpp decomposeInto).
        // The forward DFT already zeroed the LCFS R/Z entries and the axis
        // m>0 entries, so the boundary stays rigid and only the lambda
        // force is present at the LCFS (free gauge, evolved by descent).
        { dim3 bs(256), gs((p.ns*p.mnmax+255)/256);
          scalxcApplyKernel<<<gs,bs>>>(d_f_spec, rp.d_sqrtS_F, fp.basis.d_xm,
                                       p.ns, p.mnmax,
                                       sqrt(1.0 / (p.ns - 1.0)));
          checkCuda(cudaGetLastError(), "scalxc"); }

#ifdef DUMP_CUMES_VERIFY
        if (iter == 0 || iter2 == kDumpIter) {
            // Dump AFTER the decomposition scaling, matching vmecpp's dump
            // of m_decomposed_f (post-decomposeInto). Keyed on iter2 so a
            // handoff/plateau comparison can use the same effective counter
            // as vmecpp's dump blocks.
            size_t n_fspec = (size_t)6 * (size_t)p.mnmax * (size_t)p.ns;
            char fn[128];
            snprintf(fn, sizeof fn, "dump/cuMES/step_H_f_spec_iter_%d.bin",
                     iter == 0 ? 1 : iter2);
            dumpDeviceArray(fn, d_f_spec, n_fspec);
        }
        if (iter2 >= kE2Start && iter2 < kE2Start + 40) {
            char fn[128];
            snprintf(fn, sizeof fn, "dump/cuMES/fspec_invariant_iter_%d.bin", iter2);
            dumpDeviceArray(fn, d_f_spec, (size_t)6 * p.mnmax * p.ns);
        }
#endif

        // ---- m1 gauge constraint (vmecpp FourierCoeffs::m1Constraint) ----
        // Applied to the decomposed forces after the forward DFT (post
        // step_H dump), before the residuals and the preconditioner,
        // matching vmecpp ideal_mhd_model.cc. The fzcs zeroing mirrors
        // vmecpp's `fix_m1_gauge = always_fix_m1_gauge || fsqz < 1e-6 ||
        // iter2 < 2` with always_fix_m1_gauge = false (the standalone
        // default): zeroZ only on the first pass and once the previous
        // pass's invariant Z-residual dropped below 1e-6.
        { dim3 b1(256), g1((p.ns + 255) / 256);
          int zeroZ = (iter2 < 2) || (fsqz_prev < 1.0e-6);
          m1ConstraintKernel<<<g1, b1>>>(d_f_spec, p.ns, p.mnmax, p.ntor,
                                         zeroZ);
          checkCuda(cudaGetLastError(), "m1Constraint"); }

        // ---- Invariant (unpreconditioned) residuals ----
        // Used for the stopping criterion and the BAD_PROGRESS threshold,
        // matching vmecpp's fsqr/fsqz/fsql (evalFResInvar). Computed before
        // preconditioning so a false minimum (tiny preconditioned forces on
        // garbage geometry) can never be mistaken for convergence.
#ifdef DUMP_CUMES_VERIFY
        if (iter == kMaxIterEff - 1) {
            dumpDeviceArray("dump/cuMES/step_final_f_spec.bin", d_f_spec,
                            (size_t)6 * p.mnmax * p.ns);
        }
#endif
        { dim3 b3(256),g3(3); computeResidualsKernel<<<g3,b3>>>(d_f_spec,p.ns,p.mnmax,d_sq); }
        double h_sq_i[3]; checkCuda(cudaMemcpy(h_sq_i,d_sq,3*sizeof(double),cudaMemcpyDeviceToHost),"cpy sqi");
        // vmecpp evalFResInvar: fsqr = fResInvar[0]·fNormRZ·0.25 (same for
        // fsqz), fsql = fResInvar[2]·fNormL, where fResInvar are the plain
        // sums. The cuMES kernel returns ΣF²/(mnmax·ns), so undo that first.
        const double plainPerEl = (double)p.mnmax * (double)p.ns;
        double fsqr_i = h_sq_i[0] * plainPerEl * fNormRZ * 0.25;
        double fsqz_i = h_sq_i[1] * plainPerEl * fNormRZ * 0.25;
        double fsql_i = h_sq_i[2] * plainPerEl * fNormL;
        fsqz_prev = fsqz_i;  // vmecpp: m_fc_.fsqz (NORMALIZED), read by the
                             // next pass's fix_m1_gauge condition

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
        if (fsqr_i <= p.ftol && fsqz_i <= p.ftol && fsql_i <= p.ftol) {
#ifdef DUMP_CUMES_VERIFY
            recordPass(0, fsqr_i, fsqz_i, fsql_i, 0,0,0, delt,0,0,0,0);
#endif
            res.converged=true; res.iterations=iter+1;
            res.fsqr=fsqr_i;res.fsqz=fsqz_i;res.fsql=fsql_i;res.delt=delt;
            printf("  -> CONVERGED at iter %d\n",iter+1); break;
        }

        // vmecpp applyM1Preconditioner: m=1 frss scale, before the RZ solve
        { dim3 b1(256), g1((p.ns + 255) / 256);
          m1PreconScaleKernel<<<g1, b1>>>(d_f_spec, pw.d_ard, pw.d_brd,
                                          pw.d_azd, pw.d_bzd,
                                          p.ns, p.mnmax, p.ntor);
          checkCuda(cudaGetLastError(), "m1PreconScale"); }

        // Apply the radial tridiagonal + lambda preconditioners to the
        // (decomposed) spectral forces.
        preconApply(d_f_spec, p, pw, fp.basis.d_xm, fp.basis.d_xn);

#ifdef DUMP_CUMES_VERIFY
        if (iter == 0 || iter2 == 51 || (iter2 >= kDumpIter && iter2 <= kDumpIter + 2) ||
            (iter2 >= 2 && iter2 <= 4)) {
            size_t n_fspec = (size_t)6 * (size_t)p.mnmax * (size_t)p.ns;
            size_t n_spec = (size_t)p.mnmax * (size_t)p.ns;
            char fn[128];
            snprintf(fn, sizeof fn, "dump/cuMES/step_I_f_spec_iter_%d.bin",
                     iter == 0 ? 1 : iter2);
            dumpDeviceArray(fn, d_f_spec, n_fspec);
            // State + velocities at the handoff window (pre-descent of the
            // pass, matching vmecpp's dump phase at vmec.cc). Also at the
            // iter-2..4 window (first lambda != 0 passes) for the state check.
            if (iter2 >= kDumpIter || iter2 == 51 || (iter2 >= 2 && iter2 <= 4)) {
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
                snprintf(fn, sizeof fn, "dump/cuMES/state_lmncs_iter_%d.bin", iter2);
                dumpDeviceArray(fn, st.d_lmncs, n_spec);
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
                snprintf(fn, sizeof fn, "dump/cuMES/vel_vlmncs_iter_%d.bin", iter2);
                dumpDeviceArray(fn, st.d_v_lmncs, n_spec);
            }
        }
        if (iter2 >= kE2Start && iter2 < kE2Start + 40) {
            char fn[128];
            snprintf(fn, sizeof fn, "dump/cuMES/fspec_precon_iter_%d.bin", iter2);
            dumpDeviceArray(fn, d_f_spec, (size_t)6 * p.mnmax * p.ns);
        }
#endif

        // ---- Preconditioned residuals (vmecpp fsqr1/fsqz1/fsql1) ----
        { dim3 b3(256),g3(3); computeResidualsKernel<<<g3,b3>>>(d_f_spec,p.ns,p.mnmax,d_sq); }
        double h_sq[3]; checkCuda(cudaMemcpy(h_sq,d_sq,3*sizeof(double),cudaMemcpyDeviceToHost),"cpy sq");
        // vmecpp evalFResPrecd: fsqr1 = fResPrecd[0]·fNorm1 (same for fsqz1),
        // fsql1 = fResPrecd[2]·deltaS — NOTE: deltaS, not fNormL.
        double fsqr = h_sq[0] * plainPerEl * fNorm1;
        double fsqz = h_sq[1] * plainPerEl * fNorm1;
        double fsql = h_sq[2] * plainPerEl * rp.delta_s;
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
        bool doRefresh = false;  // refresh the backup AFTER the descent
        if (fsq <= res0 && (iter2 - iter1) > 10) {
            doRefresh = true;  // consistent progress: refresh rollback target
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

        // ---- Descent step (Garabedian second-order Richardson) ----------
        // Runs BEFORE the time-step control, matching vmecpp's pass order:
        // Evolve() descends, then SolveEquilibriumLoop's control block
        // refreshes the backup or restores. The backup refresh must capture
        // the POST-descent state (vmecpp's RestartIteration(NO_RESTART) runs
        // after Evolve): a pre-descent backup restores the state one descent
        // step earlier at every restart, which offsets the weakly-determined
        // lambda gauge modes by ~1e-2 (one step of their ~1e-2/pass drift) at
        // the pass-56/57 BAD_PROGRESS restore and splits the trajectory
        // (2791 vs 2953 iters; converged lambda gauge modes 1.4e-2 off).
        { dim3 bd(256), gd((p.ns*p.mnmax+255)/256);
          descentStepKernel<<<gd,bd>>>(
              st.d_rmncc,st.d_rmnss,st.d_zmnsc,st.d_zmncs,st.d_lmnsc,st.d_lmncs,
              st.d_v_rmncc,st.d_v_rmnss,st.d_v_zmnsc,st.d_v_zmncs,st.d_v_lmnsc,st.d_v_lmncs,
              d_f_spec, fp.basis.d_xm, fp.basis.d_xn,
              p.ns,p.mnmax,delt,b1,fac);
          checkCuda(cudaGetLastError(),"descent"); }

        if (doRefresh) {
            backupState();  // POST-descent state (vmecpp RestartIteration
                            // NO_RESTART semantics — see comment above)
        }

        if (reason != kNoRestart) {
            // Restore overwrites the just-descended state and zeroes the
            // velocities (vmecpp does the same: Evolve()'s descent is
            // discarded by the control block's RestartIteration).
            restoreState();
            if (reason == kBadJacobian) { delt *= 0.9; ++ijacob; }
            else { delt /= 1.03; }
            iter1 = iter2;
            printf("  -> %s (iter2=%d) delt=%.3e\n",
                   reason == kBadJacobian ? "BAD JACOBIAN" : "BAD PROGRESS",
                   iter2, delt);
        } else {
            iter2++;  // effective counter advances on good passes only
        }

        // ---- Output (every 50 iters, first 5, or late iterations) ----
        if(iter%50==0||iter<5||(iter>1950&&iter%10==0)) {
            printf("%5d | %11.3e %11.3e %11.3e | %8.2e",iter2,fsqr_i,fsqz_i,fsql_i,delt);
            // Axis R at zeta=0 (vmecpp r00 convention) and R_00 at the LCFS
            double h_rmncc_axis = axisRAtZeta0(), h_rmncc_bnd;
            checkCuda(cudaMemcpy(&h_rmncc_bnd, st.d_rmncc + (p.ns-1), sizeof(double), cudaMemcpyDeviceToHost), "cpy Rbnd");
            printf(" | Rax=%.4f Rbnd=%.4f\n", h_rmncc_axis, h_rmncc_bnd);
        }

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
    checkCuda(cudaFree(d_psum),"free psum");
    checkCuda(cudaFree(d_rzsum),"free rzsum");
    checkCuda(cudaFree(d_bk_rmncc),"free bk cc");
    checkCuda(cudaFree(d_bk_zmnsc),"free bk zsc");
    checkCuda(cudaFree(d_bk_lmnsc),"free bk lsc");
    checkCuda(cudaFree(d_bk_rmnss),"free bk ss");
    checkCuda(cudaFree(d_bk_zmncs),"free bk zcs");
    checkCuda(cudaFree(d_bk_lmncs),"free bk lcs");
    preconFree(pw); constraintFree(cw);
    printf("transform timing: inverseDFT total %.1f ms (%.3f ms/iter), "
           "forwardDFT total %.1f ms (%.3f ms/iter)\n",
           t_inv_ms, t_inv_ms / (res.iterations > 0 ? res.iterations : 1),
           t_fwd_ms, t_fwd_ms / (res.iterations > 0 ? res.iterations : 1));
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    return res;
}
