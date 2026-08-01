// constraint.cu — spectral condensation constraint force.
// Reference: vmecpp constraint_force_kernel.h and deAliasConstraintForce().
//
// The constraint force penalizes spectral condensation (unphysical
// accumulation of Fourier energy at high poloidal modes) by adding
// a bandpass-filtered contribution to the poloidal MHD forces (brmn/bzmn).
//
// Pipeline per iteration:
//   1. gConEff = R * dR/dθ + Z * dZ/dθ            (effective constraint force)
//   2. gCon = bandpass(gConEff, m=1..mpol-2)       (de-alias filter)
//   3. brmn += (R - R0) * gCon, bzmn += (Z - Z0) * gCon  (add to forces)
//      with odd-parity scaled by sqrt(s)

#include "constraint.cuh"
#include <cstdio>
#include <cmath>

static void cc(cudaError_t e, const char* t) {
    if (e != cudaSuccess) { fprintf(stderr, "CUDA[%s]: %s\n", t, cudaGetErrorString(e)); exit(1); }
}

// ---------------------------------------------------------------------------
// Allocate/free
// ---------------------------------------------------------------------------
ConstraintWorkspace constraintCreate(const GridParams& p) {
    ConstraintWorkspace cw{};
    size_t nF = p.ns * p.nZnT * sizeof(double);
    cc(cudaMalloc(&cw.d_gConEff, nF), "gConEff");
    cc(cudaMalloc(&cw.d_gCon,    nF), "gCon");
    cc(cudaMalloc(&cw.d_rCon,    nF), "rCon");
    cc(cudaMalloc(&cw.d_zCon,    nF), "zCon");
    cc(cudaMalloc(&cw.d_rCon0,   nF), "rCon0");
    cc(cudaMalloc(&cw.d_zCon0,   nF), "zCon0");
    // Initialize rCon0/zCon0 to zero
    cc(cudaMemset(cw.d_rCon0, 0, nF), "rCon0 zero");
    cc(cudaMemset(cw.d_zCon0, 0, nF), "zCon0 zero");

    // tcon profile: allocate on host and device
    cc(cudaMallocHost(&cw.h_tcon, p.ns * sizeof(double)), "tcon host");
    cc(cudaMalloc(&cw.d_tcon, p.ns * sizeof(double)), "tcon dev");
    // Zero-init: deAliasKernelFast reads tcon on iteration 0 before
    // computeTconKernel writes it — with zeros the constraint force is
    // inactive on the first iteration (deterministic, matches vmecpp where
    // the constraint has no prior tcon either).
    memset(cw.h_tcon, 0, p.ns * sizeof(double));
    cc(cudaMemset(cw.d_tcon, 0, p.ns * sizeof(double)), "tcon zero");
    // faccon: allocate on host and device
    cc(cudaMallocHost(&cw.h_faccon, p.mnmax * sizeof(double)), "faccon host");
    cc(cudaMalloc(&cw.d_faccon, p.mnmax * sizeof(double)), "faccon dev");
    // Precompute faccon[m] = -0.25 * signJ / (xmpq[m+1]^2) with
    // xmpq[m+1] = (m+1)*m, matching vmecpp (ideal_mhd_model.cc lines
    // 238-242): faccon[i] = 0.25 / (i^2 (i+1)^2) for i >= 1.
    for (int m = 0; m < p.mnmax; ++m) {
        double xmpq = (double)((m + 1) * m);
        cw.h_faccon[m] = (m > 0) ? (0.25 / (xmpq * xmpq)) : 0.0;
    }
    cc(cudaMemcpy(cw.d_faccon, cw.h_faccon, p.mnmax * sizeof(double), cudaMemcpyHostToDevice), "faccon copy");

    // Constraint-force outputs (frcon/fzcon), zero-initialized so the axis
    // surface (skipped by the add kernel) reads zero like vmecpp.
    size_t nFc = p.ns * p.nZnT * sizeof(double);
    cc(cudaMalloc(&cw.d_frcon_e, nFc), "frcon_e");
    cc(cudaMalloc(&cw.d_frcon_o, nFc), "frcon_o");
    cc(cudaMalloc(&cw.d_fzcon_e, nFc), "fzcon_e");
    cc(cudaMalloc(&cw.d_fzcon_o, nFc), "fzcon_o");
    cc(cudaMemset(cw.d_frcon_e, 0, nFc), "frcon_e zero");
    cc(cudaMemset(cw.d_frcon_o, 0, nFc), "frcon_o zero");
    cc(cudaMemset(cw.d_fzcon_e, 0, nFc), "fzcon_e zero");
    cc(cudaMemset(cw.d_fzcon_o, 0, nFc), "fzcon_o zero");

    return cw;
}

void constraintFree(ConstraintWorkspace& cw) {
    cudaFree(cw.d_gConEff); cudaFree(cw.d_gCon);
    cudaFree(cw.d_rCon);    cudaFree(cw.d_zCon);
    cudaFree(cw.d_rCon0);   cudaFree(cw.d_zCon0);
    cudaFreeHost(cw.h_tcon); cudaFree(cw.d_tcon);
    cudaFreeHost(cw.h_faccon); cudaFree(cw.d_faccon);
    cudaFree(cw.d_frcon_e); cudaFree(cw.d_frcon_o);
    cudaFree(cw.d_fzcon_e); cudaFree(cw.d_fzcon_o);
}

// ---------------------------------------------------------------------------
// vmecpp dft_FourierToReal_2d_symm's rCon/zCon: the xmpq-weighted real-space
// combination used by the spectral-condensation constraint.
//   rCon = sum_m xmpq[m] * sqrt(s)^{parity} * rmncc[m] * cos(m*theta)
//   zCon = sum_m xmpq[m] * sqrt(s)^{parity} * zmnsc[m] * sin(m*theta)
// with xmpq[m] = m*(m-1): the m=0 and m=1 contributions vanish (the m=1
// constraint), so rCon measures the deviation of the m>=2 content from the
// LCFS-extrapolated profile rCon0.
// ---------------------------------------------------------------------------
__global__ void rzConComputeKernel(
    const double* __restrict__ rmncc, const double* __restrict__ zmnsc,
    const double* __restrict__ cc, const double* __restrict__ sc,
    const double* __restrict__ sqrtS_F,
    const int* __restrict__ xm,
    int ns, int mnmax, int nZnT,
    double* __restrict__ rCon, double* __restrict__ zCon)
{
    int jF = blockIdx.y;
    int k  = threadIdx.x + blockIdx.x * blockDim.x;
    if (jF >= ns || k >= nZnT) return;

    double r = 0.0, z = 0.0;
    for (int m = 0; m < mnmax; ++m) {
        int mm = xm[m];
        double xmpq = (double)(mm * (mm - 1));
        if (xmpq == 0.0) continue;  // m=0,1: nothing (m=1 constraint)
        double scal = (mm % 2 == 0) ? 1.0 : sqrtS_F[jF];
        int idx_j = jF + m * ns;
        r += xmpq * scal * rmncc[idx_j] * cc[k + m * nZnT];
        z += xmpq * scal * zmnsc[idx_j] * sc[k + m * nZnT];
    }
    int idx = k + jF * nZnT;
    rCon[idx] = r;
    zCon[idx] = z;
}

// ---------------------------------------------------------------------------
// Step 1: Compute effective constraint force gConEff.
// gConEff = (rCon - rCon0) * ruFull + (zCon - zCon0) * zuFull  (skip axis)
// with the PHYSICAL derivatives ruFull = ru_e + sqrt(s)*ru_o (vmecpp
// combines the parity-split derivatives this way at line 1178). rCon0/zCon0
// are set by constraintResetRzCon0 (vmecpp rzConIntoVolume).
// ---------------------------------------------------------------------------
__global__ void effectiveConstraintKernel(
    const double* __restrict__ rCon,   const double* __restrict__ zCon,
    const double* __restrict__ ru_e,   const double* __restrict__ ru_o,
    const double* __restrict__ zu_e,   const double* __restrict__ zu_o,
    const double* __restrict__ sqrtS_F,
    const double* __restrict__ rCon0,  const double* __restrict__ zCon0,
    int ns, int nZnT,
    double* __restrict__ gConEff)
{
    int jF = blockIdx.y;
    int k  = threadIdx.x + blockIdx.x * blockDim.x;
    if (jF >= ns || k >= nZnT) return;
    int idx = k + jF * nZnT;

    // Skip magnetic axis (no poloidal angle)
    if (jF == 0) {
        gConEff[idx] = 0.0;
        return;
    }

    double sF = sqrtS_F[jF];
    double ruFull = ru_e[idx] + sF * ru_o[idx];
    double zuFull = zu_e[idx] + sF * zu_o[idx];

    gConEff[idx] = (rCon[idx] - rCon0[idx]) * ruFull +
                   (zCon[idx] - zCon0[idx]) * zuFull;
}

// ---------------------------------------------------------------------------
// vmecpp rzConIntoVolume: extrapolate the LCFS (r,z) into the volume as
// rCon0 = rCon_LCFS * s. Since rCon0 is subtracted from rCon, this
// effectively disables the constraint while the geometry is close to the
// LCFS-extrapolated profile, and lets it ramp in as the geometry evolves
// away from it. Runs on the first iteration and on the reinit pass after
// every restart (iter2 == iter1, vmecpp's "initialization/soft reset").
// For fixed boundary the vacuum state stays kOff, so rCon0 never decays.
// ---------------------------------------------------------------------------
__global__ void rzConIntoVolumeKernel(
    const double* __restrict__ rCon, const double* __restrict__ zCon,
    const double* __restrict__ sqrtS_F,
    int ns, int nZnT,
    double* __restrict__ rCon0, double* __restrict__ zCon0)
{
    int jF = blockIdx.y;
    int k  = threadIdx.x + blockIdx.x * blockDim.x;
    if (jF >= ns || k >= nZnT) return;
    if (jF == 0) return;  // axis: stays zero (no poloidal angle)

    int lcfs = (ns - 1) * nZnT + k;
    double sFull = sqrtS_F[jF] * sqrtS_F[jF];
    int idx = k + jF * nZnT;
    rCon0[idx] = rCon[lcfs] * sFull;
    zCon0[idx] = zCon[lcfs] * sFull;
}

// ---------------------------------------------------------------------------
// Step 2: Bandpass filter gConEff -> gCon (efficient: one block per mode/surface).
// ---------------------------------------------------------------------------

__global__ void deAliasKernelFast(
    const double* __restrict__ gConEff,
    const double* __restrict__ cc, const double* __restrict__ sc,
    const double* __restrict__ cs,
    const double* __restrict__ cos_mt_nz, const double* __restrict__ sin_mt_nz,
    const int* __restrict__ xm, const int* __restrict__ xn,
    const double* __restrict__ tcon,
    const double* __restrict__ faccon,
    int ns, int mnmax, int mpol, int ntor, int nZnT, int ntheta, int nzeta,
    double* __restrict__ gCon)
{
    // One thread per (mode, surface) for the forward DFT
    // Mode index = blockIdx.x, surface = blockIdx.y
    int mode = blockIdx.x;
    int jF = blockIdx.y;
    if (jF >= ns || jF == 0 || mode >= mnmax) return;

    int mm = xm[mode], nn = xn[mode];
    // Only bandpass modes: m = 1 .. mpol-2
    if (mm < 1 || mm >= mpol - 1) return;

    double faccon_m = faccon[mm];  // 0.25/(m*m)
    double tcon_j = tcon[jF];
    double scale = tcon_j * faccon_m;
    if (scale == 0.0) return;

    int tid = threadIdx.x;

    // Forward DFT projection of gConEff onto basis (mm, nn)
    // Sum over (theta, zeta): gConEff * basis_{mm,nn}
    double w_sc = 0.0;  // sin(mθ)cos(nζ) basis
    double w_cs = 0.0;  // cos(mθ)sin(nζ) basis

    for (int k = tid; k < nZnT; k += blockDim.x) {
        int idx = k + jF * nZnT;
        // sc = sin(mθ)cos(nζ), cs = cos(mθ)sin(nζ)
        w_sc += gConEff[idx] * sc[k + mode * nZnT];
        w_cs += gConEff[idx] * cs[k + mode * nZnT];
    }

    // Parallel reduction in shared memory
    __shared__ double s_wsc[256], s_wcs[256];
    s_wsc[tid] = w_sc; s_wcs[tid] = w_cs;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            s_wsc[tid] += s_wsc[tid + s];
            s_wcs[tid] += s_wcs[tid + s];
        }
        __syncthreads();
    }

    // Thread 0: normalize and write spectral coefficient
    if (tid == 0) {
        // Normalization: for m>0, n>=0: factor 4/nZnT
        // (since sin²(mθ)cos²(nζ) integrates to nZnT/4)
        double norm = (mm > 0 && nn > 0) ? (4.0 / nZnT) :
                      (nn == 0)            ? (2.0 / nZnT) : (4.0 / nZnT);
        double coeff = (s_wsc[0] + s_wcs[0]) * norm * scale;

        // Inverse DFT: add coeff * (sc + cs) back to gCon at each point
        for (int k = 0; k < nZnT; ++k) {
            int idx = k + jF * nZnT;
            // Use atomic add since multiple modes contribute
            atomicAdd(&gCon[idx],
                coeff * (sc[k + mode * nZnT] + cs[k + mode * nZnT]));
        }
    }
}

// ---------------------------------------------------------------------------
// Step 3: Add constraint force to brmn/bzmn forces.
// brmn_e += (rCon - rCon0) * gCon
// bzmn_e += (zCon - zCon0) * gCon
// brmn_o += (rCon - rCon0) * gCon * sqrtSF
// bzmn_o += (zCon - zCon0) * gCon * sqrtSF
// ---------------------------------------------------------------------------
__global__ void addConstraintKernel(
    const double* __restrict__ rCon,    const double* __restrict__ zCon,
    const double* __restrict__ rCon0,   const double* __restrict__ zCon0,
    const double* __restrict__ gCon,
    const double* __restrict__ sqrtS_F,
    const double* __restrict__ ru_e, const double* __restrict__ ru_o,
    const double* __restrict__ zu_e, const double* __restrict__ zu_o,
    int ns, int nZnT,
    double* __restrict__ brmn_e, double* __restrict__ brmn_o,
    double* __restrict__ bzmn_e, double* __restrict__ bzmn_o,
    double* __restrict__ frcon_e, double* __restrict__ frcon_o,
    double* __restrict__ fzcon_e, double* __restrict__ fzcon_o)
{
    int jF = blockIdx.y;
    int k  = threadIdx.x + blockIdx.x * blockDim.x;
    if (jF >= ns || k >= nZnT) return;
    int idx = k + jF * nZnT;

    if (jF == 0) return;  // no constraint on axis

    double dr = rCon[idx] - rCon0[idx];
    double dz = zCon[idx] - zCon0[idx];
    double gc = gCon[idx];
    double sF = sqrtS_F[jF];

    double brcon = dr * gc;
    double bzcon = dz * gc;

    brmn_e[idx] += brcon;
    bzmn_e[idx] += bzcon;
    brmn_o[idx] += brcon * sF;
    bzmn_o[idx] += bzcon * sF;

    // Constraint-force outputs (vmecpp frcon/fzcon): the forward DFT adds
    // xmpq[m] * frcon to frcc and xmpq[m] * fzcon to fzsc. The full
    // derivatives are ruFull = ru_e + sqrt(s)*ru_o (matching vmecpp's
    // ruFull in geometryFromFourier), with the odd-parity outputs scaled by
    // sqrt(s) on top.
    double ru_full = ru_e[idx] + sF * ru_o[idx];
    double zu_full = zu_e[idx] + sF * zu_o[idx];
    frcon_e[idx] = ru_full * gc;
    frcon_o[idx] = ru_full * gc * sF;
    fzcon_e[idx] = zu_full * gc;
    fzcon_o[idx] = zu_full * gc * sF;
}

// ---------------------------------------------------------------------------
// Update tcon profile from preconditioner elements.
// Port of vmecpp's constraintForceMultiplier: the |∇R|²/|∇Z|² surface
// averages use trapezoidal integration over the reduced poloidal grid
// [0, pi] (first ntheta/2+1 points of the full theta grid) with wInt
// weights, matching vmecpp's wInt from sizes.cc.
// ---------------------------------------------------------------------------
__global__ void computeTconKernel(
    const double* __restrict__ ru_e, const double* __restrict__ ru_o,
    const double* __restrict__ zu_e, const double* __restrict__ zu_o,
    const double* __restrict__ sqrtS_F,
    const double* __restrict__ ard, const double* __restrict__ azd,
    int ns, int nZnT, int ntheta, int nzeta, double delta_s,
    double tcon_multiplier,
    double* __restrict__ tcon)
{
    int jF = blockIdx.x * blockDim.x + threadIdx.x;
    if (jF >= ns || jF == 0) { if (jF == 0) tcon[0] = 0.0; return; }

    // Surface average of the PHYSICAL derivatives
    // |∇R|² = ruFull², |∇Z|² = zuFull² with ruFull = ru_e + sqrt(s)*ru_o
    // (vmecpp line 1178). One thread per surface; the reduced-grid sum over
    // the nThetaRed = 10 points is serial per thread (no block reduction —
    // threads of different surfaces must not share accumulators).
    double arN = 0.0, azN = 0.0;

    const int nThetaEven = 2 * (ntheta / 2);
    const int nThetaRed = nThetaEven / 2 + 1;  // reduced grid [0, pi]
    const double dnorm3 = 1.0 / (nzeta * (nThetaRed - 1));
    const double sF = sqrtS_F[jF];

    for (int k = 0; k < nThetaRed; ++k) {
        double w = dnorm3;
        if (k == 0 || k == nThetaRed - 1) w *= 0.5;
        int idx = k + jF * nZnT;
        double ruFull = ru_e[idx] + sF * ru_o[idx];
        double zuFull = zu_e[idx] + sF * zu_o[idx];
        arN += ruFull * ruFull * w;
        azN += zuFull * zuFull * w;
    }
    if (arN == 0.0) arN = 1e-10;
    if (azN == 0.0) azN = 1e-10;

    double ard_even = fabs(ard[jF * 2 + 0]);  // even parity
    double azd_even = fabs(azd[jF * 2 + 0]);
    double tcon_base = fmin(ard_even / arN, azd_even / azN);

    // 32 = 4*4 * 2 factor from vmecpp (cancels scaling in ard/azd)
    tcon[jF] = tcon_base * tcon_multiplier * 32.0 * delta_s * 32.0 * delta_s;
}

// vmecpp: tcon at the LCFS is halved ("maybe related to boundary only having
// MHD force contributions from the inside"). One thread.
__global__ void tconLcfsHalfKernel(double* __restrict__ tcon, int ns) {
    if (ns > 1) tcon[ns - 1] = 0.5 * tcon[ns - 2];
}

// ---------------------------------------------------------------------------
// Compute the xmpq-weighted real-space combination rCon/zCon from the
// current spectral state (vmecpp's rCon/zCon in dft_FourierToReal_2d_symm).
// Call every iteration before the constraint force is assembled.
// ---------------------------------------------------------------------------
void constraintRzConCompute(const GridParams& p, const FourierPlan& fp,
                            const SpectralState& st, ConstraintWorkspace& cw,
                            const double* d_sqrtS_F) {
    dim3 block(32);
    dim3 grid((p.nZnT + 31) / 32, p.ns);
    rzConComputeKernel<<<grid, block>>>(
        st.d_rmncc, st.d_zmnsc,
        fp.basis.d_cc, fp.basis.d_sc,
        d_sqrtS_F, fp.basis.d_xm,
        p.ns, p.mnmax, p.nZnT,
        cw.d_rCon, cw.d_zCon);
    cc(cudaGetLastError(), "rzConCompute");
}

// ---------------------------------------------------------------------------
// Reset rCon0/zCon0 to the LCFS-extrapolated profile (vmecpp rzConIntoVolume).
// Must be called with the current rCon/zCon, on the first iteration and on
// the reinit pass after every restart.
// ---------------------------------------------------------------------------
void constraintResetRzCon0(const GridParams& p, ConstraintWorkspace& cw,
                           const double* d_sqrtS_F) {
    dim3 block(32);
    dim3 grid((p.nZnT + 31) / 32, p.ns);
    rzConIntoVolumeKernel<<<grid, block>>>(
        cw.d_rCon, cw.d_zCon, d_sqrtS_F,
        p.ns, p.nZnT, cw.d_rCon0, cw.d_zCon0);
    cc(cudaGetLastError(), "rzConIntoVolume");
}

// ---------------------------------------------------------------------------
// Host orchestration
// ---------------------------------------------------------------------------
void constraintCompute(const GridParams& p, const FourierPlan& fp,
                       const PreconWorkspace& pw, ConstraintWorkspace& cw,
                       const double* d_sqrtS_F, bool precon_updated) {
    dim3 block(32);
    dim3 grid((p.nZnT + 31) / 32, p.ns);

    // Step 0: refresh tcon from the current preconditioner elements.
    // vmecpp recomputes tcon only when the radial preconditioner is updated
    // (constraintForceMultiplier inside the shouldUpdateRadialPreconditioner
    // branch), using the current iteration's geometry — so it is applied in
    // the same iteration it is computed.
    if (precon_updated) {
        double tcon_multiplier = 1.0 * (1.0 + p.ns * (1.0/60.0 + p.ns/(200.0*120.0))) / 16.0;
        int gridF = (p.ns + 255) / 256;
        computeTconKernel<<<gridF, 256>>>(
            fp.d_ru_e, fp.d_ru_o, fp.d_zu_e, fp.d_zu_o,
            d_sqrtS_F,
            pw.d_ard, pw.d_azd,
            p.ns, p.nZnT, p.ntheta, p.nzeta, 1.0/(p.ns-1.0), tcon_multiplier,
            cw.d_tcon);
        cc(cudaGetLastError(), "tcon");
        tconLcfsHalfKernel<<<1, 1>>>(cw.d_tcon, p.ns);
        cc(cudaGetLastError(), "tcon lcfs");
    }

    // Zero gCon before accumulation
    cc(cudaMemset(cw.d_gCon, 0, p.ns * p.nZnT * sizeof(double)), "zero gCon");

    // Step 1: Effective constraint force
    effectiveConstraintKernel<<<grid, block>>>(
        cw.d_rCon, cw.d_zCon,
        fp.d_ru_e, fp.d_ru_o, fp.d_zu_e, fp.d_zu_o,
        d_sqrtS_F,
        cw.d_rCon0, cw.d_zCon0,
        p.ns, p.nZnT, cw.d_gConEff);
    cc(cudaGetLastError(), "gConEff");

    // Step 2: Bandpass filter (de-alias)
    // One block per mode, one per surface
    dim3 gridDA(p.mnmax, p.ns);
    deAliasKernelFast<<<gridDA, 256>>>(
        cw.d_gConEff,
        fp.basis.d_cc, fp.basis.d_sc, fp.basis.d_cs,
        fp.basis.d_cos_mt_nz, fp.basis.d_sin_mt_nz,
        fp.basis.d_xm, fp.basis.d_xn,
        cw.d_tcon, cw.d_faccon,
        p.ns, p.mnmax, p.mpol, p.ntor, p.nZnT, p.ntheta, p.nzeta,
        cw.d_gCon);
    cc(cudaGetLastError(), "deAlias");
    cc(cudaDeviceSynchronize(), "deAlias sync");

    // Step 3: Add constraint force to brmn/bzmn + write frcon/fzcon outputs
    addConstraintKernel<<<grid, block>>>(
        cw.d_rCon, cw.d_zCon, cw.d_rCon0, cw.d_zCon0,
        cw.d_gCon, d_sqrtS_F,
        fp.d_ru_e, fp.d_ru_o, fp.d_zu_e, fp.d_zu_o,
        p.ns, p.nZnT,
        fp.d_brmn_e, fp.d_brmn_o, fp.d_bzmn_e, fp.d_bzmn_o,
        cw.d_frcon_e, cw.d_frcon_o, cw.d_fzcon_e, cw.d_fzcon_o);
    cc(cudaGetLastError(), "addConstraint");
}
