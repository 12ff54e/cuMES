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
static void ccf(cufftResult r, const char* t) {
    if (r != CUFFT_SUCCESS) { fprintf(stderr, "CUFFT[%s]: %d\n", t, (int)r); exit(1); }
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

    // Compact deAlias bandpass buffers + plans (2 slots x (mpol-2) x (ns-1)
    // batch elements instead of the full 12*mpol*ns).
    int n = p.nzeta, nz2 = p.nzeta / 2 + 1;
    int batchDa = 2 * (p.mpol - 2) * (p.ns - 1);
    cc(cudaMalloc(&cw.d_zeta_real_c, (size_t)batchDa * n * sizeof(double)), "zeta_real_c");
    cc(cudaMalloc(&cw.d_zeta_spectra_c, (size_t)batchDa * nz2 * sizeof(double2)), "zeta_spectra_c");
    ccf(cufftPlanMany(&cw.plan_d2z_da, 1, &n, &n, 1, n, &nz2, 1, nz2,
                      CUFFT_D2Z, batchDa), "plan d2z_da");
    ccf(cufftPlanMany(&cw.plan_z2d_da, 1, &n, &nz2, 1, nz2, &n, 1, n,
                      CUFFT_Z2D, batchDa), "plan z2d_da");

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
    cudaFree(cw.d_zeta_real_c); cudaFree(cw.d_zeta_spectra_c);
    cufftDestroy(cw.plan_d2z_da); cufftDestroy(cw.plan_z2d_da);
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
// rCon/zCon in the 3D folded product basis (vmecpp's rCon/zCon in
// dft_FourierToReal_3d_symm): the xmpq-weighted real-space combination
//   rCon = Σ xmpq[m]*sqrt(s)^{odd}*(rmncc*cos(mθ)cos(nζ) + rmnss*sin(mθ)sin(nζ))
//   zCon = Σ xmpq[m]*sqrt(s)^{odd}*(zmnsc*sin(mθ)cos(nζ) + zmncs*cos(mθ)sin(nζ))
// with xmpq[m] = m*(m-1): the m=0 and m=1 contributions vanish (the m=1
// constraint), so rCon measures the deviation of the m>=2 content from the
// LCFS-extrapolated profile rCon0.
// rCon/zCon via the cuFFT machinery: the xmpq-weighted real-space combination
// used by the spectral-condensation constraint.
//   rCon = Σ xmpq[m]*(rmncc*cos(mθ)cos(nζ) + rmnss*sin(mθ)sin(nζ))
//   zCon = Σ xmpq[m]*(zmnsc*sin(mθ)cos(nζ) + zmncs*cos(mθ)sin(nζ))
// with xmpq[m] = m*(m-1): the m=0,1 contributions vanish (the m=1
// constraint), so rCon measures the deviation of the m>=2 content from the
// LCFS-extrapolated profile rCon0.
// Odd-m: NO extra sqrt(s) factor. vmecpp's con_factor = xmpq*sqrtSF applies
// to its PHYSICAL state; cuMES's state carries the 1/scalxc decomposition
// for odd m and sqrtSF*scalxc = 1, so the factor is exactly 1.0.
// (FIXED 2026-08-02: the old sqrtS_F factor made odd-m rCon too small by
// 1/scalxc, ~sqrt(s).)
// The xmpq weighting is folded into the pack (a per-mode scalar that
// commutes with the transform); only the value slots 0/1/4/5 are used.
__global__ void rzConPackKernel(
    const double* __restrict__ rmncc, const double* __restrict__ rmnss,
    const double* __restrict__ zmnsc, const double* __restrict__ zmncs,
    const int* __restrict__ xm, const int* __restrict__ xn,
    int ns, int mpol, int ntor, int nfp, int nz2,
    double2* __restrict__ spectra)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= ns * mpol * (ntor + 1)) return;
    int j = t % ns, mode = t / ns;
    int m = xm[mode], n = xn[mode];
    double xmpq = (double)m * (m - 1);
    if (xmpq == 0.0) return;   // m=0,1 vanish; the rest stays zero (memset)
    double half  = (n == 0) ? 1.0 : 0.5;
    double shalf = (n == 0) ? 0.0 : 0.5;
    double rc = xmpq * rmncc[j + mode * ns], rs = xmpq * rmnss[j + mode * ns];
    double zs = xmpq * zmnsc[j + mode * ns], zc = xmpq * zmncs[j + mode * ns];
    size_t step = (size_t)mpol * ns * nz2;
    double2* slot = spectra + ((size_t)m * ns + j) * nz2 + n;
    slot[0 * step] = make_double2(rc * half, 0.0);
    slot[1 * step] = make_double2(0.0, -rs * shalf);
    slot[4 * step] = make_double2(zs * half, 0.0);
    slot[5 * step] = make_double2(0.0, -zc * shalf);
}

// rCon/zCon poloidal accumulation: the plain reconstruction of the value
// slots over all m (no parity split, no maxsc — rCon/zCon are full
// real-space fields, unlike the e/o-split inverse-DFT outputs).
__global__ void rzConAccumulateKernel(
    const double* __restrict__ zeta_real,
    const double* __restrict__ cos_th, const double* __restrict__ sin_th,
    int ns, int mpol, int ntheta, int nzeta, int nZnT,
    double* __restrict__ rCon, double* __restrict__ zCon)
{
    int j = blockIdx.x;
    // Thread mapping: l1 = threadIdx.x (fastest), k = threadIdx.y — the
    // rCon/zCon stores at idx = j*nZnT + k*ntheta + l then vary l fastest
    // and coalesce.
    int k = threadIdx.y, l1 = threadIdx.x;
    int nthreads = blockDim.x * blockDim.y;
    extern __shared__ double sh[];   // [4][mpol][nzeta]: slots 0,1,4,5
    for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < 4 * mpol * nzeta; i += nthreads) {
        int s = i / (mpol * nzeta), rem = i - s * mpol * nzeta;
        int m = rem / nzeta, kk = rem % nzeta;
        int slot = (s == 2) ? 4 : (s == 3) ? 5 : s;
        sh[i] = zeta_real[(((size_t)slot * mpol + m) * ns + j) * nzeta + kk];
    }
    __syncthreads();
    size_t mstride = (size_t)mpol * nzeta;
    #pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
        int l = l1 + pass * (ntheta / 2);
        double r = 0.0, z = 0.0;
        for (int m = 0; m < mpol; ++m) {
            const double* sm = sh + m * nzeta;
            double cosm = cos_th[m * ntheta + l], sinm = sin_th[m * ntheta + l];
            r += sm[0 * mstride + k] * cosm + sm[1 * mstride + k] * sinm;
            z += sm[2 * mstride + k] * sinm + sm[3 * mstride + k] * cosm;
        }
        int idx = j * nZnT + k * ntheta + l;
        rCon[idx] = r;
        zCon[idx] = z;
    }
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
// Step 2: Bandpass filter gConEff -> gCon via the cuFFT machinery.
// Analysis (full-grid uniform sums, matching deAliasKernelFast's quadrature):
//   w_sc(m,n) = Σ gConEff*sin(mθ)cos(nζ) = Re F_sc(n)
//   w_cs(m,n) = Σ gConEff*cos(mθ)sin(nζ) = -Im F_cs(n)
// where F_sc/F_cs are the 1D-ζ real FFTs of the per-(m,jF) θ-reduced signals
// s_sc[ζ] = Σ_θ gConEff*sin(mθ), s_cs[ζ] = Σ_θ gConEff*cos(mθ) (both θ and ζ
// sums are uniform over the full grid, as in the original kernel).
// Synthesis: slots 4/5 (zmksc/zmkcs) are packed with the normalized
// coefficients (norm = 4/nZnT for n>0, 2/nZnT for n=0; scale = tcon*faccon),
// and the inverse FFT + poloidal sum rebuilds
// gCon = Σ coeff_sc*sin(mθ)cos(nζ) + coeff_cs*cos(mθ)sin(nζ)
// over the bandpass modes m = 1..mpol-2, surfaces jF >= 1.
// ---------------------------------------------------------------------------

__global__ void deAliasAnalyzeKernel(
    const double* __restrict__ gConEff,
    const double* __restrict__ cos_th, const double* __restrict__ sin_th,
    int ns, int mpol, int ntheta, int nzeta, int nZnT,
    double* __restrict__ zeta_real)   // compact slots 0 (sc), 1 (cs)
{
    int jF = blockIdx.y, m1 = blockIdx.x;   // m = m1 + 1 in [1, mpol-2]
    int k = threadIdx.x;
    if (jF == 0 || k >= nzeta) return;
    int m = m1 + 1;
    const double* g = gConEff + jF * nZnT + k * ntheta;
    const double* sth = sin_th + m * ntheta;
    const double* cth = cos_th + m * ntheta;
    double s_sc = 0.0, s_cs = 0.0;
    for (int it = 0; it < ntheta; ++it) {
        s_sc += g[it] * sth[it];
        s_cs += g[it] * cth[it];
    }
    // Compact layout: ((slot*(mpol-2) + m1)*(ns-1) + (jF-1))*nzeta + k
    size_t base = ((size_t)m1 * (ns - 1) + (jF - 1)) * nzeta + k;
    size_t step = (size_t)(mpol - 2) * (ns - 1) * nzeta;
    zeta_real[0 * step + base] = s_sc;
    zeta_real[1 * step + base] = s_cs;
}

__global__ void deAliasCoeffPackKernel(
    const double2* __restrict__ spectra,   // compact analysis output (slots 0,1)
    const double* __restrict__ tcon, const double* __restrict__ faccon,
    int ns, int mpol, int ntor, int nz2, int nZnT,
    double2* __restrict__ out)             // compact slots 4,5 (same buffer, disjoint)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int nBand = (mpol - 2) * (ns - 1);
    if (t >= nBand) return;
    int m1 = t / (ns - 1), jF1 = t % (ns - 1);
    int jF = jF1 + 1, m = m1 + 1;
    // scale == 0 (tcon or faccon zero) must still produce a zero synthesis
    // element: the compact Z2D synthesizes every bin, and there is no
    // memset to zero the slots otherwise (the full-batch path relied on
    // d_zeta_real's memset).
    double scale = tcon[jF] * faccon[m];
    // Compact layout: ((slot*(mpol-2) + m1)*(ns-1) + jF1)*nz2 + n
    size_t base = ((size_t)m1 * (ns - 1) + jF1) * nz2;
    size_t step = (size_t)(mpol - 2) * (ns - 1) * nz2;
    const double2* in = spectra + base;
    double2* slot = out + base;
    for (int n = 0; n <= ntor; ++n) {
        // Normalization: 4/nZnT for n>0 (sin²(mθ)cos²(nζ) sums to nZnT/4),
        // 2/nZnT for n=0 (sin²(mθ) sums to nZnT/2) — the full-grid equivalent
        // of vmecpp's mscale*nscale*intNorm round trip; the sc/cs projections
        // are kept separate (as in vmecpp's sinmu/cosmu round trip).
        double norm = (n > 0) ? 4.0 / nZnT : 2.0 / nZnT;
        double coeff_sc = norm * scale * in[0 * step + n].x;      // Re F_sc
        double coeff_cs = norm * scale * (-in[1 * step + n].y);   // -Im F_cs
        double half = (n == 0) ? 1.0 : 0.5;
        double shalf = (n == 0) ? 0.0 : 0.5;
        // In-place: compact slots 0,1 carry the analysis (sc/cs) and are
        // overwritten with the synthesis coefficients (the full-batch path
        // wrote slots 4,5, which were disjoint from 0,1 there).
        slot[0 * step + n] = make_double2(coeff_sc * half, 0.0);
        slot[1 * step + n] = make_double2(0.0, -coeff_cs * shalf);
    }
    // Zero the unused tail bins: the compact Z2D synthesizes every bin, so
    // bins n > ntor must be zero (the full-batch path got this from the
    // d_zeta_real memset; the compact buffers have no such memset).
    for (int n = ntor + 1; n < nz2; ++n) {
        slot[0 * step + n] = make_double2(0.0, 0.0);
        slot[1 * step + n] = make_double2(0.0, 0.0);
    }
}

__global__ void deAliasSynthesizeKernel(
    const double* __restrict__ zeta_real,   // Z2D output (slots 4,5)
    const double* __restrict__ cos_th, const double* __restrict__ sin_th,
    int ns, int mpol, int ntheta, int nzeta, int nZnT,
    double* __restrict__ gCon)
{
    int jF = blockIdx.x;
    if (jF == 0) return;
    // Thread mapping: l1 = threadIdx.x (fastest), k = threadIdx.y — the
    // gCon stores at jF*nZnT + k*ntheta + l then vary l fastest and coalesce.
    int k = threadIdx.y, l1 = threadIdx.x;
    int nthreads = blockDim.x * blockDim.y;
    extern __shared__ double sh[];   // [2][mpol-2][nzeta] (compact slots 0,1)
    int nb = 2 * (mpol - 2);
    int jF1 = jF - 1;
    for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < nb * nzeta; i += nthreads) {
        int s = i / ((mpol - 2) * nzeta), rem = i - s * (mpol - 2) * nzeta;
        int m1 = rem / nzeta, kk = rem % nzeta;
        sh[i] = zeta_real[((size_t)(s * (mpol - 2) + m1) * (ns - 1) + jF1) * nzeta + kk];
    }
    __syncthreads();
    #pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
        int l = l1 + pass * (ntheta / 2);
        double g = 0.0;
        for (int m1 = 0; m1 < mpol - 2; ++m1) {
            int m = m1 + 1;
            double cosm = cos_th[m * ntheta + l], sinm = sin_th[m * ntheta + l];
            g += sh[0 * (mpol - 2) * nzeta + m1 * nzeta + k] * sinm
               + sh[1 * (mpol - 2) * nzeta + m1 * nzeta + k] * cosm;
        }
        gCon[jF * nZnT + k * ntheta + l] = g;
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
    // (vmecpp constraintForceMultiplier, using ruFull on the full grid).
    // One thread per surface; serial loop over ALL (zeta, theta-reduced)
    // points -- the layout is [jF][zeta][theta], so the index must combine
    // both. (FIXED 2026-08-02: previously only theta at the first zeta plane
    // was summed, making arN/azN ~36x too small and tcon ~21x too big.)
    double arN = 0.0, azN = 0.0;

    const int nThetaEven = 2 * (ntheta / 2);
    const int nThetaRed = nThetaEven / 2 + 1;  // reduced grid [0, pi]
    const double dnorm3 = 1.0 / (nzeta * (nThetaRed - 1));
    const double sF = sqrtS_F[jF];

    for (int iz = 0; iz < nzeta; ++iz) {
        for (int it = 0; it < nThetaRed; ++it) {
            double w = dnorm3;
            if (it == 0 || it == nThetaRed - 1) w *= 0.5;
            int idx = (jF * nzeta + iz) * ntheta + it;
            double ruFull = ru_e[idx] + sF * ru_o[idx];
            double zuFull = zu_e[idx] + sF * zu_o[idx];
            arN += ruFull * ruFull * w;
            azN += zuFull * zuFull * w;
        }
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
    (void)d_sqrtS_F;   // the odd-m factor is exactly 1.0 (see rzConPackKernel)
    // xmpq-weighted inverse transform: pack the value slots, Z2D, accumulate
    // rCon/zCon (the poloidal sum over all m with the raw cos/sin tables).
    cc(cudaMemset(fp.d_zeta_spectra, 0,
        (size_t)12 * p.mpol * p.ns * (p.nzeta / 2 + 1) * sizeof(double2)), "rzcon zero");
    int total = p.ns * p.mnmax;
    rzConPackKernel<<<(total + 255) / 256, 256>>>(
        st.d_rmncc, st.d_rmnss, st.d_zmnsc, st.d_zmncs,
        fp.basis.d_xm, fp.basis.d_xn,
        p.ns, p.mpol, p.ntor, p.nfp, p.nzeta / 2 + 1, fp.d_zeta_spectra);
    cc(cudaGetLastError(), "rzcon pack");
    ccf(cufftExecZ2D(fp.plan_z2d, fp.d_zeta_spectra, fp.d_zeta_real), "rzcon z2d");
    dim3 blk(p.ntheta / 2, p.nzeta);
    rzConAccumulateKernel<<<p.ns, blk, 4 * p.mpol * p.nzeta * sizeof(double)>>>(
        fp.d_zeta_real, fp.d_cos_th, fp.d_sin_th,
        p.ns, p.mpol, p.ntheta, p.nzeta, p.nZnT,
        cw.d_rCon, cw.d_zCon);
    cc(cudaGetLastError(), "rzcon acc");
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

    // Step 1: Effective constraint force
    effectiveConstraintKernel<<<grid, block>>>(
        cw.d_rCon, cw.d_zCon,
        fp.d_ru_e, fp.d_ru_o, fp.d_zu_e, fp.d_zu_o,
        d_sqrtS_F,
        cw.d_rCon0, cw.d_zCon0,
        p.ns, p.nZnT, cw.d_gConEff);
    cc(cudaGetLastError(), "gConEff");

    // Step 2: Bandpass filter (de-alias) — cuFFT round trip on the compact
    // batch (2 slots x (mpol-2) modes x (ns-1) surfaces instead of the full
    // 12*mpol*ns): θ-reduce gConEff into the slot-0/1 ζ-signals, D2Z, scale
    // the per-mode coefficients into slots 4/5 (the spectra tail bins n>ntor
    // are zeroed by the memset — the pack writes only the used bins), Z2D,
    // poloidal synthesis -> gCon.
    {   dim3 blkA(p.nzeta), grdA(p.mpol - 2, p.ns);
        deAliasAnalyzeKernel<<<grdA, blkA>>>(
            cw.d_gConEff, fp.d_cos_th, fp.d_sin_th,
            p.ns, p.mpol, p.ntheta, p.nzeta, p.nZnT, cw.d_zeta_real_c);
        cc(cudaGetLastError(), "deAlias analyze");
    }
    ccf(cufftExecD2Z(cw.plan_d2z_da, cw.d_zeta_real_c, cw.d_zeta_spectra_c), "deAlias d2z");
    {   int nBand = (p.mpol - 2) * (p.ns - 1);
        deAliasCoeffPackKernel<<<(nBand + 255) / 256, 256>>>(
            cw.d_zeta_spectra_c, cw.d_tcon, cw.d_faccon,
            p.ns, p.mpol, p.ntor, p.nzeta / 2 + 1, p.nZnT,
            cw.d_zeta_spectra_c);
        cc(cudaGetLastError(), "deAlias coeff");
    }
    ccf(cufftExecZ2D(cw.plan_z2d_da, cw.d_zeta_spectra_c, cw.d_zeta_real_c), "deAlias z2d");
    {   dim3 blkS(p.ntheta / 2, p.nzeta);
        deAliasSynthesizeKernel<<<p.ns, blkS,
            2 * (p.mpol - 2) * p.nzeta * sizeof(double)>>>(
            cw.d_zeta_real_c, fp.d_cos_th, fp.d_sin_th,
            p.ns, p.mpol, p.ntheta, p.nzeta, p.nZnT, cw.d_gCon);
        cc(cudaGetLastError(), "deAlias synth");
    }

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
