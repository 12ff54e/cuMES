// constraint_impl.cuh — template definitions for constraint.cuh.
// Included once per scalar type by constraint_double.cu / constraint_float.cu; see the
// explicit-instantiation split (cumes_cuda_double / cumes_cuda_float).
#pragma once
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
//
// All computation is templated on the scalar type T (double or float); the
// cuFFT plans / exec calls dispatch through FftTraits<T>.

#include "constraint.cuh"
#include <cstdio>
#include <cmath>

#include "cumes/transforms/axisymmetric_operator.hpp"


// Dynamic shared-memory base accessor. Each block reserves one extern __shared__
// region per kernel launch; the consuming kernels reinterpret that base as T*.
// NOTE: the explicit double/float instantiation split (one scalar type per TU)
// removes the ORIGINAL reason for this indirection — nvcc rejecting a direct
// `extern __shared__ T[]` in a template instantiated with two scalar types in
// one TU. It is nevertheless RETAINED here: switching to the direct form
// changes -use_fast_math FMA fusion in the consumers (opaque function return
// vs. known shared-array aliasing) and perturbs the trajectory at ~1e-10 — a
// Class B change, not the Class A bitwise-equivalence the build/library split
// must preserve. Removal is deferred to a Class B phase (re-frozen baseline).
namespace {
__device__ void* dynSharedBase() {
    extern __shared__ unsigned char smem_base[];
    return smem_base;
}
}

#include "cumes/runtime/cuda_status.hpp"
#include "cumes/runtime/device_arena.cuh"

// ζ-tile width for the accumulate/synthesize/analyze kernels (mirrors the
// helper in fourier.cu): the block no longer embeds the full theta/2 × zeta
// product, so block sizes stay bounded for larger angular grids.
static int computeKTile(int blkX, int nzeta) {
    int kt = (blkX >= 1024) ? 1 : (16 < 1024 / blkX ? 16 : 1024 / blkX);
    return kt < nzeta ? kt : nzeta;
}

// ---------------------------------------------------------------------------
// Allocate/free
// ---------------------------------------------------------------------------
template <typename T>
ConstraintWorkspace<T> constraintCreate(const GridParams<T>& p,
                                        cumes::DeviceArena* arena) {
    using Complex = typename FftTraits<T>::Complex;
    ConstraintWorkspace<T> cw{};
    size_t nF = p.ns * p.nZnT * sizeof(T);
    size_t nFull = p.ns * p.nZnT;

    auto alloc = [&](T*& dst, size_t count, const char* name) {
        if (arena) dst = arena->alloc_span<T>(name, count);
        else cumes::check_cuda(cudaMalloc(&dst, count * sizeof(T)), name);
    };
    alloc(cw.d_gConEff, nFull, "constraint/gConEff");
    alloc(cw.d_gCon,    nFull, "constraint/gCon");
    alloc(cw.d_rCon,    nFull, "constraint/rCon");
    alloc(cw.d_zCon,    nFull, "constraint/zCon");
    alloc(cw.d_rCon0,   nFull, "constraint/rCon0");
    alloc(cw.d_zCon0,   nFull, "constraint/zCon0");
    // Initialize rCon0/zCon0 to zero
    cumes::check_cuda(cudaMemset(cw.d_rCon0, 0, nF), "rCon0 zero");
    cumes::check_cuda(cudaMemset(cw.d_zCon0, 0, nF), "zCon0 zero");

    // tcon profile (device). Zero-init: deAliasKernelFast reads tcon on
    // iteration 0 before computeTconKernel writes it — with zeros the
    // constraint force is inactive on the first iteration (deterministic,
    // matches vmecpp where the constraint has no prior tcon either).
    alloc(cw.d_tcon, p.ns, "constraint/tcon");
    cumes::check_cuda(cudaMemset(cw.d_tcon, 0, p.ns * sizeof(T)), "tcon zero");
    // faccon: host (pinned, never arena) + device
    cumes::check_cuda(cudaMallocHost(&cw.h_faccon, p.mnmax * sizeof(T)), "faccon host");
    alloc(cw.d_faccon, p.mnmax, "constraint/faccon");
    // Precompute faccon[m] = -0.25 * signJ / (xmpq[m+1]^2) with
    // xmpq[m+1] = (m+1)*m, matching vmecpp (ideal_mhd_model.cc lines
    // 238-242): faccon[i] = 0.25 / (i^2 (i+1)^2) for i >= 1.
    for (int m = 0; m < p.mnmax; ++m) {
        T xmpq = T((m + 1) * m);
        cw.h_faccon[m] = (m > 0) ? (T(0.25) / (xmpq * xmpq)) : T(0.0);
    }
    cumes::check_cuda(cudaMemcpy(cw.d_faccon, cw.h_faccon, p.mnmax * sizeof(T), cudaMemcpyHostToDevice), "faccon copy");

    // Constraint-force outputs (frcon/fzcon), zero-initialized so the axis
    // surface (skipped by the add kernel) reads zero like vmecpp.
    size_t nFc = p.ns * p.nZnT * sizeof(T);
    alloc(cw.d_frcon_e, nFull, "constraint/frcon_e");
    alloc(cw.d_frcon_o, nFull, "constraint/frcon_o");
    alloc(cw.d_fzcon_e, nFull, "constraint/fzcon_e");
    alloc(cw.d_fzcon_o, nFull, "constraint/fzcon_o");
    cumes::check_cuda(cudaMemset(cw.d_frcon_e, 0, nFc), "frcon_e zero");
    cumes::check_cuda(cudaMemset(cw.d_frcon_o, 0, nFc), "frcon_o zero");
    cumes::check_cuda(cudaMemset(cw.d_fzcon_e, 0, nFc), "fzcon_e zero");
    cumes::check_cuda(cudaMemset(cw.d_fzcon_o, 0, nFc), "fzcon_o zero");

    // Compact deAlias bandpass buffers + plans (2 slots x (mpol-2) x (ns-1)
    // batch elements instead of the full 12*mpol*ns).
    int n = p.nzeta, nz2 = p.nzeta / 2 + 1;
    int batchDa = 2 * (p.mpol - 2) * (p.ns - 1);
    if (arena) {
        // cuFFT needs 16-byte-aligned data (see fourierCreate).
        cw.d_zeta_real_c = arena->alloc_span<T>("constraint/zeta_real_c", (size_t)batchDa * n, 16);
        cw.d_zeta_spectra_c = arena->alloc_span<Complex>("constraint/zeta_spectra_c", (size_t)batchDa * nz2, 16);
    } else {
        cumes::check_cuda(cudaMalloc(&cw.d_zeta_real_c, (size_t)batchDa * n * sizeof(T)), "zeta_real_c");
        cumes::check_cuda(cudaMalloc(&cw.d_zeta_spectra_c, (size_t)batchDa * nz2 * sizeof(Complex)), "zeta_spectra_c");
    }
    cumes::check_cufft(cufftPlanMany(&cw.plan_d2z_da, 1, &n, &n, 1, n, &nz2, 1, nz2,
                      FftTraits<T>::kForward, batchDa), "plan d2z_da");
    cumes::check_cufft(cufftPlanMany(&cw.plan_z2d_da, 1, &n, &nz2, 1, nz2, &n, 1, n,
                      FftTraits<T>::kInverse, batchDa), "plan z2d_da");

    // Phase 6B: disable cuFFT auto-allocation and share one max-sized work
    // area across the two constraint plans (sequential on one stream, so
    // their lifetimes never overlap). Replaces cuFFT's two per-plan auto
    // allocations (~0.5-1.4 MB each for the W7-X shape) with one buffer.
    {
        size_t wda = 0, wza = 0;
        cumes::check_cufft(cufftSetAutoAllocation(cw.plan_d2z_da, 0), "cufft noauto d2z_da");
        cumes::check_cufft(cufftSetAutoAllocation(cw.plan_z2d_da, 0), "cufft noauto z2d_da");
        cumes::check_cufft(cufftGetSize(cw.plan_d2z_da, &wda), "cufftGetSize d2z_da");
        cumes::check_cufft(cufftGetSize(cw.plan_z2d_da, &wza), "cufftGetSize z2d_da");
        cw.cufft_work_bytes = (wda > wza) ? wda : wza;
        if (cw.cufft_work_bytes > 0) {
            cumes::check_cuda(cudaMalloc(&cw.d_cufft_work, cw.cufft_work_bytes),
                              "cufft work c");
            cumes::check_cufft(cufftSetWorkArea(cw.plan_d2z_da, cw.d_cufft_work),
                               "cufft workarea d2z_da");
            cumes::check_cufft(cufftSetWorkArea(cw.plan_z2d_da, cw.d_cufft_work),
                               "cufft workarea z2d_da");
        }
    }

    cw.arena_backed = (arena != nullptr);
    return cw;
}

template <typename T>
void constraintFree(ConstraintWorkspace<T>& cw) {
    if (!cw.arena_backed) {
        cudaFree(cw.d_gConEff); cudaFree(cw.d_gCon);
        cudaFree(cw.d_rCon);    cudaFree(cw.d_zCon);
        cudaFree(cw.d_rCon0);   cudaFree(cw.d_zCon0);
        cudaFree(cw.d_tcon);
        cudaFree(cw.d_faccon);
        cudaFree(cw.d_frcon_e); cudaFree(cw.d_frcon_o);
        cudaFree(cw.d_fzcon_e); cudaFree(cw.d_fzcon_o);
        cudaFree(cw.d_zeta_real_c); cudaFree(cw.d_zeta_spectra_c);
    }
    cudaFreeHost(cw.h_faccon);
    cufftDestroy(cw.plan_d2z_da); cufftDestroy(cw.plan_z2d_da);
    if (cw.d_cufft_work) cudaFree(cw.d_cufft_work);
    cw = ConstraintWorkspace<T>{};
}

// ---------------------------------------------------------------------------
// Step 1: Compute effective constraint force gConEff.
// gConEff = (rCon - rCon0) * ruFull + (zCon - zCon0) * zuFull  (skip axis)
// with the PHYSICAL derivatives ruFull = ru_e + sqrt(s)*ru_o (vmecpp
// combines the parity-split derivatives this way at line 1178). rCon0/zCon0
// are set by constraintResetRzCon0 (vmecpp rzConIntoVolume).
// ---------------------------------------------------------------------------
template <typename T>
__global__ void effectiveConstraintKernel(
    const T* __restrict__ rCon,   const T* __restrict__ zCon,
    const T* __restrict__ ru_e,   const T* __restrict__ ru_o,
    const T* __restrict__ zu_e,   const T* __restrict__ zu_o,
    const T* __restrict__ sqrtS_F,
    const T* __restrict__ rCon0,  const T* __restrict__ zCon0,
    int ns, int nZnT,
    T* __restrict__ gConEff)
{
    int jF = blockIdx.y;
    int k  = threadIdx.x + blockIdx.x * blockDim.x;
    if (jF >= ns || k >= nZnT) return;
    int idx = k + jF * nZnT;

    // Skip magnetic axis (no poloidal angle)
    if (jF == 0) {
        gConEff[idx] = T(0.0);
        return;
    }

    T sF = sqrtS_F[jF];
    T ruFull = ru_e[idx] + sF * ru_o[idx];
    T zuFull = zu_e[idx] + sF * zu_o[idx];

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
template <typename T>
__global__ void rzConIntoVolumeKernel(
    const T* __restrict__ rCon, const T* __restrict__ zCon,
    const T* __restrict__ sqrtS_F,
    int ns, int nZnT,
    T* __restrict__ rCon0, T* __restrict__ zCon0)
{
    int jF = blockIdx.y;
    int k  = threadIdx.x + blockIdx.x * blockDim.x;
    if (jF >= ns || k >= nZnT) return;
    if (jF == 0) return;  // axis: stays zero (no poloidal angle)

    int lcfs = (ns - 1) * nZnT + k;
    T sFull = sqrtS_F[jF] * sqrtS_F[jF];
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

template <typename T>
__global__ void deAliasAnalyzeKernel(
    const T* __restrict__ gConEff,
    const T* __restrict__ cos_th, const T* __restrict__ sin_th,
    int ns, int mpol, int ntheta, int nzeta, int nZnT,
    T* __restrict__ zeta_real,   // compact slots 0 (sc), 1 (cs)
    int kTile)
{
    // 8 threads per (m, jF, k) split the theta sum (4 contiguous points
    // each), reduced by a warp shuffle tree over the 8 lanes (4 k-groups
    // per warp, width 8). The original ran one serial 30-point dot per
    // thread (36 threads/block, latency-bound at ~41 us/iter); the
    // summation order differs at the rounding level.
    // Theta coverage LOOPS over 32-point groups (8 lanes x 4 points), so
    // grids with ntheta > 32 are fully summed instead of silently dropping
    // the tail; the ζ direction is TILED (blockIdx.z selects the k-tile) so
    // the launch block stays bounded for larger angular grids. For the
    // shipped configs (ntheta <= 32) the loop runs exactly once per thread
    // — same arithmetic as the pre-fix kernel.
    int jF = blockIdx.y, m1 = blockIdx.x;   // m = m1 + 1 in [1, mpol-2]
    if (jF == 0) return;
    int t = threadIdx.x;                    // t in [0,8)
    int k = threadIdx.y + blockIdx.z * kTile;
    int m = m1 + 1;
    T s_sc = T(0.0), s_cs = T(0.0);
    if (k < nzeta) {
        const T* g = gConEff + jF * nZnT + k * ntheta;
        const T* sth = sin_th + m * ntheta;
        const T* cth = cos_th + m * ntheta;
        for (int it0 = 4 * t; it0 < ntheta; it0 += 32) {
            int itEnd = it0 + 4;
            if (itEnd > ntheta) itEnd = ntheta;
            for (int it = it0; it < itEnd; ++it) {
                s_sc += g[it] * sth[it];
                s_cs += g[it] * cth[it];
            }
        }
    }
    // Shuffle tree over the 8 theta-split lanes (width 8). All block threads
    // converge here (the k >= nzeta tail contributes zeros, discarded by the
    // store guard), so __activemask names exactly the existing lanes — a
    // 0xffffffff mask would be an invalid contract for partial warps, e.g.
    // the 8-thread blocks of the nzeta=1 Solovev grid.
    unsigned mask = __activemask();
    #pragma unroll
    for (int off = 4; off > 0; off >>= 1) {
        s_sc += __shfl_down_sync(mask, s_sc, off, 8);
        s_cs += __shfl_down_sync(mask, s_cs, off, 8);
    }
    if (t == 0 && k < nzeta) {
        // Compact layout: ((slot*(mpol-2) + m1)*(ns-1) + (jF-1))*nzeta + k
        size_t base = ((size_t)m1 * (ns - 1) + (jF - 1)) * nzeta + k;
        size_t step = (size_t)(mpol - 2) * (ns - 1) * nzeta;
        zeta_real[0 * step + base] = s_sc;
        zeta_real[1 * step + base] = s_cs;
    }
}

template <typename T>
__global__ void deAliasCoeffPackKernel(
    const typename FftTraits<T>::Complex* spectra,   // compact analysis output
                                                      // (slots 0,1) — no
    const T* __restrict__ tcon, const T* __restrict__ faccon,
    int ns, int mpol, int ntor, int nz2, int nZnT,
    typename FftTraits<T>::Complex* out)  // compact slots 4,5 — intentionally
                                          // the SAME buffer as spectra
                                          // (in-place, see below)
{
    using Complex = typename FftTraits<T>::Complex;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int nBand = (mpol - 2) * (ns - 1);
    if (t >= nBand) return;
    int m1 = t / (ns - 1), jF1 = t % (ns - 1);
    int jF = jF1 + 1, m = m1 + 1;
    // scale == 0 (tcon or faccon zero) must still produce a zero synthesis
    // element: the compact Z2D synthesizes every bin, and there is no
    // memset to zero the slots otherwise (the full-batch path relied on
    // d_zeta_real's memset).
    T scale = tcon[jF] * faccon[m];
    // Compact layout: ((slot*(mpol-2) + m1)*(ns-1) + jF1)*nz2 + n
    size_t base = ((size_t)m1 * (ns - 1) + jF1) * nz2;
    size_t step = (size_t)(mpol - 2) * (ns - 1) * nz2;
    const Complex* in = spectra + base;
    Complex* slot = out + base;
    for (int n = 0; n <= ntor; ++n) {
        // Normalization: 4/nZnT for n>0 (sin²(mθ)cos²(nζ) sums to nZnT/4),
        // 2/nZnT for n=0 (sin²(mθ) sums to nZnT/2) — the full-grid equivalent
        // of vmecpp's mscale*nscale*intNorm round trip; the sc/cs projections
        // are kept separate (as in vmecpp's sinmu/cosmu round trip).
        T norm = (n > 0) ? T(4.0) / T(nZnT) : T(2.0) / T(nZnT);
        T coeff_sc = norm * scale * in[0 * step + n].x;      // Re F_sc
        T coeff_cs = norm * scale * (-in[1 * step + n].y);   // -Im F_cs
        T half = (n == 0) ? T(1.0) : T(0.5);
        T shalf = (n == 0) ? T(0.0) : T(0.5);
        // In-place: compact slots 0,1 carry the analysis (sc/cs) and are
        // overwritten with the synthesis coefficients (the full-batch path
        // wrote slots 4,5, which were disjoint from 0,1 there).
        slot[0 * step + n] = Complex{coeff_sc * half, T(0.0)};
        slot[1 * step + n] = Complex{T(0.0), -coeff_cs * shalf};
    }
    // Zero the unused tail bins: the compact Z2D synthesizes every bin, so
    // bins n > ntor must be zero (the full-batch path got this from the
    // d_zeta_real memset; the compact buffers have no such memset).
    for (int n = ntor + 1; n < nz2; ++n) {
        slot[0 * step + n] = Complex{T(0.0), T(0.0)};
        slot[1 * step + n] = Complex{T(0.0), T(0.0)};
    }
}

template <typename T>
__global__ void deAliasSynthesizeKernel(
    const T* __restrict__ zeta_real,   // Z2D output (slots 4,5)
    const T* __restrict__ cos_th, const T* __restrict__ sin_th,
    int ns, int mpol, int ntheta, int nzeta, int nZnT,
    T* __restrict__ gCon, int kTile)
{
    int jF = blockIdx.x;
    if (jF == 0) return;
    // Thread mapping: l1 = threadIdx.x (fastest), k = threadIdx.y — the
    // gCon stores at jF*nZnT + k*ntheta + l then vary l fastest and coalesce.
    // The ζ direction is TILED (blockIdx.y selects the k-tile), so the launch
    // block stays bounded for larger grids; every (k, l) output point is
    // independent, so the per-point arithmetic is unchanged.
    int k0 = blockIdx.y * kTile;
    int k = threadIdx.y + k0;
    int l1 = threadIdx.x;
    int nthreads = blockDim.x * blockDim.y;
    T* sh = static_cast<T*>(dynSharedBase());   // [2][mpol-2][kTile] (compact slots 0,1)
    int nb = 2 * (mpol - 2);
    int jF1 = jF - 1;
    for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < nb * kTile; i += nthreads) {
        int s = i / ((mpol - 2) * kTile), rem = i - s * (mpol - 2) * kTile;
        int m1 = rem / kTile, kk = rem % kTile;
        int kk_abs = k0 + kk;
        sh[i] = (kk_abs < nzeta)
                    ? zeta_real[((size_t)(s * (mpol - 2) + m1) * (ns - 1) + jF1) * nzeta + kk_abs]
                    : T(0.0);  // tail tile: zeros (never used)
    }
    __syncthreads();
    if (k >= nzeta) return;  // tail tile past the grid
    #pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
        int l = l1 + pass * (ntheta / 2);
        T g = T(0.0);
        for (int m1 = 0; m1 < mpol - 2; ++m1) {
            int m = m1 + 1;
            T cosm = cos_th[m * ntheta + l], sinm = sin_th[m * ntheta + l];
            g += sh[0 * (mpol - 2) * kTile + m1 * kTile + (k - k0)] * sinm
               + sh[1 * (mpol - 2) * kTile + m1 * kTile + (k - k0)] * cosm;
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
template <typename T>
__global__ void addConstraintKernel(
    const T* __restrict__ rCon,    const T* __restrict__ zCon,
    const T* __restrict__ rCon0,   const T* __restrict__ zCon0,
    const T* __restrict__ gCon,
    const T* __restrict__ sqrtS_F,
    const T* __restrict__ ru_e, const T* __restrict__ ru_o,
    const T* __restrict__ zu_e, const T* __restrict__ zu_o,
    int ns, int nZnT,
    T* __restrict__ brmn_e, T* __restrict__ brmn_o,
    T* __restrict__ bzmn_e, T* __restrict__ bzmn_o,
    T* __restrict__ frcon_e, T* __restrict__ frcon_o,
    T* __restrict__ fzcon_e, T* __restrict__ fzcon_o)
{
    int jF = blockIdx.y;
    int k  = threadIdx.x + blockIdx.x * blockDim.x;
    if (jF >= ns || k >= nZnT) return;
    int idx = k + jF * nZnT;

    if (jF == 0) return;  // no constraint on axis

    T dr = rCon[idx] - rCon0[idx];
    T dz = zCon[idx] - zCon0[idx];
    T gc = gCon[idx];
    T sF = sqrtS_F[jF];

    T brcon = dr * gc;
    T bzcon = dz * gc;

    brmn_e[idx] += brcon;
    bzmn_e[idx] += bzcon;
    brmn_o[idx] += brcon * sF;
    bzmn_o[idx] += bzcon * sF;

    // Constraint-force outputs (vmecpp frcon/fzcon): the forward DFT adds
    // xmpq[m] * frcon to frcc and xmpq[m] * fzcon to fzsc. The full
    // derivatives are ruFull = ru_e + sqrt(s)*ru_o (matching vmecpp's
    // ruFull in geometryFromFourier), with the odd-parity outputs scaled by
    // sqrt(s) on top.
    T ru_full = ru_e[idx] + sF * ru_o[idx];
    T zu_full = zu_e[idx] + sF * zu_o[idx];
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
template <typename T>
__global__ void computeTconKernel(
    const T* __restrict__ ru_e, const T* __restrict__ ru_o,
    const T* __restrict__ zu_e, const T* __restrict__ zu_o,
    const T* __restrict__ sqrtS_F,
    const T* __restrict__ ard, const T* __restrict__ azd,
    int ns, int nZnT, int ntheta, int nzeta, T delta_s,
    T tcon_multiplier,
    T* __restrict__ tcon)
{
    int jF = blockIdx.x * blockDim.x + threadIdx.x;
    if (jF >= ns || jF == 0) { if (jF == 0) tcon[0] = T(0.0); return; }

    // Surface average of the PHYSICAL derivatives
    // |∇R|² = ruFull², |∇Z|² = zuFull² with ruFull = ru_e + sqrt(s)*ru_o
    // (vmecpp constraintForceMultiplier, using ruFull on the full grid).
    // One thread per surface; serial loop over ALL (zeta, theta-reduced)
    // points -- the layout is [jF][zeta][theta], so the index must combine
    // both. (FIXED 2026-08-02: previously only theta at the first zeta plane
    // was summed, making arN/azN ~36x too small and tcon ~21x too big.)
    T arN = T(0.0), azN = T(0.0);

    const int nThetaEven = 2 * (ntheta / 2);
    const int nThetaRed = nThetaEven / 2 + 1;  // reduced grid [0, pi]
    const T dnorm3 = T(1.0) / T(nzeta * (nThetaRed - 1));
    const T sF = sqrtS_F[jF];

    for (int iz = 0; iz < nzeta; ++iz) {
        for (int it = 0; it < nThetaRed; ++it) {
            T w = dnorm3;
            if (it == 0 || it == nThetaRed - 1) w *= T(0.5);
            int idx = (jF * nzeta + iz) * ntheta + it;
            T ruFull = ru_e[idx] + sF * ru_o[idx];
            T zuFull = zu_e[idx] + sF * zu_o[idx];
            arN += ruFull * ruFull * w;
            azN += zuFull * zuFull * w;
        }
    }
    if (arN == T(0.0)) arN = T(1e-10);
    if (azN == T(0.0)) azN = T(1e-10);

    T ard_even = fabs(ard[jF * 2 + 0]);  // even parity
    T azd_even = fabs(azd[jF * 2 + 0]);
    T tcon_base = fmin(ard_even / arN, azd_even / azN);

    // 32 = 4*4 * 2 factor from vmecpp (cancels scaling in ard/azd)
    tcon[jF] = tcon_base * tcon_multiplier * T(32.0) * delta_s * T(32.0) * delta_s;
}

// vmecpp: tcon at the LCFS is halved ("maybe related to boundary only having
// MHD force contributions from the inside"). One thread.
template <typename T>
__global__ void tconLcfsHalfKernel(T* __restrict__ tcon, int ns) {
    if (ns > 1) tcon[ns - 1] = T(0.5) * tcon[ns - 2];
}

// ---------------------------------------------------------------------------
// Reset rCon0/zCon0 to the LCFS-extrapolated profile (vmecpp rzConIntoVolume).
// Must be called with the current rCon/zCon, on the first iteration and on
// the reinit pass after every restart.
// ---------------------------------------------------------------------------
template <typename T>
void constraintResetRzCon0(const GridParams<T>& p, ConstraintWorkspace<T>& cw,
                           const T* d_sqrtS_F, cudaStream_t stream) {
    dim3 block(128);
    dim3 grid((p.nZnT + 127) / 128, p.ns);
    rzConIntoVolumeKernel<T><<<grid, block, 0, stream>>>(
        cw.d_rCon, cw.d_zCon, d_sqrtS_F,
        p.ns, p.nZnT, cw.d_rCon0, cw.d_zCon0);
    cumes::check_cuda(cudaGetLastError(), "rzConIntoVolume");
}

// ---------------------------------------------------------------------------
// Step 2: Bandpass filter (de-alias) — cuFFT round trip on the compact batch
// (2 slots x (mpol-2) modes x (ns-1) surfaces instead of the full 12*mpol*ns):
// θ-reduce gConEff into the slot-0/1 ζ-signals, D2Z, scale the per-mode
// coefficients into slots 4/5 (the spectra tail bins n>ntor are zeroed by the
// memset — the pack writes only the used bins), Z2D, poloidal synthesis -> gCon.
// Extracted so the bandpass is testable in isolation (the axisymmetric backend
// replaces exactly this step).
// ---------------------------------------------------------------------------
template <typename T>
void constraintDealiasBandpass(const GridParams<T>& p, const FourierPlan<T>& fp,
                               ConstraintWorkspace<T>& cw, cudaStream_t stream) {
    {   int kTileA = computeKTile(8, p.nzeta);
        int nKTilesA = (p.nzeta + kTileA - 1) / kTileA;
        dim3 blkA(8, kTileA), grdA(p.mpol - 2, p.ns, nKTilesA);
        deAliasAnalyzeKernel<T><<<grdA, blkA, 0, stream>>>(
            cw.d_gConEff, fp.d_cos_th, fp.d_sin_th,
            p.ns, p.mpol, p.ntheta, p.nzeta, p.nZnT, cw.d_zeta_real_c, kTileA);
        cumes::check_cuda(cudaGetLastError(), "deAlias analyze");
    }
    cumes::check_cufft(FftTraits<T>::execForward(cw.plan_d2z_da, cw.d_zeta_real_c, cw.d_zeta_spectra_c), "deAlias d2z");
    {   int nBand = (p.mpol - 2) * (p.ns - 1);
        deAliasCoeffPackKernel<T><<<(nBand + 255) / 256, 256, 0, stream>>>(
            cw.d_zeta_spectra_c, cw.d_tcon, cw.d_faccon,
            p.ns, p.mpol, p.ntor, p.nzeta / 2 + 1, p.nZnT,
            cw.d_zeta_spectra_c);
        cumes::check_cuda(cudaGetLastError(), "deAlias coeff");
    }
    cumes::check_cufft(FftTraits<T>::execInverse(cw.plan_z2d_da, cw.d_zeta_spectra_c, cw.d_zeta_real_c), "deAlias z2d");
    {   int kTileS = computeKTile(p.ntheta / 2, p.nzeta);
        int nKTilesS = (p.nzeta + kTileS - 1) / kTileS;
        dim3 blkS(p.ntheta / 2, kTileS);
        dim3 grdS(p.ns, nKTilesS);
        deAliasSynthesizeKernel<T><<<grdS, blkS,
            2 * (p.mpol - 2) * kTileS * sizeof(T), stream>>>(
            cw.d_zeta_real_c, fp.d_cos_th, fp.d_sin_th,
            p.ns, p.mpol, p.ntheta, p.nzeta, p.nZnT, cw.d_gCon, kTileS);
        cumes::check_cuda(cudaGetLastError(), "deAlias synth");
    }
}

// ---------------------------------------------------------------------------
// Host orchestration
// ---------------------------------------------------------------------------
// Steps 0 (tcon refresh) + 1 (effective constraint force gConEff) — shared by
// the generic and axisymmetric backends; only the step-2 bandpass differs.
template <typename T>
static void constraintComputeHead(const GridParams<T>& p, const FourierPlan<T>& fp,
                                  const PreconWorkspace<T>& pw, ConstraintWorkspace<T>& cw,
                                  const T* d_sqrtS_F, bool precon_updated,
                                  cudaStream_t stream) {
    dim3 block(128);
    dim3 grid((p.nZnT + 127) / 128, p.ns);

    // Step 0: refresh tcon from the current preconditioner elements.
    // vmecpp recomputes tcon only when the radial preconditioner is updated
    // (constraintForceMultiplier inside the shouldUpdateRadialPreconditioner
    // branch), using the current iteration's geometry — so it is applied in
    // the same iteration it is computed.
    if (precon_updated) {
        // Pure constant: compute in double, convert once to T. The vmecpp
        // indata tcon0 scales the whole profile (the parsed value reaches
        // the kernel through GridParams::tcon0; default 1.0 keeps the
        // shipped configs unchanged).
        T tcon_multiplier = p.tcon0 * T(1.0 * (1.0 + p.ns * (1.0/60.0 + p.ns/(200.0*120.0))) / 16.0);
        int gridF = (p.ns + 255) / 256;
        computeTconKernel<T><<<gridF, 256, 0, stream>>>(
            fp.d_ru_e, fp.d_ru_o, fp.d_zu_e, fp.d_zu_o,
            d_sqrtS_F,
            pw.d_ard, pw.d_azd,
            p.ns, p.nZnT, p.ntheta, p.nzeta, T(1.0)/T(p.ns-1.0), tcon_multiplier,
            cw.d_tcon);
        cumes::check_cuda(cudaGetLastError(), "tcon");
        tconLcfsHalfKernel<T><<<1, 1, 0, stream>>>(cw.d_tcon, p.ns);
        cumes::check_cuda(cudaGetLastError(), "tcon lcfs");
    }

    // Step 1: Effective constraint force
    effectiveConstraintKernel<T><<<grid, block, 0, stream>>>(
        cw.d_rCon, cw.d_zCon,
        fp.d_ru_e, fp.d_ru_o, fp.d_zu_e, fp.d_zu_o,
        d_sqrtS_F,
        cw.d_rCon0, cw.d_zCon0,
        p.ns, p.nZnT, cw.d_gConEff);
    cumes::check_cuda(cudaGetLastError(), "gConEff");
}

// Step 3: Add constraint force to brmn/bzmn + write frcon/fzcon outputs.
template <typename T>
static void constraintComputeTail(const GridParams<T>& p, const FourierPlan<T>& fp,
                                  ConstraintWorkspace<T>& cw, const T* d_sqrtS_F,
                                  cudaStream_t stream) {
    dim3 block(128);
    dim3 grid((p.nZnT + 127) / 128, p.ns);
    addConstraintKernel<T><<<grid, block, 0, stream>>>(
        cw.d_rCon, cw.d_zCon, cw.d_rCon0, cw.d_zCon0,
        cw.d_gCon, d_sqrtS_F,
        fp.d_ru_e, fp.d_ru_o, fp.d_zu_e, fp.d_zu_o,
        p.ns, p.nZnT,
        fp.d_brmn_e, fp.d_brmn_o, fp.d_bzmn_e, fp.d_bzmn_o,
        cw.d_frcon_e, cw.d_frcon_o, cw.d_fzcon_e, cw.d_fzcon_o);
    cumes::check_cuda(cudaGetLastError(), "addConstraint");
}

template <typename T>
void constraintCompute(const GridParams<T>& p, const FourierPlan<T>& fp,
                       const PreconWorkspace<T>& pw, ConstraintWorkspace<T>& cw,
                       const T* d_sqrtS_F, bool precon_updated,
                       cudaStream_t stream) {
    constraintComputeHead(p, fp, pw, cw, d_sqrtS_F, precon_updated, stream);

    // Step 2: Bandpass filter (de-alias) — cuFFT round trip on the compact
    // batch (2 slots x (mpol-2) modes x (ns-1) surfaces instead of the full
    // 12*mpol*ns): θ-reduce gConEff into the slot-0/1 ζ-signals, D2Z, scale
    // the per-mode coefficients into slots 4/5 (the spectra tail bins n>ntor
    // are zeroed by the memset — the pack writes only the used bins), Z2D,
    // poloidal synthesis -> gCon.
    constraintDealiasBandpass(p, fp, cw, stream);

    constraintComputeTail(p, fp, cw, d_sqrtS_F, stream);
}

template <typename T>
void constraintComputeAxisym(const GridParams<T>& p, const FourierPlan<T>& fp,
                             const PreconWorkspace<T>& pw, ConstraintWorkspace<T>& cw,
                             const T* d_sqrtS_F, bool precon_updated,
                             cumes::AxisymmetricOperator<T>& op,
                             cudaStream_t stream) {
    constraintComputeHead(p, fp, pw, cw, d_sqrtS_F, precon_updated, stream);

    // Step 2: direct-poloidal de-alias (ntor=0/nzeta=1). Replaces the compact
    // cuFFT round trip above; Class B ULP-equivalent (test_axisym_backend).
    op.enqueue_dealias(
        cumes::RealFieldView<const T>(cw.d_gConEff, p.ns, p.ntheta, p.nzeta),
        cw.d_tcon, cw.d_faccon,
        cumes::RealFieldView<T>(cw.d_gCon, p.ns, p.ntheta, p.nzeta), stream);

    constraintComputeTail(p, fp, cw, d_sqrtS_F, stream);
}

