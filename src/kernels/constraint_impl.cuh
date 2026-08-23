// kernels/constraint_impl.cuh — template definitions for constraint_operator.hpp.
// Included once per scalar type by constraint_double.cu / constraint_float.cu;
// see the explicit-instantiation split (cumes_cuda_double / cumes_cuda_float).
#ifndef CUMES_SRC_CONSTRAINT_IMPL_CUH_
#define CUMES_SRC_CONSTRAINT_IMPL_CUH_
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

#include "cumes/physics/constraint_operator.hpp"
#include "cumes/runtime/cuda_status.hpp"
#include "cumes/runtime/device_arena.cuh"
#include "fft_traits.h"

#include <cmath>
#include <cstdio>

// ---------------------------------------------------------------------------
// Allocate/free
// ---------------------------------------------------------------------------
template <typename T>
cumes::ConstraintOperator<T>::ConstraintOperator(const DeviceParams<T>& p,
                                                 cumes::DeviceArena* arena) {
    using Complex = typename FftTraits<T>::Complex;
    size_t nF = p.ns * p.nZnT * sizeof(T);
    size_t nFull = p.ns * p.nZnT;

    auto alloc = [&](T*& dst, size_t count, const char* name) {
        if (arena)
            dst = arena->alloc_span<T>(name, count);
        else
            cumes::check_cuda(cudaMalloc(&dst, count * sizeof(T)), name);
    };
    alloc(d_gConEff_, nFull, "constraint/gConEff");
    alloc(d_gCon_, nFull, "constraint/gCon");
    // Zero gConEff/gCon: the effective-constraint and bandpass kernels skip
    // the axis row (no constraint on axis), so those bytes are never
    // PRODUCED — zeroing keeps every full-grid read (the dump machinery,
    // tests, initcheck) defined instead of uninitialized.
    cumes::check_cuda(cudaMemset(d_gConEff_, 0, nF), "gConEff zero");
    cumes::check_cuda(cudaMemset(d_gCon_, 0, nF), "gCon zero");
    alloc(d_rCon_, nFull, "constraint/rCon");
    alloc(d_zCon_, nFull, "constraint/zCon");
    alloc(d_rCon0_, nFull, "constraint/rCon0");
    alloc(d_zCon0_, nFull, "constraint/zCon0");
    // Initialize rCon0/zCon0 to zero
    cumes::check_cuda(cudaMemset(d_rCon0_, 0, nF), "rCon0 zero");
    cumes::check_cuda(cudaMemset(d_zCon0_, 0, nF), "zCon0 zero");

    // tcon profile (device). Zero-init: deAliasKernelFast reads tcon on
    // iteration 0 before computeTconKernel writes it — with zeros the
    // constraint force is inactive on the first iteration (deterministic,
    // matches vmecpp where the constraint has no prior tcon either).
    alloc(d_tcon_, p.ns, "constraint/tcon");
    cumes::check_cuda(cudaMemset(d_tcon_, 0, p.ns * sizeof(T)), "tcon zero");
    // faccon: host (pinned, never arena) + device
    cumes::check_cuda(cudaMallocHost(&h_faccon_, p.mnmax * sizeof(T)),
                      "faccon host");
    alloc(d_faccon_, p.mnmax, "constraint/faccon");
    // Precompute faccon[m] = -0.25 * signJ / (xmpq[m+1]^2) with
    // xmpq[m+1] = (m+1)*m, matching vmecpp (ideal_mhd_model.cc lines
    // 238-242): faccon[i] = 0.25 / (i^2 (i+1)^2) for i >= 1.
    for (int m = 0; m < p.mnmax; ++m) {
        T xmpq = T((m + 1) * m);
        h_faccon_[m] = (m > 0) ? (T(0.25) / (xmpq * xmpq)) : T(0.0);
    }
    cumes::check_cuda(cudaMemcpy(d_faccon_, h_faccon_, p.mnmax * sizeof(T),
                                 cudaMemcpyHostToDevice),
                      "faccon copy");

    // Constraint-force outputs (frcon/fzcon), zero-initialized so the axis
    // surface (skipped by the add kernel) reads zero like vmecpp.
    size_t nFc = p.ns * p.nZnT * sizeof(T);
    alloc(d_frcon_e_, nFull, "constraint/frcon_e");
    alloc(d_frcon_o_, nFull, "constraint/frcon_o");
    alloc(d_fzcon_e_, nFull, "constraint/fzcon_e");
    alloc(d_fzcon_o_, nFull, "constraint/fzcon_o");
    cumes::check_cuda(cudaMemset(d_frcon_e_, 0, nFc), "frcon_e zero");
    cumes::check_cuda(cudaMemset(d_frcon_o_, 0, nFc), "frcon_o zero");
    cumes::check_cuda(cudaMemset(d_fzcon_e_, 0, nFc), "fzcon_e zero");
    cumes::check_cuda(cudaMemset(d_fzcon_o_, 0, nFc), "fzcon_o zero");

    // The compact de-alias bandpass scratch + plans live in the
    // ToroidalFftOperator (transform scratch), so the constraint owns only its
    // data fields; the bandpass is reached through the SpectralOperator
    // interface.
    arena_backed_ = (arena != nullptr);
}

template <typename T>
cumes::ConstraintOperator<T>::~ConstraintOperator() {
    if (!arena_backed_) {
        cudaFree(d_gConEff_);
        cudaFree(d_gCon_);
        cudaFree(d_rCon_);
        cudaFree(d_zCon_);
        cudaFree(d_rCon0_);
        cudaFree(d_zCon0_);
        cudaFree(d_tcon_);
        cudaFree(d_faccon_);
        cudaFree(d_frcon_e_);
        cudaFree(d_frcon_o_);
        cudaFree(d_fzcon_e_);
        cudaFree(d_fzcon_o_);
    }
    cudaFreeHost(h_faccon_);
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
    const T* __restrict__ rCon,
    const T* __restrict__ zCon,
    const T* __restrict__ ru_e,
    const T* __restrict__ ru_o,
    const T* __restrict__ zu_e,
    const T* __restrict__ zu_o,
    const T* __restrict__ sqrtS_F,
    const T* __restrict__ rCon0,
    const T* __restrict__ zCon0,
    const cumes::ControlStatus* __restrict__ status,
    int ns,
    int nZnT,
    T* __restrict__ gConEff) {
    int jF = blockIdx.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (jF >= ns || k >= nZnT) return;
    // Status guard (completion plan step 1.4): no constraint-cache/scratch
    // writes on an invalid-Jacobian pass.
    if (status != nullptr && status->jacobian_valid == 0) return;
    int idx = k + jF * nZnT;

    // Skip magnetic axis (no poloidal angle)
    if (jF == 0) {
        gConEff[idx] = T(0.0);
        return;
    }

    T sF = sqrtS_F[jF];
    T ruFull = ru_e[idx] + sF * ru_o[idx];
    T zuFull = zu_e[idx] + sF * zu_o[idx];

    gConEff[idx] =
        (rCon[idx] - rCon0[idx]) * ruFull + (zCon[idx] - zCon0[idx]) * zuFull;
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
    const T* __restrict__ rCon,
    const T* __restrict__ zCon,
    const T* __restrict__ sqrtS_F,
    const cumes::ControlStatus* __restrict__ status,
    int ns,
    int nZnT,
    T* __restrict__ rCon0,
    T* __restrict__ zCon0) {
    int jF = blockIdx.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (jF >= ns || k >= nZnT) return;
    // Status guard (completion plan step 1.4): the rCon0/zCon0 reference
    // cache is not reset on an invalid pass (the re-anchored next pass
    // re-runs the reset).
    if (status != nullptr && status->jacobian_valid == 0) return;
    if (jF == 0) return;  // axis: stays zero (no poloidal angle)

    int lcfs = (ns - 1) * nZnT + k;
    T sFull = sqrtS_F[jF] * sqrtS_F[jF];
    int idx = k + jF * nZnT;
    rCon0[idx] = rCon[lcfs] * sFull;
    zCon0[idx] = zCon[lcfs] * sFull;
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
    const T* __restrict__ rCon,
    const T* __restrict__ zCon,
    const T* __restrict__ rCon0,
    const T* __restrict__ zCon0,
    const T* __restrict__ gCon,
    const T* __restrict__ sqrtS_F,
    const T* __restrict__ ru_e,
    const T* __restrict__ ru_o,
    const T* __restrict__ zu_e,
    const T* __restrict__ zu_o,
    const cumes::ControlStatus* __restrict__ status,
    int ns,
    int nZnT,
    T* __restrict__ brmn_e,
    T* __restrict__ brmn_o,
    T* __restrict__ bzmn_e,
    T* __restrict__ bzmn_o,
    T* __restrict__ frcon_e,
    T* __restrict__ frcon_o,
    T* __restrict__ fzcon_e,
    T* __restrict__ fzcon_o) {
    int jF = blockIdx.y;
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (jF >= ns || k >= nZnT) return;
    // Status guard (completion plan step 1.4): the constraint-force scratch
    // (and its brmn/bzmn += targets) is not written on an invalid pass.
    if (status != nullptr && status->jacobian_valid == 0) return;
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
    const T* __restrict__ ru_e,
    const T* __restrict__ ru_o,
    const T* __restrict__ zu_e,
    const T* __restrict__ zu_o,
    const T* __restrict__ sqrtS_F,
    const T* __restrict__ ard,
    const T* __restrict__ azd,
    const cumes::ControlStatus* __restrict__ status,
    int ns,
    int nZnT,
    int ntheta,
    int nzeta,
    T delta_s,
    T tcon_multiplier,
    T* __restrict__ tcon) {
    int jF = blockIdx.x * blockDim.x + threadIdx.x;
    if (jF >= ns) return;
    // Status guard (completion plan step 1.4): the tcon cache is not
    // refreshed on an invalid-Jacobian pass.
    if (status != nullptr && status->jacobian_valid == 0) return;
    if (jF == 0) {
        tcon[0] = T(0.0);
        return;
    }

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
    tcon[jF] =
        tcon_base * tcon_multiplier * T(32.0) * delta_s * T(32.0) * delta_s;
}

// vmecpp: tcon at the LCFS is halved ("maybe related to boundary only having
// MHD force contributions from the inside"). One thread.
template <typename T>
__global__ void tconLcfsHalfKernel(
    const cumes::ControlStatus* __restrict__ status,
    T* __restrict__ tcon,
    int ns) {
    // Status guard (completion plan step 1.4): tcon cache untouched when
    // the Jacobian is invalid.
    if (status != nullptr && status->jacobian_valid == 0) return;
    if (ns > 1) tcon[ns - 1] = T(0.5) * tcon[ns - 2];
}

// ---------------------------------------------------------------------------
// Reset rCon0/zCon0 to the LCFS-extrapolated profile (vmecpp rzConIntoVolume).
// Must be called with the current rCon/zCon, on the first iteration and on
// the reinit pass after every restart.
// ---------------------------------------------------------------------------
template <typename T>
void cumes::ConstraintOperator<T>::reset_reference(
    const DeviceParams<T>& p,
    const T* d_sqrtS_F,
    const cumes::ControlStatus* status,
    cudaStream_t stream) {
    dim3 block(128);
    dim3 grid((p.nZnT + 127) / 128, p.ns);
    rzConIntoVolumeKernel<T><<<grid, block, 0, stream>>>(
        d_rCon_, d_zCon_, d_sqrtS_F, status, p.ns, p.nZnT, d_rCon0_, d_zCon0_);
    cumes::check_cuda(cudaGetLastError(), "rzConIntoVolume");
}

// ---------------------------------------------------------------------------
// Host orchestration
// ---------------------------------------------------------------------------
// Steps 0 (tcon refresh) + 1 (effective constraint force gConEff) — shared by
// the generic and axisymmetric backends; only the step-2 bandpass differs.
template <typename T>
void cumes::ConstraintOperator<T>::enqueue_head(
    const DeviceParams<T>& p,
    const cumes::RealSpaceStorage<T>& rs,
    const T* ard,
    const T* azd,
    const T* d_sqrtS_F,
    bool precon_updated,
    const cumes::ControlStatus* status,
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
        // the kernel through DeviceParams::tcon0; default 1.0 keeps the
        // shipped configs unchanged).
        T tcon_multiplier =
            p.tcon0 *
            T(1.0 * (1.0 + p.ns * (1.0 / 60.0 + p.ns / (200.0 * 120.0))) /
              16.0);
        int gridF = (p.ns + 255) / 256;
        computeTconKernel<T><<<gridF, 256, 0, stream>>>(
            rs.d_ru_e, rs.d_ru_o, rs.d_zu_e, rs.d_zu_o, d_sqrtS_F, ard, azd,
            status, p.ns, p.nZnT, p.ntheta, p.nzeta, T(1.0) / T(p.ns - 1.0),
            tcon_multiplier, d_tcon_);
        cumes::check_cuda(cudaGetLastError(), "tcon");
        tconLcfsHalfKernel<T><<<1, 1, 0, stream>>>(status, d_tcon_, p.ns);
        cumes::check_cuda(cudaGetLastError(), "tcon lcfs");
    }

    // Step 1: Effective constraint force
    effectiveConstraintKernel<T><<<grid, block, 0, stream>>>(
        d_rCon_, d_zCon_, rs.d_ru_e, rs.d_ru_o, rs.d_zu_e, rs.d_zu_o, d_sqrtS_F,
        d_rCon0_, d_zCon0_, status, p.ns, p.nZnT, d_gConEff_);
    cumes::check_cuda(cudaGetLastError(), "gConEff");
}

// Step 3: Add constraint force to brmn/bzmn + write frcon/fzcon outputs.
template <typename T>
void cumes::ConstraintOperator<T>::enqueue_tail(
    const DeviceParams<T>& p,
    const cumes::RealSpaceStorage<T>& rs,
    const T* d_sqrtS_F,
    const cumes::ControlStatus* status,
    cudaStream_t stream) {
    dim3 block(128);
    dim3 grid((p.nZnT + 127) / 128, p.ns);
    addConstraintKernel<T><<<grid, block, 0, stream>>>(
        d_rCon_, d_zCon_, d_rCon0_, d_zCon0_, d_gCon_, d_sqrtS_F, rs.d_ru_e,
        rs.d_ru_o, rs.d_zu_e, rs.d_zu_o, status, p.ns, p.nZnT, rs.d_brmn_e,
        rs.d_brmn_o, rs.d_bzmn_e, rs.d_bzmn_o, d_frcon_e_, d_frcon_o_,
        d_fzcon_e_, d_fzcon_o_);
    cumes::check_cuda(cudaGetLastError(), "addConstraint");
}

// ---------------------------------------------------------------------------
// ConstraintOperator (owns the constraint buffers; the bandpass goes through
// the unified SpectralOperator interface — no backend branch)
// ---------------------------------------------------------------------------
template <typename T>
void cumes::ConstraintOperator<T>::enqueue(const DeviceParams<T>& p,
                                           const cumes::RealSpaceStorage<T>& rs,
                                           const T* ard,
                                           const T* azd,
                                           const T* sqrtS_F,
                                           bool precon_updated,
                                           cumes::SpectralOperator<T>* op,
                                           const cumes::ControlStatus* status,
                                           cudaStream_t stream) {
    // Steps 0/1 (tcon refresh + gConEff) are shared by both backends; step 2
    // (de-alias bandpass) is dispatched through the transform operator's
    // unified enqueue_dealias — the generic backend runs the compact cuFFT
    // round trip, the axisymmetric backend its direct-poloidal kernel. Step 3
    // (add to brmn/bzmn + frcon/fzcon) is shared again. The head/tail kernels
    // are status-guarded (completion plan step 1.4); the bandpass writes only
    // transform scratch, which the guarded tail never consumes on an invalid
    // pass.
    enqueue_head(p, rs, ard, azd, sqrtS_F, precon_updated, status, stream);
    op->enqueue_dealias(
        cumes::RealFieldView<const T>(d_gConEff_, p.ns, p.ntheta, p.nzeta),
        d_tcon_, d_faccon_,
        cumes::RealFieldView<T>(d_gCon_, p.ns, p.ntheta, p.nzeta), stream);
    enqueue_tail(p, rs, sqrtS_F, status, stream);
}

#endif  // CUMES_SRC_CONSTRAINT_IMPL_CUH_
