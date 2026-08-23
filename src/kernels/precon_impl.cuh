// kernels/precon_impl.cuh — template definitions for precon.cuh.
// Included once per scalar type by precon_double.cu / precon_float.cu; see the
// explicit-instantiation split (cumes_cuda_double / cumes_cuda_float).
#ifndef CUMES_SRC_PRECON_IMPL_CUH_
#define CUMES_SRC_PRECON_IMPL_CUH_
// precon.cu — radial tridiagonal preconditioner for MHD force balance.
// Computes flux-surface-averaged Hessian approximation separately for R and Z,
// assembles tridiagonal systems per (m,n) mode, and solves via parallel cyclic
// reduction (tridiagSolveKernel).
// Reference: vmecpp computePreconditioningMatrix / assembleRZPreconditioner /
// TridiagonalSolveSerial.
//
// All computation is templated on the scalar type T (double or float).
#include <cmath>
#include <cstdio>
#include <limits>

// The consuming kernels declare their dynamic shared memory directly as
// `extern __shared__ T s[]` — legal per TU because the explicit
// double/float instantiation split puts exactly one scalar type in each TU.
// The old dynSharedBase() indirection (removed 2026-08-16) existed only for
// the pre-split two-types-per-TU layout; the switch to the direct form was
// expected to be a Class B re-freeze but measured BIT-IDENTICAL on both
// configs (Solovev 251->199->456 / W7-X 1877->1617->2011, full-precision
// state, identical restart sequence) — the frozen baseline stands unchanged.

#include "cumes/numerics/preconditioner.hpp"
#include "cumes/numerics/tridiagonal_backend.hpp"
#include "cumes/runtime/cuda_status.hpp"
#include "cumes/runtime/device_arena.cuh"

// ---------------------------------------------------------------------------
// Allocate
// ---------------------------------------------------------------------------
template <typename T>
cumes::Preconditioner<T>::Preconditioner(const DeviceParams<T>& p,
                                         cumes::DeviceArena* arena) {
    int nH = p.ns - 1, nF = p.ns;

    auto alloc = [&](T*& dst, size_t count, const char* name) {
        if (arena)
            dst = arena->alloc_span<T>(name, count);
        else
            cumes::check_cuda(cudaMalloc(&dst, count * sizeof(T)), name);
    };
    alloc(d_ax_R_, 4 * nH, "precon/ax_R");
    alloc(d_ax_Z_, 4 * nH, "precon/ax_Z");
    alloc(d_bx_R_, 3 * nH, "precon/bx_R");
    alloc(d_bx_Z_, 3 * nH, "precon/bx_Z");
    alloc(d_cx_, nH, "precon/cx");
    alloc(d_arm_, 2 * nH, "precon/arm");
    alloc(d_brm_, 2 * nH, "precon/brm");
    alloc(d_azm_, 2 * nH, "precon/azm");
    alloc(d_bzm_, 2 * nH, "precon/bzm");
    alloc(d_ard_, 2 * nF, "precon/ard");
    alloc(d_brd_, 2 * nF, "precon/brd");
    alloc(d_azd_, 2 * nF, "precon/azd");
    alloc(d_bzd_, 2 * nF, "precon/bzd");
    alloc(d_cxd_, nF, "precon/cxd");
    alloc(d_sm_, nH, "precon/sm");
    alloc(d_sp_, nH, "precon/sp");
    alloc(d_ar_, p.mnmax * nF, "precon/ar");
    alloc(d_dr_, p.mnmax * nF, "precon/dr");
    alloc(d_br_, p.mnmax * nF, "precon/br");
    alloc(d_az_, p.mnmax * nF, "precon/az");
    alloc(d_dz_, p.mnmax * nF, "precon/dz");
    alloc(d_bz_, p.mnmax * nF, "precon/bz");
    if (arena)
        d_jMin_ = arena->alloc_span<int>("precon/jMin", p.mnmax);
    else
        cumes::check_cuda(cudaMalloc(&d_jMin_, p.mnmax * sizeof(int)), "jMin");
    alloc(d_lambdaPrec_, p.mnmax * nF, "precon/lambda_prec");
    alloc(d_bLambda_, p.ns + 1, "precon/bLambda");
    alloc(d_dLambda_, p.ns + 1, "precon/dLambda");
    alloc(d_cLambda_, p.ns + 1, "precon/cLambda");
    alloc(d_rmsPhiP_, 1, "precon/rmsPhiP");
    alloc(d_preconScale_, p.mnmax, "precon/scale");
    if (arena)
        d_preconStatus_ = arena->alloc_span<int>("precon/status", 1);
    else
        cumes::check_cuda(cudaMalloc(&d_preconStatus_, sizeof(int)),
                          "precon status");
    // Index ns of bLambda/cLambda must stay zero: the LCFS full-grid average
    // reads it (vmecpp: array sized ns+1, last entry never written).
    cumes::check_cuda(cudaMemset(d_bLambda_, 0, (p.ns + 1) * sizeof(T)),
                      "bLambda zero");
    cumes::check_cuda(cudaMemset(d_dLambda_, 0, (p.ns + 1) * sizeof(T)),
                      "dLambda zero");
    cumes::check_cuda(cudaMemset(d_cLambda_, 0, (p.ns + 1) * sizeof(T)),
                      "cLambda zero");
    arena_backed_ = (arena != nullptr);
}

template <typename T>
cumes::Preconditioner<T>::~Preconditioner() {
    if (!arena_backed_) {
        cudaFree(d_ax_R_);
        cudaFree(d_ax_Z_);
        cudaFree(d_bx_R_);
        cudaFree(d_bx_Z_);
        cudaFree(d_cx_);
        cudaFree(d_arm_);
        cudaFree(d_brm_);
        cudaFree(d_azm_);
        cudaFree(d_bzm_);
        cudaFree(d_ard_);
        cudaFree(d_brd_);
        cudaFree(d_azd_);
        cudaFree(d_bzd_);
        cudaFree(d_cxd_);
        cudaFree(d_sm_);
        cudaFree(d_sp_);
        cudaFree(d_ar_);
        cudaFree(d_dr_);
        cudaFree(d_br_);
        cudaFree(d_az_);
        cudaFree(d_dz_);
        cudaFree(d_bz_);
        cudaFree(d_jMin_);
        cudaFree(d_lambdaPrec_);
        cudaFree(d_bLambda_);
        cudaFree(d_dLambda_);
        cudaFree(d_cLambda_);
        cudaFree(d_rmsPhiP_);
        cudaFree(d_preconScale_);
        cudaFree(d_preconStatus_);
    }
}

// ---------------------------------------------------------------------------
// Step 1: Compute ax[4], bx[3], cx[1] on half-grid via surface integrals.
// One block per half-grid surface; parallel reduction over nZnT points.
// Processes both R and Z preconditioners in one pass.
//
// For R: xs=zs, xu12=zu12, xu_e/o=zu_e/o
// For Z: xs=rs, xu12=ru12, xu_e/o=ru_e/o
// ---------------------------------------------------------------------------
template <typename T>
__global__ void precon_compute_kernel(
    // Geometry shared by both R and Z precons
    const T* __restrict__ r12,
    const T* __restrict__ tau,
    const T* __restrict__ totalP,
    const T* __restrict__ bsupv,
    const T* __restrict__ gsqrt,
    const T* __restrict__ sqrtS_H,
    // For R precon: Z geometry (derivatives + odd-m coordinate value)
    const T* __restrict__ zs,
    const T* __restrict__ zu12,
    const T* __restrict__ zu_e,
    const T* __restrict__ zu_o,
    const T* __restrict__ z_o,  // odd-m Z coordinate value (= vmecpp z1_o)
    // For Z precon: R geometry (derivatives + odd-m coordinate value)
    const T* __restrict__ rs,
    const T* __restrict__ ru12,
    const T* __restrict__ ru_e,
    const T* __restrict__ ru_o,
    const T* __restrict__ r_o,  // odd-m R coordinate value (= vmecpp r1_o)
    const cumes::ControlStatus* __restrict__ status,
    int ns,
    int nZnT,
    T delta_s,
    // Outputs
    T* __restrict__ ax_R,
    T* __restrict__ ax_Z,
    T* __restrict__ bx_R,
    T* __restrict__ bx_Z,
    T* __restrict__ cx) {
    int jH = blockIdx.x, tid = threadIdx.x;
    if (jH >= ns - 1) return;
    // Status guard (completion plan step 1.4): the preconditioner element
    // cache is not rebuilt on an invalid-Jacobian pass.
    if (status != nullptr && status->jacobian_valid == 0) return;

    T sH = sqrtS_H[jH];
    T wInt = T(1.0) / T(nZnT);
    int base = jH * nZnT;

    // Accumulators: ax[4], bx[3], cx for R and Z
    T aR[4] = {T(0)}, aZ[4] = {T(0)};
    T bR[3] = {T(0)}, bZ[3] = {T(0)};
    T cx_v = T(0);

    for (int k = tid; k < nZnT; k += blockDim.x) {
        int idx = base + k;
        // Full-grid indices (same column-major layout, j+1 vs j surfaces)
        int idxF_i = idx;         // inner full-grid (jH)
        int idxF_o = idx + nZnT;  // outer full-grid (jH+1)

        // Common pTau factor: -4 * r12 * totalP / tau * wInt
        T pTau = T(-4.0) * r12[idx] * totalP[idx] / tau[idx] * wInt;

        // sH reciprocal for odd-m corrections in bx terms
        T inv_sH = T(1.0) / sH;

        // ---- R preconditioner (uses Z geometry) ----
        {
            // t1a: radial derivative term (dominant)
            T t1a = zu12[idx] / delta_s;
            // t2a, t3a: parity-mixed corrections from full-grid
            T t2a = T(0.25) * (zu_e[idxF_o] / sH + zu_o[idxF_o]) / sH;
            T t3a = T(0.25) * (zu_e[idxF_i] / sH + zu_o[idxF_i]) / sH;

            aR[0] += pTau * t1a * t1a;
            aR[1] += pTau * (t1a + t2a) * (-t1a + t3a);
            aR[2] += pTau * (t1a + t2a) * (t1a + t2a);
            aR[3] += pTau * (-t1a + t3a) * (-t1a + t3a);

            // t1b, t2b: poloidal term, includes odd-m full-grid
            // correction matching vmecpp: 0.5*(xs + 0.5/sqrtSH * x1_o)
            // NOTE: x1_o is the coordinate VALUE (z1_o), not derivative (zu_o)
            T t1b = T(0.5) * (zs[idx] + T(0.5) * inv_sH * z_o[idxF_o]);
            T t2b = T(0.5) * (zs[idx] + T(0.5) * inv_sH * z_o[idxF_i]);
            bR[0] += pTau * t1b * t2b;
            bR[1] += pTau * t1b * t1b;
            bR[2] += pTau * t2b * t2b;
        }

        // ---- Z preconditioner (uses R geometry) ----
        {
            T t1a = ru12[idx] / delta_s;
            T t2a = T(0.25) * (ru_e[idxF_o] / sH + ru_o[idxF_o]) / sH;
            T t3a = T(0.25) * (ru_e[idxF_i] / sH + ru_o[idxF_i]) / sH;

            aZ[0] += pTau * t1a * t1a;
            aZ[1] += pTau * (t1a + t2a) * (-t1a + t3a);
            aZ[2] += pTau * (t1a + t2a) * (t1a + t2a);
            aZ[3] += pTau * (-t1a + t3a) * (-t1a + t3a);

            // t1b, t2b: radial term, includes odd-m full-grid
            // correction matching vmecpp: 0.5*(xs + 0.5/sqrtSH * x1_o)
            // NOTE: x1_o is the coordinate VALUE (r1_o), not derivative (ru_o)
            T t1b = T(0.5) * (rs[idx] + T(0.5) * inv_sH * r_o[idxF_o]);
            T t2b = T(0.5) * (rs[idx] + T(0.5) * inv_sH * r_o[idxF_i]);
            bZ[0] += pTau * t1b * t2b;
            bZ[1] += pTau * t1b * t1b;
            bZ[2] += pTau * t2b * t2b;
        }

        // cx: toroidal term (same for R and Z)
        // 0.25 * pFactor = 0.25 * (-4) = -1
        cx_v += -bsupv[idx] * bsupv[idx] * gsqrt[idx] * wInt;
    }

    // Deterministic warp-shuffle reduction of the 15 accumulators (blueprint
    // §8.8): within-warp __shfl_down_sync tree, then a fixed cross-warp combine
    // by thread 0. The tree is fixed, so the result is deterministic; the
    // summation ORDER differs from the old shared-memory binary tree, a Class B
    // (ULP-level) change.
    T acc[15] = {aR[0], aR[1], aR[2], aR[3], aZ[0], aZ[1], aZ[2], aZ[3],
                 bR[0], bR[1], bR[2], bZ[0], bZ[1], bZ[2], cx_v};
#pragma unroll
    for (int c = 0; c < 15; ++c)
        for (int o = 16; o > 0; o >>= 1)
            acc[c] += __shfl_down_sync(0xffffffffu, acc[c], o);
    const int lane = tid & 31, warp = tid >> 5;
    __shared__ T warp_sum[8 * 15];
    if (lane == 0)
#pragma unroll
        for (int c = 0; c < 15; ++c) warp_sum[warp * 15 + c] = acc[c];
    __syncthreads();
    if (tid == 0) {
        T tot[15];
#pragma unroll
        for (int c = 0; c < 15; ++c) tot[c] = warp_sum[0 * 15 + c];
        for (int w = 1; w < blockDim.x / 32; ++w)
#pragma unroll
            for (int c = 0; c < 15; ++c) tot[c] += warp_sum[w * 15 + c];
        int jH4 = jH * 4, jH3 = jH * 3;
        ax_R[jH4 + 0] = tot[0];
        ax_R[jH4 + 1] = tot[1];
        ax_R[jH4 + 2] = tot[2];
        ax_R[jH4 + 3] = tot[3];
        ax_Z[jH4 + 0] = tot[4];
        ax_Z[jH4 + 1] = tot[5];
        ax_Z[jH4 + 2] = tot[6];
        ax_Z[jH4 + 3] = tot[7];
        bx_R[jH3 + 0] = tot[8];
        bx_R[jH3 + 1] = tot[9];
        bx_R[jH3 + 2] = tot[10];
        bx_Z[jH3 + 0] = tot[11];
        bx_Z[jH3 + 1] = tot[12];
        bx_Z[jH3 + 2] = tot[13];
        cx[jH] = tot[14];
    }
}

// ---------------------------------------------------------------------------
// Step 2: Assemble half-grid → full-grid with sm/sp parity factors.
// Also computes sm/sp on the fly.
// One block handles all half-grid surfaces.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void precon_assemble_kernel(
    const T* __restrict__ ax_R,
    const T* __restrict__ ax_Z,
    const T* __restrict__ bx_R,
    const T* __restrict__ bx_Z,
    const T* __restrict__ cx,
    const T* __restrict__ sqrtS_H,
    const T* __restrict__ sqrtS_F,
    int ns,
    T* __restrict__ arm,
    T* __restrict__ brm,
    T* __restrict__ azm,
    T* __restrict__ bzm,
    T* __restrict__ ard,
    T* __restrict__ brd,
    T* __restrict__ azd,
    T* __restrict__ bzd,
    T* __restrict__ cxd,
    T* __restrict__ sm_out,
    T* __restrict__ sp_out,
    const cumes::ControlStatus* __restrict__ status) {
    int jH = blockIdx.x * blockDim.x + threadIdx.x;
    if (jH >= ns - 1) return;
    // Status guard (completion plan step 1.4): no preconditioner-cache
    // writes on an invalid-Jacobian pass.
    if (status != nullptr && status->jacobian_valid == 0) return;

    int jH4 = jH * 4, jH3 = jH * 3;
    int jH_even = jH * 2;     // even parity index
    int jH_odd = jH * 2 + 1;  // odd parity index

    // Compute sm/sp for this half-grid (vmecpp convention).
    // Half-grid jH sits between full-grid jH and jH+1.
    // sm[jH] = sqrtS_H[jH] / sqrtS_F[jH+1]  (ratio to OUTER full-grid)
    // sp[jH] = sqrtS_H[jH] / sqrtS_F[jH]    (ratio to INNER full-grid)
    T sh = sqrtS_H[jH];
    T sm = sh / sqrtS_F[jH + 1];  // outer — always safe (sqrtS_F[jH+1] > 0)
    T sp;
    if (jH == 0) {
        sp = sm;  // at innermost half-grid: inner full-grid is axis (s=0), so
                  // sp = sm
    } else {
        sp = sh / sqrtS_F[jH];  // inner — safe for jH > 0
    }
    sm_out[jH] = sm;
    sp_out[jH] = sp;

    T smsp = sm * sp;

    // ---- R preconditioner ----
    // Off-diagonal (half-grid)
    arm[jH_even] = -ax_R[jH4 + 0];       // even: -ax[0]
    arm[jH_odd] = ax_R[jH4 + 1] * smsp;  // odd:  ax[1] * sm * sp
    brm[jH_even] = bx_R[jH3 + 0];        // even: bx[0]
    brm[jH_odd] = bx_R[jH3 + 0] * smsp;  // odd:  bx[0] * sm * sp

    // ---- Z preconditioner ----
    azm[jH_even] = -ax_Z[jH4 + 0];
    azm[jH_odd] = ax_Z[jH4 + 1] * smsp;
    bzm[jH_even] = bx_Z[jH3 + 0];
    bzm[jH_odd] = bx_Z[jH3 + 0] * smsp;
}

// ---------------------------------------------------------------------------
// Step 2b: Average half-grid diagonals to full-grid.
// One thread per full-grid surface.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void precon_diag_kernel(
    const T* __restrict__ ax_R,
    const T* __restrict__ ax_Z,
    const T* __restrict__ bx_R,
    const T* __restrict__ bx_Z,
    const T* __restrict__ cx,
    const T* __restrict__ sm,
    const T* __restrict__ sp,
    int ns,
    T* __restrict__ ard,
    T* __restrict__ brd,
    T* __restrict__ azd,
    T* __restrict__ bzd,
    T* __restrict__ cxd,
    const cumes::ControlStatus* __restrict__ status) {
    int jF = blockIdx.x * blockDim.x + threadIdx.x;
    if (jF >= ns) return;
    // Status guard (completion plan step 1.4): no preconditioner-cache
    // writes on an invalid-Jacobian pass.
    if (status != nullptr && status->jacobian_valid == 0) return;

    int jF_even = jF * 2;
    int jF_odd = jF * 2 + 1;

    // Inner half-grid index (jH_i = jF-1), outer (jH_o = jF)
    int jHi = jF - 1, jHo = jF;
    bool has_i = (jF > 0), has_o = (jF < ns - 1);

    // ---- R preconditioner diagonals ----
    // axd (radial d²/ds²):
    if (has_i && has_o) {
        ard[jF_even] = ax_R[jHi * 4 + 0] + ax_R[jHo * 4 + 0];
        brd[jF_even] = bx_R[jHi * 3 + 1] + bx_R[jHo * 3 + 2];
        // odd: ax[2]*sm² (inner) + ax[3]*sp² (outer)
        ard[jF_odd] = ax_R[jHi * 4 + 2] * sm[jHi] * sm[jHi] +
                      ax_R[jHo * 4 + 3] * sp[jHo] * sp[jHo];
        brd[jF_odd] = bx_R[jHi * 3 + 1] * sm[jHi] * sm[jHi] +
                      bx_R[jHo * 3 + 2] * sp[jHo] * sp[jHo];
        cxd[jF] = cx[jHi] + cx[jHo];
    } else if (has_o) {
        // jF == 0: only outer contribution
        ard[jF_even] = ax_R[jHo * 4 + 0];
        ard[jF_odd] = ax_R[jHo * 4 + 3] * sp[jHo] * sp[jHo];
        brd[jF_even] = bx_R[jHo * 3 + 2];
        brd[jF_odd] = bx_R[jHo * 3 + 2] * sp[jHo] * sp[jHo];
        cxd[jF] = cx[jHo];
    } else {
        // jF == ns-1: only inner contribution
        ard[jF_even] = ax_R[jHi * 4 + 0];
        ard[jF_odd] = ax_R[jHi * 4 + 2] * sm[jHi] * sm[jHi];
        brd[jF_even] = bx_R[jHi * 3 + 1];
        brd[jF_odd] = bx_R[jHi * 3 + 1] * sm[jHi] * sm[jHi];
        cxd[jF] = cx[jHi];
    }

    // ---- Z preconditioner diagonals ----
    if (has_i && has_o) {
        azd[jF_even] = ax_Z[jHi * 4 + 0] + ax_Z[jHo * 4 + 0];
        bzd[jF_even] = bx_Z[jHi * 3 + 1] + bx_Z[jHo * 3 + 2];
        azd[jF_odd] = ax_Z[jHi * 4 + 2] * sm[jHi] * sm[jHi] +
                      ax_Z[jHo * 4 + 3] * sp[jHo] * sp[jHo];
        bzd[jF_odd] = bx_Z[jHi * 3 + 1] * sm[jHi] * sm[jHi] +
                      bx_Z[jHo * 3 + 2] * sp[jHo] * sp[jHo];
    } else if (has_o) {
        azd[jF_even] = ax_Z[jHo * 4 + 0];
        azd[jF_odd] = ax_Z[jHo * 4 + 3] * sp[jHo] * sp[jHo];
        bzd[jF_even] = bx_Z[jHo * 3 + 2];
        bzd[jF_odd] = bx_Z[jHo * 3 + 2] * sp[jHo] * sp[jHo];
    } else {
        azd[jF_even] = ax_Z[jHi * 4 + 0];
        azd[jF_odd] = ax_Z[jHi * 4 + 2] * sm[jHi] * sm[jHi];
        bzd[jF_even] = bx_Z[jHi * 3 + 1];
        bzd[jF_odd] = bx_Z[jHi * 3 + 1] * sm[jHi] * sm[jHi];
    }

    // Edge pedestal for boundary stability
    const T edge_pedestal = T(0.05);
    if (jF == ns - 1) {
        ard[jF_even] *= T(1.0) + edge_pedestal;
        ard[jF_odd] *= T(1.0) + edge_pedestal;
        brd[jF_even] *= T(1.0) + edge_pedestal;
        brd[jF_odd] *= T(1.0) + edge_pedestal;
        azd[jF_even] *= T(1.0) + edge_pedestal;
        azd[jF_odd] *= T(1.0) + edge_pedestal;
        bzd[jF_even] *= T(1.0) + edge_pedestal;
        bzd[jF_odd] *= T(1.0) + edge_pedestal;
        cxd[jF] *= T(1.0) + edge_pedestal;
    }
}

// ---------------------------------------------------------------------------
// Step 3: Assemble tridiagonal matrices per (m,n) mode.
// One thread per (mode, radial_surface).
//   ar[jF] : sup-diagonal on half-grid at jF (outer for forces at jF)
//   dr[jF] : diagonal on full-grid jF
//   br[jF] : sub-diagonal on half-grid at jF-1 (inner for forces at jF)
// ---------------------------------------------------------------------------
template <typename T>
__global__ void tridiag_assembly_kernel(
    const T* __restrict__ arm,
    const T* __restrict__ brm,
    const T* __restrict__ azm,
    const T* __restrict__ bzm,
    const T* __restrict__ ard,
    const T* __restrict__ brd,
    const T* __restrict__ azd,
    const T* __restrict__ bzd,
    const T* __restrict__ cxd,
    const int* __restrict__ xm,
    const int* __restrict__ xn,
    int ns,
    int mnmax,
    int nfp,
    T* __restrict__ ar,
    T* __restrict__ dr,
    T* __restrict__ br,
    T* __restrict__ az,
    T* __restrict__ dz,
    T* __restrict__ bz,
    int* __restrict__ jMin,
    const cumes::ControlStatus* __restrict__ status) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = mnmax * ns;
    if (idx >= total) return;
    // Status guard (completion plan step 1.4): no preconditioner-cache
    // writes on an invalid-Jacobian pass.
    if (status != nullptr && status->jacobian_valid == 0) return;

    int mode = idx / ns;
    int jF = idx - mode * ns;
    int mm = xm[mode], nn = xn[mode];
    int parity = mm % 2;  // m-parity: determines even/odd precon elements

    T m2 = T(mm * mm);
    T n2 = T(nn * nfp * nn * nfp);
    int nH = ns - 1;

    // Sup-diagonal: half-grid at jF (outer of forces surface jF)
    if (jF < nH) {
        int jH_par = jF * 2 + parity;
        ar[idx] = -(arm[jH_par] + brm[jH_par] * m2);
        az[idx] = -(azm[jH_par] + bzm[jH_par] * m2);
    } else {
        ar[idx] = T(0.0);
        az[idx] = T(0.0);
    }

    // Diagonal: full-grid at jF
    int jF_par = jF * 2 + parity;
    dr[idx] = -(ard[jF_par] + brd[jF_par] * m2 + cxd[jF] * n2);
    dz[idx] = -(azd[jF_par] + bzd[jF_par] * m2 + cxd[jF] * n2);

    // Sub-diagonal: half-grid at jF-1 (inner of forces surface jF)
    if (jF > 0) {
        int jH_par = (jF - 1) * 2 + parity;
        br[idx] = -(arm[jH_par] + brm[jH_par] * m2);
        bz[idx] = -(azm[jH_par] + bzm[jH_par] * m2);
    } else {
        br[idx] = T(0.0);
        bz[idx] = T(0.0);
    }

    // jMin: magnetic axis only gets m=0 contributions
    // (higher m modes vanish at axis).
    // Each mode has exactly one jF=0 entry — no race condition.
    if (jF == 0) { jMin[mode] = (mm == 0) ? 0 : 1; }

    // Special m=1 correction at jF=1: merge sub-diagonal into diagonal
    if (jF == 1 && mm == 1) {
        dr[idx] += br[idx];
        dz[idx] += bz[idx];
    }
}

// ---------------------------------------------------------------------------
// Step 4a: Lambda preconditioner — flux-surface averages on half grid.
// Port of vmecpp IdealMhdModel::updateLambdaPreconditioner (axisymmetric
// variant: n=0 only, so the guv/dLambda term drops out). The averages run
// over the reduced poloidal grid [0, pi] (the first ntheta/2+1 points of the
// full theta grid), matching vmecpp's nThetaEff under stellarator symmetry,
// with trapezoidal weights wInt from sizes.cc:
//   wInt[l] = 1/(nzeta*(nThetaReduced-1)), halved at l=0 and l=nThetaReduced-1
// Also accumulates rmsPhiP = sum(phipH^2) for lamscale.
// One block per half-grid surface; shifted layout: half-grid jH -> index jH+1.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void lambda_prec_assemble_kernel(
    const T* __restrict__ guu,
    const T* __restrict__ guv,
    const T* __restrict__ gvv,
    const T* __restrict__ gsqrt,
    const T* __restrict__ phipH,
    int ns,
    int nZnT,
    int ntheta,
    int nzeta,
    T* __restrict__ bLambda,
    T* __restrict__ dLambda,
    T* __restrict__ cLambda,
    T* __restrict__ rmsPhiP,
    const cumes::ControlStatus* __restrict__ status) {
    int jH = blockIdx.x, tid = threadIdx.x;
    if (jH >= ns - 1) return;
    // Status guard (completion plan step 1.4): no preconditioner-cache
    // writes on an invalid-Jacobian pass.
    if (status != nullptr && status->jacobian_valid == 0) return;

    const int nThetaEven = 2 * (ntheta / 2);
    const int nThetaRed = nThetaEven / 2 + 1;  // reduced grid [0, pi]
    const T dnorm3 = T(1.0) / T(nzeta * (nThetaRed - 1));

    T bsum = T(0.0), dsum = T(0.0), csum = T(0.0);
    // Loop over ALL (zeta, reduced-theta) points: the layout is
    // [jH][zeta][theta], so the index must combine both. (FIXED 2026-08-02:
    // previously only theta at the first zeta plane was summed, making the
    // bLambda/dLambda/cLambda averages ~21-36x too small and the lambda
    // preconditioner ~32x too big near the LCFS.)
    for (int k = tid; k < nThetaRed * nzeta; k += blockDim.x) {
        int iz = k / nThetaRed, it = k % nThetaRed;
        T w = dnorm3;
        if (it == 0 || it == nThetaRed - 1) w *= T(0.5);
        int idx = (jH * nzeta + iz) * ntheta + it;
        bsum += guu[idx] / gsqrt[idx] * w;
        dsum += guv[idx] / gsqrt[idx] * w;  // 3D: toroidal coupling
        csum += gvv[idx] / gsqrt[idx] * w;
    }

    // Deterministic warp-shuffle reduction (blueprint §8.8): within-warp
    // __shfl_down_sync tree, then a fixed cross-warp combine by thread 0. The
    // tree is fixed (deterministic); the summation ORDER differs from the old
    // shared-memory binary tree, a Class B (ULP-level) change.
    for (int o = 16; o > 0; o >>= 1) {
        bsum += __shfl_down_sync(0xffffffffu, bsum, o);
        dsum += __shfl_down_sync(0xffffffffu, dsum, o);
        csum += __shfl_down_sync(0xffffffffu, csum, o);
    }
    const int lane = tid & 31, warp = tid >> 5;
    __shared__ T w_b[8], w_d[8], w_c[8];
    if (lane == 0) {
        w_b[warp] = bsum;
        w_d[warp] = dsum;
        w_c[warp] = csum;
    }
    __syncthreads();
    if (tid == 0) {
        T tb = w_b[0], td = w_d[0], tc = w_c[0];
        for (int w = 1; w < blockDim.x / 32; ++w) {
            tb += w_b[w];
            td += w_d[w];
            tc += w_c[w];
        }
        bLambda[jH + 1] = tb;
        dLambda[jH + 1] = td;
        cLambda[jH + 1] = tc;
        // Deterministic rmsPhiP: one atomic from the last block, summing in
        // jH order (the original per-block atomicAdd's cross-block FP order
        // is scheduling-dependent, making rmsPhiP — and hence the lambda
        // preconditioner — vary by ~1 ulp run to run).
        if (jH == ns - 2) {
            double total = 0.0;
            for (int j = 0; j < ns - 1; ++j) total += phipH[j] * phipH[j];
            atomicAdd(rmsPhiP, total);
        }
    }
}

// ---------------------------------------------------------------------------
// Step 4b: Assemble the per-(mode, surface) lambda diagonal preconditioner.
// Port of vmecpp's lambdaPreconditioner assembly:
//   faclam = n^2*bLambda + 2mn*sign(b)*dLambda + m^2*cLambda
//          (n=0 here, so faclam = m^2*cLambda)
//   lambda_prec = pFactor / faclam * sqrt(s)^pwr
// with pFactor = LAMBDA_PRECONDITIONER_DAMPING_FACTOR / (4*lamscale^2),
// lamscale = sqrt(rmsPhiP * deltaS), and
// pwr = min(m^2 / 16^2, 8) suppressing high-m modes.
// The m=0 (n=0) mode is skipped (stays 0), matching vmecpp — lambda_00 is
// a gauge mode and is zeroed by the preconditioner.
// Full-grid averaging follows vmecpp's shifted layout:
//   full[jF] = 0.5*(b[jF+1] + b[jF]) for jF=1..ns-1 (LCFS reads b[ns] = 0).
// One block per mode.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void lambda_prec_finalize_kernel(
    const T* __restrict__ bLambda,
    const T* __restrict__ dLambda,
    const T* __restrict__ cLambda,
    const T* __restrict__ sqrtS_F,
    const T* __restrict__ rmsPhiP,
    const int* __restrict__ xm,
    const int* __restrict__ xn,
    int ns,
    int mnmax,
    T delta_s,
    int nfp,
    T* __restrict__ lambda_prec,
    const cumes::ControlStatus* __restrict__ status) {
    // One thread per (mode, jF) element — the original ran one thread per
    // mode with a serial ns loop (<<<mnmax, 1>>>). Per-element arithmetic
    // is unchanged (the per-mode scalars are recomputed per thread).
    int mode = blockIdx.x;
    int jF = blockIdx.y * blockDim.x + threadIdx.x;
    if (mode >= mnmax || jF >= ns) return;
    // Status guard (completion plan step 1.4): the lambda-preconditioner
    // cache is not rebuilt on an invalid-Jacobian pass.
    if (status != nullptr && status->jacobian_valid == 0) return;

    int m = xm[mode], n = xn[mode];
    // Guard: an all-zero phip profile (rmsPhiP == 0) would make lamscale == 0
    // and pFactor infinite. The tiny floor keeps the lambda preconditioner
    // finite; the degenerate geometry is failed earlier by the solver's
    // jacobian-stats check anyway.
    T lamscale = sqrt(fmax(rmsPhiP[0] * delta_s, T(1e-30)));
    const T pFactor = T(2.0) / (T(4.0) * lamscale * lamscale);
    const T pwr = fmin(T(m * m) / T(16.0 * 16.0), T(8.0));
    // faclam terms (vmecpp updateLambdaPreconditioner):
    //   tnn = (n*nfp)^2, tmn = 2*m*n*nfp  (nfp passed separately)
    T tnn = T(n * n) * T(nfp * nfp);
    T tmn = T(2.0) * T(m * n) * T(nfp);

    if (jF == 0) {
        lambda_prec[mode * ns] = T(0.0);
        return;
    }
    if (m == 0 && n == 0) {
        lambda_prec[mode * ns + jF] = T(0.0);
        return;
    }  // gauge mode

    T bFull = T(0.5) * (bLambda[jF + 1] + bLambda[jF]);
    T dFull = T(0.5) * (dLambda[jF + 1] + dLambda[jF]);
    T cFull = T(0.5) * (cLambda[jF + 1] + cLambda[jF]);
    T faclam = tnn * bFull + tmn * copysign(dFull, bFull) + T(m * m) * cFull;
    if (faclam == T(0.0))
        faclam = T(-1.0e-10);  // LAMBDA_PRECONDITIONER_ZERO_GUARD
    lambda_prec[mode * ns + jF] = pFactor / faclam * pow(sqrtS_F[jF], pwr);
}

// ---------------------------------------------------------------------------
// Per-mode coefficient scale for the scale-aware pivot floor (blueprint §4.9).
// scale[mode] = max over the solved rows of |lower|,|diagonal|,|upper| for the
// assembled R+Z tridiagonal systems. One block per mode; deterministic block
// tree reduction over the surface dimension. Runs once per preconditioner
// refresh (the matrix changes only on refresh); the solve kernels read it every
// pass.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void precon_scale_kernel(
    const T* __restrict__ ar,
    const T* __restrict__ dr,
    const T* __restrict__ br,
    const T* __restrict__ az,
    const T* __restrict__ dz,
    const T* __restrict__ bz,
    const cumes::ControlStatus* __restrict__ status,
    int ns,
    int mnmax,
    T* __restrict__ scale) {
    int mode = blockIdx.x, tid = threadIdx.x;
    if (mode >= mnmax) return;
    // Status guard (completion plan step 1.4): the per-mode pivot scale
    // cache is not rebuilt on an invalid-Jacobian pass.
    if (status != nullptr && status->jacobian_valid == 0) return;
    T m = T(0.0);
    for (int j = tid; j < ns; j += blockDim.x) {
        T v = fmax(fmax(fabs(ar[mode * ns + j]), fabs(dr[mode * ns + j])),
                   fabs(br[mode * ns + j]));
        v = fmax(v, fmax(fmax(fabs(az[mode * ns + j]), fabs(dz[mode * ns + j])),
                         fabs(bz[mode * ns + j])));
        m = fmax(m, v);
    }
    __shared__ T s[256];
    s[tid] = m;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) s[tid] = fmax(s[tid], s[tid + off]);
        __syncthreads();
    }
    if (tid == 0) scale[mode] = s[0];
}

// ---------------------------------------------------------------------------
// Batched parallel cyclic reduction (PCR) solve (blueprint §8.9). One block per
// mode (system), 128 threads, grid-stride over the solved rows.
// Backend-neutral: it operates on the raw lower/diagonal/upper + rhs planes of
// a StridedBatchTridiagonalView rather than the spectral slab, so the PCR and
// Thomas backends share one contract.
//
// Extracted bit-for-bit from the legacy tridiagSolveKernel: the staged
// coefficients, the grid-stride PCR rounds, and the final divide are unchanged.
// The only behavioral difference is the pivot guard: the legacy absolute 1e-30
// clamp is replaced by a scale-aware floor `kappa * eps * scale[mode]` (eps =
// machine epsilon), so a genuinely sub-scale pivot is *reported* (atomicAdd
// into status) instead of silently clamped. For the frozen trajectories the
// diagonals are O(1)..O(m^2), far above the floor, so the guard never fires and
// the solve is byte-identical. `rhs_count` is a compile-time parameter so the
// RHS loop stays fully unrolled exactly as in the original kernel (fast-math
// FMA contraction is preserved).
// ---------------------------------------------------------------------------
template <typename T, int rhs_count>
__global__ void pcr_solve_kernel(
    const T* __restrict__ lower,
    const T* __restrict__ diagonal,
    const T* __restrict__ upper,
    T* __restrict__ rhs,
    const int* __restrict__ first_surface,
    const T* __restrict__ scale,
    T pivot_floor_rel,
    int modes,
    int ns,
    int last_surface,
    size_t rhs_stride,
    int* __restrict__ status,
    const cumes::ControlStatus* __restrict__ gate) {
    int mode = blockIdx.x;
    if (mode >= modes) return;
    // Terminal gate (completion plan step 1.4): on a nonfinite/converged
    // pass the preconditioner no-ops — the RHS (the decomposed residual
    // slab) is left untouched and the preconditioned reduction marks its
    // fields not evaluated. gate is nullptr for direct (test/benchmark)
    // callers, which then behave exactly as before.
    if (gate &&
        (gate->invariant_nonfinite != 0 || gate->invariant_converged != 0))
        return;
    int tid = threadIdx.x;
    int jMin_m = first_surface[mode];
    int jMax = last_surface;   // LCFS row excluded
    int nRow = jMax - jMin_m;  // solved rows jMin_m .. jMax-1

    // Scale-aware pivot floor; a zero scale (all-zero system) falls back to
    // kappa * eps so the guard still fires and the breakdown is reported.
    T fl = pivot_floor_rel * scale[mode];
    if (scale[mode] <= T(0.0)) fl = pivot_floor_rel;

    extern __shared__ T s_tri[];  // [(6+2*rhs_count)][ns]
    T* cL = s_tri;                // [ns]
    T* cD = cL + ns;
    T* cU = cD + ns;
    T* nL = cU + ns;
    T* nD = nL + ns;
    T* nU = nD + ns;
    T* cF = nU + ns;              // [rhs_count][ns]
    T* nF = cF + rhs_count * ns;  // [rhs_count][ns]

    // stage lower=L, diag=D, upper=U + rhs planes
    for (int j = tid; j < ns; j += blockDim.x) {
        cL[j] = lower[mode * ns + j];
        cD[j] = diagonal[mode * ns + j];
        cU[j] = upper[mode * ns + j];
#pragma unroll
        for (int c = 0; c < rhs_count; ++c)
            cF[c * ns + j] = rhs[c * rhs_stride + mode * ns + j];
    }
    // zero the j < jMin region of the staged RHS (mirrors Thomas zeroing)
    for (int j = tid; j < jMin_m; j += blockDim.x)
#pragma unroll
        for (int c = 0; c < rhs_count; ++c) cF[c * ns + j] = T(0.0);
    __syncthreads();

    bool broke = false;
    for (int k = 1; k <= nRow; k <<= 1) {
        for (int r = tid; r < nRow; r += blockDim.x) {
            int j = jMin_m + r;
            bool hasL = (j - k >= jMin_m), hasR = (j + k < jMax);
            T dL = hasL ? cD[j - k] : T(0.0);
            T dR = hasR ? cD[j + k] : T(0.0);
            // Scale-aware pivot guard with sign preservation (copysign keeps a
            // negative pivot negative — the old +1e-30 clamp flipped it). Only
            // a genuinely in-range (hasL/hasR) sub-scale diagonal is a
            // breakdown; the boundary dL=0 sentinel is not.
            if (fabs(dL) < fl) {
                dL = copysign(fl, dL);
                if (hasL) broke = true;
            }
            if (fabs(dR) < fl) {
                dR = copysign(fl, dR);
                if (hasR) broke = true;
            }
            T invL = hasL ? T(1.0) / dL : T(0.0);
            T invR = hasR ? T(1.0) / dR : T(0.0);
            T L = cL[j], D = cD[j], U = cU[j];
            T Ll = hasL ? cL[j - k] : T(0.0), Ul = hasL ? cU[j - k] : T(0.0);
            T Lr = hasR ? cL[j + k] : T(0.0), Ur = hasR ? cU[j + k] : T(0.0);
            nL[j] = -L * Ll * invL;
            nU[j] = -U * Ur * invR;
            T nDv = D - L * Ul * invL - U * Lr * invR;
            if (fabs(nDv) < fl) {
                nDv = copysign(fl, nDv);
                broke = true;
            }
            nD[j] = nDv;
#pragma unroll
            for (int c = 0; c < rhs_count; ++c) {
                T fv = cF[c * ns + j];
                T fLeft = hasL ? cF[c * ns + j - k] : T(0.0);
                T fRight = hasR ? cF[c * ns + j + k] : T(0.0);
                nF[c * ns + j] = fv - L * fLeft * invL - U * fRight * invR;
            }
        }
        __syncthreads();
        T* t;
        t = cL;
        cL = nL;
        nL = t;
        t = cD;
        cD = nD;
        nD = t;
        t = cU;
        cU = nU;
        nU = t;
        t = cF;
        cF = nF;
        nF = t;
        __syncthreads();
    }
    // ---- Final solve: x = f'' / d'' (decoupled after the rounds) ----
    for (int r = tid; r < nRow; r += blockDim.x) {
        int j = jMin_m + r;
        T d = cD[j];
        if (fabs(d) < fl) {
            d = copysign(fl, d);
            broke = true;
        }
        T inv = T(1.0) / d;
#pragma unroll
        for (int c = 0; c < rhs_count; ++c)
            rhs[c * rhs_stride + mode * ns + j] = cF[c * ns + j] * inv;
    }

    if (broke) atomicAdd(status, 1);
}

// ---------------------------------------------------------------------------
// Serial Thomas solve (blueprint §8.9 "Thomas-based backend for small
// axisymmetric batches"). One thread per mode; O(ns) dynamic shared memory for
// the forward-elimination cp/dp scratch. Same scale-aware pivot floor as the
// PCR backend; a sub-floor pivot is guarded and counted into status.
// Numerically distinct from PCR at the rounding level (different elimination
// order).
// ---------------------------------------------------------------------------
template <typename T>
__global__ void thomas_solve_kernel(
    const T* __restrict__ lower,
    const T* __restrict__ diagonal,
    const T* __restrict__ upper,
    T* __restrict__ rhs,
    const int* __restrict__ first_surface,
    const T* __restrict__ scale,
    T pivot_floor_rel,
    int modes,
    int ns,
    int last_surface,
    int rhs_count,
    size_t rhs_stride,
    int* __restrict__ status,
    const cumes::ControlStatus* __restrict__ gate) {
    int mode = blockIdx.x;
    if (mode >= modes) return;
    // Terminal gate (completion plan step 1.4): see pcr_solve_kernel.
    if (gate &&
        (gate->invariant_nonfinite != 0 || gate->invariant_converged != 0))
        return;
    int jMin = first_surface[mode];
    int jMax = last_surface;
    int n = jMax - jMin;
    if (n <= 0) return;

    T fl = pivot_floor_rel * scale[mode];
    if (scale[mode] <= T(0.0)) fl = pivot_floor_rel;

    extern __shared__ T s[];  // [(1+rhs_count)][n]
    T* cp = s;                // [n]
    T* dp = cp + n;           // [rhs_count][n]

    bool broke = false;
    T denom = diagonal[mode * ns + jMin];
    if (fabs(denom) < fl) {
        denom = copysign(fl, denom);
        broke = true;
    }
    cp[0] = upper[mode * ns + jMin] / denom;
    for (int c = 0; c < rhs_count; ++c)
        dp[c * n] = rhs[c * rhs_stride + mode * ns + jMin] / denom;
    for (int i = 1; i < n; ++i) {
        int j = jMin + i;
        T d = diagonal[mode * ns + j] - lower[mode * ns + j] * cp[i - 1];
        if (fabs(d) < fl) {
            d = copysign(fl, d);
            broke = true;
        }
        cp[i] = upper[mode * ns + j] / d;
        for (int c = 0; c < rhs_count; ++c)
            dp[c * n + i] = (rhs[c * rhs_stride + mode * ns + j] -
                             lower[mode * ns + j] * dp[c * n + i - 1]) /
                            d;
    }
    // back substitution (x[jMax] = 0 boundary is implied by the dropped term)
    for (int c = 0; c < rhs_count; ++c)
        rhs[c * rhs_stride + mode * ns + jMax - 1] = dp[c * n + n - 1];
    for (int i = n - 2; i >= 0; --i) {
        int j = jMin + i;
        for (int c = 0; c < rhs_count; ++c)
            rhs[c * rhs_stride + mode * ns + j] =
                dp[c * n + i] - cp[i] * rhs[c * rhs_stride + mode * ns + j + 1];
    }

    if (broke) atomicAdd(status, 1);
}

// ---------------------------------------------------------------------------
// Boundary + lambda-diagonal finishing of the preconditioned residual.
// Extracted from the tail of the legacy tridiagSolveKernel: comps 0..4 are
// zeroed below jMin (comp 5 keeps its value), and comps 2/5 are scaled by the
// lambda diagonal on every surface. One block per mode, 128 threads,
// grid-stride over surfaces.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void precon_boundary_kernel(
    cumes::SpectralView<T, cumes::DecomposedResidualDomain> f,
    const T* __restrict__ lambda_prec,
    const int* __restrict__ jMin,
    const cumes::ControlStatus* __restrict__ gate,
    int ns,
    int mnmax) {
    int mode = blockIdx.x, tid = threadIdx.x;
    if (mode >= mnmax) return;
    // Terminal gate (completion plan step 1.4): the lambda-diagonal finishing
    // no-ops on nonfinite/converged passes (gate nullptr = direct callers).
    if (gate &&
        (gate->invariant_nonfinite != 0 || gate->invariant_converged != 0))
        return;
    int jMin_m = jMin[mode];
    for (int j = tid; j < jMin_m; j += blockDim.x) {
        f(cumes::SpectralComponent::Rcc, mode, j) = T(0.0);
        f(cumes::SpectralComponent::Zsc, mode, j) = T(0.0);
        f(cumes::SpectralComponent::Lsc, mode, j) = T(0.0);
        f(cumes::SpectralComponent::Rss, mode, j) = T(0.0);
        f(cumes::SpectralComponent::Zcs, mode, j) = T(0.0);
    }
    for (int j = tid; j < ns; j += blockDim.x) {
        T lp = lambda_prec[mode * ns + j];
        f(cumes::SpectralComponent::Lsc, mode, j) *= lp;
        f(cumes::SpectralComponent::Lcs, mode, j) *= lp;
    }
}

// ---------------------------------------------------------------------------
// Host-side orchestration
// ---------------------------------------------------------------------------
template <typename T>
void cumes::Preconditioner<T>::enqueue_compute(
    const cumes::RealSpaceStorage<T>& rs,
    const int* xm,
    const int* xn,
    const DeviceParams<T>& p,
    const cumes::RadialProfileViews<T>& rpv,
    const cumes::BaseGeometryHalfViews<T>& base,
    const cumes::MagneticFieldViews<T>& field,
    const cumes::ControlStatus* status,
    cudaStream_t stream) {
    int nH = p.ns - 1, nF = p.ns;
    int threads = 256;

    // Step 1: Compute ax, bx, cx on half-grid (the 15-accumulator reduction is
    // now a warp-shuffle + fixed cross-warp combine, so no dynamic shared mem).
    precon_compute_kernel<T><<<nH, threads, 0, stream>>>(
        base.r12.data(), base.tau.data(), field.total_pressure.data(),
        field.bsupv.data(), base.gsqrt.data(), rpv.sqrtS_H, base.zs.data(),
        base.zu12.data(), rs.d_zu_e, rs.d_zu_o, rs.d_z_o, base.rs.data(),
        base.ru12.data(), rs.d_ru_e, rs.d_ru_o, rs.d_r_o, status, p.ns, p.nZnT,
        T(1.0) / T(p.ns - 1), d_ax_R_, d_ax_Z_, d_bx_R_, d_bx_Z_, d_cx_);
    cumes::check_cuda(cudaGetLastError(), "preconCompute");

    // Step 2a: Assemble off-diagonal terms + sm/sp on half-grid
    int gridH = (nH + 255) / 256;
    precon_assemble_kernel<T><<<gridH, 256, 0, stream>>>(
        d_ax_R_, d_ax_Z_, d_bx_R_, d_bx_Z_, d_cx_, rpv.sqrtS_H, rpv.sqrtS_F,
        p.ns, d_arm_, d_brm_, d_azm_, d_bzm_, d_ard_, d_brd_, d_azd_, d_bzd_,
        d_cxd_, d_sm_, d_sp_, status);
    cumes::check_cuda(cudaGetLastError(), "preconAssemble");

    // Step 2b: Average half-grid diagonals to full-grid
    int gridF = (nF + 255) / 256;
    precon_diag_kernel<T><<<gridF, 256, 0, stream>>>(
        d_ax_R_, d_ax_Z_, d_bx_R_, d_bx_Z_, d_cx_, d_sm_, d_sp_, p.ns, d_ard_,
        d_brd_, d_azd_, d_bzd_, d_cxd_, status);
    cumes::check_cuda(cudaGetLastError(), "preconDiag");

    // Step 3: Assemble tridiagonal matrices per (m,n) mode
    int total = p.mnmax * nF;
    int gridMN = (total + 255) / 256;
    tridiag_assembly_kernel<T><<<gridMN, 256, 0, stream>>>(
        d_arm_, d_brm_, d_azm_, d_bzm_, d_ard_, d_brd_, d_azd_, d_bzd_, d_cxd_,
        xm, xn, p.ns, p.mnmax, p.nfp, d_ar_, d_dr_, d_br_, d_az_, d_dz_, d_bz_,
        d_jMin_, status);
    cumes::check_cuda(cudaGetLastError(), "tridiagAssembly");

    // Step 3b: per-mode coefficient scale for the scale-aware pivot floor
    // (blueprint §4.9). Computed once per refresh, read by the solve kernels.
    precon_scale_kernel<T>
        <<<p.mnmax, 256, 0, stream>>>(d_ar_, d_dr_, d_br_, d_az_, d_dz_, d_bz_,
                                      status, p.ns, p.mnmax, d_preconScale_);
    cumes::check_cuda(cudaGetLastError(), "preconScale");

    // Step 4a/4b: Lambda diagonal preconditioner (components 2 and 5)
    {
        cumes::check_cuda(cudaMemsetAsync(d_rmsPhiP_, 0, sizeof(T), stream),
                          "rmsPhiP zero");
        lambda_prec_assemble_kernel<T><<<nH, threads, 0, stream>>>(
            base.guu.data(), base.guv.data(), base.gvv.data(),
            base.gsqrt.data(), rpv.phip_H, p.ns, p.nZnT, p.ntheta, p.nzeta,
            d_bLambda_, d_dLambda_, d_cLambda_, d_rmsPhiP_, status);
        cumes::check_cuda(cudaGetLastError(), "lambdaPrecAssemble");
        lambda_prec_finalize_kernel<T>
            <<<dim3(p.mnmax, (p.ns + 127) / 128), 128, 0, stream>>>(
                d_bLambda_, d_dLambda_, d_cLambda_, rpv.sqrtS_F, d_rmsPhiP_, xm,
                xn, p.ns, p.mnmax, T(1.0) / T(p.ns - 1), p.nfp, d_lambdaPrec_,
                status);
        cumes::check_cuda(cudaGetLastError(), "lambdaPrecFinalize");
    }
}

// ---------------------------------------------------------------------------
// Backend implementations (blueprint §8.9).
// ---------------------------------------------------------------------------

// The production PcrBackend dispatches on rhs_count to the compile-time RHS
// count so the RHS loop stays fully unrolled (bitwise parity with the legacy
// tridiagSolveKernel's `#pragma unroll`).
template <typename T, int R>
static void launch_pcr(const cumes::StridedBatchTridiagonalView<T>& m,
                       T pivot_floor_rel,
                       int* status,
                       cudaStream_t stream,
                       const cumes::ControlStatus* gate) {
    size_t smem = (size_t)(6 + 2 * R) * m.surfaces * sizeof(T);
    pcr_solve_kernel<T, R><<<m.modes, 128, smem, stream>>>(
        m.lower, m.diagonal, m.upper, m.rhs, m.first_surface, m.scale,
        pivot_floor_rel, m.modes, m.surfaces, m.last_surface, m.rhs_stride,
        status, gate);
    cumes::check_cuda(cudaGetLastError(), "pcrSolve");
}

template <typename T>
cumes::BackendLimits cumes::PcrBackend<T>::limits() const noexcept {
    // Grid-stride PCR handles arbitrary row counts; the O(ns) dynamic shared
    // memory (10*ns*sizeof(T) for rhs_count=2) is the real bound, already
    // enforced by the ns <= 512 shape validation. 0 = unbounded by the
    // algorithm.
    cumes::BackendLimits l;
    l.max_rows = 0;
    l.max_batch = 0;
    return l;
}

template <typename T>
void cumes::PcrBackend<T>::enqueue_solve(
    const cumes::StridedBatchTridiagonalView<T>& m,
    int* status,
    cudaStream_t stream,
    const cumes::ControlStatus* gate) {
    // pivot_floor_rel = kappa * epsilon_T (the caller owns the per-mode scale).
    T pivot_floor_rel = T(policy_.kappa) * T(std::numeric_limits<T>::epsilon());
    if (m.rhs_count == 1) {
        launch_pcr<T, 1>(m, pivot_floor_rel, status, stream, gate);
    } else if (m.rhs_count == 2) {
        launch_pcr<T, 2>(m, pivot_floor_rel, status, stream, gate);
    } else {
        throw cumes::CumesError("PcrBackend: rhs_count must be 1 or 2");
    }
}

template <typename T>
cumes::BackendLimits cumes::ThomasBackend<T>::limits() const noexcept {
    cumes::BackendLimits l;
    l.max_rows = 0;   // bounded by the ns <= 512 shape cap (shared memory)
    l.max_batch = 0;  // grid-stride over modes
    return l;
}

template <typename T>
void cumes::ThomasBackend<T>::enqueue_solve(
    const cumes::StridedBatchTridiagonalView<T>& m,
    int* status,
    cudaStream_t stream,
    const cumes::ControlStatus* gate) {
    T pivot_floor_rel = T(policy_.kappa) * T(std::numeric_limits<T>::epsilon());
    // One mode per block, one thread per block; O(ns) shared for cp + dp.
    size_t smem = (size_t)(1 + m.rhs_count) * m.surfaces * sizeof(T);
    thomas_solve_kernel<T><<<m.modes, 1, smem, stream>>>(
        m.lower, m.diagonal, m.upper, m.rhs, m.first_surface, m.scale,
        pivot_floor_rel, m.modes, m.surfaces, m.last_surface, m.rhs_count,
        m.rhs_stride, status, gate);
    cumes::check_cuda(cudaGetLastError(), "thomas_solve");
}

template <typename T>
void cumes::Preconditioner<T>::enqueue_apply(
    cumes::SpectralView<T, cumes::DecomposedResidualDomain> f,
    const DeviceParams<T>& p,
    const cumes::ControlStatus* gate,
    cudaStream_t stream) const {
    // Phase 8: route the tridiagonal solve through the backend-neutral
    // PcrBackend (the extracted production PCR, bit-identical to the legacy
    // tridiagSolveKernel). The R and Z systems each carry two RHS spectral
    // components sharing one elimination (rhs_count = 2).
    const size_t comp_stride = (size_t)p.mnmax * p.ns;
    cumes::StridedBatchTridiagonalView<T> rv, zv;
    rv.lower = d_br_;
    rv.diagonal = d_dr_;
    rv.upper = d_ar_;
    rv.rhs = f.data();  // comp 0 (Rcc)
    rv.first_surface = d_jMin_;
    rv.scale = d_preconScale_;
    rv.rhs_count = 2;
    rv.rhs_stride = 3 * comp_stride;  // comp 0 -> comp 3
    rv.modes = p.mnmax;
    rv.surfaces = p.ns;
    rv.last_surface = p.ns - 1;

    zv.lower = d_bz_;
    zv.diagonal = d_dz_;
    zv.upper = d_az_;
    zv.rhs = f.data() + comp_stride;  // comp 1 (Zsc)
    zv.first_surface = d_jMin_;
    zv.scale = d_preconScale_;
    zv.rhs_count = 2;
    zv.rhs_stride = 3 * comp_stride;  // comp 1 -> comp 4
    zv.modes = p.mnmax;
    zv.surfaces = p.ns;
    zv.last_surface = p.ns - 1;

    // Reset the breakdown accumulator once, then accumulate across both solves.
    cumes::check_cuda(cudaMemsetAsync(d_preconStatus_, 0, sizeof(int), stream),
                      "preconStatus zero");

    cumes::PcrBackend<T> pcr;
    pcr.enqueue_solve(rv, d_preconStatus_, stream, gate);
    pcr.enqueue_solve(zv, d_preconStatus_, stream, gate);

    // Boundary + lambda-diagonal finishing (the tail of the legacy kernel).
    // Terminal-guarded like the solves.
    precon_boundary_kernel<T><<<p.mnmax, 128, 0, stream>>>(
        f, d_lambdaPrec_, d_jMin_, gate, p.ns, p.mnmax);
    cumes::check_cuda(cudaGetLastError(), "preconBoundary");
}

// vmecpp's applyM1Preconditioner (FourierForces): scales the m=1 frss by
// (ard+brd)/denom and fzcs by (azd+bzd)/denom using the odd-parity diagonal
// precon elements. The fzcs scale matters only when the mixed fzcs is
// nonzero (fix_m1_gauge = false), i.e. for iter2 >= 2 before convergence.
// Applied right before the RZ preconditioner (after the invariant residuals).
// (Moved from kernels/solver_impl.cuh — the operator owns the PreconWorkspace
// these elements live in.)
template <typename T>
__global__ void m1_precon_scale_kernel(
    cumes::SpectralView<T, cumes::DecomposedResidualDomain> f_spec,
    const T* __restrict__ ard,
    const T* __restrict__ brd,
    const T* __restrict__ azd,
    const T* __restrict__ bzd,
    const cumes::ControlStatus* __restrict__ gate,
    int ns,
    int mnmax,
    int ntor) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ns) return;
    // Terminal gate (completion plan step 1.4): the m=1 scale no-ops on
    // nonfinite/converged passes (gate nullptr = direct callers).
    if (gate &&
        (gate->invariant_nonfinite != 0 || gate->invariant_converged != 0))
        return;
    int m1base = ntor + 1;
    T denom = ard[j * 2 + 1] + brd[j * 2 + 1] + azd[j * 2 + 1] + bzd[j * 2 + 1];
    // Degenerate-denominator guard: all-zero odd-parity precon diagonals
    // (e.g. a zero-√g surface) would make both scales NaN. Leave the forces
    // unscaled instead — the jacobian-stats check fails such surfaces before
    // this kernel normally runs.
    if (fabs(denom) < T(1e-30)) return;
    T scaleR = (ard[j * 2 + 1] + brd[j * 2 + 1]) / denom;
    T scaleZ = (azd[j * 2 + 1] + bzd[j * 2 + 1]) / denom;
    for (int n = 0; n < ntor + 1; ++n) {
        int mn = m1base + n;
        f_spec(cumes::SpectralComponent::Rss, mn, j) *= scaleR;
        f_spec(cumes::SpectralComponent::Zcs, mn, j) *= scaleZ;
    }
}

template <typename T>
void cumes::Preconditioner<T>::enqueue_m1_scale(
    cumes::SpectralView<T, cumes::DecomposedResidualDomain> residual,
    const DeviceParams<T>& p,
    const cumes::ControlStatus* gate,
    cudaStream_t stream) const {
    dim3 b1(256), g1((p.ns + 255) / 256);
    m1_precon_scale_kernel<T><<<g1, b1, 0, stream>>>(
        residual, d_ard_, d_brd_, d_azd_, d_bzd_, gate, p.ns, p.mnmax, p.ntor);
    cumes::check_cuda(cudaGetLastError(), "m1PreconScale");
}

#endif  // CUMES_SRC_PRECON_IMPL_CUH_
