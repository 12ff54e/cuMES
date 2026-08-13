// forces_impl.cuh — template definitions for forces.cuh.
// Included once per scalar type by forces_double.cu / forces_float.cu; see the
// explicit-instantiation split (cumes_cuda_double / cumes_cuda_float).
#pragma once
// forces.cu — MHD force residuals in real space, with even/odd parity.
//
// Follows vmecpp's mhdforce_kernel.h and lambda_force_kernel.h exactly,
// adapted for the single-coefficient-per-mode representation (where even
// and odd parity components are derived from the same rmnc/zmns/lmnc via
// trigonometric decomposition rather than being independent variables).
// In 3D the poloidal forces (brmn/bzmn) gain the gbubv = √g·B^θ·B^ζ cross
// terms and the toroidal forces (crmn/czmn) appear; the λ force gains the
// guv·bsupu term in the alternative interpolation and the toroidal λ force
// clmn.
//
// All computation is templated on the scalar type T (double or float).

#include "forces.cuh"
#include <cstdio>

static void checkCuda(cudaError_t err, const char* tag) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error [%s]: %s\n", tag, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// One thread per (theta,zeta) point on one full-grid surface.
template <typename T>
__global__ void forcesKernel(
    // Full-grid geometry at this surface (parity-split)
    const T* __restrict__ r_e,   const T* __restrict__ r_o,
    const T* __restrict__ z_e,   const T* __restrict__ z_o,
    const T* __restrict__ ru_e,  const T* __restrict__ ru_o,
    const T* __restrict__ zu_e,  const T* __restrict__ zu_o,
    const T* __restrict__ rv_e,  const T* __restrict__ rv_o,
    const T* __restrict__ zv_e,  const T* __restrict__ zv_o,
    // Half-grid geometry
    const T* __restrict__ r12,
    const T* __restrict__ ru12,
    const T* __restrict__ zu12,
    const T* __restrict__ rs,
    const T* __restrict__ zs,
    const T* __restrict__ tau,
    const T* __restrict__ gsqrt,
    const T* __restrict__ guv,
    const T* __restrict__ gvv,
    const T* __restrict__ bsupu,
    const T* __restrict__ bsupv,
    const T* __restrict__ bsubu,
    const T* __restrict__ bsubv,
    const T* __restrict__ totalPressure,
    // Full-grid lambda theta (decomposed, from the inverse DFT)
    const T* __restrict__ lu_e,
    const T* __restrict__ lu_o,
    // Radial profiles for parity weighting
    const T* __restrict__ sqrtS_F,   // sqrt(s) on full grid
    const T* __restrict__ sqrtS_H,   // sqrt(s) on half grid
    const T* __restrict__ phip_F,    // dPhi/ds on full grid (negative)
    T lamscale,
    int ns, int nZnT, T delta_s,
    // Output force components (parity-split)
    T* __restrict__ d_armn_e, T* __restrict__ d_armn_o,
    T* __restrict__ d_azmn_e, T* __restrict__ d_azmn_o,
    T* __restrict__ d_brmn_e, T* __restrict__ d_brmn_o,
    T* __restrict__ d_bzmn_e, T* __restrict__ d_bzmn_o,
    T* __restrict__ d_crmn_e, T* __restrict__ d_crmn_o,
    T* __restrict__ d_czmn_e, T* __restrict__ d_czmn_o,
    T* __restrict__ d_blmn_e, T* __restrict__ d_blmn_o,
    T* __restrict__ d_clmn_e, T* __restrict__ d_clmn_o)
{
    int j = blockIdx.y;  // full-grid surface
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (j >= ns || k >= nZnT) return;

    int idx_f = k + j * nZnT;

    // Full-grid geometry at this surface
    T re_j = r_e[idx_f],  ro_j = r_o[idx_f];
    T zo_j = z_o[idx_f];
    T rue_j = ru_e[idx_f], ruo_j = ru_o[idx_f];
    T zue_j = zu_e[idx_f], zuo_j = zu_o[idx_f];
    T rve_j = rv_e[idx_f], rvo_j = rv_o[idx_f];
    T zve_j = zv_e[idx_f], zvo_j = zv_o[idx_f];

    T sF_j = sqrtS_F[j];                // sqrt(s) at full grid
    T sFull = sF_j * sF_j;              // s (normalized flux)

    // ---- Get half-grid values from inside (j-1) and outside (j) ----
    T r12_i=T(0), ru12_i=T(0), zu12_i=T(0), rs_i=T(0), zs_i=T(0), tau_i=T(0);
    T gsqrt_i=T(0), bsupu_i=T(0), bsupv_i=T(0);
    T bsubu_i=T(0), bsubv_i=T(0), totalP_i=T(0);
    T sH_i = T(0), gvv_i = T(0), guv_i = T(0);

    T r12_o=T(0), ru12_o=T(0), zu12_o=T(0), rs_o=T(0), zs_o=T(0), tau_o=T(0);
    T gsqrt_o=T(0), bsupu_o=T(0), bsupv_o=T(0);
    T bsubu_o=T(0), bsubv_o=T(0), totalP_o=T(0);
    T sH_o = T(0), gvv_o = T(0), guv_o = T(0);

    if (j > 0) {
        int h_i = k + (j - 1) * nZnT;
        r12_i=r12[h_i]; ru12_i=ru12[h_i]; zu12_i=zu12[h_i];
        rs_i=rs[h_i]; zs_i=zs[h_i]; tau_i=tau[h_i];
        gsqrt_i=gsqrt[h_i]; gvv_i=gvv[h_i]; guv_i=guv[h_i];
        bsupu_i=bsupu[h_i]; bsupv_i=bsupv[h_i];
        bsubu_i=bsubu[h_i]; bsubv_i=bsubv[h_i];
        totalP_i=totalPressure[h_i];
        sH_i = sqrtS_H[j - 1];
    }
    if (j < ns - 1) {
        int h_o = k + j * nZnT;
        r12_o=r12[h_o]; ru12_o=ru12[h_o]; zu12_o=zu12[h_o];
        rs_o=rs[h_o]; zs_o=zs[h_o]; tau_o=tau[h_o];
        gsqrt_o=gsqrt[h_o]; gvv_o=gvv[h_o]; guv_o=guv[h_o];
        bsupu_o=bsupu[h_o]; bsupv_o=bsupv[h_o];
        bsubu_o=bsubu[h_o]; bsubv_o=bsubv[h_o];
        totalP_o=totalPressure[h_o];
        sH_o = sqrtS_H[j];
    }

    // ---- Pressure-weighted intermediates --------------------------------
    T P_i = r12_i * totalP_i,  P_o = r12_o * totalP_o;
    T zup_i = zu12_i * P_i,  zup_o = zu12_o * P_o;
    T rup_i = ru12_i * P_i,  rup_o = ru12_o * P_o;
    T rsp_i = rs_i * P_i,    rsp_o = rs_o * P_o;
    T zsp_i = zs_i * P_i,    zsp_o = zs_o * P_o;
    T taup_i = tau_i * totalP_i, taup_o = tau_o * totalP_o;

    T gbubu_i = gsqrt_i * bsupu_i * bsupu_i;
    T gbubu_o = gsqrt_o * bsupu_o * bsupu_o;
    T gbvbv_i = gsqrt_i * bsupv_i * bsupv_i;
    T gbvbv_o = gsqrt_o * bsupv_o * bsupv_o;
    T gbubv_i = gsqrt_i * bsupu_i * bsupv_i;
    T gbubv_o = gsqrt_o * bsupu_o * bsupv_o;

    // ---- Arithmetic and sqrt(s)-weighted averages ----------------------
    T inv_ds = T(1.0) / delta_s;
    T inv_sH_i = (j > 0) ? (T(1.0) / sH_i) : T(0.0);
    T inv_sH_o = (j < ns - 1) ? (T(1.0) / sH_o) : T(0.0);

    T P_avg = T(0.5) * (P_o + P_i);
    T P_wavg = T(0.5) * (P_o * inv_sH_o + P_i * inv_sH_i);
    T gbubu_avg = T(0.5) * (gbubu_o + gbubu_i);
    T gbubu_wavg = T(0.5) * (gbubu_o * sH_o + gbubu_i * sH_i);
    T gbvbv_avg = T(0.5) * (gbvbv_o + gbvbv_i);
    T gbvbv_wavg = T(0.5) * (gbvbv_o * sH_o + gbvbv_i * sH_i);
    T gbubv_avg = T(0.5) * (gbubv_o + gbubv_i);
    T gbubv_wavg = T(0.5) * (gbubv_o * sH_o + gbubv_i * sH_i);

    // ---- Radial R-force (armn): even and odd parity --------------------
    T armn_e = (zup_o - zup_i) * inv_ds
                  + T(0.5) * (taup_o + taup_i)
                  - gbvbv_avg * re_j - gbvbv_wavg * ro_j;

    T armn_o = (zup_o * sH_o - zup_i * sH_i) * inv_ds
                  - T(0.5) * P_wavg * zue_j - T(0.5) * P_avg * zuo_j
                  + T(0.5) * (taup_o * sH_o + taup_i * sH_i)
                  - gbvbv_wavg * re_j - gbvbv_avg * ro_j * sFull;

    // ---- Radial Z-force (azmn) -----------------------------------------
    T azmn_e = -(rup_o - rup_i) * inv_ds;

    T azmn_o = -(rup_o * sH_o - rup_i * sH_i) * inv_ds
                  + T(0.5) * P_wavg * rue_j + T(0.5) * P_avg * ruo_j;

    // ---- Poloidal R-force (brmn) ---------------------------------------
    T brmn_e = T(0.5) * (zsp_o + zsp_i)
                  + T(0.5) * P_wavg * zo_j
                  - gbubu_avg * rue_j - gbubu_wavg * ruo_j
                  - gbubv_avg * rve_j - gbubv_wavg * rvo_j;

    T brmn_o = T(0.5) * (zsp_o * sH_o + zsp_i * sH_i)
                  + T(0.5) * P_avg * zo_j
                  - gbubu_wavg * rue_j - gbubu_avg * ruo_j * sFull
                  - gbubv_wavg * rve_j - gbubv_avg * rvo_j * sFull;

    // ---- Poloidal Z-force (bzmn) ---------------------------------------
    T bzmn_e = -T(0.5) * (rsp_o + rsp_i)
                  - T(0.5) * P_wavg * ro_j
                  - gbubu_avg * zue_j - gbubu_wavg * zuo_j
                  - gbubv_avg * zve_j - gbubv_wavg * zvo_j;

    T bzmn_o = -T(0.5) * (rsp_o * sH_o + rsp_i * sH_i)
                  - T(0.5) * P_avg * ro_j
                  - gbubu_wavg * zue_j - gbubu_avg * zuo_j * sFull
                  - gbubv_wavg * zve_j - gbubv_avg * zvo_j * sFull;

    // ---- Toroidal forces (crmn/czmn, 3D) --------------------------------
    T crmn_e = gbubv_avg * rue_j + gbubv_wavg * ruo_j
                  + gbvbv_avg * rve_j + gbvbv_wavg * rvo_j;
    T crmn_o = gbubv_wavg * rue_j + gbubv_avg * ruo_j * sFull
                  + gbvbv_wavg * rve_j + gbvbv_avg * rvo_j * sFull;
    T czmn_e = gbubv_avg * zue_j + gbubv_wavg * zuo_j
                  + gbvbv_avg * zve_j + gbvbv_wavg * zvo_j;
    T czmn_o = gbubv_wavg * zue_j + gbubv_avg * zuo_j * sFull
                  + gbvbv_wavg * zve_j + gbvbv_avg * zvo_j * sFull;

    // ---- Lambda force (blmn): hybrid with radial blending ------------
    // Matches vmecpp's lambda_force_kernel.h: ComputeHybridLambdaForce.
    // For the first iteration (lambda=0), this reduces to:
    //   blmn = -lamscale * 0.5*(bsubv_o + bsubv_i)
    // Full formula (with lambda) blends two bsubv interpolation schemes
    // with radialBlending weights and includes lamscale factor. In 3D the
    // alternative interpolation adds the guv*bsupu term.
    //
    // lamscale = sqrt(rmsPhiP * deltaS), rmsPhiP = sum phipH^2.
    T bsubv_avg = T(0.5) * (bsubv_o + bsubv_i);
    // Note: at j==0, bsubv_i=0 (no inside half-grid), so bsubv_avg=0.5*bsubv_o.

    // The "alternative" bsubv interpolation reconstructs bsubv from the
    // half-grid gvv/gsqrt (+guv/bsupu in 3D) and the NORMALIZED lambda
    // derivative: vmecpp's computeBContra mutates lu in place to
    // lamscale*lu + phipF (phipF = dPhi/ds < 0) before this kernel runs, so
    // the alternative carries the same normalization. The odd part uses the
    // sqrt(s_H)-weighted half-grid average, matching vmecpp exactly. The two
    // interpolations are blended with radialBlending = 2*kPDamp*(1-s),
    // kPDamp = 0.05.
    T gvv_gsqrt_i = (j > 0)    ? (gvv_i / gsqrt_i) : T(0.0);
    T gvv_gsqrt_o = (j < ns-1) ? (gvv_o / gsqrt_o) : T(0.0);
    T guv_bsupu_i = (j > 0)    ? (guv_i * bsupu_i) : T(0.0);
    T guv_bsupu_o = (j < ns-1) ? (guv_o * bsupu_o) : T(0.0);
    T lu_e_norm = lamscale * lu_e[idx_f] + phip_F[j];
    T lu_o_norm = lamscale * lu_o[idx_f];
    T bsubv_alt = T(0.5) * (gvv_gsqrt_i + gvv_gsqrt_o) * lu_e_norm
                     + T(0.5) * (gvv_gsqrt_i * sH_i + gvv_gsqrt_o * sH_o) * lu_o_norm
                     + T(0.5) * (guv_bsupu_i + guv_bsupu_o);
    T rb = T(2.0) * T(0.05) * (T(1.0) - sFull);
    T _blmn = bsubv_avg * (T(1.0) - rb) + bsubv_alt * rb;
    // MINUS SIGN => HESSIAN DIAGONALS ARE POSITIVE (vmecpp comment)
    if (j > 0) {
        _blmn *= -lamscale;
    }
    T blmn_e = _blmn;                  // positive at j=0, matching vmecpp
    T blmn_o = _blmn * sF_j;           // odd parity scales with sqrt(s)

    // ---- Toroidal lambda force (clmn, 3D) -------------------------------
    // clmn = -lamscale * 0.5*(bsubu_o + bsubu_i) for j>0 (no blending);
    // positive at j=0 like blmn.
    T _clmn = T(0.5) * (bsubu_o + bsubu_i);
    if (j > 0) {
        _clmn *= -lamscale;
    }
    T clmn_e = _clmn;
    T clmn_o = _clmn * sF_j;

    // ---- Store ----------------------------------------------------------
    d_armn_e[idx_f] = armn_e;  d_armn_o[idx_f] = armn_o;
    d_azmn_e[idx_f] = azmn_e;  d_azmn_o[idx_f] = azmn_o;
    d_brmn_e[idx_f] = brmn_e;  d_brmn_o[idx_f] = brmn_o;
    d_bzmn_e[idx_f] = bzmn_e;  d_bzmn_o[idx_f] = bzmn_o;
    d_crmn_e[idx_f] = crmn_e;  d_crmn_o[idx_f] = crmn_o;
    d_czmn_e[idx_f] = czmn_e;  d_czmn_o[idx_f] = czmn_o;
    d_blmn_e[idx_f] = blmn_e;  d_blmn_o[idx_f] = blmn_o;
    d_clmn_e[idx_f] = clmn_e;  d_clmn_o[idx_f] = clmn_o;
}

template <typename T>
void computeForces(const FourierPlan<T>& fp, const GridParams<T>& p,
                   const RadialProfiles<T>& rp, const MetricWorkspace<T>& mw) {
    dim3 block(128);
    dim3 grid((p.nZnT + 127) / 128, p.ns);
    forcesKernel<T><<<grid, block>>>(
        fp.d_r_e, fp.d_r_o,
        fp.d_z_e, fp.d_z_o,
        fp.d_ru_e, fp.d_ru_o,
        fp.d_zu_e, fp.d_zu_o,
        fp.d_rv_e, fp.d_rv_o,
        fp.d_zv_e, fp.d_zv_o,
        mw.d_r12, mw.d_ru12, mw.d_zu12, mw.d_rs, mw.d_zs, mw.d_tau,
        mw.d_gsqrt, mw.d_guv, mw.d_gvv,
        mw.d_bsupu, mw.d_bsupv,
        mw.d_bsubu, mw.d_bsubv,
        mw.d_totalPressure,
        fp.d_lu_e, fp.d_lu_o,
        rp.d_sqrtS_F, rp.d_sqrtS_H, rp.d_phip_F, p.lamscale,
        p.ns, p.nZnT, rp.delta_s,
        fp.d_armn_e, fp.d_armn_o,
        fp.d_azmn_e, fp.d_azmn_o,
        fp.d_brmn_e, fp.d_brmn_o,
        fp.d_bzmn_e, fp.d_bzmn_o,
        fp.d_crmn_e, fp.d_crmn_o,
        fp.d_czmn_e, fp.d_czmn_o,
        fp.d_blmn_e, fp.d_blmn_o,
        fp.d_clmn_e, fp.d_clmn_o);
    checkCuda(cudaGetLastError(), "forces kernel");
}

