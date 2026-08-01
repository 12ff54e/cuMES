// forces.cu — MHD force residuals in real space, with even/odd parity.
//
// Follows vmecpp's mhdforce_kernel.h and lambda_force_kernel.h exactly,
// adapted for the single-coefficient-per-mode representation (where even
// and odd parity components are derived from the same rmnc/zmns/lmnc via
// trigonometric decomposition rather than being independent variables).

#include "forces.cuh"
#include <cstdio>

static void checkCuda(cudaError_t err, const char* tag) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error [%s]: %s\n", tag, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// One thread per (theta,zeta) point on one full-grid surface.
__global__ void forcesKernel(
    // Full-grid geometry at this surface (parity-split)
    const double* __restrict__ r_e,   const double* __restrict__ r_o,
    const double* __restrict__ z_e,   const double* __restrict__ z_o,
    const double* __restrict__ ru_e,  const double* __restrict__ ru_o,
    const double* __restrict__ zu_e,  const double* __restrict__ zu_o,
    // Half-grid geometry
    const double* __restrict__ r12,
    const double* __restrict__ ru12,
    const double* __restrict__ zu12,
    const double* __restrict__ rs,
    const double* __restrict__ zs,
    const double* __restrict__ tau,
    const double* __restrict__ gsqrt,
    const double* __restrict__ gvv,
    const double* __restrict__ bsupu,
    const double* __restrict__ bsupv,
    const double* __restrict__ bsubu,
    const double* __restrict__ bsubv,
    const double* __restrict__ totalPressure,
    // Full-grid lambda theta (decomposed, from the inverse DFT)
    const double* __restrict__ lu_e,
    const double* __restrict__ lu_o,
    // Radial profiles for parity weighting
    const double* __restrict__ sqrtS_F,   // sqrt(s) on full grid
    const double* __restrict__ sqrtS_H,   // sqrt(s) on half grid
    const double* __restrict__ phip_F,    // dPhi/ds on full grid (negative)
    int ns, int nZnT, double delta_s,
    // Output force components (parity-split)
    double* __restrict__ d_armn_e, double* __restrict__ d_armn_o,
    double* __restrict__ d_azmn_e, double* __restrict__ d_azmn_o,
    double* __restrict__ d_brmn_e, double* __restrict__ d_brmn_o,
    double* __restrict__ d_bzmn_e, double* __restrict__ d_bzmn_o,
    double* __restrict__ d_blmn_e, double* __restrict__ d_blmn_o)
{
    int j = blockIdx.y;  // full-grid surface
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (j >= ns || k >= nZnT) return;

    int idx_f = k + j * nZnT;

    // Full-grid geometry at this surface
    double re_j = r_e[idx_f],  ro_j = r_o[idx_f];
    double ze_j = z_e[idx_f],  zo_j = z_o[idx_f];
    double rue_j = ru_e[idx_f], ruo_j = ru_o[idx_f];
    double zue_j = zu_e[idx_f], zuo_j = zu_o[idx_f];

    double sF_j = sqrtS_F[j];                // sqrt(s) at full grid
    double sFull = sF_j * sF_j;              // s (normalized flux)

    // ---- Get half-grid values from inside (j-1) and outside (j) ----
    double r12_i=0, ru12_i=0, zu12_i=0, rs_i=0, zs_i=0, tau_i=0;
    double gsqrt_i=0, bsupu_i=0, bsupv_i=0;
    double bsubu_i=0, bsubv_i=0, totalP_i=0;
    double sH_i = 0, gvv_i = 0;

    double r12_o=0, ru12_o=0, zu12_o=0, rs_o=0, zs_o=0, tau_o=0;
    double gsqrt_o=0, bsupu_o=0, bsupv_o=0;
    double bsubu_o=0, bsubv_o=0, totalP_o=0;
    double sH_o = 0, gvv_o = 0;

    if (j > 0) {
        int h_i = k + (j - 1) * nZnT;
        r12_i=r12[h_i]; ru12_i=ru12[h_i]; zu12_i=zu12[h_i];
        rs_i=rs[h_i]; zs_i=zs[h_i]; tau_i=tau[h_i];
        gsqrt_i=gsqrt[h_i]; gvv_i=gvv[h_i];
        bsupu_i=bsupu[h_i]; bsupv_i=bsupv[h_i];
        bsubu_i=bsubu[h_i]; bsubv_i=bsubv[h_i];
        totalP_i=totalPressure[h_i];
        sH_i = sqrtS_H[j - 1];
    }
    if (j < ns - 1) {
        int h_o = k + j * nZnT;
        r12_o=r12[h_o]; ru12_o=ru12[h_o]; zu12_o=zu12[h_o];
        rs_o=rs[h_o]; zs_o=zs[h_o]; tau_o=tau[h_o];
        gsqrt_o=gsqrt[h_o]; gvv_o=gvv[h_o];
        bsupu_o=bsupu[h_o]; bsupv_o=bsupv[h_o];
        bsubu_o=bsubu[h_o]; bsubv_o=bsubv[h_o];
        totalP_o=totalPressure[h_o];
        sH_o = sqrtS_H[j];
    }

    // ---- Pressure-weighted intermediates --------------------------------
    double P_i = r12_i * totalP_i,  P_o = r12_o * totalP_o;
    double zup_i = zu12_i * P_i,  zup_o = zu12_o * P_o;
    double rup_i = ru12_i * P_i,  rup_o = ru12_o * P_o;
    double rsp_i = rs_i * P_i,    rsp_o = rs_o * P_o;
    double zsp_i = zs_i * P_i,    zsp_o = zs_o * P_o;
    double taup_i = tau_i * totalP_i, taup_o = tau_o * totalP_o;

    double gbubu_i = gsqrt_i * bsupu_i * bsupu_i;
    double gbubu_o = gsqrt_o * bsupu_o * bsupu_o;
    double gbvbv_i = gsqrt_i * bsupv_i * bsupv_i;
    double gbvbv_o = gsqrt_o * bsupv_o * bsupv_o;

    // ---- Arithmetic and sqrt(s)-weighted averages ----------------------
    double inv_ds = 1.0 / delta_s;
    double inv_sH_i = (j > 0) ? (1.0 / sH_i) : 0.0;
    double inv_sH_o = (j < ns - 1) ? (1.0 / sH_o) : 0.0;

    double P_avg = 0.5 * (P_o + P_i);
    double P_wavg = 0.5 * (P_o * inv_sH_o + P_i * inv_sH_i);
    double gbubu_avg = 0.5 * (gbubu_o + gbubu_i);
    double gbubu_wavg = 0.5 * (gbubu_o * sH_o + gbubu_i * sH_i);
    double gbvbv_avg = 0.5 * (gbvbv_o + gbvbv_i);
    double gbvbv_wavg = 0.5 * (gbvbv_o * sH_o + gbvbv_i * sH_i);

    // ---- Radial R-force (armn): even and odd parity --------------------
    double armn_e = (zup_o - zup_i) * inv_ds
                  + 0.5 * (taup_o + taup_i)
                  - gbvbv_avg * re_j - gbvbv_wavg * ro_j;

    double armn_o = (zup_o * sH_o - zup_i * sH_i) * inv_ds
                  - 0.5 * P_wavg * zue_j - 0.5 * P_avg * zuo_j
                  + 0.5 * (taup_o * sH_o + taup_i * sH_i)
                  - gbvbv_wavg * re_j - gbvbv_avg * ro_j * sFull;

    // ---- Radial Z-force (azmn) -----------------------------------------
    double azmn_e = -(rup_o - rup_i) * inv_ds;

    double azmn_o = -(rup_o * sH_o - rup_i * sH_i) * inv_ds
                  + 0.5 * P_wavg * rue_j + 0.5 * P_avg * ruo_j;

    // ---- Poloidal R-force (brmn) ---------------------------------------
    double brmn_e = 0.5 * (zsp_o + zsp_i)
                  + 0.5 * P_wavg * zo_j
                  - gbubu_avg * rue_j - gbubu_wavg * ruo_j;

    double brmn_o = 0.5 * (zsp_o * sH_o + zsp_i * sH_i)
                  + 0.5 * P_avg * zo_j
                  - gbubu_wavg * rue_j - gbubu_avg * ruo_j * sFull;

    // ---- Poloidal Z-force (bzmn) ---------------------------------------
    double bzmn_e = -0.5 * (rsp_o + rsp_i)
                  - 0.5 * P_wavg * ro_j
                  - gbubu_avg * zue_j - gbubu_wavg * zuo_j;

    double bzmn_o = -0.5 * (rsp_o * sH_o + rsp_i * sH_i)
                  - 0.5 * P_avg * ro_j
                  - gbubu_wavg * zue_j - gbubu_avg * zuo_j * sFull;

    // ---- Lambda force (blmn): hybrid with radial blending ------------
    // Matches vmecpp's lambda_force_kernel.h: ComputeHybridLambdaForce.
    // For the first iteration (lambda=0), this reduces to:
    //   blmn = -lamscale * 0.5*(bsubv_o + bsubv_i)
    // Full formula (with lambda) blends two bsubv interpolation schemes
    // with radialBlending weights and includes lamscale factor.
    //
    // lamscale = sqrt(rmsPhiP * deltaS) = sqrt(phipRMS² * deltaS)
    // For phip = 1/(2*pi): rmsPhiP² ≈ 10 * (1/(2pi))², deltaS=0.1
    // lamscale ≈ sqrt(0.0253 * 0.1) ≈ 0.159
    // lamscale = sqrt(rmsPhiP * deltaS) where rmsPhiP = sum(phipH²) over half-grid
    // For phipH = 1/(2*pi): rmsPhiP = (ns-1) * (1/(2*pi))²
    // lamscale = sqrt((ns-1) * (1/(2*pi))² * deltaS)
    //          = (1/(2*pi)) * sqrt((ns-1) * deltaS)
    //          = (1/(2*pi)) * sqrt(1.0)  [since (ns-1)*deltaS = 1.0]
    //          = 1/(2*pi)
    double lamscale = 1.0 / (2.0 * M_PI);  // for constant phip = 1/(2*pi)

    double bsubv_avg = 0.5 * (bsubv_o + bsubv_i);
    // Note: at j==0, bsubv_i=0 (no inside half-grid), so bsubv_avg=0.5*bsubv_o.

    // Hybrid lambda force (vmecpp lambda_force_kernel.h, verified to machine
    // precision against kernel-internal dumps of the vmecpp binary at the
    // iter-150 handoff state). The "alternative" bsubv interpolation
    // reconstructs bsubv from the half-grid gvv/gsqrt and the NORMALIZED
    // lambda derivative: vmecpp's computeBContra mutates lu in place to
    // lamscale*lu + phipF (phipF = dPhi/ds < 0) before this kernel runs, so
    // the alternative carries the same normalization. The odd part uses the
    // sqrt(s_H)-weighted half-grid average, matching vmecpp exactly. The two
    // interpolations are blended with radialBlending = 2*kPDamp*(1-s),
    // kPDamp = 0.05 (profile [0.1, 0.09, ..., 0] confirmed in the binary).
    double gvv_gsqrt_i = (j > 0)    ? (gvv_i / gsqrt_i) : 0.0;
    double gvv_gsqrt_o = (j < ns-1) ? (gvv_o / gsqrt_o) : 0.0;
    double lu_e_norm = lamscale * lu_e[idx_f] + phip_F[j];
    double lu_o_norm = lamscale * lu_o[idx_f];
    double bsubv_alt = 0.5 * (gvv_gsqrt_i + gvv_gsqrt_o) * lu_e_norm
                     + 0.5 * (gvv_gsqrt_i * sH_i + gvv_gsqrt_o * sH_o) * lu_o_norm;
    double rb = 2.0 * 0.05 * (1.0 - sFull);
    double _blmn = bsubv_avg * (1.0 - rb) + bsubv_alt * rb;
    // MINUS SIGN => HESSIAN DIAGONALS ARE POSITIVE (vmecpp comment)
    if (j > 0) {
        _blmn *= -lamscale;
    }
    double blmn_e = _blmn;                  // positive at j=0, matching vmecpp
    double blmn_o = _blmn * sF_j;           // odd parity scales with sqrt(s)

    // ---- Store ----------------------------------------------------------
    d_armn_e[idx_f] = armn_e;  d_armn_o[idx_f] = armn_o;
    d_azmn_e[idx_f] = azmn_e;  d_azmn_o[idx_f] = azmn_o;
    d_brmn_e[idx_f] = brmn_e;  d_brmn_o[idx_f] = brmn_o;
    d_bzmn_e[idx_f] = bzmn_e;  d_bzmn_o[idx_f] = bzmn_o;
    d_blmn_e[idx_f] = blmn_e;  d_blmn_o[idx_f] = blmn_o;
}

void computeForces(const FourierPlan& fp, const GridParams& p,
                   const RadialProfiles& rp, const MetricWorkspace& mw) {
    dim3 block(32);
    dim3 grid((p.nZnT + 31) / 32, p.ns);
    forcesKernel<<<grid, block>>>(
        fp.d_r_e, fp.d_r_o,
        fp.d_z_e, fp.d_z_o,
        fp.d_ru_e, fp.d_ru_o,
        fp.d_zu_e, fp.d_zu_o,
        mw.d_r12, mw.d_ru12, mw.d_zu12, mw.d_rs, mw.d_zs, mw.d_tau,
        mw.d_gsqrt, mw.d_gvv,
        mw.d_bsupu, mw.d_bsupv,
        mw.d_bsubu, mw.d_bsubv,
        mw.d_totalPressure,
        fp.d_lu_e, fp.d_lu_o,
        rp.d_sqrtS_F, rp.d_sqrtS_H, rp.d_phip_F,
        p.ns, p.nZnT, rp.delta_s,
        fp.d_armn_e, fp.d_armn_o,
        fp.d_azmn_e, fp.d_azmn_o,
        fp.d_brmn_e, fp.d_brmn_o,
        fp.d_bzmn_e, fp.d_bzmn_o,
        fp.d_blmn_e, fp.d_blmn_o);
    checkCuda(cudaGetLastError(), "forces kernel");
    checkCuda(cudaDeviceSynchronize(), "forces sync");
}
