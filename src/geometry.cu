// geometry.cu — Jacobian, metric, magnetic field, and total pressure
// on the staggered half-grid with even/odd parity decomposition.
//
// Follows the full VMEC formulation from:
//   jacobian_kernel.h  — tau = tau1 + dSHalfDsInterp * tau2
//   metric_kernel.h    — parity-mixed metric elements
//   bcontra_kernel.h   — B^contra from lambda derivatives + flux
//   bco_kernel.h       — index lowering B^contra → B_cov
//   pressure_kernel.h  — total pressure = |B|^2/2 + p

#include "geometry.cuh"
#include "fourier.cuh"
#include "profiles.cuh"
#include <cstdio>

static void checkCuda(cudaError_t err, const char* tag) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error [%s]: %s\n", tag, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

MetricWorkspace metricCreate(const GridParams& p) {
    MetricWorkspace mw{};
    size_t nH = (p.ns - 1) * p.nZnT * sizeof(double);

    checkCuda(cudaMalloc(&mw.d_r12,  nH), "r12");
    checkCuda(cudaMalloc(&mw.d_ru12, nH), "ru12");
    checkCuda(cudaMalloc(&mw.d_zu12, nH), "zu12");
    checkCuda(cudaMalloc(&mw.d_rs,   nH), "rs");
    checkCuda(cudaMalloc(&mw.d_zs,   nH), "zs");
    checkCuda(cudaMalloc(&mw.d_tau,  nH), "tau");
    checkCuda(cudaMalloc(&mw.d_gsqrt, nH), "gsqrt");
    checkCuda(cudaMalloc(&mw.d_guu,   nH), "guu");
    checkCuda(cudaMalloc(&mw.d_guv,   nH), "guv");
    checkCuda(cudaMalloc(&mw.d_gvv,   nH), "gvv");
    checkCuda(cudaMalloc(&mw.d_bsupu, nH), "bsupu");
    checkCuda(cudaMalloc(&mw.d_bsupv, nH), "bsupv");
    checkCuda(cudaMalloc(&mw.d_bsubu, nH), "bsubu");
    checkCuda(cudaMalloc(&mw.d_bsubv, nH), "bsubv");
    checkCuda(cudaMalloc(&mw.d_totalPressure, nH), "totalP");

    return mw;
}

void metricFree(MetricWorkspace& mw) {
    cudaFree(mw.d_r12);  cudaFree(mw.d_ru12); cudaFree(mw.d_zu12);
    cudaFree(mw.d_rs);   cudaFree(mw.d_zs);   cudaFree(mw.d_tau);
    cudaFree(mw.d_gsqrt);
    cudaFree(mw.d_guu);  cudaFree(mw.d_guv);  cudaFree(mw.d_gvv);
    cudaFree(mw.d_bsupu); cudaFree(mw.d_bsupv);
    cudaFree(mw.d_bsubu); cudaFree(mw.d_bsubv);
    cudaFree(mw.d_totalPressure);
}

// ---- geometry kernel ----------------------------------------------------
// One thread per (theta,zeta) point, one block per half-grid surface.
//
// Parity mixing: half-grid values combine even and odd parity components
// from neighbouring full-grid surfaces with sqrtSH weighting.
//
//   r12  = 0.5*[(re_i + re_o) + sH*(ro_i + ro_o)]
//   tau1 = ru12*zs - rs*zu12
//   tau2 = (ruo_o*zo_o + ruo_i*zo_i - zuo_o*ro_o - zuo_i*ro_i)
//        + (rue_o*zo_o + rue_i*zo_i - zue_o*ro_o - zue_i*ro_i)/sH
//   tau  = tau1 + 0.25 * tau2
//   gsqrt = tau * r12

__global__ void geometryKernel(
    // Full-grid geometry, even parity: (nZnT, ns) col-major
    const double* __restrict__ r_e,  const double* __restrict__ r_o,
    const double* __restrict__ z_e,  const double* __restrict__ z_o,
    const double* __restrict__ ru_e, const double* __restrict__ ru_o,
    const double* __restrict__ zu_e, const double* __restrict__ zu_o,
    const double* __restrict__ lu_e, const double* __restrict__ lu_o,
    // Full-grid sqrt(s) for parity mixing
    const double* __restrict__ sqrtS_F,   // (ns,)
    // Half-grid profiles
    const double* __restrict__ sqrtS_H,   // (ns-1,)
    const double* __restrict__ iota_H,
    const double* __restrict__ phip_H,
    const double* __restrict__ pres_H,
    int ns, int nZnT, double delta_s,
    // half-grid outputs (all (ns-1, nZnT) col-major)
    double* __restrict__ r12,   double* __restrict__ ru12,
    double* __restrict__ zu12,  double* __restrict__ rs,
    double* __restrict__ zs,    double* __restrict__ tau,
    double* __restrict__ gsqrt, double* __restrict__ guu,
    double* __restrict__ guv,   double* __restrict__ gvv,
    double* __restrict__ bsupu, double* __restrict__ bsupv,
    double* __restrict__ bsubu, double* __restrict__ bsubv,
    double* __restrict__ totalPressure)
{
    int jH = blockIdx.y;   // half-grid surface index (0 .. ns-2)
    int k   = threadIdx.x + blockIdx.x * blockDim.x;
    if (jH >= ns - 1 || k >= nZnT) return;

    // Full-grid surface indices: jH (inside), jH+1 (outside)
    int i_in  = k + (jH) * nZnT;
    int i_out = k + (jH + 1) * nZnT;

    // s = sqrt(s) values
    double sH = sqrtS_H[jH];                   // half-grid sqrt(s)
    double sF_i = sqrtS_F[jH];                 // full-grid sqrt(s) at jH
    double sF_o = sqrtS_F[jH + 1];             // full-grid sqrt(s) at jH+1

    // ---- half-grid interpolation with parity mixing --------------------
    double r12_v  = 0.5 * ((r_e[i_in]  + r_e[i_out]) + sH * (r_o[i_in]  + r_o[i_out]));
    double ru12_v = 0.5 * ((ru_e[i_in] + ru_e[i_out])+ sH * (ru_o[i_in] + ru_o[i_out]));
    double zu12_v = 0.5 * ((zu_e[i_in] + zu_e[i_out])+ sH * (zu_o[i_in] + zu_o[i_out]));

    // Radial derivatives with parity mixing
    double rs_v = ((r_e[i_out] - r_e[i_in]) + sH * (r_o[i_out] - r_o[i_in])) / delta_s;
    double zs_v = ((z_e[i_out] - z_e[i_in]) + sH * (z_o[i_out] - z_o[i_in])) / delta_s;

    // ---- Jacobian: tau = tau1 + dSHalfDsInterp * tau2 ------------------
    double tau1 = ru12_v * zs_v - rs_v * zu12_v;

    // tau2: odd-parity contribution that keeps Jacobian positive everywhere
    // (prevents sign change at the inboard side of the torus)
    double tau2 =
        ru_o[i_out] * z_o[i_out] + ru_o[i_in] * z_o[i_in]
      - zu_o[i_out] * r_o[i_out] - zu_o[i_in] * r_o[i_in]
      + (ru_e[i_out] * z_o[i_out] + ru_e[i_in] * z_o[i_in]
      -  zu_e[i_out] * r_o[i_out] - zu_e[i_in] * r_o[i_in]) / sH;

    constexpr double dSHalfDsInterp = 0.25;  // 1/2 * 1/2 from interpolation
    double tau_v = tau1 + dSHalfDsInterp * tau2;

    // Regularized Jacobian: sqrt(g) = R * sqrt(tau^2 + eps^2)
    // where tau = ru*zs - rs*zu (+ parity corrections).
    // The regularization eps ensures g > 0 even at the magnetic axis
    // or on the inboard side where tau -> 0.
    // eps = 0.01 is small enough to not affect the equilibrium
    // but large enough to prevent near-zero Jacobian singularities.
    // Jacobian: gsqrt = tau * r12  (matching vmecpp metric_kernel.h)
    // vmecpp allows negative gsqrt (signOfJacobian = -1 convention).
    // The sign carries through to gbvbv = gsqrt*bsupv² and affects forces.
    double gsqrt_v = tau_v * r12_v;

    // ---- covariant metric with parity mixing ---------------------------
    // sF = s = sqrtSF^2
    double sFi_sq = sF_i * sF_i;
    double sFo_sq = sF_o * sF_o;

    double guu_v = 0.5 * ((ru_e[i_in]*ru_e[i_in] + zu_e[i_in]*zu_e[i_in]) +
                          (ru_e[i_out]*ru_e[i_out] + zu_e[i_out]*zu_e[i_out]) +
                          sFi_sq * (ru_o[i_in]*ru_o[i_in] + zu_o[i_in]*zu_o[i_in]) +
                          sFo_sq * (ru_o[i_out]*ru_o[i_out] + zu_o[i_out]*zu_o[i_out]))
                 + sH * ((ru_e[i_in]*ru_o[i_in] + zu_e[i_in]*zu_o[i_in]) +
                         (ru_e[i_out]*ru_o[i_out] + zu_e[i_out]*zu_o[i_out]));

    double gvv_v = 0.5 * (r_e[i_in]*r_e[i_in] + r_e[i_out]*r_e[i_out] +
                          sFi_sq * r_o[i_in]*r_o[i_in] +
                          sFo_sq * r_o[i_out]*r_o[i_out])
                 + sH * (r_e[i_in]*r_o[i_in] + r_e[i_out]*r_o[i_out]);

    double guv_v = 0.0;  // axisymmetric: no toroidal coupling in metric

    // ---- contravariant B with lambda + flux contribution ---------------
    // lu (lambda derivative) on half-grid with parity mixing
    double lu_h = 0.5 * ((lu_e[i_in] + lu_e[i_out]) + sH * (lu_o[i_in] + lu_o[i_out]));

    double iota = iota_H[jH];
    double phip = phip_H[jH];
    double pres = pres_H[jH];

    // lamscale = sqrt(rmsPhiP * deltaS); for the constant-phip Solovev case
    // rmsPhiP = (ns-1)*(1/(2pi))^2 and (ns-1)*deltaS = 1 -> lamscale = 1/(2pi)
    // (matches vmecpp constants_.lamscale and forces.cu's local lamscale).
    double lamscale = 1.0 / (2.0 * M_PI);

    // B^θ = chi' / √g = ι·Φ' / √g
    double bsupu_v = iota * phip / gsqrt_v;
    // B^ζ = (lamscale·dλ/dθ + Φ') / √g, matching vmecpp computeBContra.
    // The lambda derivatives lu_e/lu_o already carry the mscale = sqrt(2)
    // basis factor and the odd-m decomposition (applied in the inverse DFT,
    // matching vmecpp's real-space convention) — no extra factor here.
    double bsupv_v = (lamscale * lu_h + phip) / gsqrt_v;

    // ---- covariant B (index lowering) -----------------------------------
    double bsubu_v = guu_v * bsupu_v + guv_v * bsupv_v;
    double bsubv_v = guv_v * bsupu_v + gvv_v * bsupv_v;

    // ---- total pressure: |B|²/2 + p ------------------------------------
    double bsq_half = 0.5 * (bsupu_v * bsubu_v + bsupv_v * bsubv_v);
    double totalP = bsq_half + pres;

    // ---- store ----------------------------------------------------------
    int idx_out = k + jH * nZnT;
    r12[idx_out]  = r12_v;
    ru12[idx_out] = ru12_v;
    zu12[idx_out] = zu12_v;
    rs[idx_out]   = rs_v;
    zs[idx_out]   = zs_v;
    tau[idx_out]  = tau_v;
    gsqrt[idx_out] = gsqrt_v;
    guu[idx_out]   = guu_v;
    guv[idx_out]   = guv_v;
    gvv[idx_out]   = gvv_v;
    bsupu[idx_out] = bsupu_v;
    bsupv[idx_out] = bsupv_v;
    bsubu[idx_out] = bsubu_v;
    bsubv[idx_out] = bsubv_v;
    totalPressure[idx_out] = totalP;
}

void computeGeometry(const FourierPlan& fp, const GridParams& p,
                     const RadialProfiles& rp, MetricWorkspace& mw) {
    dim3 block(32);
    dim3 grid((p.nZnT + 31) / 32, p.ns - 1);
    geometryKernel<<<grid, block>>>(
        fp.d_r_e, fp.d_r_o,
        fp.d_z_e, fp.d_z_o,
        fp.d_ru_e, fp.d_ru_o,
        fp.d_zu_e, fp.d_zu_o,
        fp.d_lu_e, fp.d_lu_o,
        rp.d_sqrtS_F,
        rp.d_sqrtS_H, rp.d_iota_H, rp.d_phip_H, rp.d_pres_H,
        p.ns, p.nZnT, rp.delta_s,
        mw.d_r12, mw.d_ru12, mw.d_zu12, mw.d_rs, mw.d_zs, mw.d_tau,
        mw.d_gsqrt, mw.d_guu, mw.d_guv, mw.d_gvv,
        mw.d_bsupu, mw.d_bsupv,
        mw.d_bsubu, mw.d_bsubv,
        mw.d_totalPressure);
    checkCuda(cudaGetLastError(), "geometry kernel");
    checkCuda(cudaDeviceSynchronize(), "geometry sync");
}
