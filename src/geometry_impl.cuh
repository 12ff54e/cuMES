// geometry_impl.cuh — template definitions for geometry.cuh.
// Included once per scalar type by geometry_double.cu / geometry_float.cu; see the
// explicit-instantiation split (cumes_cuda_double / cumes_cuda_float).
#pragma once
// geometry.cu — Jacobian, metric, magnetic field, and total pressure
// on the staggered half-grid with even/odd parity decomposition.
//
// Follows the full VMEC formulation from:
//   jacobian_kernel.h  — tau = tau1 + dSHalfDsInterp * tau2
//   metric_kernel.h    — parity-mixed metric elements (guv and the 3D part
//                        of gvv only when lthreed)
//   bcontra_kernel.h   — B^contra from lambda derivatives + flux
//   bco_kernel.h       — index lowering B^contra → B_cov
//   pressure_kernel.h  — total pressure = |B|^2/2 + p
// For ncurr=1 (prescribed toroidal current) the poloidal flux derivative
// chipH and iotaH are recomputed each iteration from the current profile and
// the λ-only part of the field (ideal_mhd_model.cc computeBContra).
//
// All computation is templated on the scalar type T (double or float).

#include "cumes/state/real_space_storage.hpp"
#include <cstdio>
#include <math_constants.h>


// The consuming kernels declare their dynamic shared memory directly as
// `extern __shared__ T s_buf[]` — legal per TU because the explicit
// double/float instantiation split puts exactly one scalar type in each TU.
// The old dynSharedBase() indirection (removed 2026-08-16) existed only for
// the pre-split two-types-per-TU layout; the switch to the direct form was
// expected to be a Class B re-freeze but measured BIT-IDENTICAL on both
// configs (Solovev 251->199->456 / W7-X 1877->1617->2011, full-precision
// state, identical restart sequence) — the frozen baseline stands unchanged.

#include "cumes/runtime/cuda_status.hpp"
#include "cumes/runtime/device_arena.cuh"
#include "cumes/state/real_fields.cuh"
#include "cumes/physics/geometry_operator.hpp"
#include "cumes/physics/magnetic_field_operator.hpp"

// The geometryParityViews factory (typed real-space view bundle over the
// workspace structs) is the single shared inline definition in
// cumes/state/real_fields.cuh (review finding 4.2 — was duplicated
// byte-identically here and in forces_impl.cuh).
template <typename T>
cumes::GeometryOperator<T>::GeometryOperator(const DeviceParams<T>& p, cumes::DeviceArena* arena) {
    size_t nH = (p.ns - 1) * p.nZnT;

    auto alloc = [&](T*& dst, const char* name) {
        if (arena) dst = arena->alloc_span<T>(name, nH);
        else cumes::check_cuda(cudaMalloc(&dst, nH * sizeof(T)), name);
    };
    alloc(d_r12_,  "metric/r12");
    alloc(d_ru12_, "metric/ru12");
    alloc(d_zu12_, "metric/zu12");
    alloc(d_rs_,   "metric/rs");
    alloc(d_zs_,   "metric/zs");
    alloc(d_tau_,  "metric/tau");
    alloc(d_gsqrt_, "metric/gsqrt");
    alloc(d_guu_,  "metric/guu");
    alloc(d_guv_,  "metric/guv");
    alloc(d_gvv_,  "metric/gvv");
    alloc(d_bsupu_, "metric/bsupu");
    alloc(d_bsupv_, "metric/bsupv");
    alloc(d_bsubu_, "metric/bsubu");
    alloc(d_bsubv_, "metric/bsubv");
    alloc(d_totalPressure_, "metric/totalPressure");
    arena_backed_ = (arena != nullptr);
}

template <typename T>
cumes::GeometryOperator<T>::~GeometryOperator() {
    if (!arena_backed_) {
        cudaFree(d_r12_);  cudaFree(d_ru12_); cudaFree(d_zu12_);
        cudaFree(d_rs_);   cudaFree(d_zs_);   cudaFree(d_tau_);
        cudaFree(d_gsqrt_);
        cudaFree(d_guu_);  cudaFree(d_guv_);  cudaFree(d_gvv_);
        cudaFree(d_bsupu_); cudaFree(d_bsupv_);
        cudaFree(d_bsubu_); cudaFree(d_bsubv_);
        cudaFree(d_totalPressure_);
    }
}

// ---- base geometry kernel (blueprint §6.7) -----------------------------
// One thread per (theta,zeta) point, one block per half-grid surface. Computes
// the staggered half-grid interpolation, the Jacobian (tau, √g), and the
// covariant metric g_uu/g_uv/g_vv — NO 1/√g division and NO magnetic field.
// The field (B^θ/B^ζ, B_θ/B_ζ, total pressure) is the separate
// magneticFieldKernel below, ordered after this kernel (and, in the solver,
// after the Jacobian-status chain) on the same stream.
template <typename T>
__global__ void baseGeometryKernel(
    cumes::GeometryParityViews<T> full,
    cumes::RadialProfileViews<T> radial,
    cumes::BaseGeometryHalfViews<T> half,
    int ns, int nZnT, T delta_s)
{
    // Full-grid geometry, even/odd parity (R/Z only — λ enters the field kernel)
    const T* r_e = full.r_e.data(); const T* r_o = full.r_o.data();
    const T* z_e = full.z_e.data(); const T* z_o = full.z_o.data();
    const T* ru_e = full.ru_e.data(); const T* ru_o = full.ru_o.data();
    const T* zu_e = full.zu_e.data(); const T* zu_o = full.zu_o.data();
    const T* rv_e = full.rv_e.data(); const T* rv_o = full.rv_o.data();
    const T* zv_e = full.zv_e.data(); const T* zv_o = full.zv_o.data();
    // Radial profiles
    const T* sqrtS_F = radial.sqrtS_F;
    const T* sqrtS_H = radial.sqrtS_H;
    // Half-grid outputs
    T* r12 = half.r12.data(); T* ru12 = half.ru12.data();
    T* zu12 = half.zu12.data(); T* rs = half.rs.data();
    T* zs = half.zs.data(); T* tau = half.tau.data();
    T* gsqrt = half.gsqrt.data(); T* guu = half.guu.data();
    T* guv = half.guv.data(); T* gvv = half.gvv.data();

    int jH = blockIdx.y;   // half-grid surface index (0 .. ns-2)
    int k   = threadIdx.x + blockIdx.x * blockDim.x;
    if (jH >= ns - 1 || k >= nZnT) return;

    // Full-grid surface indices: jH (inside), jH+1 (outside)
    int i_in  = k + (jH) * nZnT;
    int i_out = k + (jH + 1) * nZnT;

    // s = sqrt(s) values
    T sH = sqrtS_H[jH];                   // half-grid sqrt(s)
    T sF_i = sqrtS_F[jH];                 // full-grid sqrt(s) at jH
    T sF_o = sqrtS_F[jH + 1];             // full-grid sqrt(s) at jH+1

    // ---- half-grid interpolation with parity mixing --------------------
    T r12_v  = T(0.5) * ((r_e[i_in]  + r_e[i_out]) + sH * (r_o[i_in]  + r_o[i_out]));
    T ru12_v = T(0.5) * ((ru_e[i_in] + ru_e[i_out])+ sH * (ru_o[i_in] + ru_o[i_out]));
    T zu12_v = T(0.5) * ((zu_e[i_in] + zu_e[i_out])+ sH * (zu_o[i_in] + zu_o[i_out]));

    // Radial derivatives with parity mixing
    T rs_v = ((r_e[i_out] - r_e[i_in]) + sH * (r_o[i_out] - r_o[i_in])) / delta_s;
    T zs_v = ((z_e[i_out] - z_e[i_in]) + sH * (z_o[i_out] - z_o[i_in])) / delta_s;

    // ---- Jacobian: tau = tau1 + dSHalfDsInterp * tau2 ------------------
    T tau1 = ru12_v * zs_v - rs_v * zu12_v;

    // tau2: odd-parity contribution that keeps Jacobian positive everywhere
    // (prevents sign change at the inboard side of the torus)
    T tau2 =
        ru_o[i_out] * z_o[i_out] + ru_o[i_in] * z_o[i_in]
      - zu_o[i_out] * r_o[i_out] - zu_o[i_in] * r_o[i_in]
      + (ru_e[i_out] * z_o[i_out] + ru_e[i_in] * z_o[i_in]
      -  zu_e[i_out] * r_o[i_out] - zu_e[i_in] * r_o[i_in]) / sH;

    constexpr T dSHalfDsInterp = T(0.25);  // 1/2 * 1/2 from interpolation
    T tau_v = tau1 + dSHalfDsInterp * tau2;

    T gsqrt_v = tau_v * r12_v;

    // ---- covariant metric with parity mixing ---------------------------
    T sFi_sq = sF_i * sF_i;
    T sFo_sq = sF_o * sF_o;

    T guu_v = T(0.5) * ((ru_e[i_in]*ru_e[i_in] + zu_e[i_in]*zu_e[i_in]) +
                          (ru_e[i_out]*ru_e[i_out] + zu_e[i_out]*zu_e[i_out]) +
                          sFi_sq * (ru_o[i_in]*ru_o[i_in] + zu_o[i_in]*zu_o[i_in]) +
                          sFo_sq * (ru_o[i_out]*ru_o[i_out] + zu_o[i_out]*zu_o[i_out]))
                 + sH * ((ru_e[i_in]*ru_o[i_in] + zu_e[i_in]*zu_o[i_in]) +
                         (ru_e[i_out]*ru_o[i_out] + zu_e[i_out]*zu_o[i_out]));

    T gvv_v = T(0.5) * (r_e[i_in]*r_e[i_in] + r_e[i_out]*r_e[i_out] +
                          sFi_sq * r_o[i_in]*r_o[i_in] +
                          sFo_sq * r_o[i_out]*r_o[i_out])
                 + sH * (r_e[i_in]*r_o[i_in] + r_e[i_out]*r_o[i_out]);

    // 3D toroidal coupling (rv/zv = R_ζ/Z_ζ): guv and the 3D part of gvv
    // (vmecpp metric_kernel.h ComputeMetricElements, lthreed block).
    // NOTE: the sH-weighted cross terms sit INSIDE the 0.5, matching vmecpp's
    // ComputeMetricElements exactly (metric_kernel.h) — putting them outside
    // doubles the cross-term weight and was caught by the iter-1 dump match.
    T guv_v = T(0.5) * ((ru_e[i_in]*rv_e[i_in] + zu_e[i_in]*zv_e[i_in]) +
                          (ru_e[i_out]*rv_e[i_out] + zu_e[i_out]*zv_e[i_out]) +
                          sFi_sq * (ru_o[i_in]*rv_o[i_in] + zu_o[i_in]*zv_o[i_in]) +
                          sFo_sq * (ru_o[i_out]*rv_o[i_out] + zu_o[i_out]*zv_o[i_out]) +
                          sH * ((ru_e[i_in]*rv_o[i_in] + zu_e[i_in]*zv_o[i_in]) +
                                (ru_e[i_out]*rv_o[i_out] + zu_e[i_out]*zv_o[i_out]) +
                                (rv_e[i_in]*ru_o[i_in] + zv_e[i_in]*zu_o[i_in]) +
                                (rv_e[i_out]*ru_o[i_out] + zv_e[i_out]*zu_o[i_out])));
    gvv_v += T(0.5) * ((rv_e[i_in]*rv_e[i_in] + zv_e[i_in]*zv_e[i_in]) +
                    (rv_e[i_out]*rv_e[i_out] + zv_e[i_out]*zv_e[i_out]) +
                    sFi_sq * (rv_o[i_in]*rv_o[i_in] + zv_o[i_in]*zv_o[i_in]) +
                    sFo_sq * (rv_o[i_out]*rv_o[i_out] + zv_o[i_out]*zv_o[i_out]))
           + sH * ((rv_e[i_in]*rv_o[i_in] + zv_e[i_in]*zv_o[i_in]) +
                   (rv_e[i_out]*rv_o[i_out] + zv_e[i_out]*zv_o[i_out]));

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
}

// ---- magnetic field kernel (blueprint §6.7) -----------------------------
// One thread per (theta,zeta) point, one block per half-grid surface. Reads the
// base geometry (√g, g_uu/g_uv/g_vv) and recomputes the half-grid λ_θ/λ_ζ and
// Φ' averages, then forms the contravariant B^θ/B^ζ with the 1/√g division and
// (ncurr=0) the covariant B_θ/B_ζ + total pressure. For ncurr=1 the λ-only part
// of bsupu/bsupv is produced here and the current-constraint solve happens in
// ncurr1FinalizeKernel (it needs surface integrals of the λ-only field).
template <typename T>
__global__ void magneticFieldKernel(
    cumes::GeometryParityViews<T> full,
    cumes::RadialProfileViews<T> radial,
    cumes::BaseGeometryHalfViews<T> half,
    cumes::MagneticFieldViews<T> field,
    T lamscale, int ncurr, int ns, int nZnT)
{
    // λ full-grid derivatives (the R/Z geometry is consumed by baseGeometryKernel)
    const T* lu_e = full.lu_e.data(); const T* lu_o = full.lu_o.data();
    const T* lv_e = full.lv_e.data(); const T* lv_o = full.lv_o.data();
    // Radial profiles
    const T* sqrtS_H = radial.sqrtS_H;
    const T* pres_H = radial.pres_H;
    const T* phip_F = radial.phip_F;
    const T* chip_H = radial.chip_H;
    // Base geometry (read-only)
    const T* gsqrt = half.gsqrt.data();
    const T* guu = half.guu.data();
    const T* guv = half.guv.data();
    const T* gvv = half.gvv.data();
    // Field outputs
    T* bsupu = field.bsupu.data(); T* bsupv = field.bsupv.data();
    T* bsubu = field.bsubu.data(); T* bsubv = field.bsubv.data();
    T* totalPressure = field.total_pressure.data();

    int jH = blockIdx.y;
    int k   = threadIdx.x + blockIdx.x * blockDim.x;
    if (jH >= ns - 1 || k >= nZnT) return;

    int i_in  = k + (jH) * nZnT;
    int i_out = k + (jH + 1) * nZnT;
    T sH = sqrtS_H[jH];

    // ---- contravariant B with lambda + flux contribution ---------------
    // Normalized angular derivatives on the half grid with parity mixing.
    // vmecpp normalizes per FULL-grid point (lu*lamscale + phipF[jF]) before
    // the half-grid average (computeBContra); the λ_ζ part is stored as
    // -∂λ/∂ζ (lv) by the inverse DFT.
    T lu_h = T(0.5) * ((lu_e[i_in] + lu_e[i_out]) + sH * (lu_o[i_in] + lu_o[i_out]));
    T lv_h = T(0.5) * ((lv_e[i_in] + lv_e[i_out]) + sH * (lv_o[i_in] + lv_o[i_out]));
    T phipF_avg = T(0.5) * (phip_F[jH] + phip_F[jH + 1]);

    // B^ζ = (lamscale·λ_θ + Φ') / √g, B^θ = (lamscale·λ_ζ + χ') / √g.
    // Degenerate-Jacobian guard: a non-finite or ~zero √g (e.g. a surface
    // whose metric collapsed mid-run) would otherwise seed infinities into
    // every downstream kernel (forces, constraint, preconditioner) and only
    // surface much later as non-finite residuals. Writing zero keeps the
    // field arrays finite; the solver's jacobian-stats check
    // (computeJacobianStats) then fails the iteration early via the
    // BAD_JACOBIAN restore path. The guard branches on validity instead of
    // computing a reciprocal, so the valid branch keeps the exact x/y
    // rounding (x * (1/y) would round twice and shift the trajectory).
    int idx = k + jH * nZnT;
    T gsqrt_v = gsqrt[idx];
    T guu_v = guu[idx], guv_v = guv[idx], gvv_v = gvv[idx];

    T bsupv_v, bsupu_v;
    if (std::isfinite(gsqrt_v) && fabs(gsqrt_v) > T(1e-30)) {
        bsupv_v = (lamscale * lu_h + phipF_avg) / gsqrt_v;
        // the χ' part (chipH) is added below (ncurr=1) or taken from the
        // fixed profile (ncurr=0).
        bsupu_v = lamscale * lv_h / gsqrt_v;
    } else {
        bsupv_v = T(0.0);
        bsupu_v = T(0.0);
    }

    bsupu[idx] = bsupu_v;
    bsupv[idx] = bsupv_v;

    if (ncurr == 0) {
        // Fixed iota profile: chipH is precomputed in profiles.
        // (chip_H[jH] / gsqrt_v — the same validity branch as above.)
        if (std::isfinite(gsqrt_v) && fabs(gsqrt_v) > T(1e-30)) {
            bsupu_v += chip_H[jH] / gsqrt_v;
        }
        bsupu[idx] = bsupu_v;
        T bsubu_v = guu_v * bsupu_v + guv_v * bsupv_v;
        T bsubv_v = guv_v * bsupu_v + gvv_v * bsupv_v;
        bsubu[idx] = bsubu_v;
        bsubv[idx] = bsubv_v;
        T bsq_half = T(0.5) * (bsupu_v * bsubu_v + bsupv_v * bsubv_v);
        totalPressure[idx] = bsq_half + pres_H[jH];
    }
}

// ---- ncurr=1 current-constraint solve (vmecpp computeBContra) -----------
// Per half-grid surface: jvPlasma = Σ (guu·bsupu + guv·bsupv),
// avg_guu_gsqrt = Σ guu/√g over the surface, then
//   chipH = (currH − jvPlasma) / avg_guu_gsqrt,  iotaH = chipH / phipH,
// and bsupu += chipH/√g, followed by the covariant field and total pressure.
// The surface sums are fold-invariant: cuMES's uniform full-grid weights
// give the same ratios as vmecpp's reduced-grid trapezoid (wInt).
template <typename T>
__global__ void ncurr1FinalizeKernel(
    const T* __restrict__ guu, const T* __restrict__ guv,
    const T* __restrict__ gsqrt, const T* __restrict__ gvv,
    T* __restrict__ bsupu, const T* __restrict__ bsupv,
    const T* __restrict__ currH, const T* __restrict__ phipH,
    const T* __restrict__ presH, const T* __restrict__ sqrtSH,
    int ns, int nZnT, int ntheta, int nzeta, T lamscale,
    T* __restrict__ bsubu, T* __restrict__ bsubv,
    T* __restrict__ totalPressure, T* __restrict__ chipH_out,
    T* __restrict__ iotaH_out)
{
    int jH = blockIdx.x;
    if (jH >= ns - 1) return;
    int tid = threadIdx.x;
    extern __shared__ T s_buf[];   // [2][blockDim.x]
    T* s_jv = s_buf;             // blockDim.x
    T* s_avg = s_buf + blockDim.x;

    // vmecpp's wInt surface averages: trapezoid over the reduced [0,pi]
    // poloidal grid with dnorm3 = 1/(nZeta*(nThetaReduced-1)) (sizes.cc).
    const int nThetaRed = ntheta / 2 + 1;
    const T dnorm3 = T(1.0) / T(nzeta * (nThetaRed - 1));

    T jv = T(0), avg = T(0);
    int base = jH * nZnT;
    // Compact reduced-grid loop: only the nzeta*nThetaRed points of the
    // trapezoid subset (the old stride over nZnT skipped half the points).
    // Same (iz, it) visit order as the old k = iz*ntheta + it sequence.
    int nRed = nzeta * nThetaRed;
    for (int k = tid; k < nRed; k += blockDim.x) {
        int iz = k / nThetaRed, it = k - iz * nThetaRed;
        T w = dnorm3;
        if (it == 0 || it == nThetaRed - 1) w *= T(0.5);
        int idx = base + iz * ntheta + it;
        T guu_v = guu[idx], gsqrt_v = gsqrt[idx];
        jv  += (guu_v * bsupu[idx] + guv[idx] * bsupv[idx]) * w;
        // Same degenerate-√g guard as geometryKernel: a non-finite or ~zero
        // √g would otherwise make the surface average NaN and the chipH
        // solve below degenerate before the jacobian-stats check runs.
        if (std::isfinite(gsqrt_v) && fabs(gsqrt_v) > T(1e-30)) {
            avg += guu_v / gsqrt_v * w;
        }
    }
    s_jv[tid] = jv; s_avg[tid] = avg;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { s_jv[tid] += s_jv[tid + s]; s_avg[tid] += s_avg[tid + s]; }
        __syncthreads();
    }
    __shared__ T s_chip;
    if (tid == 0) {
        T chip = T(0.0);
        if (s_avg[0] != T(0.0)) chip = (currH[jH] - s_jv[0]) / s_avg[0];
        s_chip = chip;
        chipH_out[jH] = chip;
        if (phipH[jH] != T(0.0)) iotaH_out[jH] = chip / phipH[jH];
    }
    __syncthreads();
    T chip = s_chip;
    for (int k = tid; k < nZnT; k += blockDim.x) {
        int idx = base + k;
        T gsqrt_v = gsqrt[idx];
        // Same validity branch as geometryKernel (chip / √g exactly).
        T bsupu_v = bsupu[idx];
        if (std::isfinite(gsqrt_v) && fabs(gsqrt_v) > T(1e-30)) {
            bsupu_v += chip / gsqrt_v;
        }
        T bsubu_v = guu[idx] * bsupu_v + guv[idx] * bsupv[idx];
        T bsubv_v = guv[idx] * bsupu_v + gvv[idx] * bsupv[idx];
        bsupu[idx] = bsupu_v;
        bsubu[idx] = bsubu_v;
        bsubv[idx] = bsubv_v;
        T bsq_half = T(0.5) * (bsupu_v * bsubu_v + bsupv[idx] * bsubv_v);
        totalPressure[idx] = bsq_half + presH[jH];
    }
}

// ---- force-norm partial sums (vmecpp computeForceNorms) -----------------
// One block per half-grid surface jH. Over the reduced poloidal subset
// (it < nThetaRed, trapezoid weights wInt — the same quadrature as vmecpp's
// wInt[l], sizes.cc) each block accumulates:
//   s_RZ  = guu * r12^2            (per surface)
//   s_L   = bsubu^2 + bsubv^2      (per surface)
//   s_Mag = gsqrt * |B|^2/2        (per surface; |B|^2/2 = bsup*bsub/2)
//   s_G   = gsqrt                  (per surface; dVdsH = signJ * s_G)
// and writes dVdsH[jH] = signJ*s_G (vmecpp: dVdsH *= signOfJacobian) plus
// the four partials to psum (4 * (ns-1), row jH at 4*jH + c). The host
// assembles the totals, energies, volume and the fNormRZ/fNormL factors,
// matching vmecpp ideal_mhd_model.cc computeForceNorms.
template <typename T>
__global__ void computeNormPartialsKernel(
    const T* __restrict__ gsqrt, const T* __restrict__ guu,
    const T* __restrict__ r12, const T* __restrict__ bsupu,
    const T* __restrict__ bsupv, const T* __restrict__ bsubu,
    const T* __restrict__ bsubv,
    int ntheta, int nzeta, int ns,
    T* __restrict__ dVdsH,   // (ns-1): signJ * sum(gsqrt * wInt)
    T* __restrict__ psum)    // 4*(ns-1): sRZ sL sMag sG per surface
{
    int jH = blockIdx.x;
    if (jH >= ns - 1) return;
    int tid = threadIdx.x;
    const int nThetaRed = ntheta / 2 + 1;
    const T dnorm3 = T(1.0) / T(nzeta * (nThetaRed - 1));

    extern __shared__ T s_buf[];   // [4][blockDim.x]
    T* s_RZ = s_buf;
    T* s_L = s_buf + blockDim.x;
    T* s_M = s_buf + 2 * blockDim.x;
    T* s_G = s_buf + 3 * blockDim.x;
    s_RZ[tid] = s_L[tid] = s_M[tid] = s_G[tid] = T(0.0);

    int base = jH * (ntheta * nzeta);
    for (int k = tid; k < ntheta * nzeta; k += blockDim.x) {
        int it = k % ntheta;
        if (it >= nThetaRed) continue;
        T w = dnorm3;
        if (it == 0 || it == nThetaRed - 1) w *= T(0.5);
        int idx = base + k;
        T g = gsqrt[idx];
        T bsubu_v = bsubu[idx], bsubv_v = bsubv[idx];
        T bmag2 = T(0.5) * (bsupu[idx] * bsubu_v + bsupv[idx] * bsubv_v);
        s_RZ[tid] += guu[idx] * r12[idx] * r12[idx] * w;
        s_L[tid]  += (bsubu_v * bsubu_v + bsubv_v * bsubv_v) * w;
        s_M[tid]  += g * bmag2 * w;
        s_G[tid]  += g * w;
    }
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_RZ[tid] += s_RZ[tid + s]; s_L[tid] += s_L[tid + s];
            s_M[tid] += s_M[tid + s];   s_G[tid] += s_G[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        dVdsH[jH] = DeviceParams<T>::kSignJacobian * s_G[0];
        psum[4 * jH + 0] = s_RZ[0];
        psum[4 * jH + 1] = s_L[0];
        psum[4 * jH + 2] = s_M[0];
        psum[4 * jH + 3] = s_G[0];
    }
}

// ---- full-grid iota/chip update (vmecpp ideal_mhd_model.cc) -------------
// iotaF[0]    = 1.5*iotaH[0] - 0.5*iotaH[1]          (axis extrapolation)
// iotaF[j]    = 0.5*(iotaH[j] + iotaH[j-1])          (interior average)
// iotaF[ns-1] = 1.5*iotaH[ns-2] - 0.5*iotaH[ns-3]    (LCFS extrapolation)
// chipF likewise (the axis chipF keeps its initial value, matching vmecpp).
template <typename T>
__global__ void updateIotaChipFKernel(
    const T* __restrict__ iotaH, const T* __restrict__ chipH,
    int ns, T* __restrict__ iotaF, T* __restrict__ chipF)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j <= 0 || j >= ns) return;
    // The LCFS row (j = ns-1) has no outside half-grid neighbor: the interior
    // average would read iotaH[ns-1]/chipH[ns-1], one element PAST the
    // (ns-1)-element half-grid arrays (an out-of-bounds device read, flagged
    // by Compute Sanitizer on W7-X ns=33). The LCFS value comes from the
    // extrapolation below instead, which only reads iotaH[ns-2]/iotaH[ns-3].
    if (j == ns - 1) {
        iotaF[ns - 1] = T(1.5) * iotaH[ns - 2] - T(0.5) * iotaH[ns - 3];
        chipF[ns - 1] = T(2.0) * chipH[ns - 2] - chipH[ns - 3];
        return;
    }
    if (j == 1) {
        iotaF[0] = T(1.5) * iotaH[0] - T(0.5) * iotaH[1];
    }
    iotaF[j] = T(0.5) * (iotaH[j] + iotaH[j - 1]);
    chipF[j] = T(0.5) * (chipH[j] + chipH[j - 1]);
}

// ---- Jacobian validity stats (vmecpp's bad-jacobian detection) -----------
// Reduces over the half-grid: the ORIENTED minimum signJ·√g (signJ = ±1, the
// sign convention of the coordinates), the max |√g|, the non-finite count,
// and the index of the min element (jH*stride + point, for the error
// message). Tracking signJ·√g rather than fabs(√g) makes a genuine Jacobian
// SIGN FLIP (an interior collapse, √g crossing zero) fail the validity test
// directly — |√g| would keep the value positive and hide the flip. On a valid
// run √g has constant sign everywhere, so signJ·√g = |√g| and the reported
// minimum is identical; the change is trajectory-neutral there. The solver
// checks these after every computeGeometry call and fails the iteration
// through the BAD_JACOBIAN restore path BEFORE the forces/constraint/
// preconditioner kernels consume the geometry — so a collapsed surface can
// never silently poison the iteration with infinities (see the inv_gsqrt
// guards in geometryKernel/ncurr1FinalizeKernel).
// NOTE on the axis: √g -> 0 at the innermost half-grid is EXPECTED (the
// magnetic axis is a coordinate singularity), so the solver's threshold must
// be relative to the run's own scale (see solverRun), not an absolute zero.
template <typename T>
__global__ void jacobianStatsKernel(
    const T* __restrict__ gsqrt, int nHalf, int stride, T signJ,
    double* __restrict__ out)  // [4]: min signJ·√g, max |√g|, nonfinite count,
                               // min-signJ·√g linear index (as double)
{
    // Reduction identities: min starts at +inf (NOT 0) and max at 0, and a
    // lane that saw no finite data contributes the identity via a `seen` flag
    // rather than a zero that could win the minimum. The old code initialized
    // vmin=0 and relied on `first`; on grids where nHalf*stride < blockDim
    // (e.g. (ns-1)*nZnT < 256), idle lanes kept vmin=0 and could win the tree
    // minimum, poisoning the reported gmin (masked only because the solver
    // suppresses gminIdx < nZnT).
    // Device-safe +inf: numeric_limits<T>::infinity() is host-only constexpr
    // in nvcc; CUDART_INF_F / CUDART_INF are the device constants. Pick by T.
    const T kInf = (sizeof(T) == sizeof(double)) ? T(CUDART_INF) : T(CUDART_INF_F);
    T vmin = kInf, vmax = T(0.0), vbad = T(0.0);
    int argmin = 0;
    bool seen = false;
    for (int i = threadIdx.x; i < nHalf; i += blockDim.x) {
        for (int s = 0; s < stride; ++s) {
            T g = gsqrt[i + s * nHalf];
            if (!std::isfinite(g)) { vbad += T(1.0); continue; }
            T a = fabs(g);            // scale statistic
            T ov = signJ * g;         // ORIENTED value: = |g| when √g keeps
                                      // the expected sign, negative on a flip
            // vmax tracks max |√g| in EVERY branch: a sign-flipped element
            // (ov = -a < vmin) still contributes its magnitude |g| = a to the
            // BAD-JACOBIAN diagnostic (review finding 2.2).
            if (!seen) { vmin = vmax = a; argmin = i + s * nHalf; seen = true; }
            else if (ov < vmin) { vmin = ov; argmin = i + s * nHalf; vmax = fmax(vmax, a); }
            else { vmax = fmax(vmax, a); }
        }
    }
    __shared__ T s_min[256], s_max[256], s_bad[256];
    __shared__ int s_arg[256];
    __shared__ char s_seen[256];
    int tid = threadIdx.x;
    s_min[tid] = seen ? vmin : kInf;
    s_max[tid] = vmax; s_bad[tid] = vbad;
    s_arg[tid] = seen ? argmin : 0;
    s_seen[tid] = seen ? 1 : 0;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_seen[tid + s] && (!s_seen[tid] || s_min[tid + s] < s_min[tid])) {
                s_min[tid] = s_min[tid + s];
                s_arg[tid] = s_arg[tid + s];
                s_seen[tid] = 1;
            }
            s_max[tid] = fmax(s_max[tid], s_max[tid + s]);
            s_bad[tid] += s_bad[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        // A fully-empty grid (no finite data anywhere) keeps the +inf identity;
        // the solver treats max <= 0 (or here inf) as invalid.
        out[0] = s_seen[0] ? s_min[0] : kInf;
        out[1] = s_max[0]; out[2] = s_bad[0];
        out[3] = (double)s_arg[0];
    }
}

// ---------------------------------------------------------------------------
// GeometryOperator (owns the half-grid buffers; launches the base-geometry
// kernel + stats). The magnetic field is the separate MagneticFieldOperator.
// ---------------------------------------------------------------------------
template <typename T>
void cumes::GeometryOperator<T>::enqueue(const cumes::RealSpaceStorage<T>& rs,
                                         const DeviceParams<T>& p,
                                         const cumes::RadialProfileViews<T>& rpv,
                                         cudaStream_t stream) {
    dim3 block(128);
    dim3 grid((p.nZnT + 127) / 128, p.ns - 1);
    baseGeometryKernel<T><<<grid, block, 0, stream>>>(
        geometryParityViews(rs, p), rpv,
        base_geometry_views(p), p.ns, p.nZnT, T(1.0) / T(p.ns - 1));
    cumes::check_cuda(cudaGetLastError(), "base geometry kernel");
}

template <typename T>
void cumes::GeometryOperator<T>::jacobian_stats(const DeviceParams<T>& p, double* d_stats,
                                                cudaStream_t stream) const {
    const int nHalf = (p.ns - 1) * p.nZnT;
    jacobianStatsKernel<T><<<1, 256, 0, stream>>>(d_gsqrt_, nHalf, 1,
                                       T(p.kSignJacobian), d_stats);
    cumes::check_cuda(cudaGetLastError(), "jacobianStats");
}

template <typename T>
void cumes::GeometryOperator<T>::force_norm_partials(const DeviceParams<T>& p, T* dVdsH,
                                                     T* psum, cudaStream_t stream) const {
    dim3 block(256), grid(p.ns - 1);
    size_t shmem = 4 * block.x * sizeof(T);
    computeNormPartialsKernel<T><<<grid, block, shmem, stream>>>(
        d_gsqrt_, d_guu_, d_r12_, d_bsupu_, d_bsupv_,
        d_bsubu_, d_bsubv_,
        p.ntheta, p.nzeta, p.ns,
        dVdsH, psum);
    cumes::check_cuda(cudaGetLastError(), "norm partials");
}

template <typename T>
cumes::BaseGeometryHalfViews<T> cumes::GeometryOperator<T>::base_geometry_views(const DeviceParams<T>& p) const {
    auto h = [&](T* d) { return cumes::RealFieldView<T>(d, p.ns - 1, p.ntheta, p.nzeta); };
    cumes::BaseGeometryHalfViews<T> v;
    v.r12 = h(d_r12_); v.ru12 = h(d_ru12_); v.zu12 = h(d_zu12_);
    v.rs = h(d_rs_); v.zs = h(d_zs_); v.tau = h(d_tau_);
    v.gsqrt = h(d_gsqrt_); v.guu = h(d_guu_); v.guv = h(d_guv_); v.gvv = h(d_gvv_);
    return v;
}

template <typename T>
cumes::MagneticFieldViews<T> cumes::GeometryOperator<T>::magnetic_field_views(const DeviceParams<T>& p) const {
    auto h = [&](T* d) { return cumes::RealFieldView<T>(d, p.ns - 1, p.ntheta, p.nzeta); };
    cumes::MagneticFieldViews<T> v;
    v.bsupu = h(d_bsupu_); v.bsupv = h(d_bsupv_);
    v.bsubu = h(d_bsubu_); v.bsubv = h(d_bsubv_);
    v.total_pressure = h(d_totalPressure_);
    return v;
}

// ---------------------------------------------------------------------------
// MagneticFieldOperator (stateless; launches the field kernel + ncurr closure
// + full-grid iota/chip update over the GeometryOperator's typed views).
// ---------------------------------------------------------------------------
template <typename T>
void cumes::MagneticFieldOperator<T>::enqueue(const cumes::RealSpaceStorage<T>& rs,
                                              const DeviceParams<T>& p,
                                              const cumes::RadialProfileViews<T>& rpv,
                                              const cumes::BaseGeometryHalfViews<T>& base,
                                              cumes::MagneticFieldViews<T> field,
                                              cudaStream_t stream, bool update_iota_chi) const {
    dim3 block(128);
    dim3 grid((p.nZnT + 127) / 128, p.ns - 1);
    magneticFieldKernel<T><<<grid, block, 0, stream>>>(
        geometryParityViews(rs, p), rpv, base, field,
        p.lamscale, p.ncurr, p.ns, p.nZnT);
    cumes::check_cuda(cudaGetLastError(), "magnetic field kernel");

    if (p.ncurr == 1) {
        dim3 fb(256), fg(p.ns - 1);
        size_t shmem = 2 * 256 * sizeof(T);
        ncurr1FinalizeKernel<T><<<fg, fb, shmem, stream>>>(
            base.guu.data(), base.guv.data(), base.gsqrt.data(), base.gvv.data(),
            field.bsupu.data(), field.bsupv.data(),
            rpv.curr_H, rpv.phip_H, rpv.pres_H, rpv.sqrtS_H,
            p.ns, p.nZnT, p.ntheta, p.nzeta, p.lamscale,
            field.bsubu.data(), field.bsubv.data(), field.total_pressure.data(),
            rpv.chip_H, rpv.iota_H);
        cumes::check_cuda(cudaGetLastError(), "ncurr1 kernel");
        }

    if (update_iota_chi) {
        dim3 ib(256), ig((p.ns + 255) / 256);
        updateIotaChipFKernel<T><<<ig, ib, 0, stream>>>(
            rpv.iota_H, rpv.chip_H, p.ns, rpv.iota_F, rpv.chi_F);
        cumes::check_cuda(cudaGetLastError(), "iotaChipF kernel");
    }
}

