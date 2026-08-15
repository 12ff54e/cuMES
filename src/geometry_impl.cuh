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

#include "geometry.cuh"
#include "fourier.cuh"
#include "profiles.cuh"
#include <cstdio>
#include <math_constants.h>


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
#include "cumes/state/real_fields.cuh"

// ---- typed real-space view bundles over the workspace structs ------------
// Constructed at the operator boundary (real_fields.cuh's intended use); the
// kernels then read the raw pointers back out of the bundles, keeping the flat
// `surface*nZnT + zeta*ntheta + theta` arithmetic bit-for-bit identical.
template <typename T>
static cumes::GeometryParityViews<T> geometryParityViews(const FourierPlan<T>& fp,
                                                         const GridParams<T>& p) {
    auto f = [&](T* d) { return cumes::RealFieldView<T>(d, p.ns, p.ntheta, p.nzeta); };
    cumes::GeometryParityViews<T> v;
    v.r_e = f(fp.d_r_e); v.z_e = f(fp.d_z_e); v.l_e = f(fp.d_l_e);
    v.ru_e = f(fp.d_ru_e); v.zu_e = f(fp.d_zu_e); v.lu_e = f(fp.d_lu_e);
    v.r_o = f(fp.d_r_o); v.z_o = f(fp.d_z_o); v.l_o = f(fp.d_l_o);
    v.ru_o = f(fp.d_ru_o); v.zu_o = f(fp.d_zu_o); v.lu_o = f(fp.d_lu_o);
    v.rv_e = f(fp.d_rv_e); v.zv_e = f(fp.d_zv_e); v.lv_e = f(fp.d_lv_e);
    v.rv_o = f(fp.d_rv_o); v.zv_o = f(fp.d_zv_o); v.lv_o = f(fp.d_lv_o);
    return v;
}

template <typename T>
static cumes::BaseGeometryHalfViews<T> baseGeometryHalfViews(const MetricWorkspace<T>& mw,
                                                             const GridParams<T>& p) {
    auto h = [&](T* d) { return cumes::RealFieldView<T>(d, p.ns - 1, p.ntheta, p.nzeta); };
    cumes::BaseGeometryHalfViews<T> v;
    v.r12 = h(mw.d_r12); v.ru12 = h(mw.d_ru12); v.zu12 = h(mw.d_zu12);
    v.rs = h(mw.d_rs); v.zs = h(mw.d_zs); v.tau = h(mw.d_tau);
    v.gsqrt = h(mw.d_gsqrt); v.guu = h(mw.d_guu); v.guv = h(mw.d_guv); v.gvv = h(mw.d_gvv);
    return v;
}

template <typename T>
static cumes::MagneticFieldViews<T> magneticFieldViews(const MetricWorkspace<T>& mw,
                                                       const GridParams<T>& p) {
    auto h = [&](T* d) { return cumes::RealFieldView<T>(d, p.ns - 1, p.ntheta, p.nzeta); };
    cumes::MagneticFieldViews<T> v;
    v.bsupu = h(mw.d_bsupu); v.bsupv = h(mw.d_bsupv);
    v.bsubu = h(mw.d_bsubu); v.bsubv = h(mw.d_bsubv);
    v.total_pressure = h(mw.d_totalPressure);
    return v;
}

template <typename T>
static cumes::RadialProfileViews<T> radialProfileViews(const RadialProfiles<T>& rp) {
    cumes::RadialProfileViews<T> v;
    v.iota_F = rp.d_iota_F; v.phip_F = rp.d_phip_F; v.chi_F = rp.d_chi_F; v.sqrtS_F = rp.d_sqrtS_F;
    v.iota_H = rp.d_iota_H; v.pres_H = rp.d_pres_H; v.phip_H = rp.d_phip_H;
    v.dVds_H = rp.d_dVds_H; v.sqrtS_H = rp.d_sqrtS_H;
    v.curr_H = rp.d_curr_H; v.chip_H = rp.d_chip_H;
    return v;
}

template <typename T>
MetricWorkspace<T> metricCreate(const GridParams<T>& p, cumes::DeviceArena* arena) {
    MetricWorkspace<T> mw{};
    size_t nH = (p.ns - 1) * p.nZnT;

    auto alloc = [&](T*& dst, const char* name) {
        if (arena) dst = arena->alloc_span<T>(name, nH);
        else cumes::check_cuda(cudaMalloc(&dst, nH * sizeof(T)), name);
    };
    alloc(mw.d_r12,  "metric/r12");
    alloc(mw.d_ru12, "metric/ru12");
    alloc(mw.d_zu12, "metric/zu12");
    alloc(mw.d_rs,   "metric/rs");
    alloc(mw.d_zs,   "metric/zs");
    alloc(mw.d_tau,  "metric/tau");
    alloc(mw.d_gsqrt, "metric/gsqrt");
    alloc(mw.d_guu,  "metric/guu");
    alloc(mw.d_guv,  "metric/guv");
    alloc(mw.d_gvv,  "metric/gvv");
    alloc(mw.d_bsupu, "metric/bsupu");
    alloc(mw.d_bsupv, "metric/bsupv");
    alloc(mw.d_bsubu, "metric/bsubu");
    alloc(mw.d_bsubv, "metric/bsubv");
    alloc(mw.d_totalPressure, "metric/totalPressure");
    mw.arena_backed = (arena != nullptr);

    return mw;
}

template <typename T>
void metricFree(MetricWorkspace<T>& mw) {
    if (!mw.arena_backed) {
        cudaFree(mw.d_r12);  cudaFree(mw.d_ru12); cudaFree(mw.d_zu12);
        cudaFree(mw.d_rs);   cudaFree(mw.d_zs);   cudaFree(mw.d_tau);
        cudaFree(mw.d_gsqrt);
        cudaFree(mw.d_guu);  cudaFree(mw.d_guv);  cudaFree(mw.d_gvv);
        cudaFree(mw.d_bsupu); cudaFree(mw.d_bsupv);
        cudaFree(mw.d_bsubu); cudaFree(mw.d_bsubv);
        cudaFree(mw.d_totalPressure);
    }
    mw = MetricWorkspace<T>{};
}

// ---- geometry kernel ----------------------------------------------------
// One thread per (theta,zeta) point, one block per half-grid surface.
// For ncurr=0 the kernel also finalizes bsupu (adds chipH/√g) and computes
// the covariant field + total pressure. For ncurr=1 the λ-only part of
// bsupu/bsupv is computed here and the current-constraint solve happens in
// ncurr1FinalizeKernel (it needs surface integrals of the λ-only field).
template <typename T>
__global__ void geometryKernel(
    cumes::GeometryParityViews<T> full,
    cumes::RadialProfileViews<T> radial,
    cumes::BaseGeometryHalfViews<T> half,
    cumes::MagneticFieldViews<T> field,
    T lamscale, int ncurr, int ns, int nZnT, T delta_s)
{
    // Full-grid geometry, even/odd parity
    const T* r_e = full.r_e.data(); const T* r_o = full.r_o.data();
    const T* z_e = full.z_e.data(); const T* z_o = full.z_o.data();
    const T* ru_e = full.ru_e.data(); const T* ru_o = full.ru_o.data();
    const T* zu_e = full.zu_e.data(); const T* zu_o = full.zu_o.data();
    const T* lu_e = full.lu_e.data(); const T* lu_o = full.lu_o.data();
    const T* lv_e = full.lv_e.data(); const T* lv_o = full.lv_o.data();
    const T* rv_e = full.rv_e.data(); const T* rv_o = full.rv_o.data();
    const T* zv_e = full.zv_e.data(); const T* zv_o = full.zv_o.data();
    // Radial profiles
    const T* sqrtS_F = radial.sqrtS_F;
    const T* sqrtS_H = radial.sqrtS_H;
    const T* phip_H = radial.phip_H;
    const T* pres_H = radial.pres_H;
    const T* phip_F = radial.phip_F;
    const T* chip_H = radial.chip_H;
    // Half-grid outputs
    T* r12 = half.r12.data(); T* ru12 = half.ru12.data();
    T* zu12 = half.zu12.data(); T* rs = half.rs.data();
    T* zs = half.zs.data(); T* tau = half.tau.data();
    T* gsqrt = half.gsqrt.data(); T* guu = half.guu.data();
    T* guv = half.guv.data(); T* gvv = half.gvv.data();
    T* bsupu = field.bsupu.data(); T* bsupv = field.bsupv.data();
    T* bsubu = field.bsubu.data(); T* bsubv = field.bsubv.data();
    T* totalPressure = field.total_pressure.data();

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

    if (ncurr == 0) {
        // Fixed iota profile: chipH is precomputed in profiles.
        // (chip_H[jH] / gsqrt_v — the same validity branch as above.)
        if (std::isfinite(gsqrt_v) && fabs(gsqrt_v) > T(1e-30)) {
            bsupu_v += chip_H[jH] / gsqrt_v;
        }
        bsupu[idx_out] = bsupu_v;
        T bsubu_v = guu_v * bsupu_v + guv_v * bsupv_v;
        T bsubv_v = guv_v * bsupu_v + gvv_v * bsupv_v;
        bsubu[idx_out] = bsubu_v;
        bsubv[idx_out] = bsubv_v;
        T bsq_half = T(0.5) * (bsupu_v * bsubu_v + bsupv_v * bsubv_v);
        totalPressure[idx_out] = bsq_half + pres_H[jH];
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
    T* s_buf = static_cast<T*>(dynSharedBase());
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

    T* s_buf = static_cast<T*>(dynSharedBase());
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
        dVdsH[jH] = GridParams<T>::kSignJacobian * s_G[0];
        psum[4 * jH + 0] = s_RZ[0];
        psum[4 * jH + 1] = s_L[0];
        psum[4 * jH + 2] = s_M[0];
        psum[4 * jH + 3] = s_G[0];
    }
}

template <typename T>
void computeForceNormPartials(const GridParams<T>& p, const MetricWorkspace<T>& mw,
                              T* dVdsH, T* psum, cudaStream_t stream) {
    dim3 block(256), grid(p.ns - 1);
    size_t shmem = 4 * block.x * sizeof(T);
    computeNormPartialsKernel<T><<<grid, block, shmem, stream>>>(
        mw.d_gsqrt, mw.d_guu, mw.d_r12, mw.d_bsupu, mw.d_bsupv,
        mw.d_bsubu, mw.d_bsubv,
        p.ntheta, p.nzeta, p.ns,
        dVdsH, psum);
    cumes::check_cuda(cudaGetLastError(), "norm partials");
    cumes::check_cuda(cudaStreamSynchronize(stream), "norm partials sync");
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
    T* __restrict__ out)  // [4]: min signJ·√g, max |√g|, nonfinite count,
                          // min-signJ·√g linear index (as T)
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
            if (!seen) { vmin = vmax = a; argmin = i + s * nHalf; seen = true; }
            else if (ov < vmin) { vmin = ov; argmin = i + s * nHalf; }
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
        out[3] = T(s_arg[0]);
    }
}

// Host wrapper: d_stats is a caller-owned 4-element device scratch
// (allocated once by the solver, not per call — no cudaMalloc in the hot
// loop); h_stats receives {min|√g|, max|√g|, nonfinite count, min index}.
// Synchronous (the caller consumes the values immediately).
template <typename T>
void computeJacobianStats(const GridParams<T>& p, const MetricWorkspace<T>& mw,
                          T* d_stats, T* h_stats, cudaStream_t stream) {
    const int nHalf = (p.ns - 1) * p.nZnT;
    jacobianStatsKernel<T><<<1, 256, 0, stream>>>(mw.d_gsqrt, nHalf, 1,
                                       T(p.kSignJacobian), d_stats);
    cumes::check_cuda(cudaGetLastError(), "jacobianStats");
    cumes::check_cuda(cudaMemcpyAsync(h_stats, d_stats, 4 * sizeof(T),
                                      cudaMemcpyDeviceToHost, stream), "jac stats cpy");
    cumes::check_cuda(cudaStreamSynchronize(stream), "jac stats sync");
}

template <typename T>
void computeGeometry(const FourierPlan<T>& fp, const GridParams<T>& p,
                     const RadialProfiles<T>& rp, MetricWorkspace<T>& mw,
                     cudaStream_t stream) {
    dim3 block(128);
    dim3 grid((p.nZnT + 127) / 128, p.ns - 1);
    geometryKernel<T><<<grid, block, 0, stream>>>(
        geometryParityViews(fp, p), radialProfileViews(rp),
        baseGeometryHalfViews(mw, p), magneticFieldViews(mw, p),
        p.lamscale, p.ncurr, p.ns, p.nZnT, rp.delta_s);
    cumes::check_cuda(cudaGetLastError(), "geometry kernel");

    if (p.ncurr == 1) {
        dim3 fb(256), fg(p.ns - 1);
        size_t shmem = 2 * 256 * sizeof(T);
        ncurr1FinalizeKernel<T><<<fg, fb, shmem, stream>>>(
            mw.d_guu, mw.d_guv, mw.d_gsqrt, mw.d_gvv,
            mw.d_bsupu, mw.d_bsupv,
            rp.d_curr_H, rp.d_phip_H, rp.d_pres_H, rp.d_sqrtS_H,
            p.ns, p.nZnT, p.ntheta, p.nzeta, p.lamscale,
            mw.d_bsubu, mw.d_bsubv, mw.d_totalPressure,
            rp.d_chip_H, rp.d_iota_H);
        cumes::check_cuda(cudaGetLastError(), "ncurr1 kernel");
        }

    dim3 ib(256), ig((p.ns + 255) / 256);
    updateIotaChipFKernel<T><<<ig, ib, 0, stream>>>(
        rp.d_iota_H, rp.d_chip_H, p.ns, rp.d_iota_F, rp.d_chi_F);
    cumes::check_cuda(cudaGetLastError(), "iotaChipF kernel");

}

