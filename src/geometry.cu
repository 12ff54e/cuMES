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

// nvcc rejects `extern __shared__` arrays in a function template that is
// instantiated with different element types in one TU ("declaration is
// incompatible with previous" — both the double and float instantiations
// collide on the variable name). Route the dynamic shared-memory base through
// one non-templated device function instead: the base address is the same for
// every function in a block, so the kernels reinterpret it as T*.
namespace {
__device__ void* dynSharedBase() {
    extern __shared__ unsigned char smem_base[];
    return smem_base;
}
}


static void checkCuda(cudaError_t err, const char* tag) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error [%s]: %s\n", tag, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

template <typename T>
MetricWorkspace<T> metricCreate(const GridParams<T>& p) {
    MetricWorkspace<T> mw{};
    size_t nH = (p.ns - 1) * p.nZnT * sizeof(T);

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

template <typename T>
void metricFree(MetricWorkspace<T>& mw) {
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
// For ncurr=0 the kernel also finalizes bsupu (adds chipH/√g) and computes
// the covariant field + total pressure. For ncurr=1 the λ-only part of
// bsupu/bsupv is computed here and the current-constraint solve happens in
// ncurr1FinalizeKernel (it needs surface integrals of the λ-only field).
template <typename T>
__global__ void geometryKernel(
    // Full-grid geometry, even/odd parity: (nZnT, ns) col-major
    const T* __restrict__ r_e,  const T* __restrict__ r_o,
    const T* __restrict__ z_e,  const T* __restrict__ z_o,
    const T* __restrict__ ru_e, const T* __restrict__ ru_o,
    const T* __restrict__ zu_e, const T* __restrict__ zu_o,
    const T* __restrict__ lu_e, const T* __restrict__ lu_o,
    const T* __restrict__ lv_e, const T* __restrict__ lv_o,
    const T* __restrict__ rv_e, const T* __restrict__ rv_o,
    const T* __restrict__ zv_e, const T* __restrict__ zv_o,
    // Full-grid sqrt(s) for parity mixing
    const T* __restrict__ sqrtS_F,   // (ns,)
    // Half-grid profiles
    const T* __restrict__ sqrtS_H,   // (ns-1,)
    const T* __restrict__ phip_H,    // (ns-1,)
    const T* __restrict__ pres_H,    // (ns-1,)
    const T* __restrict__ phip_F,    // (ns,) full grid (bsupv norm)
    const T* __restrict__ chip_H,    // (ns-1,) dχ/ds (ncurr=0: fixed)
    T lamscale, int ncurr,
    int ns, int nZnT, T delta_s,
    // half-grid outputs (all (ns-1, nZnT) col-major)
    T* __restrict__ r12,   T* __restrict__ ru12,
    T* __restrict__ zu12,  T* __restrict__ rs,
    T* __restrict__ zs,    T* __restrict__ tau,
    T* __restrict__ gsqrt, T* __restrict__ guu,
    T* __restrict__ guv,   T* __restrict__ gvv,
    T* __restrict__ bsupu, T* __restrict__ bsupv,
    T* __restrict__ bsubu, T* __restrict__ bsubv,
    T* __restrict__ totalPressure)
{
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

    // B^ζ = (lamscale·λ_θ + Φ') / √g
    T bsupv_v = (lamscale * lu_h + phipF_avg) / gsqrt_v;
    // B^θ = (lamscale·λ_ζ + χ') / √g; the χ' part (chipH) is added below
    // (ncurr=1) or taken from the fixed profile (ncurr=0).
    T bsupu_v = lamscale * lv_h / gsqrt_v;

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
        bsupu_v += chip_H[jH] / gsqrt_v;
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
        avg += guu_v / gsqrt_v * w;
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
        T bsupu_v = bsupu[idx] + chip / gsqrt[idx];
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
                              T* dVdsH, T* psum) {
    dim3 block(256), grid(p.ns - 1);
    size_t shmem = 4 * block.x * sizeof(T);
    computeNormPartialsKernel<T><<<grid, block, shmem>>>(
        mw.d_gsqrt, mw.d_guu, mw.d_r12, mw.d_bsupu, mw.d_bsupv,
        mw.d_bsubu, mw.d_bsubv,
        p.ntheta, p.nzeta, p.ns,
        dVdsH, psum);
    checkCuda(cudaGetLastError(), "norm partials");
    checkCuda(cudaDeviceSynchronize(), "norm partials sync");
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
    if (j == 1) {
        iotaF[0] = T(1.5) * iotaH[0] - T(0.5) * iotaH[1];
    }
    iotaF[j] = T(0.5) * (iotaH[j] + iotaH[j - 1]);
    chipF[j] = T(0.5) * (chipH[j] + chipH[j - 1]);
    if (j == ns - 1) {
        iotaF[ns - 1] = T(1.5) * iotaH[ns - 2] - T(0.5) * iotaH[ns - 3];
        chipF[ns - 1] = T(2.0) * chipH[ns - 2] - chipH[ns - 3];
    }
}

template <typename T>
void computeGeometry(const FourierPlan<T>& fp, const GridParams<T>& p,
                     const RadialProfiles<T>& rp, MetricWorkspace<T>& mw) {
    dim3 block(128);
    dim3 grid((p.nZnT + 127) / 128, p.ns - 1);
    geometryKernel<T><<<grid, block>>>(
        fp.d_r_e, fp.d_r_o,
        fp.d_z_e, fp.d_z_o,
        fp.d_ru_e, fp.d_ru_o,
        fp.d_zu_e, fp.d_zu_o,
        fp.d_lu_e, fp.d_lu_o,
        fp.d_lv_e, fp.d_lv_o,
        fp.d_rv_e, fp.d_rv_o,
        fp.d_zv_e, fp.d_zv_o,
        rp.d_sqrtS_F,
        rp.d_sqrtS_H, rp.d_phip_H, rp.d_pres_H,
        rp.d_phip_F, rp.d_chip_H,
        p.lamscale, p.ncurr,
        p.ns, p.nZnT, rp.delta_s,
        mw.d_r12, mw.d_ru12, mw.d_zu12, mw.d_rs, mw.d_zs, mw.d_tau,
        mw.d_gsqrt, mw.d_guu, mw.d_guv, mw.d_gvv,
        mw.d_bsupu, mw.d_bsupv,
        mw.d_bsubu, mw.d_bsubv,
        mw.d_totalPressure);
    checkCuda(cudaGetLastError(), "geometry kernel");

    if (p.ncurr == 1) {
        dim3 fb(256), fg(p.ns - 1);
        size_t shmem = 2 * 256 * sizeof(T);
        ncurr1FinalizeKernel<T><<<fg, fb, shmem>>>(
            mw.d_guu, mw.d_guv, mw.d_gsqrt, mw.d_gvv,
            mw.d_bsupu, mw.d_bsupv,
            rp.d_curr_H, rp.d_phip_H, rp.d_pres_H, rp.d_sqrtS_H,
            p.ns, p.nZnT, p.ntheta, p.nzeta, p.lamscale,
            mw.d_bsubu, mw.d_bsubv, mw.d_totalPressure,
            rp.d_chip_H, rp.d_iota_H);
        checkCuda(cudaGetLastError(), "ncurr1 kernel");
        }

    dim3 ib(256), ig((p.ns + 255) / 256);
    updateIotaChipFKernel<T><<<ig, ib>>>(
        rp.d_iota_H, rp.d_chip_H, p.ns, rp.d_iota_F, rp.d_chi_F);
    checkCuda(cudaGetLastError(), "iotaChipF kernel");

}

// ---- Explicit instantiation (double + float) ----------------------------
template MetricWorkspace<double> metricCreate<double>(const GridParams<double>&);
template MetricWorkspace<float>  metricCreate<float>(const GridParams<float>&);
template void metricFree<double>(MetricWorkspace<double>&);
template void metricFree<float>(MetricWorkspace<float>&);
template void computeGeometry<double>(const FourierPlan<double>&, const GridParams<double>&, const RadialProfiles<double>&, MetricWorkspace<double>&);
template void computeGeometry<float>(const FourierPlan<float>&, const GridParams<float>&, const RadialProfiles<float>&, MetricWorkspace<float>&);
template void computeForceNormPartials<double>(const GridParams<double>&, const MetricWorkspace<double>&, double*, double*);
template void computeForceNormPartials<float>(const GridParams<float>&, const MetricWorkspace<float>&, float*, float*);
