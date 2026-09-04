// kernels/axisymmetric_impl.cuh — template definitions for
// axisymmetric_operator.hpp. Included once per scalar type by
// axisymmetric_double.cu / axisymmetric_float.cu (the explicit-instantiation
// split, one scalar type per TU).
//
// The axisymmetric backend replaces the cuFFT length-one Z2D/D2Z round trip
// with direct poloidal synthesis/projection. For ntor=0, nzeta=1 every folded
// mode has n=0, so (from the generic pack kernels' n=0 branch) the only nonzero
// spectral slots are rmkcc/zmksc/lmksc; the sin(nζ) value/derivative slots are
// exactly zero and the toroidal derivatives vanish. The kernels below reproduce
// that arithmetic with the same per-mode poloidal table reads and the same
// ascending-m accumulation order, differing from the cuFFT path only in that
// the length-one transform is elided — a Class B difference (the differential
// test bounds it).
#ifndef CUMES_SRC_AXISYMMETRIC_IMPL_CUH_
#define CUMES_SRC_AXISYMMETRIC_IMPL_CUH_

#include "cumes/runtime/cuda_status.hpp"
#include "cumes/transforms/axisymmetric_operator.hpp"

#include <cmath>
#include <vector>

namespace cumes {
namespace axisym_detail {

using std::sqrt;

// --- inverse -------------------------------------------------------------
// One thread per (surface j, theta l). Sums the poloidal basis over m in
// ascending order. Odd-m contributions are divided by maxsc (vmecpp scalxc
// odd decomposition); the e/o accumulators match the generic
// inverse_accumulate_kernel's m%2 split exactly.
template <class T>
__global__ void inverse_kernel(SpectralView<const T, PhysicalStateDomain> coeff,
                               int ns,
                               int mpol,
                               int ntheta,
                               int nZnT,
                               const T* __restrict__ cos_th,
                               const T* __restrict__ sin_th,
                               const T* __restrict__ mcos_th,
                               const T* __restrict__ msin_th,
                               T* __restrict__ r_e,
                               T* __restrict__ z_e,
                               T* __restrict__ l_e,
                               T* __restrict__ ru_e,
                               T* __restrict__ zu_e,
                               T* __restrict__ lu_e,
                               T* __restrict__ r_o,
                               T* __restrict__ z_o,
                               T* __restrict__ l_o,
                               T* __restrict__ ru_o,
                               T* __restrict__ zu_o,
                               T* __restrict__ lu_o,
                               T* __restrict__ rv_e,
                               T* __restrict__ zv_e,
                               T* __restrict__ lv_e,
                               T* __restrict__ rv_o,
                               T* __restrict__ zv_o,
                               T* __restrict__ lv_o) {
    const int j = blockIdx.x;
    const int l = blockIdx.y * blockDim.x + threadIdx.x;
    if (l >= ntheta) return;
    const T maxsc = fmax(sqrt(T(j) / T(ns - 1)), sqrt(T(1.0) / T(ns - 1)));
    const T facO = T(1.0) / maxsc;
    T re = 0, ze = 0, le = 0, rue = 0, zue = 0, lue = 0;
    T ro = 0, zo = 0, lo = 0, ruo = 0, zuo = 0, luo = 0;
    for (int m = 0; m < mpol; ++m) {
        const T cosm = cos_th[m * ntheta + l], sinm = sin_th[m * ntheta + l];
        const T mcos = mcos_th[m * ntheta + l], msin = msin_th[m * ntheta + l];
        const T rc = coeff(SpectralComponent::Rcc, m, j);
        const T zs = coeff(SpectralComponent::Zsc, m, j);
        const T ls = coeff(SpectralComponent::Lsc, m, j);
        const T fac = (m % 2 == 1) ? facO : T(1.0);
        // R: value rc·cos(mθ), θ-deriv -m·rc·sin(mθ)  (msin = -m·sin)
        // Z, λ: value ·sin(mθ), θ-deriv +m·cos(mθ)     (mcos = +m·cos)
        const T rv = fac * rc * cosm;
        const T ruv = fac * rc * msin;
        const T zv = fac * zs * sinm;
        const T zuv = fac * zs * mcos;
        const T lv = fac * ls * sinm;
        const T luv = fac * ls * mcos;
        if (m % 2 == 1) {
            ro += rv;
            ruo += ruv;
            zo += zv;
            zuo += zuv;
            lo += lv;
            luo += luv;
        } else {
            re += rv;
            rue += ruv;
            ze += zv;
            zue += zuv;
            le += lv;
            lue += luv;
        }
    }
    const int idx = j * nZnT + l;  // nZnT == ntheta (nzeta == 1)
    r_e[idx] = re;
    ru_e[idx] = rue;
    rv_e[idx] = T(0);
    z_e[idx] = ze;
    zu_e[idx] = zue;
    zv_e[idx] = T(0);
    l_e[idx] = le;
    lu_e[idx] = lue;
    lv_e[idx] = T(0);
    r_o[idx] = ro;
    ru_o[idx] = ruo;
    rv_o[idx] = T(0);
    z_o[idx] = zo;
    zu_o[idx] = zuo;
    zv_o[idx] = T(0);
    l_o[idx] = lo;
    lu_o[idx] = luo;
    lv_o[idx] = T(0);
}

// --- forward -------------------------------------------------------------
// One thread per (surface j, mode m). Serial reduced-θ trapezoid sum; the
// weight is folded into the table reads exactly as forward_reduce_kernel does
// (cosm = w·cos(mθ), msin = w·(-m)·sin(mθ), …). The sin(nζ) families are zero
// (no n=0 basis function); crmn/czmn/clmn carry nf=0 weights and drop out.
template <class T>
__global__ void forward_kernel(
    const T* __restrict__ armn_e,
    const T* __restrict__ armn_o,
    const T* __restrict__ azmn_e,
    const T* __restrict__ azmn_o,
    const T* __restrict__ brmn_e,
    const T* __restrict__ brmn_o,
    const T* __restrict__ bzmn_e,
    const T* __restrict__ bzmn_o,
    const T* __restrict__ blmn_e,
    const T* __restrict__ blmn_o,
    const T* __restrict__ frcon_e,
    const T* __restrict__ frcon_o,
    const T* __restrict__ fzcon_e,
    const T* __restrict__ fzcon_o,
    const T* __restrict__ cos_th,
    const T* __restrict__ sin_th,
    const T* __restrict__ mcos_th,
    const T* __restrict__ msin_th,
    const T* __restrict__ fwd_w,
    int ns,
    int mpol,
    int ntheta,
    int nThetaRed,
    int nZnT,
    int include_lcfs,
    SpectralView<T, DecomposedResidualDomain> f_spec) {
    const int j = blockIdx.x;
    const int m = blockIdx.y;
    const bool m_even = (m % 2 == 0);
    const T* armn = m_even ? armn_e : armn_o;
    const T* azmn = m_even ? azmn_e : azmn_o;
    const T* brmn = m_even ? brmn_e : brmn_o;
    const T* bzmn = m_even ? bzmn_e : bzmn_o;
    const T* blmn = m_even ? blmn_e : blmn_o;
    const T* frcon = m_even ? frcon_e : frcon_o;
    const T* fzcon = m_even ? fzcon_e : fzcon_o;
    const T xmpq = T(m) * T(m - 1);
    const T* cth = cos_th + m * ntheta;
    const T* sth = sin_th + m * ntheta;
    const T* mcth = mcos_th + m * ntheta;
    const T* msth = msin_th + m * ntheta;
    T v0 = 0, v4 = 0, v8 = 0;  // the n=0 slots rmkcc / zmksc / lmksc
    for (int l = 0; l < nThetaRed; ++l) {
        const T w = fwd_w[l];
        const T cosm = w * cth[l], sinm = w * sth[l];
        const T mcos = w * mcth[l], msin = w * msth[l];
        const int idx = j * nZnT + l;  // zeta == 0 (nzeta == 1)
        const T tempR = armn[idx] + xmpq * frcon[idx];
        const T tempZ = azmn[idx] + xmpq * fzcon[idx];
        v0 += tempR * cosm + brmn[idx] * msin;
        v4 += tempZ * sinm + bzmn[idx] * mcos;
        v8 += blmn[idx] * mcos;
    }
    const T mscale = (m == 0) ? T(1.0) : sqrt(T(2.0));
    if (j == 0) {
        // axis: m=0 keeps frcc only (nf=0 drops the fzcs term); all else zero.
        f_spec(SpectralComponent::Rcc, m, j) = (m == 0) ? mscale * v0 : T(0);
        f_spec(SpectralComponent::Zsc, m, j) = T(0);
        f_spec(SpectralComponent::Lsc, m, j) = T(0);
        f_spec(SpectralComponent::Rss, m, j) = T(0);
        f_spec(SpectralComponent::Zcs, m, j) = T(0);
        f_spec(SpectralComponent::Lcs, m, j) = T(0);
    } else if (j == ns - 1 && !include_lcfs) {
        // LCFS: λ only (R/Z are fixed-boundary).
        f_spec(SpectralComponent::Rcc, m, j) = T(0);
        f_spec(SpectralComponent::Zsc, m, j) = T(0);
        f_spec(SpectralComponent::Lsc, m, j) = mscale * v8;
        f_spec(SpectralComponent::Rss, m, j) = T(0);
        f_spec(SpectralComponent::Zcs, m, j) = T(0);
        f_spec(SpectralComponent::Lcs, m, j) = T(0);
    } else {
        f_spec(SpectralComponent::Rcc, m, j) = mscale * v0;
        f_spec(SpectralComponent::Zsc, m, j) = mscale * v4;
        f_spec(SpectralComponent::Lsc, m, j) = mscale * v8;
        f_spec(SpectralComponent::Rss, m, j) = T(0);
        f_spec(SpectralComponent::Zcs, m, j) = T(0);
        f_spec(SpectralComponent::Lcs, m, j) = T(0);
    }
}

// --- constraint: xmpq-weighted rCon/zCon ----------------------------------
// rCon = Σ m(m-1)·rmncc·cos(mθ), zCon = Σ m(m-1)·zmnsc·sin(mθ) — a full
// real-space field (no parity split, no scalxc), matching the fused inverse
// DFT's rCon/zCon accumulation on the n=0 compact pack (slots 0/2 survive).
template <class T>
__global__ void rzcon_kernel(SpectralView<const T, PhysicalStateDomain> coeff,
                             int ns,
                             int mpol,
                             int ntheta,
                             int nZnT,
                             const T* __restrict__ cos_th,
                             const T* __restrict__ sin_th,
                             T* __restrict__ rCon,
                             T* __restrict__ zCon) {
    const int j = blockIdx.x;
    const int l = blockIdx.y * blockDim.x + threadIdx.x;
    if (l >= ntheta) return;
    T r = 0, z = 0;
    for (int m = 0; m < mpol; ++m) {
        const T xmpq = T(m) * T(m - 1);
        if (xmpq == T(0.0)) continue;  // m=0,1 vanish
        const T rc = xmpq * coeff(SpectralComponent::Rcc, m, j);
        const T zs = xmpq * coeff(SpectralComponent::Zsc, m, j);
        r += rc * cos_th[m * ntheta + l];
        z += zs * sin_th[m * ntheta + l];
    }
    const int idx = j * nZnT + l;
    // Null guards: the interface documents null views as "skip that output"
    // (review finding 3.3); one-null callers must not fault.
    if (rCon != nullptr) rCon[idx] = r;
    if (zCon != nullptr) zCon[idx] = z;
}

// --- constraint: de-alias bandpass ----------------------------------------
// For n=0 the analysis D2Z (length one) keeps only the real bin, the coeff pack
// keeps only the sc slot (the cs slot vanishes with shalf=0), and the synthesis
// rebuilds gCon = Σ (2/nZnT)·tcon·faccon[m]·(Σ_θ gConEff·sin(mθ))·sin(mθ) over
// m = 1..mpol-2. The axis (surface 0) is skipped (never consumed downstream).
template <class T>
__global__ void dealias_kernel(const T* __restrict__ gConEff,
                               const T* __restrict__ tcon,
                               const T* __restrict__ faccon,
                               const T* __restrict__ sin_th,
                               int ns,
                               int mpol,
                               int ntheta,
                               int nZnT,
                               T* __restrict__ gCon) {
    const int jF = blockIdx.x;
    const int l = blockIdx.y * blockDim.x + threadIdx.x;
    if (jF == 0 || l >= ntheta) return;
    const T norm = T(2.0) / T(nZnT);  // n=0 normalization (nZnT == ntheta)
    const T scale = tcon[jF];
    T g = 0;
    for (int m = 1; m <= mpol - 2; ++m) {
        const T* sth = sin_th + m * ntheta;
        T s_sc = 0;
        for (int lp = 0; lp < ntheta; ++lp) {
            s_sc += gConEff[jF * nZnT + lp] * sth[lp];
        }
        g += norm * scale * faccon[m] * s_sc * sth[l];
    }
    gCon[jF * nZnT + l] = g;
}

}  // namespace axisym_detail

// ---------------------------------------------------------------------------
// AxisymmetricOperator member definitions
// ---------------------------------------------------------------------------
template <class T>
AxisymmetricOperator<T>::AxisymmetricOperator(const DeviceParams<T>& p)
    : p_(p) {
    if (p.ntor != 0 || p.nzeta != 1) {
        throw CumesError(
            "AxisymmetricOperator: requires ntor=0, nzeta=1 "
            "(got ntor=" +
            std::to_string(p.ntor) + ", nzeta=" + std::to_string(p.nzeta) +
            ")");
    }
    const int mpol = p.mpol, ntheta = p.ntheta;
    cos_th_.allocate((size_t)mpol * ntheta);
    sin_th_.allocate((size_t)mpol * ntheta);
    mcos_th_.allocate((size_t)mpol * ntheta);
    msin_th_.allocate((size_t)mpol * ntheta);
    fwd_w_.allocate((size_t)(ntheta / 2 + 1));

    std::vector<T> h_cos((size_t)mpol * ntheta), h_sin((size_t)mpol * ntheta);
    std::vector<T> h_mcos((size_t)mpol * ntheta), h_msin((size_t)mpol * ntheta);
    for (int m = 0; m < mpol; ++m)
        for (int l = 0; l < ntheta; ++l) {
            const T th = T(2.0 * M_PI) * T(l) / T(ntheta);
            h_cos[m * ntheta + l] = cos(m * th);
            h_sin[m * ntheta + l] = sin(m * th);
            h_mcos[m * ntheta + l] = m * cos(m * th);
            h_msin[m * ntheta + l] = -m * sin(m * th);
        }
    const int nThetaRed = ntheta / 2 + 1;
    std::vector<T> h_fwd((size_t)nThetaRed);
    const T intNorm = T(1.0) / T(p.nzeta * (nThetaRed - 1));
    for (int l = 0; l < nThetaRed; ++l) {
        h_fwd[l] = intNorm;
        if (l == 0 || l == nThetaRed - 1) h_fwd[l] *= T(0.5);
    }
    check_cuda(cudaMemcpy(cos_th_.data(), h_cos.data(),
                          h_cos.size() * sizeof(T), cudaMemcpyHostToDevice),
               "axisym cos_th");
    check_cuda(cudaMemcpy(sin_th_.data(), h_sin.data(),
                          h_sin.size() * sizeof(T), cudaMemcpyHostToDevice),
               "axisym sin_th");
    check_cuda(cudaMemcpy(mcos_th_.data(), h_mcos.data(),
                          h_mcos.size() * sizeof(T), cudaMemcpyHostToDevice),
               "axisym mcos_th");
    check_cuda(cudaMemcpy(msin_th_.data(), h_msin.data(),
                          h_msin.size() * sizeof(T), cudaMemcpyHostToDevice),
               "axisym msin_th");
    check_cuda(cudaMemcpy(fwd_w_.data(), h_fwd.data(), h_fwd.size() * sizeof(T),
                          cudaMemcpyHostToDevice),
               "axisym fwd_w");
}

template <class T>
void AxisymmetricOperator<T>::enqueue_inverse(
    SpectralView<const T, PhysicalStateDomain> coefficients,
    GeometryParityViews<T> g,
    RealFieldView<T> rCon,
    RealFieldView<T> zCon,
    cudaStream_t stream) {
    const int ntheta = p_.ntheta, nZnT = p_.nZnT;
    const int blk = 32;  // bounded block (theta point lanes)
    const dim3 grid(p_.ns, (ntheta + blk - 1) / blk);
    axisym_detail::inverse_kernel<T><<<grid, blk, 0, stream>>>(
        coefficients, p_.ns, p_.mpol, ntheta, nZnT, cos_th_.data(),
        sin_th_.data(), mcos_th_.data(), msin_th_.data(), g.r_e.data(),
        g.z_e.data(), g.l_e.data(), g.ru_e.data(), g.zu_e.data(), g.lu_e.data(),
        g.r_o.data(), g.z_o.data(), g.l_o.data(), g.ru_o.data(), g.zu_o.data(),
        g.lu_o.data(), g.rv_e.data(), g.zv_e.data(), g.lv_e.data(),
        g.rv_o.data(), g.zv_o.data(), g.lv_o.data());
    check_cuda(cudaGetLastError(), "axisym inverse");
    // Fused rCon/zCon (blueprint §8.4): the generic backend accumulates them in
    // the same launch as the geometry; the axisymmetric backend runs its direct
    // poloidal rzcon kernel right after the synthesis, in the same stream order
    // the solver previously used (enqueue_inverse then enqueue_rzcon).
    if (rCon.data() != nullptr || zCon.data() != nullptr)
        enqueue_rzcon(coefficients, rCon, zCon, stream);
}

template <class T>
void AxisymmetricOperator<T>::enqueue_forward(
    ForceParityViews<const T> f,
    ConstraintForceViews<const T> cf,
    SpectralView<T, DecomposedResidualDomain> residual,
    cudaStream_t stream,
    bool include_lcfs) {
    const int nThetaRed = p_.ntheta / 2 + 1;
    const dim3 grid(p_.ns, p_.mpol);  // one thread per (surface, mode)
    axisym_detail::forward_kernel<T><<<grid, 1, 0, stream>>>(
        f.armn_e.data(), f.armn_o.data(), f.azmn_e.data(), f.azmn_o.data(),
        f.brmn_e.data(), f.brmn_o.data(), f.bzmn_e.data(), f.bzmn_o.data(),
        f.blmn_e.data(), f.blmn_o.data(), cf.frcon_e.data(), cf.frcon_o.data(),
        cf.fzcon_e.data(), cf.fzcon_o.data(), cos_th_.data(), sin_th_.data(),
        mcos_th_.data(), msin_th_.data(), fwd_w_.data(), p_.ns, p_.mpol,
        p_.ntheta, nThetaRed, p_.nZnT, include_lcfs ? 1 : 0, residual);
    check_cuda(cudaGetLastError(), "axisym forward");
}

template <class T>
void AxisymmetricOperator<T>::enqueue_rzcon(
    SpectralView<const T, PhysicalStateDomain> coefficients,
    RealFieldView<T> rCon,
    RealFieldView<T> zCon,
    cudaStream_t stream) {
    const int ntheta = p_.ntheta;
    const int blk = 32;
    const dim3 grid(p_.ns, (ntheta + blk - 1) / blk);
    axisym_detail::rzcon_kernel<T><<<grid, blk, 0, stream>>>(
        coefficients, p_.ns, p_.mpol, ntheta, p_.nZnT, cos_th_.data(),
        sin_th_.data(), rCon.data(), zCon.data());
    check_cuda(cudaGetLastError(), "axisym rzcon");
}

template <class T>
void AxisymmetricOperator<T>::enqueue_dealias(RealFieldView<const T> gConEff,
                                              const T* tcon,
                                              const T* faccon,
                                              RealFieldView<T> gCon,
                                              cudaStream_t stream) {
    const int ntheta = p_.ntheta;
    const int blk = 32;
    const dim3 grid(p_.ns, (ntheta + blk - 1) / blk);
    axisym_detail::dealias_kernel<T><<<grid, blk, 0, stream>>>(
        gConEff.data(), tcon, faccon, sin_th_.data(), p_.ns, p_.mpol, ntheta,
        p_.nZnT, gCon.data());
    check_cuda(cudaGetLastError(), "axisym dealias");
}

}  // namespace cumes

#endif  // CUMES_SRC_AXISYMMETRIC_IMPL_CUH_
