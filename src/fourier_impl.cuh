// fourier_impl.cuh — template definitions for fourier.cuh.
// Included once per scalar type by fourier_double.cu / fourier_float.cu; see the
// explicit-instantiation split (cumes_cuda_double / cumes_cuda_float).
#pragma once
// fourier.cu — DFT transforms with vmecpp m-parity convention.
//
// Internal (folded, n>=0) product basis, matching vmecpp's
// FourierToReal3DSymmFastPoloidal (dft_toroidal.cc):
//   R = rmncc*cos(mθ)cos(nζ) + rmnss*sin(mθ)sin(nζ)
//   Z = zmnsc*sin(mθ)cos(nζ) + zmncs*cos(mθ)sin(nζ)
//   λ = lmnsc*sin(mθ)cos(nζ) + lmncs*cos(mθ)sin(nζ)
// Mode index: mode = m*(ntor+1) + n, m = 0..mpol-1, n = 0..ntor;
// xn = n*nfp (toroidal mode number per field period).
//
// The R/Z coefficients are the plain physical cos(mθ-nζ) amplitudes (unlike
// vmecpp's internal state, which stores them divided by mscale*nscale; the
// real-space reconstruction is identical because vmecpp's basis tables carry
// the mscale/nscale). Lambda is physical in both codes; its real-space
// reconstruction carries mscale*nscale (the λ basis is mscale'd in vmecpp
// and cuMES compensates with the per-mode factor in the inverse DFT below).
//
// The toroidal derivative of lambda is stored as -∂λ/∂ζ (vmecpp convention:
// lv = -(lmksc_n*sinmu + lmkcs_n*cosmu) with sinnvn = -n*nfp*sin(nζ) and
// cosnvn = +n*nfp*cos(nζ)); bsupu = (lamscale*lv + chip')/√g.
//
// All computation is templated on the scalar type T (double or float); the
// cuFFT plan types / exec calls dispatch through FftTraits<T>.

#include "fourier.cuh"
#include "constraint.cuh"
#include <cstdio>
#include <cmath>

#include "cumes/state/mode_table.cuh"
#include "cumes/transforms/toroidal_fft_operator.hpp"


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

template <typename T>
cumes::DeviceModeTable cumes::modeTableCreate(const DeviceParams<T>& p, cumes::DeviceArena* arena) {
    cumes::DeviceModeTable mt{};
    const int mnmax = p.mnmax;
    auto* h_xm = new int[mnmax], *h_xn = new int[mnmax];

    // Folded mode table: m = 0..mpol-1, n = 0..ntor, mode = m*(ntor+1)+n.
    // xn = the RAW toroidal mode number n. The zeta grid covers one field
    // period (zeta = 2*pi*k/nzeta, physical phi = zeta/nfp), so the basis is
    // cos(m*theta - n*zeta) — matching vmecpp (fourier_basis.cc:122,137).
    // The toroidal DERIVATIVES carry the nfp factor (cosnvn = n*nfp*cosnv).
    for (int m = 0; m < p.mpol; ++m)
        for (int n = 0; n < p.ntor + 1; ++n) {
            int mode = m * (p.ntor + 1) + n;
            h_xm[mode] = m; h_xn[mode] = n;
        }
    auto allocInt = [&](int*& dst, size_t count, const char* name) {
        if (arena) dst = arena->alloc_span<int>(name, count);
        else cumes::check_cuda(cudaMalloc(&dst, count * sizeof(int)), name);
    };
    allocInt(mt.d_xm, mnmax, "mode_table/xm");
    allocInt(mt.d_xn, mnmax, "mode_table/xn");
    cumes::check_cuda(cudaMemcpy(mt.d_xm, h_xm, (size_t)mnmax * sizeof(int), cudaMemcpyHostToDevice), "xm");
    cumes::check_cuda(cudaMemcpy(mt.d_xn, h_xn, (size_t)mnmax * sizeof(int), cudaMemcpyHostToDevice), "xn");
    delete[] h_xm; delete[] h_xn;
    mt.arena_backed = (arena != nullptr);
    return mt;
}

template <typename T>
FourierPlan<T> fourierCreate(const DeviceParams<T>& p, cumes::DeviceArena* arena) {
    FourierPlan<T> fp{};

    // The mode table (d_xm/d_xn) is built by cumes::modeTableCreate (resolution
    // metadata, not transform scratch); the real-space geometry/force/combined
    // arrays live in the stage-owned RealSpaceStorage (realSpaceCreate). The
    // FourierPlan therefore owns only transform scratch below (blueprint §6.2/§6.6).

    // ---- cuFFT backend: poloidal tables, scratch, plans ----
    {
        using Complex = typename FftTraits<T>::Complex;
        size_t n_spectra = (size_t)12 * p.mpol * p.ns * (p.nzeta / 2 + 1);
        size_t n_zeta    = (size_t)12 * p.mpol * p.ns * p.nzeta;
        if (arena) {
            // cuFFT requires 16-byte-aligned data (cufftDoubleComplex is a
            // 16-byte vector; the real input of a D2Z is processed as double2
            // chunks). cudaMalloc's 256-byte alignment masked this in the
            // legacy path; the arena must request it explicitly.
            fp.d_zeta_spectra = arena->alloc_span<Complex>("fourier/zeta_spectra", n_spectra, 16);
            fp.d_zeta_real    = arena->alloc_span<T>("fourier/zeta_real", n_zeta, 16);
        } else {
            cumes::check_cuda(cudaMalloc(&fp.d_zeta_spectra,
                n_spectra * sizeof(Complex)), "spectra");
            cumes::check_cuda(cudaMalloc(&fp.d_zeta_real,
                n_zeta * sizeof(T)), "zeta");
        }
    }
    auto amt = [&](T*& q, const char* n, size_t cnt) {
        if (arena) q = arena->alloc_span<T>(n, cnt);
        else cumes::check_cuda(cudaMalloc(&q, cnt * sizeof(T)), n);
    };
    amt(fp.d_cos_th, "costh", p.mpol * p.ntheta);  amt(fp.d_sin_th, "sinth", p.mpol * p.ntheta);
    amt(fp.d_mcos_th, "mcosth", p.mpol * p.ntheta); amt(fp.d_msin_th, "msinth", p.mpol * p.ntheta);
    amt(fp.d_fwd_w, "fwdw", p.ntheta / 2 + 1);

    auto* h_cos_th  = new T[p.mpol * p.ntheta];
    auto* h_sin_th  = new T[p.mpol * p.ntheta];
    auto* h_mcos_th = new T[p.mpol * p.ntheta];
    auto* h_msin_th = new T[p.mpol * p.ntheta];
    for (int m = 0; m < p.mpol; ++m)
        for (int l = 0; l < p.ntheta; ++l) {
            T th = T(2.0 * M_PI) * T(l) / T(p.ntheta);
            T c = cos(m * th), s = sin(m * th);
            h_cos_th[m * p.ntheta + l]  = c;
            h_sin_th[m * p.ntheta + l]  = s;
            h_mcos_th[m * p.ntheta + l] = m * c;
            h_msin_th[m * p.ntheta + l] = -m * s;
        }
    // Reduced-grid trapezoid weights for the forward quadrature (vmecpp
    // intNorm = 1/(nZeta*(nThetaRed-1)), endpoint-halved).
    int nThetaRed = p.ntheta / 2 + 1;
    auto* h_fwd_w = new T[nThetaRed];
    T intNorm = T(1.0) / T(p.nzeta * (nThetaRed - 1));
    for (int l = 0; l < nThetaRed; ++l) {
        h_fwd_w[l] = intNorm;
        if (l == 0 || l == nThetaRed - 1) h_fwd_w[l] *= T(0.5);
    }
    cumes::check_cuda(cudaMemcpy(fp.d_cos_th, h_cos_th, (size_t)p.mpol * p.ntheta * sizeof(T),
                         cudaMemcpyHostToDevice), "cp costh");
    cumes::check_cuda(cudaMemcpy(fp.d_sin_th, h_sin_th, (size_t)p.mpol * p.ntheta * sizeof(T),
                         cudaMemcpyHostToDevice), "cp sinth");
    cumes::check_cuda(cudaMemcpy(fp.d_mcos_th, h_mcos_th, (size_t)p.mpol * p.ntheta * sizeof(T),
                         cudaMemcpyHostToDevice), "cp mcosth");
    cumes::check_cuda(cudaMemcpy(fp.d_msin_th, h_msin_th, (size_t)p.mpol * p.ntheta * sizeof(T),
                         cudaMemcpyHostToDevice), "cp msinth");
    cumes::check_cuda(cudaMemcpy(fp.d_fwd_w, h_fwd_w, (size_t)nThetaRed * sizeof(T),
                         cudaMemcpyHostToDevice), "cp fwdw");
    delete[] h_cos_th; delete[] h_sin_th; delete[] h_mcos_th; delete[] h_msin_th;
    delete[] h_fwd_w;

    // Batched 1D real FFTs of length nzeta, one batch element per (slot, m, j).
    int n = p.nzeta, nz2 = p.nzeta / 2 + 1;
    int batch = 12 * p.mpol * p.ns;
    int inemb = nz2, outemb = n;
    cumes::check_cufft(cufftPlanMany(&fp.plan_z2d, 1, &n, &inemb, 1, nz2,
                             &outemb, 1, n, FftTraits<T>::kInverse, batch), "plan z2d");
    cumes::check_cufft(cufftPlanMany(&fp.plan_d2z, 1, &n, &outemb, 1, n,
                             &inemb, 1, nz2, FftTraits<T>::kForward, batch), "plan d2z");

    // Phase 6B: disable cuFFT auto-allocation and share one max-sized work
    // area across the two Fourier plans (sequential on one stream, so their
    // lifetimes never overlap). This replaces cuFFT's two per-plan auto
    // allocations (~4 MB each for the W7-X shape) with one buffer.
    {
        size_t wz = 0, wd = 0;
        cumes::check_cufft(cufftSetAutoAllocation(fp.plan_z2d, 0), "cufft noauto z2d");
        cumes::check_cufft(cufftSetAutoAllocation(fp.plan_d2z, 0), "cufft noauto d2z");
        cumes::check_cufft(cufftGetSize(fp.plan_z2d, &wz), "cufftGetSize z2d");
        cumes::check_cufft(cufftGetSize(fp.plan_d2z, &wd), "cufftGetSize d2z");
        fp.cufft_work_bytes = (wz > wd) ? wz : wd;
        if (fp.cufft_work_bytes > 0) {
            cumes::check_cuda(cudaMalloc(&fp.d_cufft_work, fp.cufft_work_bytes),
                              "cufft work");
            cumes::check_cufft(cufftSetWorkArea(fp.plan_z2d, fp.d_cufft_work),
                               "cufft workarea z2d");
            cumes::check_cufft(cufftSetWorkArea(fp.plan_d2z, fp.d_cufft_work),
                               "cufft workarea d2z");
        }
    }

    // ---- compact de-alias bandpass scratch + plans (constraint step 2) -----
    // Moved here from constraintCreate so the transform operator owns all
    // transform scratch (blueprint §5.1/§6.8). The bandpass is a D2Z/Z2D round
    // trip over 2*(mpol-2)*(ns-1) batch elements (slots 0/1 analysis, 4/5
    // synthesis); its work area is shared the same way as the main plans'.
    {
        using Complex = typename FftTraits<T>::Complex;
        int batchDa = 2 * (p.mpol - 2) * (p.ns - 1);
        if (arena) {
            fp.d_zeta_real_c = arena->alloc_span<T>("fourier/zeta_real_c", (size_t)batchDa * n, 16);
            fp.d_zeta_spectra_c = arena->alloc_span<Complex>("fourier/zeta_spectra_c", (size_t)batchDa * nz2, 16);
        } else {
            cumes::check_cuda(cudaMalloc(&fp.d_zeta_real_c, (size_t)batchDa * n * sizeof(T)), "zeta_real_c");
            cumes::check_cuda(cudaMalloc(&fp.d_zeta_spectra_c, (size_t)batchDa * nz2 * sizeof(Complex)), "zeta_spectra_c");
        }
        cumes::check_cufft(cufftPlanMany(&fp.plan_d2z_da, 1, &n, &n, 1, n,
                                 &nz2, 1, nz2, FftTraits<T>::kForward, batchDa), "plan d2z_da");
        cumes::check_cufft(cufftPlanMany(&fp.plan_z2d_da, 1, &n, &nz2, 1, nz2,
                                 &n, 1, n, FftTraits<T>::kInverse, batchDa), "plan z2d_da");
        size_t wda = 0, wza = 0;
        cumes::check_cufft(cufftSetAutoAllocation(fp.plan_d2z_da, 0), "cufft noauto d2z_da");
        cumes::check_cufft(cufftSetAutoAllocation(fp.plan_z2d_da, 0), "cufft noauto z2d_da");
        cumes::check_cufft(cufftGetSize(fp.plan_d2z_da, &wda), "cufftGetSize d2z_da");
        cumes::check_cufft(cufftGetSize(fp.plan_z2d_da, &wza), "cufftGetSize z2d_da");
        fp.cufft_work_bytes_c = (wda > wza) ? wda : wza;
        if (fp.cufft_work_bytes_c > 0) {
            cumes::check_cuda(cudaMalloc(&fp.d_cufft_work_c, fp.cufft_work_bytes_c),
                              "cufft work c");
            cumes::check_cufft(cufftSetWorkArea(fp.plan_d2z_da, fp.d_cufft_work_c),
                               "cufft workarea d2z_da");
            cumes::check_cufft(cufftSetWorkArea(fp.plan_z2d_da, fp.d_cufft_work_c),
                               "cufft workarea z2d_da");
        }
    }
    fp.arena_backed = (arena != nullptr);
    return fp;
}

template <typename T>
void fourierFree(FourierPlan<T>& fp) {
    if (!fp.arena_backed) {
        cudaFree(fp.d_zeta_spectra); cudaFree(fp.d_zeta_real);
        cudaFree(fp.d_cos_th); cudaFree(fp.d_sin_th);
        cudaFree(fp.d_mcos_th); cudaFree(fp.d_msin_th); cudaFree(fp.d_fwd_w);
        cudaFree(fp.d_zeta_real_c); cudaFree(fp.d_zeta_spectra_c);
    }
    cufftDestroy(fp.plan_z2d); cufftDestroy(fp.plan_d2z);
    cufftDestroy(fp.plan_d2z_da); cufftDestroy(fp.plan_z2d_da);
    if (fp.d_cufft_work) cudaFree(fp.d_cufft_work);
    if (fp.d_cufft_work_c) cudaFree(fp.d_cufft_work_c);
    fp = FourierPlan<T>{};
}

// ---------------------------------------------------------------------------
// RealSpaceStorage (stage-owned geometry/force/combined buffers)
// ---------------------------------------------------------------------------
template <typename T>
cumes::RealSpaceStorage<T> realSpaceCreate(const DeviceParams<T>& p,
                                           cumes::DeviceArena* arena) {
    cumes::RealSpaceStorage<T> rs{};
    const int nZnT = p.nZnT;
    const size_t nbytes_real = p.ns * nZnT * sizeof(T);
    auto am = [&](T*& q, const char* n) {
        if (arena) q = arena->alloc_span<T>(n, (size_t)p.ns * nZnT);
        else cumes::check_cuda(cudaMalloc(&q, nbytes_real), n);
    };
    am(rs.d_r_e, "r_e"); am(rs.d_z_e, "z_e"); am(rs.d_l_e, "l_e");
    am(rs.d_ru_e, "ru_e"); am(rs.d_zu_e, "zu_e"); am(rs.d_lu_e, "lu_e");
    am(rs.d_r_o, "r_o"); am(rs.d_z_o, "z_o"); am(rs.d_l_o, "l_o");
    am(rs.d_ru_o, "ru_o"); am(rs.d_zu_o, "zu_o"); am(rs.d_lu_o, "lu_o");
    am(rs.d_r_real, "r_r"); am(rs.d_z_real, "z_r"); am(rs.d_l_real, "l_r");
    am(rs.d_ru_real, "ru_r"); am(rs.d_zu_real, "zu_r"); am(rs.d_lu_real, "lu_r");
    am(rs.d_rv_real, "rv_r"); am(rs.d_zv_real, "zv_r"); am(rs.d_lv_real, "lv_r");
    // The combined (e+o) *_real buffers are only produced by
    // fourierCombineParity / inverseDFT(do_combine=true). Zero them here so a
    // diagnostic dump taken before the first combine is deterministic (the old
    // code dumped uninitialized memory, whose bytes shifted with the allocation
    // order — see the step_A_l_real_iter_1 dump in solver_impl.cuh). Parity
    // arrays are left as-is: inverseDFT writes them before any consumer reads.
    cumes::check_cuda(cudaMemset(rs.d_r_real, 0, nbytes_real), "zero r_r");
    cumes::check_cuda(cudaMemset(rs.d_z_real, 0, nbytes_real), "zero z_r");
    cumes::check_cuda(cudaMemset(rs.d_l_real, 0, nbytes_real), "zero l_r");
    cumes::check_cuda(cudaMemset(rs.d_ru_real, 0, nbytes_real), "zero ru_r");
    cumes::check_cuda(cudaMemset(rs.d_zu_real, 0, nbytes_real), "zero zu_r");
    cumes::check_cuda(cudaMemset(rs.d_lu_real, 0, nbytes_real), "zero lu_r");
    cumes::check_cuda(cudaMemset(rs.d_rv_real, 0, nbytes_real), "zero rv_r");
    cumes::check_cuda(cudaMemset(rs.d_zv_real, 0, nbytes_real), "zero zv_r");
    cumes::check_cuda(cudaMemset(rs.d_lv_real, 0, nbytes_real), "zero lv_r");
    // Parity-split toroidal derivatives (3D geometry needs them separately)
    am(rs.d_rv_e, "rve"); am(rs.d_rv_o, "rvo");
    am(rs.d_zv_e, "zve"); am(rs.d_zv_o, "zvo");
    am(rs.d_lv_e, "lve"); am(rs.d_lv_o, "lvo");
    am(rs.d_armn_e, "ae"); am(rs.d_armn_o, "ao");
    am(rs.d_azmn_e, "aze"); am(rs.d_azmn_o, "azo");
    am(rs.d_brmn_e, "be"); am(rs.d_brmn_o, "bo");
    am(rs.d_bzmn_e, "bze"); am(rs.d_bzmn_o, "bzo");
    am(rs.d_blmn_e, "ble"); am(rs.d_blmn_o, "blo");
    am(rs.d_crmn_e, "ce"); am(rs.d_crmn_o, "co");
    am(rs.d_czmn_e, "cze"); am(rs.d_czmn_o, "czo");
    am(rs.d_clmn_e, "cle"); am(rs.d_clmn_o, "clo");
    rs.arena_backed = (arena != nullptr);
    return rs;
}

template <typename T>
void realSpaceFree(cumes::RealSpaceStorage<T>& rs) {
    if (!rs.arena_backed) {
        auto cuFree = [](T* p) { cudaFree(p); };
        cuFree(rs.d_r_e); cuFree(rs.d_z_e); cuFree(rs.d_l_e);
        cuFree(rs.d_ru_e); cuFree(rs.d_zu_e); cuFree(rs.d_lu_e);
        cuFree(rs.d_r_o); cuFree(rs.d_z_o); cuFree(rs.d_l_o);
        cuFree(rs.d_ru_o); cuFree(rs.d_zu_o); cuFree(rs.d_lu_o);
        cuFree(rs.d_r_real); cuFree(rs.d_z_real); cuFree(rs.d_l_real);
        cuFree(rs.d_ru_real); cuFree(rs.d_zu_real); cuFree(rs.d_lu_real);
        cuFree(rs.d_rv_real); cuFree(rs.d_zv_real); cuFree(rs.d_lv_real);
        cuFree(rs.d_rv_e); cuFree(rs.d_rv_o);
        cuFree(rs.d_zv_e); cuFree(rs.d_zv_o);
        cuFree(rs.d_lv_e); cuFree(rs.d_lv_o);
        cuFree(rs.d_armn_e); cuFree(rs.d_armn_o); cuFree(rs.d_azmn_e); cuFree(rs.d_azmn_o);
        cuFree(rs.d_brmn_e); cuFree(rs.d_brmn_o); cuFree(rs.d_bzmn_e); cuFree(rs.d_bzmn_o);
        cuFree(rs.d_blmn_e); cuFree(rs.d_blmn_o);
        cuFree(rs.d_crmn_e); cuFree(rs.d_crmn_o); cuFree(rs.d_czmn_e); cuFree(rs.d_czmn_o);
        cuFree(rs.d_clmn_e); cuFree(rs.d_clmn_o);
    }
    rs = cumes::RealSpaceStorage<T>{};
}

// ---- cuFFT backend: inverse ---------------------------------------------
// Mirror of vmecpp's fft_toroidal.cc (FourierToReal3DSymmFastPoloidal):
// the ζ direction is a real inverse FFT (cuFFT Z2D), the θ direction stays a
// direct basis-table sum. All nine real-space outputs (r/z/l, their θ and ζ
// derivatives) come from the 12 slots; the m-parity split into e/o arrays and
// the odd-m scalxc division (maxsc) happen on the target side, as in vmecpp.
template <typename T>
__global__ void inversePackKernel(
    cumes::SpectralView<const T, cumes::PhysicalStateDomain> coeff,
    const int* __restrict__ xm, const int* __restrict__ xn,
    int ns, int mpol, int ntor, int nfp, int nz2,
    typename FftTraits<T>::Complex* __restrict__ spectra)
{
    using Complex = typename FftTraits<T>::Complex;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= ns * mpol * (ntor + 1)) return;
    int j = t % ns, mode = t / ns;
    int m = xm[mode], n = xn[mode];
    T nf = T(n * nfp);
    T rc = coeff(cumes::SpectralComponent::Rcc, mode, j);
    T rs = coeff(cumes::SpectralComponent::Rss, mode, j);
    T zs = coeff(cumes::SpectralComponent::Zsc, mode, j);
    T zc = coeff(cumes::SpectralComponent::Zcs, mode, j);
    T lsc = coeff(cumes::SpectralComponent::Lsc, mode, j);
    T lcs = coeff(cumes::SpectralComponent::Lcs, mode, j);
    // cuFFT's Z2D synthesis is f[k] = X[0] + 2*Σ Re(X[n])cos - 2*Σ Im(X[n])sin,
    // so the n>=1 bins are halved (cancelling the 2× exactly) and the DST
    // slots carry a minus (vmecpp FillDct/FillDst). The ζ-derivative slots
    // carry the n*nfp factor with vmecpp's signs (FillDctDeriv: Im = +spec*n*nfp,
    // FillDstDeriv: Re = +spec*n*nfp). The n=0 bins of the DST/derivative
    // slots stay 0 (sin(nζ) = n*nfp = 0); no mscale/nscale — the cuMES state
    // is plain physical (λ carries ms*ns inside the state values).
    T half   = (n == 0) ? T(1.0) : T(0.5);
    T shalf  = (n == 0) ? T(0.0) : T(0.5);
    T dhalf  = (n == 0) ? T(0.0) : T(0.5) * nf;
    size_t step = (size_t)mpol * ns * nz2;
    Complex* slot = spectra + ((size_t)m * ns + j) * nz2 + n;
    slot[0 * step] = Complex{rc * half, T(0.0)};
    slot[1 * step] = Complex{T(0.0), -rs * shalf};
    slot[2 * step] = Complex{T(0.0), +rc * dhalf};
    slot[3 * step] = Complex{+rs * dhalf, T(0.0)};
    slot[4 * step] = Complex{zs * half, T(0.0)};
    slot[5 * step] = Complex{T(0.0), -zc * shalf};
    slot[6 * step] = Complex{T(0.0), +zs * dhalf};
    slot[7 * step] = Complex{+zc * dhalf, T(0.0)};
    slot[8 * step] = Complex{lsc * half, T(0.0)};
    slot[9 * step] = Complex{T(0.0), -lcs * shalf};
    slot[10 * step] = Complex{T(0.0), +lsc * dhalf};
    slot[11 * step] = Complex{+lcs * dhalf, T(0.0)};
}

// Poloidal accumulation over the FULL θ grid. Grid (ns, 3): blockIdx.y
// selects the slot group (0 = R slots 0-3, 1 = Z slots 4-7, 2 = λ slots
// 8-11), blockIdx.x the surface j. Threads (l1=θ-half, k=ζ); each thread
// handles θ points l1 and l1 + ntheta/2 and sums the m-loop serially — the
// same accumulation order as inverseDFTKernel — then writes each output
// point exactly once. Outputs the e/o parity arrays: even m -> *_e, odd m
// -> *_o divided by maxsc (vmecpp's scalxc odd decomposition). Splitting
// the 12 slots into 3 groups cuts the shared memory per block 41.5 KB ->
// 13.8 KB, raising occupancy from 1 to 3 blocks/SM (was latency-bound at
// ~425 us/iter; the group selection is uniform per block, so the per-thread
// arithmetic is unchanged — bit-identical).
//
// Slot-to-output mapping (verified against inverseDFTKernel and
// fft_toroidal.cc). Per group, slots are (cos-slot, sin-slot, cosN-slot,
// sinN-slot) and the θ basis differs: R multiplies (cos, sin), Z and λ
// multiply (sin, cos), and λ's ζ-derivative is negated:
//   R: v = c0*cos + c1*sin      vu = c0*msin + c1*mcos
//      vv = c2*cos + c3*sin
//   Z: v = c0*sin + c1*cos      vu = c0*mcos + c1*msin
//      vv = c2*sin + c3*cos
//   λ: v = c0*sin + c1*cos      vu = c0*mcos + c1*msin
//      vv = -(c2*sin + c3*cos)
template <typename T, bool FuseRzCon = false>
__global__ void inverseAccumulateKernel(
    const T* __restrict__ zeta_real,
    const T* __restrict__ cos_th, const T* __restrict__ sin_th,
    const T* __restrict__ mcos_th, const T* __restrict__ msin_th,
    int ns, int mpol, int ntheta, int nzeta, int nZnT, int slot0,
    T* __restrict__ e0, T* __restrict__ e1, T* __restrict__ e2,
    T* __restrict__ o0, T* __restrict__ o1, T* __restrict__ o2,
    int kTile, T* __restrict__ rCon, T* __restrict__ zCon)
{
    // slot0: 0 = R slots 0-3, 4 = Z slots 4-7, 8 = λ slots 8-11
    // Thread mapping: l1 = threadIdx.x (fastest), k = threadIdx.y — the
    // output stores at idx = j*nZnT + k*ntheta + l then vary l fastest and
    // coalesce; the m-loop shared reads (sm[.. + k]) become broadcasts.
    // The ζ direction is TILED (blockIdx.y selects the k-tile, kTile-wide):
    // the launch block no longer embeds the full nzeta product, so the block
    // size stays bounded for larger angular grids. Every output point is
    // computed by the same arithmetic as the untiled kernel (bit-identical).
    // FuseRzCon (blueprint §8.4) additionally accumulates the xmpq = m(m-1)
    // weighted R/Z sums into rCon/zCon (full fields, no parity split, no
    // scalxc) — the arithmetic the retired reference path performed as a
    // separate xmpq-weighted inverse transform, moved into the main accumulator
    // and gated at compile time so the FuseRzCon=false path is bit-identical.
    int j = blockIdx.x;
    int k0 = blockIdx.y * kTile;
    int k = threadIdx.y + k0;
    int l1 = threadIdx.x;
    int nthreads = blockDim.x * blockDim.y;
    T* sh = static_cast<T*>(dynSharedBase());   // [4][mpol][kTile]
    for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < 4 * mpol * kTile; i += nthreads) {
        int s = i / (mpol * kTile), rem = i - s * mpol * kTile;
        int m = rem / kTile, kk = rem % kTile;
        int kk_abs = k0 + kk;
        sh[i] = (kk_abs < nzeta) ? zeta_real[(((size_t)(slot0 + s) * mpol + m) * ns + j) * nzeta + kk_abs]
                                 : T(0.0);  // tail tile: zeros (never used)
    }
    __syncthreads();
    if (k >= nzeta) return;  // tail tile past the grid
    T maxsc = fmax(sqrt(T(j) / T(ns - 1)), sqrt(T(1.0) / T(ns - 1)));
    T facO = T(1.0) / maxsc;
    bool isR = (slot0 == 0);
    T signV = (slot0 == 8) ? T(-1.0) : T(1.0);
    size_t mstride = (size_t)mpol * kTile;
    #pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
        int l = l1 + pass * (ntheta / 2);
        T v0e = T(0), v1e = T(0), v2e = T(0);
        T v0o = T(0), v1o = T(0), v2o = T(0);
        T rcon = T(0), zcon = T(0);
        for (int m = 0; m < mpol; ++m) {
            const T* sm = sh + m * kTile;
            T c0 = sm[0 * mstride + k - k0], c1 = sm[1 * mstride + k - k0];
            T c2 = sm[2 * mstride + k - k0], c3 = sm[3 * mstride + k - k0];
            T cosm = cos_th[m * ntheta + l], sinm = sin_th[m * ntheta + l];
            T mcos = mcos_th[m * ntheta + l], msin = msin_th[m * ntheta + l];
            T fac = (m % 2 == 1) ? facO : T(1.0);
            T t0 = isR ? cosm : sinm, t1 = isR ? sinm : cosm;
            T u0 = isR ? msin : mcos, u1 = isR ? mcos : msin;
            T v0 = fac * (c0 * t0 + c1 * t1);
            T v1 = fac * (c0 * u0 + c1 * u1);
            T v2 = signV * fac * (c2 * t0 + c3 * t1);
            if constexpr (FuseRzCon) {
                // rCon from the R slots (c0=Rcc cos, c1=Rss sin), zCon from
                // the Z slots (c0=Zsc sin, c1=Zcs cos) — no fac/maxsc, the
                // full-field reconstruction.
                T xmpq = T(m) * T(m - 1);
                rcon += xmpq * (c0 * cosm + c1 * sinm);
                zcon += xmpq * (c0 * sinm + c1 * cosm);
            }
            if (m % 2 == 1) { v0o += v0; v1o += v1; v2o += v2; }
            else            { v0e += v0; v1e += v1; v2e += v2; }
        }
        int idx = j * nZnT + k * ntheta + l;
        e0[idx] = v0e; e1[idx] = v1e; e2[idx] = v2e;
        o0[idx] = v0o; o1[idx] = v1o; o2[idx] = v2o;
        if constexpr (FuseRzCon) {
            if (rCon != nullptr) rCon[idx] = rcon;   // R-slot launch only
            if (zCon != nullptr) zCon[idx] = zcon;   // Z-slot launch only
        }
    }
}

// The 9 combined (e+o) real-space arrays (used by the dump machinery and the
// legacy test-only forwardDFTDirect).
template <typename T>
__global__ void combineParityKernel(
    const T* __restrict__ r_e,  const T* __restrict__ z_e,
    const T* __restrict__ l_e,  const T* __restrict__ ru_e,
    const T* __restrict__ zu_e, const T* __restrict__ lu_e,
    const T* __restrict__ rv_e, const T* __restrict__ zv_e,
    const T* __restrict__ lv_e, const T* __restrict__ r_o,
    const T* __restrict__ z_o,  const T* __restrict__ l_o,
    const T* __restrict__ ru_o, const T* __restrict__ zu_o,
    const T* __restrict__ lu_o, const T* __restrict__ rv_o,
    const T* __restrict__ zv_o, const T* __restrict__ lv_o,
    int nZnT, int ns,
    T* __restrict__ r_real,  T* __restrict__ z_real,
    T* __restrict__ l_real,  T* __restrict__ ru_real,
    T* __restrict__ zu_real, T* __restrict__ lu_real,
    T* __restrict__ rv_real, T* __restrict__ zv_real,
    T* __restrict__ lv_real)
{
    int j = blockIdx.y, k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nZnT) return;
    int idx = k + j * nZnT;
    r_real[idx] = r_e[idx] + r_o[idx];   z_real[idx] = z_e[idx] + z_o[idx];
    l_real[idx] = l_e[idx] + l_o[idx];   ru_real[idx] = ru_e[idx] + ru_o[idx];
    zu_real[idx] = zu_e[idx] + zu_o[idx]; lu_real[idx] = lu_e[idx] + lu_o[idx];
    rv_real[idx] = rv_e[idx] + rv_o[idx]; zv_real[idx] = zv_e[idx] + zv_o[idx];
    lv_real[idx] = lv_e[idx] + lv_o[idx];
}

// ζ-tile width for the accumulate/reduce kernels: the block no longer embeds
// the full ntheta/2 × nzeta product (unbounded blocks for larger grids).
// blockDim.y = kTile, capped so blockDim.x*blockDim.y <= 1024 (and near 512
// for occupancy); ntheta is input-capped at 256, so blockDim.x <= 128.
static int computeKTile(int blkX, int nzeta) {
    int kt = (blkX >= 1024) ? 1 : std::min(16, 1024 / blkX);
    return std::min(kt, nzeta);
}

template <typename T, bool FuseRzCon = false>
static void inverseDFTCufft(const FourierPlan<T>& fp, cumes::RealSpaceStorage<T>& rs,
                            cumes::SpectralView<const T, cumes::PhysicalStateDomain> coeff,
                            const DeviceParams<T>& p, const int* xm, const int* xn,
                            bool do_combine, T* rCon, T* zCon, cudaStream_t stream) {
    // Zero the half-spectra (only bins n <= ntor are filled).
    cumes::check_cuda(cudaMemsetAsync(fp.d_zeta_spectra, 0,
        (size_t)12 * p.mpol * p.ns * (p.nzeta / 2 + 1) * sizeof(typename FftTraits<T>::Complex), stream), "inv zero");
    int total = p.ns * p.mnmax;
    inversePackKernel<T><<<(total + 255) / 256, 256, 0, stream>>>(
        coeff, xm, xn,
        p.ns, p.mpol, p.ntor, p.nfp, p.nzeta / 2 + 1, fp.d_zeta_spectra);
    cumes::check_cufft(FftTraits<T>::execInverse(fp.plan_z2d, fp.d_zeta_spectra, fp.d_zeta_real), "inv z2d");
    // ζ-tiled accumulate (see inverseAccumulateKernel): block (ntheta/2,
    // kTile), one grid row per (surface, k-tile).
    int kTile = computeKTile(p.ntheta / 2, p.nzeta);
    int nKTiles = (p.nzeta + kTile - 1) / kTile;
    dim3 blk(p.ntheta / 2, kTile);
    dim3 grd(p.ns, nKTiles);
    size_t invSmem = 4 * p.mpol * kTile * sizeof(T);
    // R slots 0-3 -> r/ru/rv (and fused rCon), Z slots 4-7 -> z/zu/zv (and
    // fused zCon), λ slots 8-11 -> l/lu/lv.
    inverseAccumulateKernel<T, FuseRzCon><<<grd, blk, invSmem, stream>>>(
        fp.d_zeta_real,
        fp.d_cos_th, fp.d_sin_th, fp.d_mcos_th, fp.d_msin_th,
        p.ns, p.mpol, p.ntheta, p.nzeta, p.nZnT, 0,
        rs.d_r_e, rs.d_ru_e, rs.d_rv_e, rs.d_r_o, rs.d_ru_o, rs.d_rv_o,
        kTile, rCon, nullptr);
    inverseAccumulateKernel<T, FuseRzCon><<<grd, blk, invSmem, stream>>>(
        fp.d_zeta_real,
        fp.d_cos_th, fp.d_sin_th, fp.d_mcos_th, fp.d_msin_th,
        p.ns, p.mpol, p.ntheta, p.nzeta, p.nZnT, 4,
        rs.d_z_e, rs.d_zu_e, rs.d_zv_e, rs.d_z_o, rs.d_zu_o, rs.d_zv_o,
        kTile, nullptr, zCon);
    inverseAccumulateKernel<T, FuseRzCon><<<grd, blk, invSmem, stream>>>(
        fp.d_zeta_real,
        fp.d_cos_th, fp.d_sin_th, fp.d_mcos_th, fp.d_msin_th,
        p.ns, p.mpol, p.ntheta, p.nzeta, p.nZnT, 8,
        rs.d_l_e, rs.d_lu_e, rs.d_lv_e, rs.d_l_o, rs.d_lu_o, rs.d_lv_o,
        kTile, nullptr, nullptr);
    if (do_combine) {
        dim3 cblk(32), cgrd((p.nZnT + 31) / 32, p.ns);
        combineParityKernel<T><<<cgrd, cblk, 0, stream>>>(
            rs.d_r_e, rs.d_z_e, rs.d_l_e, rs.d_ru_e, rs.d_zu_e, rs.d_lu_e,
            rs.d_rv_e, rs.d_zv_e, rs.d_lv_e,
            rs.d_r_o, rs.d_z_o, rs.d_l_o, rs.d_ru_o, rs.d_zu_o, rs.d_lu_o,
            rs.d_rv_o, rs.d_zv_o, rs.d_lv_o,
            p.nZnT, p.ns,
            rs.d_r_real, rs.d_z_real, rs.d_l_real,
            rs.d_ru_real, rs.d_zu_real, rs.d_lu_real,
            rs.d_rv_real, rs.d_zv_real, rs.d_lv_real);
    }
    cumes::check_cuda(cudaGetLastError(), "inv cuFFT");
}

// Public snapshot: refresh the 9 combined arrays from the CURRENT parity
// arrays (the hot loop runs with do_combine=false and never refreshes them;
// call this before reading any *_real array after a do_combine=false pass).
template <typename T>
void fourierCombineParity(const FourierPlan<T>& fp, cumes::RealSpaceStorage<T>& rs,
                          const DeviceParams<T>& p, cudaStream_t stream) {
    dim3 cblk(32), cgrd((p.nZnT + 31) / 32, p.ns);
    combineParityKernel<T><<<cgrd, cblk, 0, stream>>>(
        rs.d_r_e, rs.d_z_e, rs.d_l_e, rs.d_ru_e, rs.d_zu_e, rs.d_lu_e,
        rs.d_rv_e, rs.d_zv_e, rs.d_lv_e,
        rs.d_r_o, rs.d_z_o, rs.d_l_o, rs.d_ru_o, rs.d_zu_o, rs.d_lu_o,
        rs.d_rv_o, rs.d_zv_o, rs.d_lv_o,
        p.nZnT, p.ns,
        rs.d_r_real, rs.d_z_real, rs.d_l_real,
        rs.d_ru_real, rs.d_zu_real, rs.d_lu_real,
        rs.d_rv_real, rs.d_zv_real, rs.d_lv_real);
    cumes::check_cuda(cudaGetLastError(), "combine parity");
}

template <typename T>
void inverseDFT(const FourierPlan<T>& fp, cumes::RealSpaceStorage<T>& rs,
                cumes::SpectralView<const T, cumes::PhysicalStateDomain> coeff,
                const DeviceParams<T>& p, const int* xm, const int* xn,
                bool do_combine, cudaStream_t stream) {
    inverseDFTCufft<T, false>(fp, rs, coeff, p, xm, xn, do_combine, nullptr, nullptr, stream);
}

template <typename T>
void inverseDFTFused(const FourierPlan<T>& fp, cumes::RealSpaceStorage<T>& rs,
                     cumes::SpectralView<const T, cumes::PhysicalStateDomain> coeff,
                     const DeviceParams<T>& p, const int* xm, const int* xn,
                     bool do_combine, T* rCon, T* zCon,
                     cudaStream_t stream) {
    inverseDFTCufft<T, true>(fp, rs, coeff, p, xm, xn, do_combine, rCon, zCon, stream);
}

// ---- cuFFT backend: forward ----------------------------------------------
// Mirror of vmecpp's dft_ForcesToFourier_3d_symm (fft_toroidal.cc): a fused
// poloidal reduction over the REDUCED θ grid [0, π] builds 12 real ζ-signals
// per (m, j) (the m-parity selects the _e/_o force arrays, xmpq = m(m-1)
// folds the spectral-condensation constraint into tempR/tempZ), a batched
// real FFT gives their half-spectra, and the recovery recombines the bins.
// Slot definitions (signs verified against the reference kernels):
//   rmkcc  += tempR*cos + brmn*(-m*sin)     rmkss += tempR*sin + brmn*(+m*cos)
//   zmksc  += tempZ*sin + bzmn*(+m*cos)     zmkcs += tempZ*cos + bzmn*(-m*sin)
//   lmksc  += blmn*(+m*cos)                 lmkcs += blmn*(-m*sin)
//   rmkccN -= crmn*cos;  rmkssN -= crmn*sin
//   zmkscN -= czmn*sin;  zmkcsN -= czmn*cos
//   lmkscN -= clmn*sin;  lmkcsN -= clmn*cos
// with the θ tables carrying the trapezoid weight (d_fwd_w, intNorm with
// endpoint ½ — the θ > π points are not on vmecpp's reduced grid).
template <typename T>
__global__ void forwardReduceKernel(
    const T* __restrict__ armn_e, const T* __restrict__ armn_o,
    const T* __restrict__ azmn_e, const T* __restrict__ azmn_o,
    const T* __restrict__ brmn_e, const T* __restrict__ brmn_o,
    const T* __restrict__ bzmn_e, const T* __restrict__ bzmn_o,
    const T* __restrict__ crmn_e, const T* __restrict__ crmn_o,
    const T* __restrict__ czmn_e, const T* __restrict__ czmn_o,
    const T* __restrict__ blmn_e, const T* __restrict__ blmn_o,
    const T* __restrict__ clmn_e, const T* __restrict__ clmn_o,
    const T* __restrict__ frcon_e, const T* __restrict__ frcon_o,
    const T* __restrict__ fzcon_e, const T* __restrict__ fzcon_o,
    const T* __restrict__ cos_th, const T* __restrict__ sin_th,
    const T* __restrict__ mcos_th, const T* __restrict__ msin_th,
    const T* __restrict__ fwd_w,
    int ns, int mpol, int ntheta, int nThetaRed, int nzeta, int nZnT,
    T* __restrict__ zeta_real, int kTile)
{
    int j = blockIdx.y, m = blockIdx.x;
    // Thread mapping: l = threadIdx.x (fastest), k = threadIdx.y — the 14
    // force-array loads at idx = j*nZnT + k*ntheta + l then vary l fastest
    // and coalesce. blockDim.x is padded to 16 lanes; threads with
    // l >= nThetaRed contribute zero. The per-(slot, k) sum over the 16
    // l-values is a warp shuffle tree (two k groups per warp, width 16) —
    // replacing the previous shared-memory atomicAdd, which after the
    // dimension swap serialized 16-way per address (2.5 ms/iter).
    // The ζ direction is TILED (blockIdx.z selects the k-tile, kTile-wide)
    // so the launch block stays bounded for larger grids; every (m, j, k)
    // is independent, so the per-point arithmetic is unchanged.
    int k = threadIdx.y + blockIdx.z * kTile;
    int l = threadIdx.x;
    // The k guard covers the tail tile (k >= nzeta): those threads contribute
    // zeros to the shuffle tree, whose result is discarded by the store guard.
    bool active = (l < nThetaRed && k < nzeta);
    T v0 = T(0), v1 = T(0), v2 = T(0), v3 = T(0), v4 = T(0), v5 = T(0);
    T v6 = T(0), v7 = T(0), v8 = T(0), v9 = T(0), v10 = T(0), v11 = T(0);
    if (active) {
        bool m_even = (m % 2 == 0);
        const T* armn = m_even ? armn_e : armn_o;
        const T* azmn = m_even ? azmn_e : azmn_o;
        const T* brmn = m_even ? brmn_e : brmn_o;
        const T* bzmn = m_even ? bzmn_e : bzmn_o;
        const T* crmn = m_even ? crmn_e : crmn_o;
        const T* czmn = m_even ? czmn_e : czmn_o;
        const T* blmn = m_even ? blmn_e : blmn_o;
        const T* clmn = m_even ? clmn_e : clmn_o;
        const T* frcon = m_even ? frcon_e : frcon_o;
        const T* fzcon = m_even ? fzcon_e : fzcon_o;
        T xmpq = T(m) * T(m - 1);
        int idx = j * nZnT + k * ntheta + l;
        T w = fwd_w[l];
        T cosm = w * cos_th[m * ntheta + l], sinm = w * sin_th[m * ntheta + l];
        T mcos = w * mcos_th[m * ntheta + l], msin = w * msin_th[m * ntheta + l];
        T tempR = armn[idx] + xmpq * frcon[idx];
        T tempZ = azmn[idx] + xmpq * fzcon[idx];
        T br = brmn[idx], bz = bzmn[idx];
        T cr = crmn[idx], cz = czmn[idx];
        T bl = blmn[idx], cl = clmn[idx];
        v0 = tempR * cosm + br * msin;
        v1 = tempR * sinm + br * mcos;
        v2 = -cr * cosm;
        v3 = -cr * sinm;
        v4 = tempZ * sinm + bz * mcos;
        v5 = tempZ * cosm + bz * msin;
        v6 = -cz * sinm;
        v7 = -cz * cosm;
        v8 = bl * mcos;
        v9 = bl * msin;
        v10 = -cl * sinm;
        v11 = -cl * cosm;
    }
    // Warp reduction over the 16 l-values (two k groups per warp, width 16).
    // All block threads converge here (threads with l >= nThetaRed only
    // zeroed their contributions), so the active mask names exactly the
    // existing lanes — 0xffffffff would be an invalid mask contract for
    // partial warps (e.g. the 16-thread blocks of the nzeta=1 Solovev grid).
    unsigned mask = __activemask();
    #pragma unroll
    for (int off = 8; off > 0; off >>= 1) {
        v0 += __shfl_down_sync(mask, v0, off, 16);
        v1 += __shfl_down_sync(mask, v1, off, 16);
        v2 += __shfl_down_sync(mask, v2, off, 16);
        v3 += __shfl_down_sync(mask, v3, off, 16);
        v4 += __shfl_down_sync(mask, v4, off, 16);
        v5 += __shfl_down_sync(mask, v5, off, 16);
        v6 += __shfl_down_sync(mask, v6, off, 16);
        v7 += __shfl_down_sync(mask, v7, off, 16);
        v8 += __shfl_down_sync(mask, v8, off, 16);
        v9 += __shfl_down_sync(mask, v9, off, 16);
        v10 += __shfl_down_sync(mask, v10, off, 16);
        v11 += __shfl_down_sync(mask, v11, off, 16);
    }
    if (l == 0 && k < nzeta) {   // k guard: tail tile past the grid
        T* base = zeta_real + ((size_t)m * ns + j) * nzeta;
        size_t step = (size_t)mpol * ns * nzeta;
        #pragma unroll
        for (int s = 0; s < 12; ++s) {
            T v = (s == 0) ? v0 : (s == 1) ? v1 : (s == 2) ? v2 : (s == 3) ? v3
                 : (s == 4) ? v4 : (s == 5) ? v5 : (s == 6) ? v6 : (s == 7) ? v7
                 : (s == 8) ? v8 : (s == 9) ? v9 : (s == 10) ? v10 : v11;
            base[s * step + k] = v;
        }
    }
}

// Coefficient recovery from the 12 half-spectra (bin n of each slot).
// With nf = n*nfp and mn = mscale*nscale:
//   frcc = mn*(Re F_rmkcc + nf*Im F_rmkccN)   frss = mn*(-Im F_rmkss + nf*Re F_rmkssN)
//   fzsc = mn*(Re F_zmksc + nf*Im F_zmkscN)   fzcs = mn*(-Im F_zmkcs + nf*Re F_zmkcsN)
//   flsc = mn*(Re F_lmksc + nf*Im F_lmkscN)   flcs = mn*(-Im F_lmkcs + nf*Re F_lmkcsN)
// Surface coverage (vmecpp dft_ForcesToFourier_3d_symm): axis j=0 keeps only
// the m=0 frcc/fzcs (incl. the crmn/czmn toroidal terms); the LCFS j=ns-1
// keeps only the λ components. Every thread writes ALL SIX families — the
// skipped entries are written as explicit zeros, so the caller no longer needs
// to pre-zero the residual slab (Phase 6A removes the per-iteration memset).
template <typename T>
__global__ void forwardRecoverKernel(
    const typename FftTraits<T>::Complex* __restrict__ spectra,
    const int* __restrict__ xm, const int* __restrict__ xn,
    int ns, int mpol, int mnmax, int nfp, int nz2,
    cumes::SpectralView<T, cumes::DecomposedResidualDomain> f_spec)
{
    using Complex = typename FftTraits<T>::Complex;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= ns * mnmax) return;
    int j = t % ns, mode = t / ns;
    int m = xm[mode], n = xn[mode];
    T nf = T(n * nfp);
    T ms = (m == 0) ? T(1.0) : std::sqrt(T(2.0));
    T nsq = (n == 0) ? T(1.0) : std::sqrt(T(2.0));
    T mn = ms * nsq;
    size_t step = (size_t)mpol * ns * nz2;
    const Complex* slot = spectra + ((size_t)m * ns + j) * nz2 + n;
    Complex F0 = slot[0 * step],  F1 = slot[1 * step],  F2 = slot[2 * step],  F3 = slot[3 * step];
    Complex F4 = slot[4 * step],  F5 = slot[5 * step],  F6 = slot[6 * step],  F7 = slot[7 * step];
    Complex F8 = slot[8 * step],  F9 = slot[9 * step],  F10 = slot[10 * step], F11 = slot[11 * step];
    if (j == 0) {
        // axis: m=0 keeps frcc/fzcs; m>0 and the remaining families are zero
        // (decomposed forces vanish at the magnetic axis).
        f_spec(cumes::SpectralComponent::Rcc, mode, j) =
            (m == 0) ? mn * (F0.x + nf * F2.y) : T(0);
        f_spec(cumes::SpectralComponent::Zsc, mode, j) = T(0);
        f_spec(cumes::SpectralComponent::Lsc, mode, j) = T(0);
        f_spec(cumes::SpectralComponent::Rss, mode, j) = T(0);
        f_spec(cumes::SpectralComponent::Zcs, mode, j) =
            (m == 0) ? mn * (-F5.y + nf * F7.x) : T(0);
        f_spec(cumes::SpectralComponent::Lcs, mode, j) = T(0);
        return;
    }
    if (j == ns - 1) {       // LCFS: λ only (R/Z are fixed-boundary)
        f_spec(cumes::SpectralComponent::Rcc, mode, j) = T(0);
        f_spec(cumes::SpectralComponent::Zsc, mode, j) = T(0);
        f_spec(cumes::SpectralComponent::Lsc, mode, j) = mn * (F8.x + nf * F10.y);
        f_spec(cumes::SpectralComponent::Rss, mode, j) = T(0);
        f_spec(cumes::SpectralComponent::Zcs, mode, j) = T(0);
        f_spec(cumes::SpectralComponent::Lcs, mode, j) = mn * (-F9.y + nf * F11.x);
        return;
    }
    f_spec(cumes::SpectralComponent::Rcc, mode, j) = mn * (F0.x + nf * F2.y);
    f_spec(cumes::SpectralComponent::Zsc, mode, j) = mn * (F4.x + nf * F6.y);
    f_spec(cumes::SpectralComponent::Lsc, mode, j) = mn * (F8.x + nf * F10.y);
    f_spec(cumes::SpectralComponent::Rss, mode, j) = mn * (-F1.y + nf * F3.x);
    f_spec(cumes::SpectralComponent::Zcs, mode, j) = mn * (-F5.y + nf * F7.x);
    f_spec(cumes::SpectralComponent::Lcs, mode, j) = mn * (-F9.y + nf * F11.x);
}

template <typename T>
static void forwardDFTCufft(const FourierPlan<T>& fp, cumes::RealSpaceStorage<T>& rs,
                            cumes::SpectralView<T, cumes::DecomposedResidualDomain> f_spec,
                            const DeviceParams<T>& p, const int* xm, const int* xn,
                            const T* frcon_e, const T* frcon_o,
                            const T* fzcon_e, const T* fzcon_o,
                            cudaStream_t stream) {
    // The recover kernel writes all six families (explicit axis/LCFS zeros), so
    // no pre-zero of the residual slab is needed (Phase 6A removes the memset).
    // ζ-tiled reduce (see forwardReduceKernel): block (16 lanes, kTile),
    // grid (mpol, ns, k-tiles).
    int kTile = computeKTile(16, p.nzeta);
    int nKTiles = (p.nzeta + kTile - 1) / kTile;
    dim3 blk(16, kTile);  // x padded to 16 lanes (warp shuffle width)
    dim3 grd(p.mpol, p.ns, nKTiles);
    forwardReduceKernel<T><<<grd, blk, 0, stream>>>(
        rs.d_armn_e, rs.d_armn_o, rs.d_azmn_e, rs.d_azmn_o,
        rs.d_brmn_e, rs.d_brmn_o, rs.d_bzmn_e, rs.d_bzmn_o,
        rs.d_crmn_e, rs.d_crmn_o, rs.d_czmn_e, rs.d_czmn_o,
        rs.d_blmn_e, rs.d_blmn_o, rs.d_clmn_e, rs.d_clmn_o,
        frcon_e, frcon_o, fzcon_e, fzcon_o,
        fp.d_cos_th, fp.d_sin_th, fp.d_mcos_th, fp.d_msin_th, fp.d_fwd_w,
        p.ns, p.mpol, p.ntheta, p.ntheta / 2 + 1, p.nzeta, p.nZnT,
        fp.d_zeta_real, kTile);
    cumes::check_cufft(FftTraits<T>::execForward(fp.plan_d2z, fp.d_zeta_real, fp.d_zeta_spectra), "fwd d2z");
    int total = p.ns * p.mnmax;
    forwardRecoverKernel<T><<<(total + 255) / 256, 256, 0, stream>>>(
        fp.d_zeta_spectra, xm, xn,
        p.ns, p.mpol, p.mnmax, p.nfp, p.nzeta / 2 + 1, f_spec);
    cumes::check_cuda(cudaGetLastError(), "fwd cuFFT");
}

template <typename T>
void forwardDFT(const FourierPlan<T>& fp, cumes::RealSpaceStorage<T>& rs,
                cumes::SpectralView<T, cumes::DecomposedResidualDomain> f_spec,
                const DeviceParams<T>& p, const int* xm, const int* xn,
                const ConstraintWorkspace<T>& cw, cudaStream_t stream) {
    forwardDFTCufft(fp, rs, f_spec, p, xm, xn, cw.d_frcon_e, cw.d_frcon_o,
                    cw.d_fzcon_e, cw.d_fzcon_o, stream);
}

// ---------------------------------------------------------------------------
// ToroidalFftOperator (owns the FourierPlan; a SpectralOperator backend)
// ---------------------------------------------------------------------------
template <typename T>
void cumes::ToroidalFftOperator<T>::enqueue_inverse(
    cumes::SpectralView<const T, cumes::PhysicalStateDomain> coeff,
    cumes::GeometryParityViews<T> geometry, cumes::RealFieldView<T> rCon,
    cumes::RealFieldView<T> zCon, cudaStream_t stream) {
    // The geometry views alias *rs_ (the stage-owned RealSpaceStorage this
    // operator was constructed with); the fused inverse writes directly into
    // those same arrays. do_combine is false on the hot loop (combined buffers
    // are dump-only and materialized on demand).
    (void)geometry;
    inverseDFTFused(fp_, *rs_, coeff, p_, mt_->d_xm, mt_->d_xn,
                    /*do_combine=*/false, rCon.data(), zCon.data(), stream);
}

template <typename T>
void cumes::ToroidalFftOperator<T>::enqueue_forward(
    cumes::ForceParityViews<const T> real_force,
    cumes::ConstraintForceViews<const T> constraint_force,
    cumes::SpectralView<T, cumes::DecomposedResidualDomain> residual,
    cudaStream_t stream) {
    // `real_force` aliases *rs_ (same arrays); the constraint force is read
    // from the constraint_force views (the constraint owns those buffers).
    (void)real_force;
    forwardDFTCufft(fp_, *rs_, residual, p_, mt_->d_xm, mt_->d_xn,
                    constraint_force.frcon_e.data(),
                    constraint_force.frcon_o.data(),
                    constraint_force.fzcon_e.data(),
                    constraint_force.fzcon_o.data(), stream);
}

template <typename T>
void cumes::ToroidalFftOperator<T>::enqueue_dealias(
    cumes::RealFieldView<const T> gConEff, const T* tcon, const T* faccon,
    cumes::RealFieldView<T> gCon, cudaStream_t stream) {
    constraintDealiasBandpass(p_, fp_, gConEff.data(), tcon, faccon,
                              gCon.data(), stream);
}

template <typename T>
void cumes::ToroidalFftOperator<T>::bind_stream(cudaStream_t stream) {
    cumes::check_cufft(cufftSetStream(fp_.plan_z2d, stream), "set stream z2d");
    cumes::check_cufft(cufftSetStream(fp_.plan_d2z, stream), "set stream d2z");
    cumes::check_cufft(cufftSetStream(fp_.plan_d2z_da, stream), "set stream d2z_da");
    cumes::check_cufft(cufftSetStream(fp_.plan_z2d_da, stream), "set stream z2d_da");
}

template <typename T>
void cumes::ToroidalFftOperator<T>::enqueue_inverse_dump(
    cumes::SpectralView<const T, cumes::PhysicalStateDomain> coeff,
    cudaStream_t stream) {
    inverseDFT(fp_, *rs_, coeff, p_, mt_->d_xm, mt_->d_xn, /*do_combine=*/false,
               stream);
}

template <typename T>
void cumes::ToroidalFftOperator<T>::combine_parity(cudaStream_t stream) {
    fourierCombineParity(fp_, *rs_, p_, stream);
}

