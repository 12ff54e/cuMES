// fourier.cuh — DFT transforms between spectral (parity-split) and real space.
#pragma once
#include "vmec_types.h"
#include "fft_traits.h"
#include <cublas_v2.h>
#include <cufft.h>

template <typename T> struct ConstraintWorkspace;  // defined in constraint.cuh

// Transforms use a batched 1D real FFT in the toroidal (ζ) direction plus
// direct poloidal synthesis, mirroring vmecpp's FFTX fft_toroidal.cc.
template <typename T>
struct FourierPlan {
    FourierBasis basis;
    cublasHandle_t handle;

    // Batched 1D real FFTs of length nzeta, one batch element per
    // (slot, m, j) with slot in 0..11 (the vmecpp kBatch layout below) and
    // element index ((slot*mpol + m)*ns + j). Slot layout per poloidal
    // mode m (fft_toroidal.cc kBatch):
    //   0 rmkcc   1 rmkss   2 rmkccN   3 rmkssN     (R value + ζ-derivs)
    //   4 zmksc   5 zmkcs   6 zmkscN   7 zmkcsN     (Z value + ζ-derivs)
    //   8 lmksc   9 lmkcs  10 lmkscN  11 lmkcsN     (λ value + ζ-derivs)
    // The d_zeta_* scratch is also reused by the constraint module
    // (constraint.cu: rCon/zCon and the de-aliasing bandpass filter).
    cufftHandle plan_z2d;    // inverse:  half-spectrum -> real
    cufftHandle plan_d2z;    // forward:  real -> half-spectrum
    typename FftTraits<T>::Complex* d_zeta_spectra; // [12*mpol*ns][nzeta/2+1]
    T* d_zeta_real;    // [12*mpol*ns][nzeta] real
    // Poloidal tables (per mode m over the full θ grid): cos(mθ), sin(mθ),
    // m*cos(mθ), -m*sin(mθ). The forward path multiplies by the reduced-grid
    // trapezoid weights in d_fwd_w.
    T* d_cos_th; T* d_sin_th;
    T* d_mcos_th; T* d_msin_th;
    T* d_fwd_w;         // [ntheta/2+1] intNorm weights (endpoints 1/2)

    // Real-space geometry (parity-split, full grid)
    T* d_r_e;  T* d_z_e;  T* d_l_e;
    T* d_ru_e; T* d_zu_e; T* d_lu_e;
    T* d_r_o;  T* d_z_o;  T* d_l_o;
    T* d_ru_o; T* d_zu_o; T* d_lu_o;
    T* d_r_real; T* d_z_real; T* d_l_real;
    T* d_ru_real; T* d_zu_real; T* d_lu_real;
    T* d_rv_real; T* d_zv_real; T* d_lv_real;
    // Parity-split toroidal derivatives (the 3D metric/forces need them)
    T* d_rv_e; T* d_rv_o;
    T* d_zv_e; T* d_zv_o;
    T* d_lv_e; T* d_lv_o;

    // Force components (parity-split). crmn/czmn/clmn are the toroidal force
    // components (3D only; zero for axisymmetric).
    T* d_armn_e; T* d_armn_o;
    T* d_azmn_e; T* d_azmn_o;
    T* d_brmn_e; T* d_brmn_o;
    T* d_bzmn_e; T* d_bzmn_o;
    T* d_blmn_e; T* d_blmn_o;
    T* d_crmn_e; T* d_crmn_o;
    T* d_czmn_e; T* d_czmn_o;
    T* d_clmn_e; T* d_clmn_o;

    // Combined forces (backward compat)
    T* d_fr_real; T* d_fz_real; T* d_fl_real;
};

template <typename T>
FourierPlan<T> fourierCreate(const GridParams<T>& p, cublasHandle_t handle);
template <typename T>
void fourierFree(FourierPlan<T>& fp);

// do_combine=false skips the e/o parity combination (only needed for the
// dump machinery and tests); the hot loop passes false.
template <typename T>
void inverseDFT(const FourierPlan<T>& fp, const SpectralState<T>& st,
                const GridParams<T>& p, bool do_combine = true);

// forwardDFT: parity forces → 6-component spectral forces
// Layout: (6*mnmax, ns) col-major
//   [0*mnmax*ns .. 1*mnmax*ns): f_rmncc
//   [1*mnmax*ns .. 2*mnmax*ns): f_zmnsc
//   [2*mnmax*ns .. 3*mnmax*ns): f_lmnsc
//   [3*mnmax*ns .. 4*mnmax*ns): f_rmnss
//   [4*mnmax*ns .. 5*mnmax*ns): f_zmncs
//   [5*mnmax*ns .. 6*mnmax*ns): f_lmncs
template <typename T>
void forwardDFT(const FourierPlan<T>& fp, T* d_f_spectral,
                const GridParams<T>& p, const ConstraintWorkspace<T>& cw);
