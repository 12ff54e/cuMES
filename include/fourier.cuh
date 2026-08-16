// fourier.cuh — DFT transforms between spectral (parity-split) and real space.
#pragma once
#include "vmec_types.h"
#include "fft_traits.h"
#include "cumes/core/tensor_view.cuh"
#include "cumes/state/real_space_storage.hpp"
#include <cufft.h>

namespace cumes { class DeviceArena; }


// Transforms use a batched 1D real FFT in the toroidal (ζ) direction plus
// direct poloidal synthesis, mirroring vmecpp's FFTX fft_toroidal.cc.
template <typename T>
struct FourierPlan {
    // Batched 1D real FFTs of length nzeta, one batch element per
    // (slot, m, j) with slot in 0..11 (the vmecpp kBatch layout below) and
    // element index ((slot*mpol + m)*ns + j). Slot layout per poloidal
    // mode m (fft_toroidal.cc kBatch):
    //   0 rmkcc   1 rmkss   2 rmkccN   3 rmkssN     (R value + ζ-derivs)
    //   4 zmksc   5 zmkcs   6 zmkscN   7 zmkcsN     (Z value + ζ-derivs)
    //   8 lmksc   9 lmkcs  10 lmkscN  11 lmkcsN     (λ value + ζ-derivs)
    // Note: the constraint module does NOT reuse these scratch buffers — it
    // allocates its own compact d_zeta_real_c/d_zeta_spectra_c and plans in
    // constraintCreate (constraint.cuh). These scratch arrays serve only the
    // main transform.
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

    // true when the device arrays above are subspans of a shared DeviceArena
    // (fourierFree then destroys only the cuFFT plans; the arena owns memory).
    bool arena_backed = false;

    // Phase 6B: one shared cuFFT work area for the two Fourier plans (z2d/d2z),
    // with auto-allocation disabled. The two transforms are sequential on one
    // stream, so their work-area lifetimes never overlap; a single max-sized
    // buffer replaces cuFFT's two auto-allocated per-plan areas (~4 MB each for
    // the W7-X shape). Owned here, freed in fourierFree after the plans.
    void* d_cufft_work = nullptr;
    size_t cufft_work_bytes = 0;

    // Compact de-alias bandpass scratch + plans (the constraint's step-2 D2Z/Z2D
    // round trip, blueprint §6.8). Moved here from the ConstraintWorkspace so
    // the transform operator owns all transform tables/plans/scratch and the
    // constraint reaches the bandpass through the SpectralOperator interface
    // (blueprint §5.1 dependency rule). Batch elements: 2*(mpol-2)*(ns-1) —
    // only slots 0/1 (analysis) and 4/5 (synthesis), modes m = 1..mpol-2,
    // surfaces jF = 1..ns-1. Element order: ((slot*(mpol-2)+(m-1))*(ns-1)+(jF-1)),
    // then nzeta (real) / nz2 (spectra) contiguous.
    T* d_zeta_real_c = nullptr;     // [2*(mpol-2)*(ns-1) * nzeta]
    typename FftTraits<T>::Complex* d_zeta_spectra_c = nullptr;  // [... * nz2]
    cufftHandle plan_d2z_da;
    cufftHandle plan_z2d_da;
    void* d_cufft_work_c = nullptr;  // shared work area for the two compact plans
    size_t cufft_work_bytes_c = 0;
};

// fourierCreate allocates the transform scratch (zeta spectra/real, poloidal
// tables, compact de-alias scratch) and creates the four cuFFT plans. The
// folded mode table (d_xm/d_xn) is now built separately by cumes::modeTableCreate
// (resolution-scoped metadata, not transform scratch — blueprint §6.2). With
// `arena == nullptr` every array is its own cudaMalloc (legacy); with an arena
// the arrays are aligned named subspans of one stage allocation.
template <typename T>
FourierPlan<T> fourierCreate(const DeviceParams<T>& p,
                             cumes::DeviceArena* arena = nullptr);
template <typename T>
void fourierFree(FourierPlan<T>& fp);

// Stage-owned real-space storage (geometry + force + combined buffers), split
// out of FourierPlan so the transform operator owns only transform scratch
// (blueprint §6.6). Same pointers/layout the FourierPlan used to hold.
template <typename T>
cumes::RealSpaceStorage<T> realSpaceCreate(const DeviceParams<T>& p,
                                           cumes::DeviceArena* arena = nullptr);
template <typename T>
void realSpaceFree(cumes::RealSpaceStorage<T>& rs);

// do_combine=false skips the e/o parity combination (only needed for the
// dump machinery and tests); the hot loop passes false.
// `coeff` is a component-major view over the contiguous state slab (the
// SpectralStorage::physical() layout); it replaces the legacy 6-pointer
// SpectralState input and indexes bit-for-bit identically.
template <typename T>
void inverseDFT(const FourierPlan<T>& fp, cumes::RealSpaceStorage<T>& rs,
                cumes::SpectralView<const T, cumes::PhysicalStateDomain> coeff,
                const DeviceParams<T>& p, const int* xm, const int* xn,
                bool do_combine = true, cudaStream_t stream = 0);

// Fused inverse DFT (blueprint §8.4): in addition to the 18 parity-split
// geometry arrays, accumulate the xmpq = m(m-1)-weighted R/Z sums into `rCon`
// and `zCon` at the same time as the main inverse poloidal accumulation. This
// eliminates the constraint's duplicate pack + zeta inverse + accumulation
// (a separate xmpq-weighted inverse transform over the compact 4-slot batch,
// retired in Phase 10).
//
// `rCon`/`zCon` are full real-space fields [ns * nZnT] (no parity split, no
// scalxc), matching the ConstraintWorkspace d_rCon/d_zCon layout. `rCon` is
// written by the R-slot launch, `zCon` by the Z-slot launch; either may be
// null to skip that output. Class B vs the retired reference path (the xmpq
// weight moves across the reconstruction, changing summation order at the ULP
// level).
template <typename T>
void inverseDFTFused(const FourierPlan<T>& fp, cumes::RealSpaceStorage<T>& rs,
                     cumes::SpectralView<const T, cumes::PhysicalStateDomain> coeff,
                     const DeviceParams<T>& p, const int* xm, const int* xn,
                     bool do_combine, T* rCon, T* zCon,
                     cudaStream_t stream = 0);

// Materialize the 9 combined (e+o) real-space arrays from the current parity
// arrays. The combined buffers hold whatever the last do_combine=true
// inverseDFT (or this call) produced — the hot loop runs with do_combine=false
// and never refreshes them, so any dump/consumer must call this first to get
// a snapshot of the CURRENT state (never read the combined arrays after a
// do_combine=false pass and assume they are fresh).
template <typename T>
void fourierCombineParity(const FourierPlan<T>& fp, cumes::RealSpaceStorage<T>& rs,
                          const DeviceParams<T>& p, cudaStream_t stream = 0);

// De-alias bandpass (constraint step 2, blueprint §6.8): gConEff -> gCon over
// the bandpass modes m = 1..mpol-2, scaled by tcon/faccon. The compact cuFFT
// round trip uses this FourierPlan's scratch + plans. (Defined in
// constraint_impl.cuh; reached through ToroidalFftOperator::enqueue_dealias.)
template <typename T>
void constraintDealiasBandpass(const DeviceParams<T>& p, const FourierPlan<T>& fp,
                               const T* gConEff, const T* tcon, const T* faccon,
                               T* gCon, cudaStream_t stream = 0);

// forwardDFT: parity forces → 6-component spectral forces
// Layout: (6*mnmax, ns) col-major
//   [0*mnmax*ns .. 1*mnmax*ns): f_rmncc
//   [1*mnmax*ns .. 2*mnmax*ns): f_zmnsc
//   [2*mnmax*ns .. 3*mnmax*ns): f_lmnsc
//   [3*mnmax*ns .. 4*mnmax*ns): f_rmnss
//   [4*mnmax*ns .. 5*mnmax*ns): f_zmncs
//   [5*mnmax*ns .. 6*mnmax*ns): f_lmncs
// `f_spec` is the component-major residual view (DecomposedResidualDomain);
// the layout matches the legacy `d_f_spectral` 6*mnmax*ns slab exactly.
template <typename T>
void forwardDFT(const FourierPlan<T>& fp, cumes::RealSpaceStorage<T>& rs,
                cumes::SpectralView<T, cumes::DecomposedResidualDomain> f_spec,
                const DeviceParams<T>& p, const int* xm, const int* xn,
                const T* frcon_e, const T* frcon_o, const T* fzcon_e, const T* fzcon_o, cudaStream_t stream = 0);
