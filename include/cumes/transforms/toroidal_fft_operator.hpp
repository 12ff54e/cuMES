// toroidal_fft_operator.hpp — the generic cuFFT transform backend (blueprint
// §6.6, §8.5). A SpectralOperator<T> peer of AxisymmetricOperator.
//
// The operator OWNS the cuFFT plans + transform scratch + poloidal tables
// directly (the legacy FourierPlan struct + fourierCreate/fourierFree and the
// inverseDFT/inverseDFTFused/forwardDFT/fourierCombineParity/constraintDealiasBandpass
// free functions are gone — migration step 13.3); the parity-split
// geometry/force arrays live in the stage-owned RealSpaceStorage (non-owning),
// which the operator references so the unified enqueue_inverse/enqueue_forward
// view parameters alias that storage. The folded mode table (xm/xn) is
// resolution-scoped metadata shared via DeviceModeTable.
#pragma once

#include <cstddef>
#include <cufft.h>

#include "cumes/state/mode_table.cuh"
#include "cumes/state/real_space_storage.hpp"
#include "cumes/state/spectral_storage.hpp"
#include "cumes/transforms/spectral_operator.hpp"
#include "fft_traits.h"
#include "vmec_types.h"

namespace cumes {

class DeviceArena;

template <class T>
class ToroidalFftOperator : public SpectralOperator<T> {
 public:
  ToroidalFftOperator(const DeviceParams<T>& p, RealSpaceStorage<T>& rs,
                      const DeviceModeTable& mt, DeviceArena* arena = nullptr);
  ~ToroidalFftOperator() override;

  // Non-movable: the destructor frees the owned cuFFT plans/scratch without
  // nulling, so a defaulted move would double-free (review finding 3.2).
  ToroidalFftOperator(const ToroidalFftOperator&) = delete;
  ToroidalFftOperator& operator=(const ToroidalFftOperator&) = delete;
  ToroidalFftOperator(ToroidalFftOperator&&) noexcept = delete;
  ToroidalFftOperator& operator=(ToroidalFftOperator&&) noexcept = delete;

  // The shared folded-mode table (non-owning; stage/resolution-scoped).
  const int* xm() const { return mt_->d_xm; }
  const int* xn() const { return mt_->d_xn; }

  // The shared cuFFT work-area size (Phase 6B), reported by the benchmark.
  std::size_t cufft_work_bytes() const { return cufft_work_bytes_; }

  // Bind every cuFFT plan (main z2d/d2z + compact de-alias d2z_da/z2d_da) to
  // the explicit compute stream (blueprint §6.6). Called once per stage before
  // any transform runs; keeps the plan handles out of the solver.
  void bind_stream(cudaStream_t stream);

  // ---- transform primitives (were the fourier.cuh free functions) --------
  // Plain inverse DFT: spectral coefficients -> parity-split geometry (+ the
  // combined e+o buffers when do_combine is true). The geometry views alias the
  // stage-owned `rs` this operator was constructed with.
  void inverse(SpectralView<const T, PhysicalStateDomain> coeff, bool do_combine,
               cudaStream_t stream = 0);

  // Fused inverse (blueprint §8.4): parity-split geometry + xmpq-weighted
  // rCon/zCon. rCon/zCon may be null to skip either output.
  void inverse_fused(SpectralView<const T, PhysicalStateDomain> coeff,
                     bool do_combine, T* rCon, T* zCon, cudaStream_t stream = 0);

  // Forward projection: parity forces (aliasing `rs`) + constraint force -> six
  // spectral forces (component-major DecomposedResidualDomain).
  void forward(SpectralView<T, DecomposedResidualDomain> f_spec,
               const T* frcon_e, const T* frcon_o, const T* fzcon_e,
               const T* fzcon_o, cudaStream_t stream = 0);

  // Refresh the 9 combined (e+o) real-space arrays from the CURRENT parity
  // arrays (the hot loop runs do_combine=false and never refreshes them).
  void combine_parity(cudaStream_t stream = 0);

  // De-alias bandpass: the compact cuFFT round trip over gConEff (the
  // xmpq-weighted constraint reconstruction), scaled by tcon/faccon.
  void dealias_bandpass(const T* gConEff, const T* tcon, const T* faccon,
                        T* gCon, cudaStream_t stream = 0);

  // ---- SpectralOperator interface ----------------------------------------
  void enqueue_inverse(SpectralView<const T, PhysicalStateDomain> coefficients,
                       GeometryParityViews<T> geometry,
                       RealFieldView<T> rCon, RealFieldView<T> zCon,
                       cudaStream_t stream) override;

  void enqueue_forward(ForceParityViews<const T> real_force,
                       ConstraintForceViews<const T> constraint_force,
                       SpectralView<T, DecomposedResidualDomain> residual,
                       cudaStream_t stream) override;

  void enqueue_dealias(RealFieldView<const T> gConEff, const T* tcon,
                       const T* faccon, RealFieldView<T> gCon,
                       cudaStream_t stream) override;

  // ---- dump-only accessors (observability, not the hot loop) -------------
  // The solver's DUMP_CUMES_VERIFY diagnostics materialize the parity-split
  // geometry and the combined (e+o) buffers without naming the transform
  // internals. Both are gated on dumpEnabled() at the call site.
  void enqueue_inverse_dump(SpectralView<const T, PhysicalStateDomain> coeff,
                            cudaStream_t stream);

 private:
  template <bool FuseRzCon>
  void inverse_impl(SpectralView<const T, PhysicalStateDomain> coeff,
                    bool do_combine, T* rCon, T* zCon, cudaStream_t stream);
  void forward_impl(SpectralView<T, DecomposedResidualDomain> f_spec,
                    const T* frcon_e, const T* frcon_o, const T* fzcon_e,
                    const T* fzcon_o, cudaStream_t stream);

  // ---- cuFFT plans + transform scratch + poloidal tables (was FourierPlan) --
  cufftHandle plan_z2d_ = 0;  // inverse: half-spectrum -> real
  cufftHandle plan_d2z_ = 0;  // forward: real -> half-spectrum
  typename FftTraits<T>::Complex* d_zeta_spectra_ = nullptr;  // [12*mpol*ns][nz2]
  T* d_zeta_real_ = nullptr;                                  // [12*mpol*ns][nzeta]
  T* d_cos_th_ = nullptr; T* d_sin_th_ = nullptr;
  T* d_mcos_th_ = nullptr; T* d_msin_th_ = nullptr;
  T* d_fwd_w_ = nullptr;  // [ntheta/2+1] intNorm weights (endpoints 1/2)
  bool arena_backed_ = false;

  // Phase 6B: one shared cuFFT work area for the two main plans.
  void* d_cufft_work_ = nullptr;
  std::size_t cufft_work_bytes_ = 0;

  // Compact de-alias bandpass scratch + plans (constraint step 2).
  T* d_zeta_real_c_ = nullptr;      // [2*(mpol-2)*(ns-1) * nzeta]
  typename FftTraits<T>::Complex* d_zeta_spectra_c_ = nullptr;  // [... * nz2]
  cufftHandle plan_d2z_da_ = 0;
  cufftHandle plan_z2d_da_ = 0;
  void* d_cufft_work_c_ = nullptr;
  std::size_t cufft_work_bytes_c_ = 0;

  DeviceParams<T> p_{};
  RealSpaceStorage<T>* rs_ = nullptr;  // non-owning (stage-owned)
  const DeviceModeTable* mt_ = nullptr;  // non-owning (stage/resolution-owned)
};

}  // namespace cumes
