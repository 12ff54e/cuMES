// toroidal_fft_operator.hpp — the generic cuFFT transform backend (blueprint
// §6.6, §8.5). Owns the FourierPlan and is a SpectralOperator<T> peer of
// AxisymmetricOperator.
//
// Transitional strangler-fig form: the operator references the stage-owned
// RealSpaceStorage (non-owning) where the parity-split geometry/force arrays
// live, so the unified enqueue_inverse/enqueue_forward view parameters alias
// that storage; the operator still exposes fourier_plan() for the solver's
// mode-table / cuFFT-stream setup. The mode-table extraction (blueprint §6.2)
// is the follow-up that lets the solver stop naming FourierPlan.
#pragma once

#include <cuda_runtime.h>

#include "cumes/state/spectral_storage.hpp"
#include "cumes/transforms/spectral_operator.hpp"
#include "constraint.cuh"
#include "fourier.cuh"

namespace cumes {

class DeviceArena;

template <class T>
class ToroidalFftOperator : public SpectralOperator<T> {
 public:
  ToroidalFftOperator(const GridParams<T>& p, RealSpaceStorage<T>& rs,
                      DeviceArena* arena)
      : fp_(fourierCreate(p, arena)), p_(p), rs_(&rs) {}
  ~ToroidalFftOperator() override { fourierFree(fp_); }

  ToroidalFftOperator(const ToroidalFftOperator&) = delete;
  ToroidalFftOperator& operator=(const ToroidalFftOperator&) = delete;
  ToroidalFftOperator(ToroidalFftOperator&&) noexcept = default;
  ToroidalFftOperator& operator=(ToroidalFftOperator&&) noexcept = default;

  FourierPlan<T>& fourier_plan() { return fp_; }
  const FourierPlan<T>& fourier_plan() const { return fp_; }

  // Fused inverse (blueprint §8.4): parity-split geometry + xmpq-weighted
  // rCon/zCon. The geometry views alias the stage-owned `rs` this operator was
  // constructed with; rCon/zCon may be null to skip either output.
  void enqueue_inverse(SpectralView<const T, PhysicalStateDomain> coefficients,
                       GeometryParityViews<T> geometry,
                       RealFieldView<T> rCon, RealFieldView<T> zCon,
                       cudaStream_t stream) override;

  // Forward projection: parity forces (aliasing `rs`) + constraint force -> six
  // spectral forces.
  void enqueue_forward(ForceParityViews<const T> real_force,
                       ConstraintForceViews<const T> constraint_force,
                       SpectralView<T, DecomposedResidualDomain> residual,
                       cudaStream_t stream) override;

  // De-alias bandpass: the compact cuFFT round trip (constraintDealiasBandpass)
  // using this operator's FourierPlan scratch + tables.
  void enqueue_dealias(RealFieldView<const T> gConEff, const T* tcon,
                       const T* faccon, RealFieldView<T> gCon,
                       cudaStream_t stream) override;

 private:
  FourierPlan<T> fp_;
  GridParams<T> p_{};
  RealSpaceStorage<T>* rs_ = nullptr;  // non-owning (stage-owned)
};

}  // namespace cumes
