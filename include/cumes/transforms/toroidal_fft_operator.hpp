// toroidal_fft_operator.hpp — the generic cuFFT transform backend (blueprint
// §6.6, §8.5). Owns the FourierPlan and wraps inverseDFTFused/forwardDFT.
//
// Transitional strangler-fig form: the FourierPlan still carries the real-space
// geometry/force arrays alongside the transform scratch, so the operator exposes
// fourier_plan() for the geometry/force/constraint operators that read/write
// them. The split (transform scratch vs stage-owned RealSpaceStorage) is the
// deferred next step, after which this operator becomes a clean
// SpectralOperator<T> peer of AxisymmetricOperator.
#pragma once

#include <cuda_runtime.h>

#include "cumes/state/spectral_storage.hpp"
#include "constraint.cuh"
#include "fourier.cuh"

namespace cumes {

class DeviceArena;

template <class T>
class ToroidalFftOperator {
 public:
  ToroidalFftOperator(const GridParams<T>& p, DeviceArena* arena)
      : fp_(fourierCreate(p, arena)) {}
  ~ToroidalFftOperator() { fourierFree(fp_); }

  ToroidalFftOperator(const ToroidalFftOperator&) = delete;
  ToroidalFftOperator& operator=(const ToroidalFftOperator&) = delete;
  ToroidalFftOperator(ToroidalFftOperator&&) noexcept = default;
  ToroidalFftOperator& operator=(ToroidalFftOperator&&) noexcept = default;

  FourierPlan<T>& fourier_plan() { return fp_; }
  const FourierPlan<T>& fourier_plan() const { return fp_; }

  // Fused inverse (blueprint §8.4): parity-split geometry + xmpq-weighted
  // rCon/zCon (null skips either output). `rs` is the stage-owned real-space
  // storage the geometry arrays are written into.
  void enqueue_inverse(RealSpaceStorage<T>& rs,
                       SpectralView<const T, PhysicalStateDomain> coeff,
                       const GridParams<T>& p, bool do_combine, T* rCon, T* zCon,
                       cudaStream_t stream);

  // Forward projection: parity forces (in `rs`) + constraint force -> six
  // spectral forces.
  void enqueue_forward(RealSpaceStorage<T>& rs,
                       SpectralView<T, DecomposedResidualDomain> f_spec,
                       const GridParams<T>& p, const ConstraintWorkspace<T>& cw,
                       cudaStream_t stream);

 private:
  FourierPlan<T> fp_;
};

}  // namespace cumes
