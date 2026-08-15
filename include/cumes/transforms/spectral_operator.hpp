// spectral_operator.hpp — transform-only operator interface (blueprint §6.6).
//
// One interface, two backends (introduced in Phase 7): the AxisymmetricOperator
// (ntor=0, nzeta=1 — direct poloidal synthesis/projection, no length-one cuFFT
// plans) and the ToroidalFftOperator (batched 1D zeta cuFFT + tiled direct
// poloidal accumulation). The operator owns only transform tables/plans/scratch
// — never geometry, force, or diagnostics (blueprint §5.1 dependency rule).
//
// The legacy inverseDFT/forwardDFT free functions remain the reference backend
// until the backends pass differential tests (Phase 7); this header is the
// typed contract the backends implement.
#pragma once

#include <cuda_runtime.h>

#include "cumes/core/tensor_view.cuh"
#include "cumes/state/real_fields.cuh"

namespace cumes {

template <class T>
class SpectralOperator {
 public:
  virtual ~SpectralOperator() = default;

  // Spectral coefficients -> parity-split real-space geometry + derivatives.
  virtual void enqueue_inverse(SpectralView<const T, PhysicalStateDomain> coefficients,
                               GeometryParityViews<T> geometry,
                               cudaStream_t stream) = 0;

  // Parity-split real-space forces -> six spectral-force families.
  virtual void enqueue_forward(ForceParityViews<const T> real_force,
                               SpectralView<T, DecomposedResidualDomain> residual,
                               cudaStream_t stream) = 0;
};

}  // namespace cumes
