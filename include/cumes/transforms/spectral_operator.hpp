// spectral_operator.hpp — transform-only operator interface (blueprint §6.6).
//
// One interface, two backends (introduced in Phase 7): the AxisymmetricOperator
// (ntor=0, nzeta=1 — direct poloidal synthesis/projection, no length-one cuFFT
// plans) and the ToroidalFftOperator (batched 1D zeta cuFFT + tiled direct
// poloidal accumulation). The operator owns only transform tables/plans/scratch
// — never geometry, force, or diagnostics (blueprint §5.1 dependency rule).
//
// The interface grew (Phase 11 tail) to cover the full transform surface the
// solver needs: the fused inverse carries the xmpq-weighted rCon/zCon constraint
// reconstruction (the generic backend fuses them into the inverse accumulate;
// the axisymmetric backend runs a direct-poloidal rzCon kernel right after its
// synthesis), and the de-alias bandpass is dispatched through the same interface
// (generic: compact cuFFT round trip; axisymmetric: direct poloidal).
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

  // Spectral coefficients -> parity-split real-space geometry + derivatives,
  // and (fused) the xmpq = m(m-1)-weighted constraint reconstruction rCon/zCon
  // (full real-space fields, no parity split / scalxc — blueprint §4.8). `rCon`
  // /`zCon` may be null views to skip that output.
  virtual void enqueue_inverse(SpectralView<const T, PhysicalStateDomain> coefficients,
                               GeometryParityViews<T> geometry,
                               RealFieldView<T> rCon, RealFieldView<T> zCon,
                               cudaStream_t stream) = 0;

  // Parity-split real-space forces + the constraint force (frcon/fzcon) ->
  // six spectral-force families. The constraint force folds into the R/Z
  // projections with the xmpq = m(m-1) weight (blueprint §4.8, §7 pipeline).
  virtual void enqueue_forward(ForceParityViews<const T> real_force,
                               ConstraintForceViews<const T> constraint_force,
                               SpectralView<T, DecomposedResidualDomain> residual,
                               cudaStream_t stream) = 0;

  // De-alias bandpass (constraint step 2, blueprint §6.8): gConEff -> gCon over
  // the bandpass modes m = 1..mpol-2, scaled by tcon/faccon. The generic backend
  // runs the compact cuFFT round trip; the axisymmetric backend a direct
  // poloidal sum.
  virtual void enqueue_dealias(RealFieldView<const T> gConEff, const T* tcon,
                               const T* faccon, RealFieldView<T> gCon,
                               cudaStream_t stream) = 0;
};

}  // namespace cumes
