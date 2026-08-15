// preconditioner.hpp — radial tridiagonal + lambda preconditioner boundary
// (blueprint §6.9).
//
// Computes the per-(m,n) lower/diagonal/upper coefficients and the lambda
// diagonal (the "ar"/"br"/"dr" naming trap is replaced with lower/diagonal/upper
// here), then solves the batched systems through a TridiagonalBackend. The
// legacy preconCompute/preconApply are the reference implementation.
#pragma once

#include <cuda_runtime.h>

#include "cumes/numerics/tridiagonal_backend.hpp"
#include "cumes/state/real_fields.cuh"

namespace cumes {

template <class T>
class Preconditioner {
 public:
  // Refresh the matrix coefficients from the current geometry/field.
  void enqueue_compute(const BaseGeometryHalfViews<T>& geometry,
                       const MagneticFieldViews<T>& field,
                       const RadialProfileViews<T>& profiles,
                       cudaStream_t stream) const;

  // Apply the preconditioner in place to the decomposed residual.
  void enqueue_apply(SpectralView<T, DecomposedResidualDomain> residual,
                     cudaStream_t stream) const;
};

}  // namespace cumes
