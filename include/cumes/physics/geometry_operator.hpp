// geometry_operator.hpp — base-geometry operator boundary (blueprint §6.7).
//
// Splits base geometry (tau, sqrt(g), covariant metric) from every Jacobian
// division. The legacy computeGeometry is the reference implementation; this is
// the typed enqueue contract the solver/equilibrium operator will drive. The
// device-side Jacobian-status finalization (reset -> reduce -> finalize, with an
// event gating every dependent stream) is the Phase 6A concern and is not part
// of this boundary yet.
#pragma once

#include <cuda_runtime.h>

#include "cumes/state/real_fields.cuh"

namespace cumes {

template <class T>
class GeometryOperator {
 public:
  // full: parity-split full-grid geometry (inverse-DFT output); radial: the
  // immutable radial profiles; half: the (ns-1, ntheta, nzeta) outputs.
  void enqueue(const GeometryParityViews<T>& full,
               const RadialProfileViews<T>& radial,
               BaseGeometryHalfViews<T> half, cudaStream_t stream) const;
};

}  // namespace cumes
