// force_operator.hpp — real-space MHD force boundary (blueprint §6.7).
//
// Evaluates the weak-form MHD + hybrid-lambda force residuals into the
// parity-split force views. The legacy computeForces is the reference; a scalar
// CPU reference with the same operation order is the gate before any split/fuse.
#pragma once

#include <cuda_runtime.h>

#include "cumes/state/real_fields.cuh"

namespace cumes {

template <class T>
class ForceOperator {
 public:
  void enqueue(const BaseGeometryHalfViews<T>& geometry,
               const MagneticFieldViews<T>& field,
               const RadialProfileViews<T>& profiles,
               ForceParityViews<T> force, cudaStream_t stream) const;
};

}  // namespace cumes
