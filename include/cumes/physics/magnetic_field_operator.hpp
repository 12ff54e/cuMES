// magnetic_field_operator.hpp — contravariant/covariant B + total pressure
// boundary (blueprint §6.7).
//
// Consumes base geometry + profiles, produces B^theta/B^zeta, B_theta/B_zeta and
// the total pressure on the half grid. The ncurr=0 (fixed iota) and ncurr=1
// (prescribed current) flows are separate policy paths; the legacy
// geometryKernel/ncurr1FinalizeKernel are the reference implementation.
#pragma once

#include <cuda_runtime.h>

#include "cumes/state/real_fields.cuh"

namespace cumes {

template <class T>
class MagneticFieldOperator {
 public:
  void enqueue(const BaseGeometryHalfViews<T>& geometry,
               const RadialProfileViews<T>& profiles,
               MagneticFieldViews<T> field, cudaStream_t stream) const;
};

}  // namespace cumes
