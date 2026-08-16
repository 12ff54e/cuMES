// force_operator.hpp — real-space MHD force boundary (blueprint §6.7).
//
// Evaluates the weak-form MHD + hybrid-lambda force residuals into the
// parity-split force views. The legacy computeForces is the reference; a scalar
// CPU reference with the same operation order is the gate before any split/fuse.
//
// (Migration step 13.3: a stateless operator over typed base-geometry + field +
// profile views — the legacy MetricWorkspace/computeForces are gone.)
#pragma once

#include <cuda_runtime.h>

#include "cumes/config/device_params.hpp"
#include "cumes/state/real_fields.cuh"
#include "cumes/state/real_space_storage.hpp"

namespace cumes {

template <class T>
class ForceOperator {
 public:
  void enqueue(const RealSpaceStorage<T>& rs, const DeviceParams<T>& p,
               const RadialProfileViews<T>& rpv,
               const BaseGeometryHalfViews<T>& base, const MagneticFieldViews<T>& field,
               cudaStream_t stream) const;
};

}  // namespace cumes
