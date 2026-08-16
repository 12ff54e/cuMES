// force_operator.hpp — real-space MHD force boundary (blueprint §6.7).
//
// Evaluates the weak-form MHD + hybrid-lambda force residuals into the
// parity-split force views. The legacy computeForces is the reference; a scalar
// CPU reference with the same operation order is the gate before any split/fuse.
//
// Strangler-fig form: a stateless thin wrapper over computeForces (verbatim —
// migration step 6). It takes the same workspace inputs computeForces does and
// forwards; the view-based boundary (base geometry + field + profiles → force)
// is the post base-geometry-split follow-up (migration step 5/6).
#pragma once

#include <cuda_runtime.h>

#include "cumes/state/real_space_storage.hpp"
#include "forces.cuh"

namespace cumes {

template <class T>
class ForceOperator {
 public:
  void enqueue(const RealSpaceStorage<T>& rs, const DeviceParams<T>& p,
               const cumes::RadialProfileViews<T>& rp, const MetricWorkspace<T>& mw,
               cudaStream_t stream) const {
    computeForces(rs, p, rp, mw, stream);
  }
};

}  // namespace cumes
