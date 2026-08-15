// descent_operator.hpp — Garabedian accelerated descent + checkpoint boundary
// (blueprint §6.10, §6.11).
//
// Applies the ordered decision: optional descent, optional post-descent
// checkpoint capture, then optional post-descent restore + velocity zero. The
// legacy descentStepKernel + backupState/restoreState are the reference
// implementation; the velocity/state slabs make checkpoint capture one copy.
#pragma once

#include <cuda_runtime.h>

#include "cumes/core/tensor_view.cuh"

namespace cumes {

// The descent/checkpoint action derived from the controller decision (blueprint
// §6.10). For the nonfinite exceptional path perform_descent is false.
struct DescentAction {
  bool perform_descent = false;
  bool refresh_checkpoint_after_descent = false;
  bool restore_checkpoint_after_descent = false;
  double delta_t = 0.0;
  double damping_b1 = 0.0;
  double damping_fac = 0.0;
};

template <class T>
class DescentOperator {
 public:
  void enqueue(SpectralView<const T, PhysicalStateDomain> state,
               SpectralView<T, DecomposedVelocityDomain> velocity,
               SpectralView<const T, DecomposedResidualDomain> residual,
               const DescentAction& action, cudaStream_t stream) const;
};

}  // namespace cumes
