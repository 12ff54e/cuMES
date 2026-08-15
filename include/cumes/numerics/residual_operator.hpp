// residual_operator.hpp — invariant/preconditioned residual boundary
// (blueprint §6.9).
//
// Reduces the six spectral-force families to the invariant (FSQR/FSQZ/FSQL) and
// preconditioned (FSQR1/FSQZ1/FSQL1) triples. The target reduces entirely on the
// device into a ControlRecord (double accumulation even for a float-state
// configuration), with one asynchronous copy to a pinned mirror; the legacy
// computeResidualsKernel + host scaling are the reference implementation.
#pragma once

#include <cuda_runtime.h>

#include "cumes/core/tensor_view.cuh"
#include "cumes/solver/control_record.hpp"

namespace cumes {

template <class T>
class ResidualOperator {
 public:
  // Reduce the decomposed residual into the invariant triple (before
  // preconditioning) and the preconditioned triple (after), writing the
  // controller scalars into `record` (device side).
  void enqueue_invariant(SpectralView<const T, DecomposedResidualDomain> residual,
                         ControlRecord<T>& record, cudaStream_t stream) const;

  void enqueue_preconditioned(
      SpectralView<const T, DecomposedResidualDomain> residual,
      ControlRecord<T>& record, cudaStream_t stream) const;
};

}  // namespace cumes
