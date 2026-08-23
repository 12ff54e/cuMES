// descent_operator.hpp — Garabedian accelerated descent + checkpoint boundary
// (blueprint §6.10, §6.11).
//
// Applies the ordered decision: optional descent, optional post-descent
// checkpoint capture, then optional post-descent restore + velocity zero. The
// legacy descentStepKernel + backupState/restoreState are the reference
// implementation; the velocity/state slabs make checkpoint capture one copy.
//
// Strangler-fig form: a stateless thin wrapper over descentStepKernel
// (verbatim — migration step 10). It launches the accelerated descent step; the
// single-copy checkpoint capture/restore stays with the solver's state slab
// (the blueprint §6.10 keeps the checkpoint as a distinct operator).
#ifndef CUMES_INCLUDE_CUMES_NUMERICS_DESCENT_OPERATOR_HPP_
#define CUMES_INCLUDE_CUMES_NUMERICS_DESCENT_OPERATOR_HPP_

#include "cumes/core/tensor_view.cuh"

#include <cuda_runtime.h>

namespace cumes {

// The descent/checkpoint action derived from the controller decision (blueprint
// §6.10). The descent operator consumes the damping/delta_t fields; the solver
// consumes the checkpoint flags (capture/restore) after the descent, in the
// exact frozen order (descent → post-descent capture → post-descent restore +
// velocity zero).
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
    using val_type = T;

    void enqueue(SpectralView<T, PhysicalStateDomain> state,
                 SpectralView<T, DecomposedVelocityDomain> velocity,
                 SpectralView<const T, DecomposedResidualDomain> residual,
                 const int* xm,
                 const int* xn,
                 int ns,
                 int mnmax,
                 const DescentAction& action,
                 cudaStream_t stream) const;
};

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_NUMERICS_DESCENT_OPERATOR_HPP_
