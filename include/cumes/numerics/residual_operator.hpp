// residual_operator.hpp — invariant/preconditioned residual boundary
// (blueprint §6.9).
//
// Reduces the six spectral-force families to the invariant (FSQR/FSQZ/FSQL) and
// preconditioned (FSQR1/FSQZ1/FSQL1) triples. The legacy
// compute_residuals_kernel
// + host scaling are the reference implementation; the device-only
// ControlRecord with double accumulation (§6.9) is a later refactor.
//
// Strangler-fig form: a stateless thin wrapper over compute_residuals_kernel
// (verbatim — migration step 8). It reduces the decomposed residual into the
// caller's 3-element device `sq_out` (ΣF²/(mnmax·ns) per group); the invariant
// vs preconditioned distinction is only the host-side scaling (fNormRZ/fNormL
// vs fNorm1/delta_s), which stays with the solver.
#ifndef CUMES_INCLUDE_CUMES_NUMERICS_RESIDUAL_OPERATOR_HPP_
#define CUMES_INCLUDE_CUMES_NUMERICS_RESIDUAL_OPERATOR_HPP_

#include "cumes/core/tensor_view.cuh"
#include "cumes/solver/control_record.hpp"

#include <cuda_runtime.h>

namespace cumes {

template <class T>
class ResidualOperator {
   public:
    using val_type = T;

    // Reduce the decomposed residual into `sq_out` (3 elements: the fsqr/fsqz/
    // fsql group sums, ΣF²/(mnmax·ns)). One kernel, three output groups. The
    // output is DOUBLE in both builds (ADR-0001 follow-up): the kernel
    // accumulates in NormAccum<T>::type (double for mixed-float) and stores
    // without rounding to T.
    void enqueue(SpectralView<const T, DecomposedResidualDomain> residual,
                 int ns,
                 int mnmax,
                 double* sq_out,
                 cudaStream_t stream) const;

    // Preconditioned-residual reduction with the device terminal gate
    // (completion plan step 1.4): on a nonfinite/converged pass the
    // preconditioner no-op'd, so this reduction stores the zero sentinel into
    // rec->preconditioned_raw and leaves preconditioned_evaluated clear. On a
    // continuing pass it reduces normally and sets the evaluated bit.
    void enqueue_preconditioned(
        SpectralView<const T, DecomposedResidualDomain> residual,
        int ns,
        int mnmax,
        ControlRecord* rec,
        cudaStream_t stream) const;
};

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_NUMERICS_RESIDUAL_OPERATOR_HPP_
