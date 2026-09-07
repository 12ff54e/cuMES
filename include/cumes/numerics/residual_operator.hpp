// residual_operator.hpp — invariant/preconditioned residual boundary
// (blueprint §6.9).
//
// Reduces decomposed spectral forces to raw group sums. Both device inputs
// and outputs use T; float sums use float-float accumulation before rounding
// to float. Normalization and terminal predicates share the typed device
// control record, which the host widens only after the single control fence.
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

    // Three T outputs: group sums divided by mnmax*ns. Float-float retains
    // small summands without introducing FP64 operations in float kernels.
    void enqueue(SpectralView<const T, DecomposedResidualDomain> residual,
                 int ns,
                 int mnmax,
                 bool include_edge_rz,
                 T* sq_out,
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
        DeviceControlRecord<T>* rec,
        cudaStream_t stream) const;
};

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_NUMERICS_RESIDUAL_OPERATOR_HPP_
