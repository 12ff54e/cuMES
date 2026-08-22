// force_operator.hpp — real-space MHD force boundary (blueprint §6.7).
//
// Evaluates the weak-form MHD + hybrid-lambda force residuals into the
// parity-split force views. The legacy computeForces is the reference; a scalar
// CPU reference with the same operation order is the gate before any
// split/fuse.
//
// (Migration step 13.3: a stateless operator over typed base-geometry + field +
// profile views — the legacy MetricWorkspace/computeForces are gone.)
#pragma once

#include "cumes/config/device_params.hpp"
#include "cumes/solver/control_record.hpp"
#include "cumes/state/real_fields.cuh"
#include "cumes/state/real_space_storage.hpp"

#include <cuda_runtime.h>

namespace cumes {

template <class T>
class ForceOperator {
   public:
    // Status-guarded (completion plan step 1.4): the kernel no-ops on an
    // invalid-Jacobian pass (status->jacobian_valid == 0), writing no force
    // buffers.
    void enqueue(const RealSpaceStorage<T>& rs,
                 const DeviceParams<T>& p,
                 const RadialProfileViews<T>& rpv,
                 const BaseGeometryHalfViews<T>& base,
                 const MagneticFieldViews<T>& field,
                 const ControlStatus* status,
                 cudaStream_t stream) const;
};

}  // namespace cumes
