// magnetic_field_operator.hpp — contravariant/covariant B + total pressure
// boundary (blueprint §6.7).
//
// Consumes base geometry (√g, covariant metric) + profiles + the full-grid λ
// derivatives, produces B^θ/B^ζ, B_θ/B_ζ and the total pressure on the half
// grid with the 1/√g division. The ncurr=0 (fixed iota) and ncurr=1
// (prescribed current) flows are separate policy paths; the legacy
// geometryKernel/ncurr1FinalizeKernel are the reference implementation (the
// field was fused into the base geometry kernel until Phase 11 step 5).
//
// (Migration step 13.3: it reads/writes the typed BaseGeometryHalfViews /
// MagneticFieldViews owned by the GeometryOperator, not the deleted
// MetricWorkspace.)
#pragma once

#include "cumes/config/device_params.hpp"
#include "cumes/solver/control_record.hpp"
#include "cumes/state/real_fields.cuh"
#include "cumes/state/real_space_storage.hpp"

#include <cuda_runtime.h>

namespace cumes {

template <class T>
class MagneticFieldOperator {
   public:
    // Half-grid magnetic field + total pressure + ncurr closure; update the
    // full-grid iota/chip on the first pass (ncurr=0) or every pass (ncurr=1).
    // Reads the parity-split λ derivatives from `rs`, the base geometry from
    // `base`, writes `field`, and reads the radial profiles from `rpv`.
    // Status-guarded (completion plan step 1.4): when `status->jacobian_valid`
    // is clear the kernels no-op — neither the field arrays nor the evolved
    // iotaH/chipH cache are written on an invalid-Jacobian pass.
    void enqueue(const RealSpaceStorage<T>& rs,
                 const DeviceParams<T>& p,
                 const RadialProfileViews<T>& rpv,
                 const BaseGeometryHalfViews<T>& base,
                 MagneticFieldViews<T> field,
                 const ControlStatus* status,
                 cudaStream_t stream,
                 bool update_iota_chi) const;
};

}  // namespace cumes
