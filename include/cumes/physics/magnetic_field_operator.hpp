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
// Strangler-fig form: a stateless thin wrapper over computeMagneticField
// (verbatim — migration step 5). It reads/writes the MetricWorkspace owned by
// the GeometryOperator (the base-geometry and field arrays share one workspace);
// the view-based boundary (base geometry + field views) is a later follow-up.
#pragma once

#include <cuda_runtime.h>

#include "cumes/state/real_space_storage.hpp"
#include "geometry.cuh"
#include "profiles.cuh"

namespace cumes {

template <class T>
class MagneticFieldOperator {
 public:
  // Half-grid magnetic field + total pressure + ncurr closure; update the
  // full-grid iota/chip on the first pass (ncurr=0) or every pass (ncurr=1).
  // Reads the parity-split λ derivatives from `rs`, the base geometry from
  // `mw`, and the radial profiles from `rp`.
  void enqueue(const RealSpaceStorage<T>& rs, const DeviceParams<T>& p,
               const RadialProfiles<T>& rp, MetricWorkspace<T>& mw,
               cudaStream_t stream, bool update_iota_chi) const {
    computeMagneticField(rs, p, rp, mw, stream, update_iota_chi);
  }
};

}  // namespace cumes
