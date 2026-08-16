// geometry_operator.hpp — base-geometry operator boundary (blueprint §6.7).
//
// Owns the MetricWorkspace and wraps the legacy computeGeometry /
// computeJacobianStats / computeForceNormPartials. Transitional strangler-fig
// form: it still names the legacy FourierPlan/RadialProfiles in the enqueue
// signature (for the parity-split full-grid geometry and radial profiles);
// those become typed views once the FourierPlan split lands. The base-geometry
// vs Jacobian-division decomposition (blueprint §6.7) is a separate follow-up.
#pragma once

#include <cuda_runtime.h>

#include "cumes/state/real_fields.cuh"
#include "fourier.cuh"
#include "geometry.cuh"
#include "profiles.cuh"

namespace cumes {

class DeviceArena;

template <class T>
class GeometryOperator {
 public:
  GeometryOperator(const GridParams<T>& p, DeviceArena* arena)
      : mw_(metricCreate(p, arena)) {}
  ~GeometryOperator() { metricFree(mw_); }

  GeometryOperator(const GeometryOperator&) = delete;
  GeometryOperator& operator=(const GeometryOperator&) = delete;
  GeometryOperator(GeometryOperator&&) noexcept = default;
  GeometryOperator& operator=(GeometryOperator&&) noexcept = default;

  // Half-grid geometry, metric, field and current closure; update the full-grid
  // iota/chip on the first pass (ncurr=0) or every pass (ncurr=1). Reads the
  // parity-split geometry from the stage-owned `rs`.
  void enqueue(const RealSpaceStorage<T>& rs, const GridParams<T>& p,
               const RadialProfiles<T>& rp, cudaStream_t stream, bool update_iota_chi);

  // Oriented-Jacobian statistics into a caller-owned 4-element device scratch.
  void jacobian_stats(const GridParams<T>& p, T* d_stats, cudaStream_t stream) const;

  // Force-norm partial sums (dVdsH + psum) for the residual normalization.
  void force_norm_partials(const GridParams<T>& p, T* dVdsH, T* psum,
                           cudaStream_t stream) const;

  MetricWorkspace<T>& workspace() { return mw_; }
  const MetricWorkspace<T>& workspace() const { return mw_; }

 private:
  MetricWorkspace<T> mw_;
};

}  // namespace cumes
