// geometry_operator.hpp — base-geometry operator boundary (blueprint §6.7).
//
// Owns the 15 half-grid buffers (staggered interpolation, Jacobian, covariant
// metric, contravariant/covariant B, total pressure) and exposes them as typed
// BaseGeometryHalfViews / MagneticFieldViews. (Migration step 13.3: the legacy
// MetricWorkspace struct + metricCreate/metricFree/compute* free functions are
// gone; the operator owns the buffers directly and launches the kernels.)
#pragma once

#include <cuda_runtime.h>

#include "cumes/config/device_params.hpp"
#include "cumes/state/real_fields.cuh"
#include "cumes/state/real_space_storage.hpp"

namespace cumes {

class DeviceArena;

template <class T>
class GeometryOperator {
 public:
  GeometryOperator(const DeviceParams<T>& p, DeviceArena* arena);
  ~GeometryOperator();

  GeometryOperator(const GeometryOperator&) = delete;
  GeometryOperator& operator=(const GeometryOperator&) = delete;
  GeometryOperator(GeometryOperator&&) noexcept = default;
  GeometryOperator& operator=(GeometryOperator&&) noexcept = default;

  // Half-grid base geometry: staggered interpolation, Jacobian, covariant
  // metric — no 1/√g division. Reads the parity-split geometry from `rs`.
  void enqueue(const RealSpaceStorage<T>& rs, const DeviceParams<T>& p,
               const RadialProfileViews<T>& rpv, cudaStream_t stream);

  // Oriented-Jacobian statistics into a caller-owned 4-element device scratch
  // (DOUBLE in both builds — ADR-0001 control-record follow-up).
  void jacobian_stats(const DeviceParams<T>& p, double* d_stats, cudaStream_t stream) const;

  // Force-norm partial sums (dVdsH + psum) for the residual normalization.
  void force_norm_partials(const DeviceParams<T>& p, T* dVdsH, T* psum,
                           cudaStream_t stream) const;

  // Typed view bundles over the owned buffers (the field/force/precon
  // operators consume these instead of the deleted MetricWorkspace).
  BaseGeometryHalfViews<T> base_geometry_views(const DeviceParams<T>& p) const;
  MagneticFieldViews<T> magnetic_field_views(const DeviceParams<T>& p) const;

 private:
  T* d_r12_ = nullptr;
  T* d_ru12_ = nullptr;
  T* d_zu12_ = nullptr;
  T* d_rs_ = nullptr;
  T* d_zs_ = nullptr;
  T* d_tau_ = nullptr;
  T* d_gsqrt_ = nullptr;
  T* d_guu_ = nullptr;
  T* d_guv_ = nullptr;
  T* d_gvv_ = nullptr;
  T* d_bsupu_ = nullptr;
  T* d_bsupv_ = nullptr;
  T* d_bsubu_ = nullptr;
  T* d_bsubv_ = nullptr;
  T* d_totalPressure_ = nullptr;
  bool arena_backed_ = false;
};

}  // namespace cumes
