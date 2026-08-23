// geometry_operator.hpp — base-geometry operator boundary (blueprint §6.7).
//
// Owns the 15 half-grid buffers (staggered interpolation, Jacobian, covariant
// metric, contravariant/covariant B, total pressure) and exposes them as typed
// BaseGeometryHalfViews / MagneticFieldViews. (Migration step 13.3: the legacy
// MetricWorkspace struct + metricCreate/metricFree/compute* free functions are
// gone; the operator owns the buffers directly and launches the kernels.)
#ifndef CUMES_INCLUDE_CUMES_PHYSICS_GEOMETRY_OPERATOR_HPP_
#define CUMES_INCLUDE_CUMES_PHYSICS_GEOMETRY_OPERATOR_HPP_

#include "cumes/config/device_params.hpp"
#include "cumes/solver/control_record.hpp"
#include "cumes/state/real_fields.cuh"
#include "cumes/state/real_space_storage.hpp"

#include <cuda_runtime.h>

namespace cumes {

class DeviceArena;

template <class T>
class GeometryOperator {
   public:
    GeometryOperator(const DeviceParams<T>& p, DeviceArena* arena);
    ~GeometryOperator();

    // Non-movable: the destructor frees the owned half-grid buffers without
    // nulling, so a defaulted move would double-free (review finding 3.2).
    GeometryOperator(const GeometryOperator&) = delete;
    GeometryOperator& operator=(const GeometryOperator&) = delete;
    GeometryOperator(GeometryOperator&&) noexcept = delete;
    GeometryOperator& operator=(GeometryOperator&&) noexcept = delete;

    // Half-grid base geometry: staggered interpolation, Jacobian, covariant
    // metric — no 1/√g division. Reads the parity-split geometry from `rs`.
    void enqueue(const RealSpaceStorage<T>& rs,
                 const DeviceParams<T>& p,
                 const RadialProfileViews<T>& rpv,
                 cudaStream_t stream);

    // Oriented-Jacobian statistics into the typed control record's four
    // jacobian_* slots (DOUBLE in both builds — ADR-0001 control-record
    // follow-up). The finalize step that turns the stats into
    // status.jacobian_valid lives with the solver (jacobianFinalizeKernel in
    // kernels/solver_impl.cuh); both use the shared kJacobianEps rule.
    void jacobian_stats(const DeviceParams<T>& p,
                        ControlRecord* rec,
                        cudaStream_t stream) const;

    // Force-norm partial sums (dVdsH + psum) for the residual normalization.
    void force_norm_partials(const DeviceParams<T>& p,
                             T* dVdsH,
                             T* psum,
                             cudaStream_t stream) const;

    // Typed view bundles over the owned buffers (the field/force/precon
    // operators consume these instead of the deleted MetricWorkspace).
    BaseGeometryHalfViews<T> base_geometry_views(
        const DeviceParams<T>& p) const;
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

#endif  // CUMES_INCLUDE_CUMES_PHYSICS_GEOMETRY_OPERATOR_HPP_
