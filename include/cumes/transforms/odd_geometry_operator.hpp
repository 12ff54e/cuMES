#ifndef CUMES_INCLUDE_CUMES_TRANSFORMS_ODD_GEOMETRY_OPERATOR_HPP_
#define CUMES_INCLUDE_CUMES_TRANSFORMS_ODD_GEOMETRY_OPERATOR_HPP_

#include "cumes/config/device_params.hpp"
#include "cumes/numerics/float_float.cuh"
#include "cumes/runtime/device_buffer.cuh"
#include "cumes/state/real_fields.cuh"

namespace cumes {
// Reconstruct only r_o/z_o. Plans/tables/scratch are prepared at stage setup;
// enqueue has no allocations and is safe inside CUDA Graph capture.
template <class A>
class OddGeometryOperator {
   public:
    using val_type = A;
    explicit OddGeometryOperator(const DeviceParams<float>& p);
    void enqueue(SpectralView<const float, PhysicalStateDomain> coeff,
                 GeometryParityViews<float> geometry,
                 cudaStream_t stream);

    // COMPENSATED: four m=1 channel sums; FLOAT_FLOAT: all odd m. The
    // scratch retains paired values for consumption by the poloidal kernel.
    void enqueue_toroidal(SpectralView<const float, PhysicalStateDomain> coeff,
                          cudaStream_t stream);
    const A* toroidal() const { return d_scratch_.data(); }

    const A* scale() const { return d_scale_.data(); }

   private:
    DeviceParams<float> p_;
    DeviceBuffer<A> d_cos_zeta_, d_sin_zeta_, d_cos_theta_, d_sin_theta_;
    DeviceBuffer<A> d_scale_, d_scratch_;
};
}  // namespace cumes
#endif
