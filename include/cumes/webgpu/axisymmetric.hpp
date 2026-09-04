#ifndef CUMES_INCLUDE_CUMES_WEBGPU_AXISYMMETRIC_HPP_
#define CUMES_INCLUDE_CUMES_WEBGPU_AXISYMMETRIC_HPP_

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include <webgpu/webgpu_cpp.h>

namespace cumes::webgpu {

inline constexpr std::size_t SPECTRAL_COMPONENT_COUNT = 6;
inline constexpr std::size_t GEOMETRY_PARITY_FIELD_COUNT = 18;

struct AxisymmetricInverseCase {
    int ns = 0;
    int mpol = 0;
    int ntheta = 0;
    // Component-major spectral layout: [component][mode][surface].
    std::vector<float> state;
};

struct AxisymmetricInverseResult {
    // Field-major real-space layout: [field][surface][theta]. Field order is
    // r/z/l, ru/zu/lu even; r/z/l, ru/zu/lu odd; then the six zero toroidal
    // derivatives, matching GeometryParityViews.
    std::vector<float> geometry;
    std::vector<float> r_con;
    std::vector<float> z_con;
};

using AxisymmetricInverseCallback =
    std::function<void(std::string, AxisymmetricInverseResult)>;

void enqueue_axisymmetric_inverse(const wgpu::Device& device,
                                  const AxisymmetricInverseCase& input,
                                  AxisymmetricInverseCallback callback);

AxisymmetricInverseResult axisymmetric_inverse_reference(
    const AxisymmetricInverseCase& input);

}  // namespace cumes::webgpu

#endif  // CUMES_INCLUDE_CUMES_WEBGPU_AXISYMMETRIC_HPP_
