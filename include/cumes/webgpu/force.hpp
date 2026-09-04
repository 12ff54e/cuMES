#ifndef CUMES_INCLUDE_CUMES_WEBGPU_FORCE_HPP_
#define CUMES_INCLUDE_CUMES_WEBGPU_FORCE_HPP_

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include <webgpu/webgpu_cpp.h>

namespace cumes::webgpu {

inline constexpr std::size_t AXISYMMETRIC_FORCE_FIELD_COUNT = 10;

struct AxisymmetricForceCase {
    int ns = 0;
    int ntheta = 0;
    float delta_s = 0.0F;
    float lamscale = 0.0F;
    std::vector<float> geometry;
    std::vector<float> base_geometry;
    std::vector<float> magnetic_field;
    std::vector<float> sqrt_s_f;
    std::vector<float> sqrt_s_h;
    std::vector<float> phip_f;
};

struct AxisymmetricForceResult {
    // Field-major full-grid order: armn e/o, azmn e/o, brmn e/o, bzmn e/o,
    // blmn e/o.
    std::vector<float> fields;
};

using AxisymmetricForceCallback =
    std::function<void(std::string, AxisymmetricForceResult)>;

void enqueue_axisymmetric_force(const wgpu::Device& device,
                                const AxisymmetricForceCase& input,
                                AxisymmetricForceCallback callback);

AxisymmetricForceResult axisymmetric_force_reference(
    const AxisymmetricForceCase& input);

}  // namespace cumes::webgpu

#endif  // CUMES_INCLUDE_CUMES_WEBGPU_FORCE_HPP_
