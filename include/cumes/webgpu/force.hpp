#ifndef CUMES_INCLUDE_CUMES_WEBGPU_FORCE_HPP_
#define CUMES_INCLUDE_CUMES_WEBGPU_FORCE_HPP_

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include <webgpu/webgpu_cpp.h>

namespace cumes::webgpu {

inline constexpr std::size_t FORCE_FIELD_COUNT = 16;

struct AxisymmetricForceCase {
    int ns = 0;
    int ntheta = 0;
    int nzeta = 1;
    float delta_s = 0.0F;
    float delta_s_lo = 0.0F;
    float lamscale = 0.0F;
    float lamscale_lo = 0.0F;
    bool double_single = false;
    std::vector<float> geometry;
    std::vector<float> geometry_lo;
    std::vector<float> base_geometry;
    std::vector<float> base_geometry_lo;
    std::vector<float> magnetic_field;
    std::vector<float> magnetic_field_lo;
    std::vector<float> sqrt_s_f;
    std::vector<float> sqrt_s_f_lo;
    std::vector<float> sqrt_s_h;
    std::vector<float> sqrt_s_h_lo;
    std::vector<float> phip_f;
    std::vector<float> phip_f_lo;
};

struct AxisymmetricForceResult {
    // Field-major full-grid order consumed by the direct forward transform:
    // armn e/o, azmn e/o, brmn e/o, bzmn e/o, blmn e/o, crmn e/o,
    // czmn e/o, clmn e/o. The historical type name is retained while the
    // operator supports both axisymmetric and 3-D angular grids.
    std::vector<float> fields;
    std::vector<float> fields_lo;
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
