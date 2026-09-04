#ifndef CUMES_INCLUDE_CUMES_WEBGPU_GEOMETRY_HPP_
#define CUMES_INCLUDE_CUMES_WEBGPU_GEOMETRY_HPP_

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include <webgpu/webgpu_cpp.h>

namespace cumes::webgpu {

inline constexpr std::size_t BASE_GEOMETRY_FIELD_COUNT = 10;

struct BaseGeometryCase {
    int ns = 0;
    int ntheta = 0;
    float delta_s = 0.0F;
    // The 18 field-major full-grid fields produced by the inverse transform.
    std::vector<float> geometry;
    std::vector<float> sqrt_s_f;
    std::vector<float> sqrt_s_h;
};

struct BaseGeometryResult {
    // Field-major half-grid order: r12, ru12, zu12, rs, zs, tau, gsqrt,
    // guu, guv, gvv.
    std::vector<float> fields;
};

using BaseGeometryCallback =
    std::function<void(std::string, BaseGeometryResult)>;

void enqueue_base_geometry(const wgpu::Device& device,
                           const BaseGeometryCase& input,
                           BaseGeometryCallback callback);

BaseGeometryResult base_geometry_reference(const BaseGeometryCase& input);

}  // namespace cumes::webgpu

#endif  // CUMES_INCLUDE_CUMES_WEBGPU_GEOMETRY_HPP_
