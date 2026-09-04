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
    int nzeta = 1;
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

inline constexpr std::size_t MAGNETIC_FIELD_COUNT = 5;

struct MagneticFieldCase {
    int ns = 0;
    int ntheta = 0;
    int nzeta = 1;
    float lamscale = 0.0F;
    bool prescribed_current = false;
    std::vector<float> geometry;
    std::vector<float> base_geometry;
    std::vector<float> sqrt_s_h;
    std::vector<float> phip_f;
    std::vector<float> chip_h;
    std::vector<float> pres_h;
    std::vector<float> curr_h;
    std::vector<float> phip_h;
    std::vector<float> iota_h;
};

struct MagneticFieldResult {
    // Field-major half-grid order: B^theta, B^zeta, B_theta, B_zeta,
    // total pressure.
    std::vector<float> fields;
    std::vector<float> chip_h;
    std::vector<float> iota_h;
};

using MagneticFieldCallback =
    std::function<void(std::string, MagneticFieldResult)>;

void enqueue_magnetic_field(const wgpu::Device& device,
                            const MagneticFieldCase& input,
                            MagneticFieldCallback callback);

MagneticFieldResult magnetic_field_reference(const MagneticFieldCase& input);

}  // namespace cumes::webgpu

#endif  // CUMES_INCLUDE_CUMES_WEBGPU_GEOMETRY_HPP_
