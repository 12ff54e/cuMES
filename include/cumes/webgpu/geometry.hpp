#ifndef CUMES_INCLUDE_CUMES_WEBGPU_GEOMETRY_HPP_
#define CUMES_INCLUDE_CUMES_WEBGPU_GEOMETRY_HPP_
#include "cumes/webgpu/device_fields.hpp"
#include "cumes/webgpu/float_geometry.hpp"
#include "cumes/webgpu/reduction.hpp"

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include <webgpu/webgpu_cpp.h>

namespace cumes::webgpu {

inline constexpr std::size_t BASE_GEOMETRY_FIELD_COUNT = 10;

struct BaseGeometryResult;

struct BaseGeometryCase {
    bool device_control = false;
    bool readback_values = true;
    // Batched host validity checks need only gsqrt and guv. Retain full
    // readback with device_control or above the GPU finite-scan range.
    bool readback_validity = false;
    bool axisymmetric = false;
    BatchedReadback<BaseGeometryResult> readback;
    DeviceFields device_geometry;
    int ns = 0;
    int ntheta = 0;
    int nzeta = 1;
    float delta_s = 0.0F;
    bool double_single = false;
    FloatRadiusReferencePtr radius_reference;
    // The 18 field-major full-grid fields produced by the inverse transform.
    std::vector<float> geometry;
    std::vector<float> geometry_lo;
    std::vector<float> sqrt_s_f;
    std::vector<float> sqrt_s_h;
};

struct BaseGeometryResult {
    bool fields_are_validity = false;
    bool fields_finite = true;
    GeometryControlResult control;
    DeviceFields device_fields;
    // Field-major half-grid order: r12, ru12, zu12, rs, zs, tau, gsqrt,
    // guu, guv, gvv. If fields_are_validity, pack only gsqrt then guv; device
    // fields always retain the full ten-field layout.
    std::vector<float> fields;
    std::vector<float> fields_lo;
};

using BaseGeometryCallback =
    std::function<void(std::string, BaseGeometryResult)>;

void enqueue_base_geometry(const wgpu::Device& device,
                           const BaseGeometryCase& input,
                           BaseGeometryCallback callback);

BaseGeometryResult base_geometry_reference(const BaseGeometryCase& input);

inline constexpr std::size_t MAGNETIC_FIELD_COUNT = 5;

struct MagneticFieldResult;

struct MagneticFieldCase {
    // Keep radial profiles but replace full-field vectors with a finite flag.
    bool readback_values = true;
    // With batched readback, retain just the outer two half-grid rows of
    // B_theta, B_zeta and total pressure, plus all radial profiles. Requires
    // ns>=3 and the GPU finite-scan range; otherwise retain full readback.
    bool readback_vacuum = false;
    BatchedReadback<MagneticFieldResult> readback;
    DeviceFields device_geometry;
    DeviceFields device_base_geometry;
    int ns = 0;
    int ntheta = 0;
    int nzeta = 1;
    float lamscale = 0.0F;
    float lamscale_lo = 0.0F;
    bool prescribed_current = false;
    bool double_single = false;
    std::vector<float> geometry;
    std::vector<float> geometry_lo;
    std::vector<float> base_geometry;
    std::vector<float> base_geometry_lo;
    std::vector<float> sqrt_s_h;
    std::vector<float> sqrt_s_h_lo;
    std::vector<float> phip_f;
    std::vector<float> phip_f_lo;
    std::vector<float> chip_h;
    std::vector<float> chip_h_lo;
    std::vector<float> pres_h;
    std::vector<float> pres_h_lo;
    std::vector<float> curr_h;
    std::vector<float> curr_h_lo;
    std::vector<float> phip_h;
    std::vector<float> phip_h_lo;
    std::vector<float> iota_h;
    std::vector<float> iota_h_lo;
};

struct MagneticFieldResult {
    bool fields_finite = true;
    bool fields_are_vacuum = false;
    DeviceFields device_fields;
    // Field-major half-grid order: B^theta, B^zeta, B_theta, B_zeta,
    // total pressure. If fields_are_vacuum, pack only the last two half-grid
    // rows of B_theta, B_zeta and pressure, in that order. Device fields always
    // retain the full five-field layout; profile-only readbacks leave these
    // host vectors empty.
    std::vector<float> fields;
    std::vector<float> fields_lo;
    std::vector<float> chip_h;
    std::vector<float> chip_h_lo;
    std::vector<float> iota_h;
    std::vector<float> iota_h_lo;
};

using MagneticFieldCallback =
    std::function<void(std::string, MagneticFieldResult)>;

void enqueue_magnetic_field(const wgpu::Device& device,
                            const MagneticFieldCase& input,
                            MagneticFieldCallback callback);

MagneticFieldResult magnetic_field_reference(const MagneticFieldCase& input);

}  // namespace cumes::webgpu

#endif  // CUMES_INCLUDE_CUMES_WEBGPU_GEOMETRY_HPP_
