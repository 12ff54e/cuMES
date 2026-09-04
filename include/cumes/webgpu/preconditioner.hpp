#ifndef CUMES_INCLUDE_CUMES_WEBGPU_PRECONDITIONER_HPP_
#define CUMES_INCLUDE_CUMES_WEBGPU_PRECONDITIONER_HPP_

#include <functional>
#include <string>
#include <vector>

#include <webgpu/webgpu_cpp.h>

namespace cumes::webgpu {

struct AxisymmetricPreconditionerElementCase {
    int ns = 0;
    int ntheta = 0;
    float delta_s = 0.0F;
    bool free_boundary = false;
    std::vector<float> geometry;
    std::vector<float> base_geometry;
    std::vector<float> magnetic_field;
    std::vector<float> sqrt_s_f;
    std::vector<float> sqrt_s_h;
};

struct AxisymmetricPreconditionerElements {
    // Surface-major [surface][even, odd].
    std::vector<float> ard;
    std::vector<float> brd;
    std::vector<float> azd;
    std::vector<float> bzd;
    std::vector<float> cxd;
};

using AxisymmetricPreconditionerElementCallback =
    std::function<void(std::string, AxisymmetricPreconditionerElements)>;

void enqueue_axisymmetric_preconditioner_elements(
    const wgpu::Device& device,
    const AxisymmetricPreconditionerElementCase& input,
    AxisymmetricPreconditionerElementCallback callback);

AxisymmetricPreconditionerElements
axisymmetric_preconditioner_element_reference(
    const AxisymmetricPreconditionerElementCase& input);

}  // namespace cumes::webgpu

#endif  // CUMES_INCLUDE_CUMES_WEBGPU_PRECONDITIONER_HPP_
