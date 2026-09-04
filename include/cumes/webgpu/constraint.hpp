#ifndef CUMES_INCLUDE_CUMES_WEBGPU_CONSTRAINT_HPP_
#define CUMES_INCLUDE_CUMES_WEBGPU_CONSTRAINT_HPP_

#include <functional>
#include <string>
#include <vector>

#include <webgpu/webgpu_cpp.h>

namespace cumes::webgpu {

struct AxisymmetricConstraintCase {
    int ns = 0;
    int mpol = 0;
    int ntheta = 0;
    float delta_s = 0.0F;
    float tcon0 = 1.0F;
    bool reset_reference = false;
    bool refresh_preconditioner = false;

    // GeometryParityViews field-major data and the inverse transform's
    // xmpq-weighted constraint reconstructions.
    std::vector<float> geometry;
    std::vector<float> r_con;
    std::vector<float> z_con;

    // Persistent constraint state entering this pass.
    std::vector<float> r_con0;
    std::vector<float> z_con0;
    std::vector<float> tcon;

    // Even/odd m=1 radial-preconditioner diagonals: [surface][parity].
    std::vector<float> ard;
    std::vector<float> azd;
    std::vector<float> sqrt_s_f;

    // MHD force fields: armn e/o, azmn e/o, brmn e/o, bzmn e/o, blmn e/o.
    std::vector<float> force_fields;
};

struct AxisymmetricConstraintResult {
    // Forward-transform fields: the ten updated MHD fields followed by
    // frcon e/o and fzcon e/o.
    std::vector<float> fields;
    std::vector<float> r_con0;
    std::vector<float> z_con0;
    std::vector<float> tcon;
    std::vector<float> g_con_eff;
    std::vector<float> g_con;
};

using AxisymmetricConstraintCallback =
    std::function<void(std::string, AxisymmetricConstraintResult)>;

void enqueue_axisymmetric_constraint(const wgpu::Device& device,
                                     const AxisymmetricConstraintCase& input,
                                     AxisymmetricConstraintCallback callback);

AxisymmetricConstraintResult axisymmetric_constraint_reference(
    const AxisymmetricConstraintCase& input);

}  // namespace cumes::webgpu

#endif  // CUMES_INCLUDE_CUMES_WEBGPU_CONSTRAINT_HPP_
