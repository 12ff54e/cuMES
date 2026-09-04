#ifndef CUMES_INCLUDE_CUMES_WEBGPU_NUMERICS_HPP_
#define CUMES_INCLUDE_CUMES_WEBGPU_NUMERICS_HPP_

#include <array>
#include <functional>
#include <string>
#include <vector>

#include <webgpu/webgpu_cpp.h>

namespace cumes::webgpu {

struct ResidualDecompositionCase {
    int ns = 0;
    int mpol = 0;
    int ntor = 0;
    bool include_edge_rz = false;
    bool zero_m1_z = false;
    bool double_single = false;
    std::vector<float> residual;
    std::vector<float> residual_lo;
    std::vector<float> sqrt_s_f;
    std::vector<float> sqrt_s_f_lo;
};

struct ResidualDecompositionResult {
    std::vector<float> residual;
    std::vector<float> residual_lo;
    std::array<double, 3> raw_norm{};
};

using ResidualDecompositionCallback =
    std::function<void(std::string, ResidualDecompositionResult)>;

void enqueue_residual_decomposition(const wgpu::Device& device,
                                    const ResidualDecompositionCase& input,
                                    ResidualDecompositionCallback callback);

ResidualDecompositionResult residual_decomposition_reference(
    const ResidualDecompositionCase& input);

struct AxisymmetricForceNormalizationCase {
    int ns = 0;
    int mpol = 0;
    int ntor = 0;
    int ntheta = 0;
    int nzeta = 1;
    float delta_s = 0.0F;
    float lamscale = 0.0F;
    std::vector<float> state;
    std::vector<float> base_geometry;
    std::vector<float> magnetic_field;
    std::vector<float> pres_h;
};

struct ForceNormalizationResult {
    // sRZ, sL, sMag, eTherm, volume, rzNorm before delta-s finalization.
    std::array<double, 6> raw{};
    double f_norm_rz = 1.0;
    double f_norm_l = 1.0;
    double f_norm_1 = 1.0;
};

ForceNormalizationResult axisymmetric_force_normalization(
    const AxisymmetricForceNormalizationCase& input);

std::array<double, 3> residual_raw_norms(const std::vector<float>& residual,
                                         int ns,
                                         int mpol,
                                         int ntor,
                                         bool include_edge_rz);

}  // namespace cumes::webgpu

#endif  // CUMES_INCLUDE_CUMES_WEBGPU_NUMERICS_HPP_
