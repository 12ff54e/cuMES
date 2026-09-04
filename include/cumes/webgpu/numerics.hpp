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
    bool include_edge_rz = false;
    bool zero_m1_z = false;
    std::vector<float> residual;
    std::vector<float> sqrt_s_f;
};

struct ResidualDecompositionResult {
    std::vector<float> residual;
    std::array<double, 3> raw_norm{};
};

using ResidualDecompositionCallback =
    std::function<void(std::string, ResidualDecompositionResult)>;

void enqueue_residual_decomposition(const wgpu::Device& device,
                                    const ResidualDecompositionCase& input,
                                    ResidualDecompositionCallback callback);

ResidualDecompositionResult residual_decomposition_reference(
    const ResidualDecompositionCase& input);

}  // namespace cumes::webgpu

#endif  // CUMES_INCLUDE_CUMES_WEBGPU_NUMERICS_HPP_
