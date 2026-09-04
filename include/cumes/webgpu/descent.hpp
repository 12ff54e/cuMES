#ifndef CUMES_INCLUDE_CUMES_WEBGPU_DESCENT_HPP_
#define CUMES_INCLUDE_CUMES_WEBGPU_DESCENT_HPP_

#include <functional>
#include <string>
#include <vector>

#include <webgpu/webgpu_cpp.h>

namespace cumes::webgpu {

struct AxisymmetricDescentCase {
    int ns = 0;
    int mpol = 0;
    bool move_lcfs = false;
    float delta_t = 0.0F;
    float damping_b1 = 0.0F;
    float damping_fac = 0.0F;
    std::vector<float> state;
    std::vector<float> velocity;
    std::vector<float> residual;
};

struct AxisymmetricDescentResult {
    std::vector<float> state;
    std::vector<float> velocity;
};

using AxisymmetricDescentCallback =
    std::function<void(std::string, AxisymmetricDescentResult)>;

void enqueue_axisymmetric_descent(const wgpu::Device& device,
                                  const AxisymmetricDescentCase& input,
                                  AxisymmetricDescentCallback callback);

AxisymmetricDescentResult axisymmetric_descent_reference(
    const AxisymmetricDescentCase& input);

}  // namespace cumes::webgpu

#endif  // CUMES_INCLUDE_CUMES_WEBGPU_DESCENT_HPP_
