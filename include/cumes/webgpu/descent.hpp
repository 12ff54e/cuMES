#ifndef CUMES_INCLUDE_CUMES_WEBGPU_DESCENT_HPP_
#define CUMES_INCLUDE_CUMES_WEBGPU_DESCENT_HPP_

#include "cumes/webgpu/device_fields.hpp"

#include <functional>
#include <string>
#include <vector>

#include <webgpu/webgpu_cpp.h>

namespace cumes::webgpu {

struct AxisymmetricDescentResult;

struct AxisymmetricDescentCase {
    BatchedReadback<AxisymmetricDescentResult> readback;
    DeviceFields device_state, device_velocity, device_residual;
    // A paired descent may consume an ordinary-f32 preconditioned direction.
    bool residual_is_f32 = false;
    bool extrapolate_axis = false;
    int ns = 0;
    int mpol = 0;
    int ntor = 0;
    bool move_lcfs = false;
    float delta_t = 0.0F;
    float damping_b1 = 0.0F;
    float damping_fac = 0.0F;
    bool double_single = false;
    std::vector<float> state;
    std::vector<float> state_lo;
    std::vector<float> velocity;
    std::vector<float> velocity_lo;
    std::vector<float> residual;
    std::vector<float> residual_lo;
};

struct AxisymmetricDescentResult {
    DeviceFields device_state;
    DeviceFields device_velocity;
    std::vector<float> state;
    std::vector<float> state_lo;
    std::vector<float> velocity;
    std::vector<float> velocity_lo;
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
