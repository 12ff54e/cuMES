#pragma once

#include "cumes/webgpu/device_fields.hpp"

#include <array>
#include <functional>

namespace cumes::webgpu {

struct ResidualNormResult {
    DeviceFields device_norm;
    std::array<double, 3> raw{};
    bool finite = true;
};

struct ResidualNormCase {
    BatchedReadback<ResidualNormResult> readback;
    DeviceFields residual;
    int ns = 0;
    bool paired = false;
    bool include_edge_rz = false;
};

// GPU-only input; two dispatches reduce all six parity families to a paired
// triple. At most 2^24 samples/family keeps the f32 divisor exact. The compact
// readback is appended to the caller's existing fence. Nonfinite inputs and
// unrepresentable squared norms are invalid, never a convergence signal.
void enqueue_residual_norm(
    const wgpu::Device& device,
    const ResidualNormCase& input,
    std::function<void(std::string, ResidualNormResult)> callback);

}  // namespace cumes::webgpu
