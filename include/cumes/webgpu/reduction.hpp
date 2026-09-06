#pragma once

#include "cumes/solver/control_record.hpp"
#include "cumes/webgpu/device_fields.hpp"

#include <array>
#include <functional>

namespace cumes::webgpu {

struct GeometryControlResult {
    JacobianStatus<double> jacobian;
    bool guards_valid = true;
    bool invalid = false;
    bool fallback = false;
    bool present = false;
};

// Reduce the ten base-geometry fields and execute the oriented-Jacobian gate.
// Ambiguous ordering/range cases request full host validation, never
// acceptance.
void enqueue_geometry_control(
    const wgpu::Device& device,
    const DeviceFields& fields,
    bool paired,
    bool axisymmetric,
    int axis_points,
    const std::shared_ptr<ReadbackBatch>& batch,
    std::function<void(std::string, GeometryControlResult)> callback);

// Scan one f32 plane without downloading it. Integer exponent tests also catch
// NaNs on backends whose floating-point optimizations assume finite operands.
// The callback is deferred until batch->map(), like other batched operators.
void enqueue_field_finite(const wgpu::Device& device,
                          const DeviceFields& fields,
                          const std::shared_ptr<ReadbackBatch>& batch,
                          std::function<void(std::string, bool)> callback);

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
