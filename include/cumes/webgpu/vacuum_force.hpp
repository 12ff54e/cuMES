#pragma once

#include "cumes/webgpu/device_fields.hpp"

#include <functional>
#include <string>

namespace cumes::webgpu {

struct VacuumPressureFields {
    wgpu::Buffer buffer;
    std::uint64_t byte_offset = 0;
    std::size_t count = 0;
};

struct VacuumForceResult {
    DeviceFields device_fields;
    double delbsq = 0.0;
    bool finite = true;
};

struct VacuumForceCase {
    BatchedReadback<VacuumForceResult> readback;
    DeviceFields geometry;
    DeviceFields magnetic_field;
    DeviceFields force;
    // Interleaved high/low words on NESTOR's reduced, theta-major grid.
    VacuumPressureFields vacuum_pressure;
    int ns = 0;
    int ntheta = 0;
    int nzeta = 1;
    bool paired = false;
    double delta_s = 0.0;
    double edge_pressure = 0.0;
};

// Correct the first four force planes at the LCFS in place. Paired arithmetic
// is used for both plasma precisions, matching the existing Wasm-double
// correction to paired accuracy. Interior words remain untouched. The ordered
// host sum for delbsq and finite flags join the caller's existing readback.
void enqueue_vacuum_force(
    const wgpu::Device& device,
    const VacuumForceCase& input,
    std::function<void(std::string, VacuumForceResult)> callback);

}  // namespace cumes::webgpu
