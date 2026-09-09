#include "cumes/webgpu/vacuum_force.hpp"

#include "cumes/webgpu/axisymmetric.hpp"
#include "cumes/webgpu/float_float.hpp"
#include "cumes/webgpu/force.hpp"
#include "cumes/webgpu/geometry.hpp"
#include "pipeline_cache.hpp"
#include "shader_source.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

namespace cumes::webgpu {
namespace {
constexpr std::uint32_t WORKGROUP_SIZE = 256;

bool valid_fields(const DeviceFields& fields, std::size_t count, bool paired) {
    const auto size = fields ? fields.buffer.GetSize() : 0;
    const auto fits = [&](std::uint64_t offset) {
        return offset % sizeof(float) == 0 && offset <= size &&
               count <= (size - offset) / sizeof(float);
    };
    return fields && fields.values == count &&
           size / sizeof(float) <= std::numeric_limits<std::uint32_t>::max() &&
           fits(fields.high_offset) && (!paired || fits(fields.low_offset));
}
}  // namespace

void enqueue_vacuum_force(
    const wgpu::Device& device,
    const VacuumForceCase& input,
    std::function<void(std::string, VacuumForceResult)> callback) {
    if (!device || !input.readback.batch || input.ns < 3 || input.ntheta < 2 ||
        input.ntheta % 2 != 0 || input.nzeta < 1 ||
        !std::isfinite(input.delta_s) || input.delta_s <= 0.0 ||
        !std::isfinite(input.edge_pressure)) {
        callback("invalid vacuum force grid, scalars, or readback", {});
        return;
    }
    const auto angular = std::size_t(input.ntheta) * input.nzeta;
    const auto full = std::size_t(input.ns) * angular;
    const auto half = std::size_t(input.ns - 1) * angular;
    const auto vacuum_count = std::size_t(input.ntheta / 2 + 1) * input.nzeta;
    const auto& vacuum = input.vacuum_pressure;
    const auto vacuum_size = vacuum.buffer ? vacuum.buffer.GetSize() : 0;
    if (angular > 65535U * WORKGROUP_SIZE ||
        full > std::numeric_limits<std::uint32_t>::max() /
                   GEOMETRY_PARITY_FIELD_COUNT ||
        !valid_fields(input.geometry, GEOMETRY_PARITY_FIELD_COUNT * full,
                      input.paired) ||
        !valid_fields(input.magnetic_field, MAGNETIC_FIELD_COUNT * half,
                      input.paired) ||
        !valid_fields(input.force, FORCE_FIELD_COUNT * full, input.paired) ||
        !vacuum.buffer || vacuum.count != vacuum_count ||
        vacuum_size / sizeof(float) >
            std::numeric_limits<std::uint32_t>::max() ||
        vacuum.byte_offset % sizeof(FloatFloat) != 0 ||
        vacuum.byte_offset > vacuum_size ||
        vacuum_count >
            (vacuum_size - vacuum.byte_offset) / sizeof(FloatFloat)) {
        callback("invalid vacuum force device field shape or range", {});
        return;
    }
    const auto delta_s = split(input.delta_s);
    const auto edge_pressure = split(input.edge_pressure);
    if (!std::isfinite(delta_s.hi) || delta_s.hi == 0.0F ||
        !std::isfinite(edge_pressure.hi)) {
        callback("vacuum force scalars exceed paired-f32 range", {});
        return;
    }
    const auto& source =
        detail::cached_shader_source("/shaders/vacuum_force.wgsl");
    if (source.empty()) {
        callback("cannot load embedded WebGPU vacuum force shader", {});
        return;
    }
    struct Params {
        std::uint32_t ns, ntheta, nzeta, angular;
        std::uint32_t geometry_high, geometry_low, magnetic_high, magnetic_low;
        std::uint32_t force_high, force_low, vacuum_offset, paired;
        FloatFloat delta_s, edge_pressure;
    };
    static_assert(sizeof(Params) == 64);
    const Params params{
        static_cast<std::uint32_t>(input.ns),
        static_cast<std::uint32_t>(input.ntheta),
        static_cast<std::uint32_t>(input.nzeta),
        static_cast<std::uint32_t>(angular),
        static_cast<std::uint32_t>(input.geometry.high_offset / sizeof(float)),
        static_cast<std::uint32_t>(input.geometry.low_offset / sizeof(float)),
        static_cast<std::uint32_t>(input.magnetic_field.high_offset /
                                   sizeof(float)),
        static_cast<std::uint32_t>(input.magnetic_field.low_offset /
                                   sizeof(float)),
        static_cast<std::uint32_t>(input.force.high_offset / sizeof(float)),
        static_cast<std::uint32_t>(input.force.low_offset / sizeof(float)),
        static_cast<std::uint32_t>(vacuum.byte_offset / sizeof(float)),
        input.paired ? 1U : 0U,
        delta_s,
        edge_pressure};
    const auto bytes = 3 * angular * sizeof(float);
    const auto diagnostic = detail::cached_buffer(
        device, bytes, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc,
        "vacuum force pressure error and validity");
    const auto uniform = detail::cached_buffer(
        device, sizeof(params),
        wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
        "vacuum force params");
    device.GetQueue().WriteBuffer(uniform, 0, &params, sizeof(params));
    const auto& pipeline = detail::cached_compute_pipeline(
        device, "vacuum-force", source, "cuMES vacuum edge force");
    const wgpu::BindGroupEntry entries[] = {
        {nullptr, 0, input.geometry.buffer, 0, input.geometry.buffer.GetSize(),
         nullptr, nullptr},
        {nullptr, 1, input.magnetic_field.buffer, 0,
         input.magnetic_field.buffer.GetSize(), nullptr, nullptr},
        {nullptr, 2, input.force.buffer, 0, input.force.buffer.GetSize(),
         nullptr, nullptr},
        {nullptr, 3, vacuum.buffer, 0, vacuum_size, nullptr, nullptr},
        {nullptr, 4, diagnostic, 0, bytes, nullptr, nullptr},
        {nullptr, 5, uniform, 0, sizeof(params), nullptr, nullptr}};
    wgpu::BindGroupDescriptor descriptor{};
    descriptor.layout = pipeline.GetBindGroupLayout(0);
    descriptor.entries = entries;
    descriptor.entryCount = 6;
    const auto group = device.CreateBindGroup(&descriptor);
    const auto encoder = device.CreateCommandEncoder();
    auto pass = encoder.BeginComputePass();
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, group);
    pass.DispatchWorkgroups(
        (static_cast<std::uint32_t>(angular) + WORKGROUP_SIZE - 1) /
        WORKGROUP_SIZE);
    pass.End();
    const auto force = input.force;
    input.readback.batch->append(
        encoder, diagnostic, 0, bytes,
        [callback = std::move(callback), force,
         angular](std::span<const float> words) {
            VacuumForceResult result;
            result.device_fields = force;
            for (std::size_t point = 0; point < angular; ++point) {
                const double contribution =
                    double(words[3 * point]) + words[3 * point + 1];
                result.delbsq += contribution;
                result.finite &=
                    std::isfinite(contribution) && words[3 * point + 2] == 0.0F;
            }
            callback({}, std::move(result));
        });
    const auto commands = encoder.Finish();
    device.GetQueue().Submit(1, &commands);
    input.readback.publish_device({force, 0.0, true});
}

}  // namespace cumes::webgpu
