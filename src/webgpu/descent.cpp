#include "cumes/webgpu/descent.hpp"

#include "pipeline_cache.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <sstream>
#include <utility>

namespace cumes::webgpu {
namespace {

constexpr std::uint32_t WORKGROUP_SIZE = 256;

struct Params {
    std::uint32_t ns, ntor_plus_one, points, move_lcfs;
    float delta_t, damping_b1, damping_fac, padding;
};
static_assert(sizeof(Params) == 32);

std::string validate_case(const AxisymmetricDescentCase& in) {
    if (in.ns < 2 || in.mpol < 2 || in.ntor < 0 || !std::isfinite(in.delta_t) ||
        !std::isfinite(in.damping_b1) || !std::isfinite(in.damping_fac)) {
        return "axisymmetric descent has invalid dimensions or scalars";
    }
    const std::size_t points =
        static_cast<std::size_t>(in.ns) * in.mpol * (in.ntor + 1);
    if (points > std::numeric_limits<std::uint32_t>::max() ||
        in.state.size() != 6 * points || in.velocity.size() != 6 * points ||
        in.residual.size() != 6 * points) {
        return "axisymmetric descent input shape mismatch";
    }
    return {};
}

std::string load_shader() {
    std::ifstream stream("/shaders/axisymmetric_descent.wgsl",
                         std::ios::binary);
    if (!stream) return {};
    std::ostringstream text;
    text << stream.rdbuf();
    return text.str();
}

wgpu::Buffer make_buffer(const wgpu::Device& device,
                         std::uint64_t size,
                         wgpu::BufferUsage usage,
                         const char* label) {
    wgpu::BufferDescriptor descriptor{};
    descriptor.label = label;
    descriptor.size = size;
    descriptor.usage = usage;
    return device.CreateBuffer(&descriptor);
}

struct Dispatch {
    AxisymmetricDescentCallback callback;
    wgpu::Buffer output, readback;
    std::size_t bytes = 0, values = 0;
};

}  // namespace

AxisymmetricDescentResult axisymmetric_descent_reference(
    const AxisymmetricDescentCase& input) {
    if (!validate_case(input).empty()) return {};
    const int mode_count = input.mpol * (input.ntor + 1);
    const std::size_t points = static_cast<std::size_t>(input.ns) * mode_count;
    AxisymmetricDescentResult out{input.state, input.velocity};
    const auto index = [points, &input](int component, int mode, int surface) {
        return static_cast<std::size_t>(component) * points +
               static_cast<std::size_t>(mode) * input.ns + surface;
    };
    const auto update_velocity = [&](int component, int mode, int surface) {
        const auto i = index(component, mode, surface);
        return input.damping_fac * (input.damping_b1 * input.velocity[i] +
                                    input.delta_t * input.residual[i]);
    };
    const int j_max = input.move_lcfs ? input.ns : input.ns - 1;
    for (int mode = 0; mode < mode_count; ++mode) {
        const int m = mode / (input.ntor + 1);
        const int n = mode % (input.ntor + 1);
        const float m_scale = m == 0 ? 1.0F : 1.4142135623730951F;
        const float n_scale = n == 0 ? 1.0F : 1.4142135623730951F;
        const float basis_scale = m_scale * n_scale;
        for (int surface = 0; surface < input.ns; ++surface) {
            if (surface == 0 && m > 0) continue;
            if (surface < j_max) {
                const float vr = update_velocity(0, mode, surface);
                const float vz = update_velocity(1, mode, surface);
                const float vrs = update_velocity(3, mode, surface);
                const float vzc = update_velocity(4, mode, surface);
                out.velocity[index(0, mode, surface)] = vr;
                out.velocity[index(1, mode, surface)] = vz;
                out.velocity[index(3, mode, surface)] = vrs;
                out.velocity[index(4, mode, surface)] = vzc;
                out.state[index(0, mode, surface)] +=
                    input.delta_t * vr * basis_scale;
                out.state[index(1, mode, surface)] +=
                    input.delta_t * vz * basis_scale;
                if (m == 1) {
                    out.state[index(3, mode, surface)] +=
                        input.delta_t * (vrs + vzc) * basis_scale;
                    out.state[index(4, mode, surface)] +=
                        input.delta_t * (vrs - vzc) * basis_scale;
                } else {
                    out.state[index(3, mode, surface)] +=
                        input.delta_t * vrs * basis_scale;
                    out.state[index(4, mode, surface)] +=
                        input.delta_t * vzc * basis_scale;
                }
            }
            const float vl = update_velocity(2, mode, surface);
            const float vlc = update_velocity(5, mode, surface);
            out.velocity[index(2, mode, surface)] = vl;
            out.velocity[index(5, mode, surface)] = vlc;
            out.state[index(2, mode, surface)] +=
                input.delta_t * vl * basis_scale;
            out.state[index(5, mode, surface)] +=
                input.delta_t * vlc * basis_scale;
        }
    }
    return out;
}

void enqueue_axisymmetric_descent(const wgpu::Device& device,
                                  const AxisymmetricDescentCase& input,
                                  AxisymmetricDescentCallback callback) {
    const auto error = validate_case(input);
    if (!error.empty()) {
        callback(error, {});
        return;
    }
    const auto shader_text = load_shader();
    if (shader_text.empty()) {
        callback("cannot load axisymmetric descent shader", {});
        return;
    }
    const std::size_t points =
        static_cast<std::size_t>(input.ns) * input.mpol * (input.ntor + 1);
    const auto input_bytes = input.state.size() * sizeof(float);
    const auto output_values = 12 * points;
    const auto output_bytes = output_values * sizeof(float);
    auto state_buffer =
        make_buffer(device, input_bytes,
                    wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                    "descent state");
    auto velocity_buffer =
        make_buffer(device, input_bytes,
                    wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                    "descent velocity");
    auto residual_buffer =
        make_buffer(device, input_bytes,
                    wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                    "descent residual");
    auto output_buffer =
        make_buffer(device, output_bytes,
                    wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc,
                    "descent output");
    auto readback =
        make_buffer(device, output_bytes,
                    wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead,
                    "descent readback");
    auto params_buffer =
        make_buffer(device, sizeof(Params),
                    wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
                    "descent params");
    const auto& pipeline = detail::cached_compute_pipeline(
        device, "accelerated-descent", shader_text,
        "cuMES accelerated descent pipeline");
    const Params params{static_cast<std::uint32_t>(input.ns),
                        static_cast<std::uint32_t>(input.ntor + 1),
                        static_cast<std::uint32_t>(points),
                        input.move_lcfs ? 1U : 0U,
                        input.delta_t,
                        input.damping_b1,
                        input.damping_fac,
                        0.0F};
    auto queue = device.GetQueue();
    queue.WriteBuffer(state_buffer, 0, input.state.data(), input_bytes);
    queue.WriteBuffer(velocity_buffer, 0, input.velocity.data(), input_bytes);
    queue.WriteBuffer(residual_buffer, 0, input.residual.data(), input_bytes);
    queue.WriteBuffer(params_buffer, 0, &params, sizeof(params));
    auto layout = pipeline.GetBindGroupLayout(0);
    const wgpu::BindGroupEntry entries[] = {
        {nullptr, 0, state_buffer, 0, input_bytes, nullptr, nullptr},
        {nullptr, 1, velocity_buffer, 0, input_bytes, nullptr, nullptr},
        {nullptr, 2, residual_buffer, 0, input_bytes, nullptr, nullptr},
        {nullptr, 3, output_buffer, 0, output_bytes, nullptr, nullptr},
        {nullptr, 4, params_buffer, 0, sizeof(params), nullptr, nullptr}};
    wgpu::BindGroupDescriptor bind_descriptor{};
    bind_descriptor.layout = layout;
    bind_descriptor.entryCount = std::size(entries);
    bind_descriptor.entries = entries;
    auto bind_group = device.CreateBindGroup(&bind_descriptor);
    auto encoder = device.CreateCommandEncoder();
    wgpu::ComputePassDescriptor pass_descriptor{};
    auto pass = encoder.BeginComputePass(&pass_descriptor);
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bind_group);
    pass.DispatchWorkgroups(
        (static_cast<std::uint32_t>(points) + WORKGROUP_SIZE - 1) /
        WORKGROUP_SIZE);
    pass.End();
    encoder.CopyBufferToBuffer(output_buffer, 0, readback, 0, output_bytes);
    auto commands = encoder.Finish();
    queue.Submit(1, &commands);
    auto dispatch = std::make_shared<Dispatch>();
    dispatch->callback = std::move(callback);
    dispatch->output = output_buffer;
    dispatch->readback = readback;
    dispatch->bytes = output_bytes;
    dispatch->values = 6 * points;
    readback.MapAsync(
        wgpu::MapMode::Read, 0, output_bytes,
        wgpu::CallbackMode::AllowSpontaneous,
        [dispatch](wgpu::MapAsyncStatus status, wgpu::StringView message) {
            if (status != wgpu::MapAsyncStatus::Success) {
                const std::string detail =
                    message.length == 0
                        ? std::string{}
                        : std::string(message.data, message.length);
                dispatch->callback("descent mapping failed: " + detail, {});
                return;
            }
            const auto* values = static_cast<const float*>(
                dispatch->readback.GetConstMappedRange(0, dispatch->bytes));
            if (values == nullptr) {
                dispatch->callback("descent mapped range is null", {});
                return;
            }
            AxisymmetricDescentResult out;
            out.state.assign(values, values + dispatch->values);
            out.velocity.assign(values + dispatch->values,
                                values + 2 * dispatch->values);
            dispatch->readback.Unmap();
            dispatch->callback({}, std::move(out));
        });
}

}  // namespace cumes::webgpu
