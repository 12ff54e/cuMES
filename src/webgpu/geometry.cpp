#include "cumes/webgpu/geometry.hpp"

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
constexpr std::size_t GEOMETRY_INPUT_FIELD_COUNT = 18;

struct ShaderParams {
    std::uint32_t ns;
    std::uint32_t ntheta;
    std::uint32_t full_points;
    std::uint32_t half_points;
    float delta_s;
    std::uint32_t padding[3];
};
static_assert(sizeof(ShaderParams) == 32);

std::string validate_case(const BaseGeometryCase& input) {
    if (input.ns < 2 || input.ntheta < 2 || input.ntheta % 2 != 0 ||
        !std::isfinite(input.delta_s) || input.delta_s <= 0.0F) {
        return "base geometry requires ns>=2, even ntheta>=2, and positive "
               "finite delta_s";
    }
    const std::size_t full_points =
        static_cast<std::size_t>(input.ns) * input.ntheta;
    const std::size_t half_points =
        static_cast<std::size_t>(input.ns - 1) * input.ntheta;
    if (full_points > std::numeric_limits<std::uint32_t>::max() ||
        half_points > std::numeric_limits<std::uint32_t>::max()) {
        return "base geometry exceeds WebGPU indexing limits";
    }
    if (input.geometry.size() != GEOMETRY_INPUT_FIELD_COUNT * full_points ||
        input.sqrt_s_f.size() != static_cast<std::size_t>(input.ns) ||
        input.sqrt_s_h.size() != static_cast<std::size_t>(input.ns - 1)) {
        return "base geometry input shape mismatch";
    }
    return {};
}

std::string load_shader() {
    std::ifstream stream("/shaders/base_geometry.wgsl", std::ios::binary);
    if (!stream) return {};
    std::ostringstream text;
    text << stream.rdbuf();
    return text.str();
}

wgpu::Buffer create_buffer(const wgpu::Device& device,
                           std::uint64_t size,
                           wgpu::BufferUsage usage,
                           const char* label) {
    wgpu::BufferDescriptor descriptor{};
    descriptor.label = label;
    descriptor.size = size;
    descriptor.usage = usage;
    return device.CreateBuffer(&descriptor);
}

struct DispatchState {
    BaseGeometryCallback callback;
    wgpu::Buffer result_buffer;
    wgpu::Buffer readback_buffer;
    std::size_t result_values = 0;
    std::size_t result_bytes = 0;
};

}  // namespace

BaseGeometryResult base_geometry_reference(const BaseGeometryCase& input) {
    if (!validate_case(input).empty()) return {};
    const std::size_t full_points =
        static_cast<std::size_t>(input.ns) * input.ntheta;
    const std::size_t half_points =
        static_cast<std::size_t>(input.ns - 1) * input.ntheta;
    BaseGeometryResult result;
    result.fields.resize(BASE_GEOMETRY_FIELD_COUNT * half_points);
    const auto full = [&](std::size_t field, std::size_t point) {
        return input.geometry[field * full_points + point];
    };
    const auto store = [&](std::size_t field, std::size_t point, float value) {
        result.fields[field * half_points + point] = value;
    };

    for (int surface = 0; surface < input.ns - 1; ++surface) {
        const float sqrt_h = input.sqrt_s_h[surface];
        const float sqrt_i = input.sqrt_s_f[surface];
        const float sqrt_o = input.sqrt_s_f[surface + 1];
        const float sqrt_i_squared = sqrt_i * sqrt_i;
        const float sqrt_o_squared = sqrt_o * sqrt_o;
        for (int theta = 0; theta < input.ntheta; ++theta) {
            const std::size_t point =
                static_cast<std::size_t>(surface) * input.ntheta + theta;
            const std::size_t inside = point;
            const std::size_t outside = point + input.ntheta;
            const float r12 =
                0.5F * ((full(0, inside) + full(0, outside)) +
                        sqrt_h * (full(6, inside) + full(6, outside)));
            const float ru12 =
                0.5F * ((full(3, inside) + full(3, outside)) +
                        sqrt_h * (full(9, inside) + full(9, outside)));
            const float zu12 =
                0.5F * ((full(4, inside) + full(4, outside)) +
                        sqrt_h * (full(10, inside) + full(10, outside)));
            const float rs = ((full(0, outside) - full(0, inside)) +
                              sqrt_h * (full(6, outside) - full(6, inside))) /
                             input.delta_s;
            const float zs = ((full(1, outside) - full(1, inside)) +
                              sqrt_h * (full(7, outside) - full(7, inside))) /
                             input.delta_s;
            const float tau1 = ru12 * zs - rs * zu12;
            const float tau2 = full(9, outside) * full(7, outside) +
                               full(9, inside) * full(7, inside) -
                               full(10, outside) * full(6, outside) -
                               full(10, inside) * full(6, inside) +
                               (full(3, outside) * full(7, outside) +
                                full(3, inside) * full(7, inside) -
                                full(4, outside) * full(6, outside) -
                                full(4, inside) * full(6, inside)) /
                                   sqrt_h;
            const float tau = tau1 + 0.25F * tau2;
            const float gsqrt = tau * r12;
            const float guu =
                0.5F *
                    (full(3, inside) * full(3, inside) +
                     full(4, inside) * full(4, inside) +
                     full(3, outside) * full(3, outside) +
                     full(4, outside) * full(4, outside) +
                     sqrt_i_squared * (full(9, inside) * full(9, inside) +
                                       full(10, inside) * full(10, inside)) +
                     sqrt_o_squared * (full(9, outside) * full(9, outside) +
                                       full(10, outside) * full(10, outside))) +
                sqrt_h * (full(3, inside) * full(9, inside) +
                          full(4, inside) * full(10, inside) +
                          full(3, outside) * full(9, outside) +
                          full(4, outside) * full(10, outside));
            float gvv =
                0.5F * (full(0, inside) * full(0, inside) +
                        full(0, outside) * full(0, outside) +
                        sqrt_i_squared * full(6, inside) * full(6, inside) +
                        sqrt_o_squared * full(6, outside) * full(6, outside)) +
                sqrt_h * (full(0, inside) * full(6, inside) +
                          full(0, outside) * full(6, outside));
            const float guv =
                0.5F *
                (full(3, inside) * full(12, inside) +
                 full(4, inside) * full(13, inside) +
                 full(3, outside) * full(12, outside) +
                 full(4, outside) * full(13, outside) +
                 sqrt_i_squared * (full(9, inside) * full(15, inside) +
                                   full(10, inside) * full(16, inside)) +
                 sqrt_o_squared * (full(9, outside) * full(15, outside) +
                                   full(10, outside) * full(16, outside)) +
                 sqrt_h * (full(3, inside) * full(15, inside) +
                           full(4, inside) * full(16, inside) +
                           full(3, outside) * full(15, outside) +
                           full(4, outside) * full(16, outside) +
                           full(12, inside) * full(9, inside) +
                           full(13, inside) * full(10, inside) +
                           full(12, outside) * full(9, outside) +
                           full(13, outside) * full(10, outside)));
            gvv +=
                0.5F *
                    (full(12, inside) * full(12, inside) +
                     full(13, inside) * full(13, inside) +
                     full(12, outside) * full(12, outside) +
                     full(13, outside) * full(13, outside) +
                     sqrt_i_squared * (full(15, inside) * full(15, inside) +
                                       full(16, inside) * full(16, inside)) +
                     sqrt_o_squared * (full(15, outside) * full(15, outside) +
                                       full(16, outside) * full(16, outside))) +
                sqrt_h * (full(12, inside) * full(15, inside) +
                          full(13, inside) * full(16, inside) +
                          full(12, outside) * full(15, outside) +
                          full(13, outside) * full(16, outside));
            store(0, point, r12);
            store(1, point, ru12);
            store(2, point, zu12);
            store(3, point, rs);
            store(4, point, zs);
            store(5, point, tau);
            store(6, point, gsqrt);
            store(7, point, guu);
            store(8, point, guv);
            store(9, point, gvv);
        }
    }
    return result;
}

void enqueue_base_geometry(const wgpu::Device& device,
                           const BaseGeometryCase& input,
                           BaseGeometryCallback callback) {
    const std::string validation_error = validate_case(input);
    if (!validation_error.empty()) {
        callback(validation_error, {});
        return;
    }
    const std::string shader_text = load_shader();
    if (shader_text.empty()) {
        callback("cannot load embedded /shaders/base_geometry.wgsl", {});
        return;
    }
    const std::size_t full_points =
        static_cast<std::size_t>(input.ns) * input.ntheta;
    const std::size_t half_points =
        static_cast<std::size_t>(input.ns - 1) * input.ntheta;
    std::vector<float> radial;
    radial.reserve(input.sqrt_s_f.size() + input.sqrt_s_h.size());
    radial.insert(radial.end(), input.sqrt_s_f.begin(), input.sqrt_s_f.end());
    radial.insert(radial.end(), input.sqrt_s_h.begin(), input.sqrt_s_h.end());
    const std::size_t input_bytes = input.geometry.size() * sizeof(float);
    const std::size_t radial_bytes = radial.size() * sizeof(float);
    const std::size_t result_values = BASE_GEOMETRY_FIELD_COUNT * half_points;
    const std::size_t result_bytes = result_values * sizeof(float);
    const wgpu::Buffer input_buffer =
        create_buffer(device, input_bytes,
                      wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                      "cuMES full-grid geometry");
    const wgpu::Buffer radial_buffer =
        create_buffer(device, radial_bytes,
                      wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                      "cuMES geometry radial profiles");
    const wgpu::Buffer result_buffer =
        create_buffer(device, result_bytes,
                      wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc,
                      "cuMES half-grid base geometry");
    const wgpu::Buffer readback_buffer =
        create_buffer(device, result_bytes,
                      wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead,
                      "cuMES half-grid geometry readback");
    const wgpu::Buffer params_buffer =
        create_buffer(device, sizeof(ShaderParams),
                      wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
                      "cuMES base geometry parameters");

    wgpu::ShaderSourceWGSL wgsl{};
    wgsl.code = shader_text.c_str();
    wgpu::ShaderModuleDescriptor shader_descriptor{};
    shader_descriptor.label = "cuMES base geometry";
    shader_descriptor.nextInChain = &wgsl;
    const wgpu::ShaderModule shader =
        device.CreateShaderModule(&shader_descriptor);
    wgpu::ComputePipelineDescriptor pipeline_descriptor{};
    pipeline_descriptor.label = "cuMES base geometry pipeline";
    pipeline_descriptor.compute.module = shader;
    pipeline_descriptor.compute.entryPoint = "main";
    const wgpu::ComputePipeline pipeline =
        device.CreateComputePipeline(&pipeline_descriptor);

    const ShaderParams params{static_cast<std::uint32_t>(input.ns),
                              static_cast<std::uint32_t>(input.ntheta),
                              static_cast<std::uint32_t>(full_points),
                              static_cast<std::uint32_t>(half_points),
                              input.delta_s,
                              {0, 0, 0}};
    const wgpu::Queue queue = device.GetQueue();
    queue.WriteBuffer(input_buffer, 0, input.geometry.data(), input_bytes);
    queue.WriteBuffer(radial_buffer, 0, radial.data(), radial_bytes);
    queue.WriteBuffer(params_buffer, 0, &params, sizeof(params));
    const wgpu::BindGroupLayout layout = pipeline.GetBindGroupLayout(0);
    const wgpu::BindGroupEntry entries[] = {
        {nullptr, 0, input_buffer, 0, input_bytes, nullptr, nullptr},
        {nullptr, 1, radial_buffer, 0, radial_bytes, nullptr, nullptr},
        {nullptr, 2, result_buffer, 0, result_bytes, nullptr, nullptr},
        {nullptr, 3, params_buffer, 0, sizeof(params), nullptr, nullptr},
    };
    wgpu::BindGroupDescriptor bind_group_descriptor{};
    bind_group_descriptor.label = "cuMES base geometry bindings";
    bind_group_descriptor.layout = layout;
    bind_group_descriptor.entryCount = std::size(entries);
    bind_group_descriptor.entries = entries;
    const wgpu::BindGroup bind_group =
        device.CreateBindGroup(&bind_group_descriptor);
    const wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    wgpu::ComputePassDescriptor pass_descriptor{};
    const wgpu::ComputePassEncoder pass =
        encoder.BeginComputePass(&pass_descriptor);
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bind_group);
    pass.DispatchWorkgroups(
        (static_cast<std::uint32_t>(half_points) + WORKGROUP_SIZE - 1) /
        WORKGROUP_SIZE);
    pass.End();
    encoder.CopyBufferToBuffer(result_buffer, 0, readback_buffer, 0,
                               result_bytes);
    const wgpu::CommandBuffer commands = encoder.Finish();
    queue.Submit(1, &commands);

    auto dispatch = std::make_shared<DispatchState>();
    dispatch->callback = std::move(callback);
    dispatch->result_buffer = result_buffer;
    dispatch->readback_buffer = readback_buffer;
    dispatch->result_values = result_values;
    dispatch->result_bytes = result_bytes;
    readback_buffer.MapAsync(
        wgpu::MapMode::Read, 0, result_bytes,
        wgpu::CallbackMode::AllowSpontaneous,
        [dispatch](wgpu::MapAsyncStatus status, wgpu::StringView message) {
            if (status != wgpu::MapAsyncStatus::Success) {
                const std::string detail =
                    message.length == 0
                        ? std::string{}
                        : std::string(message.data, message.length);
                dispatch->callback("WebGPU geometry mapping failed: " + detail,
                                   {});
                return;
            }
            const void* mapped = dispatch->readback_buffer.GetConstMappedRange(
                0, dispatch->result_bytes);
            if (mapped == nullptr) {
                dispatch->callback("WebGPU returned a null mapped range", {});
                return;
            }
            const auto* values = static_cast<const float*>(mapped);
            BaseGeometryResult result;
            result.fields.assign(values, values + dispatch->result_values);
            dispatch->readback_buffer.Unmap();
            dispatch->callback({}, std::move(result));
        });
}

}  // namespace cumes::webgpu
