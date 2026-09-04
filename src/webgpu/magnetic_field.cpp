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
constexpr std::size_t FULL_GEOMETRY_FIELD_COUNT = 18;

struct ShaderParams {
    std::uint32_t ns;
    std::uint32_t ntheta;
    std::uint32_t full_points;
    std::uint32_t half_points;
    float lamscale;
    std::uint32_t padding[3];
};
static_assert(sizeof(ShaderParams) == 32);

std::string validate_case(const MagneticFieldCase& input) {
    if (input.ns < 2 || input.ntheta < 2 || input.ntheta % 2 != 0 ||
        !std::isfinite(input.lamscale)) {
        return "magnetic field requires ns>=2, even ntheta>=2, and finite "
               "lamscale";
    }
    const std::size_t full_points =
        static_cast<std::size_t>(input.ns) * input.ntheta;
    const std::size_t half_points =
        static_cast<std::size_t>(input.ns - 1) * input.ntheta;
    if (full_points > std::numeric_limits<std::uint32_t>::max() ||
        half_points > std::numeric_limits<std::uint32_t>::max()) {
        return "magnetic field exceeds WebGPU indexing limits";
    }
    const auto half_surfaces = static_cast<std::size_t>(input.ns - 1);
    if (input.geometry.size() != FULL_GEOMETRY_FIELD_COUNT * full_points ||
        input.base_geometry.size() != BASE_GEOMETRY_FIELD_COUNT * half_points ||
        input.sqrt_s_h.size() != half_surfaces ||
        input.phip_f.size() != static_cast<std::size_t>(input.ns) ||
        input.chip_h.size() != half_surfaces ||
        input.pres_h.size() != half_surfaces) {
        return "magnetic field input shape mismatch";
    }
    return {};
}

std::string load_shader() {
    std::ifstream stream("/shaders/magnetic_field.wgsl", std::ios::binary);
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
    MagneticFieldCallback callback;
    wgpu::Buffer result_buffer;
    wgpu::Buffer readback_buffer;
    std::size_t result_values = 0;
    std::size_t result_bytes = 0;
};

}  // namespace

MagneticFieldResult magnetic_field_reference(const MagneticFieldCase& input) {
    if (!validate_case(input).empty()) return {};
    const std::size_t full_points =
        static_cast<std::size_t>(input.ns) * input.ntheta;
    const std::size_t half_points =
        static_cast<std::size_t>(input.ns - 1) * input.ntheta;
    MagneticFieldResult result;
    result.fields.resize(MAGNETIC_FIELD_COUNT * half_points);
    const auto full = [&](std::size_t field, std::size_t point) {
        return input.geometry[field * full_points + point];
    };
    const auto half = [&](std::size_t field, std::size_t point) {
        return input.base_geometry[field * half_points + point];
    };
    const auto store = [&](std::size_t field, std::size_t point, float value) {
        result.fields[field * half_points + point] = value;
    };
    for (int surface = 0; surface < input.ns - 1; ++surface) {
        const float sqrt_h = input.sqrt_s_h[surface];
        const float phip_average =
            0.5F * (input.phip_f[surface] + input.phip_f[surface + 1]);
        for (int theta = 0; theta < input.ntheta; ++theta) {
            const std::size_t point =
                static_cast<std::size_t>(surface) * input.ntheta + theta;
            const std::size_t inside = point;
            const std::size_t outside = point + input.ntheta;
            const float lu_h =
                0.5F * ((full(5, inside) + full(5, outside)) +
                        sqrt_h * (full(11, inside) + full(11, outside)));
            const float lv_h =
                0.5F * ((full(14, inside) + full(14, outside)) +
                        sqrt_h * (full(17, inside) + full(17, outside)));
            const float gsqrt = half(6, point);
            float bsupu = 0.0F;
            float bsupv = 0.0F;
            if (std::isfinite(gsqrt) && std::abs(gsqrt) > 1.0e-30F) {
                bsupv = (input.lamscale * lu_h + phip_average) / gsqrt;
                bsupu = (input.lamscale * lv_h + input.chip_h[surface]) / gsqrt;
            }
            const float bsubu = half(7, point) * bsupu + half(8, point) * bsupv;
            const float bsubv = half(8, point) * bsupu + half(9, point) * bsupv;
            const float magnetic_pressure =
                0.5F * (bsupu * bsubu + bsupv * bsubv);
            store(0, point, bsupu);
            store(1, point, bsupv);
            store(2, point, bsubu);
            store(3, point, bsubv);
            store(4, point, magnetic_pressure + input.pres_h[surface]);
        }
    }
    return result;
}

void enqueue_magnetic_field(const wgpu::Device& device,
                            const MagneticFieldCase& input,
                            MagneticFieldCallback callback) {
    const std::string validation_error = validate_case(input);
    if (!validation_error.empty()) {
        callback(validation_error, {});
        return;
    }
    const std::string shader_text = load_shader();
    if (shader_text.empty()) {
        callback("cannot load embedded /shaders/magnetic_field.wgsl", {});
        return;
    }
    const std::size_t full_points =
        static_cast<std::size_t>(input.ns) * input.ntheta;
    const std::size_t half_points =
        static_cast<std::size_t>(input.ns - 1) * input.ntheta;
    std::vector<float> profiles;
    profiles.reserve(input.sqrt_s_h.size() + input.phip_f.size() +
                     input.chip_h.size() + input.pres_h.size());
    profiles.insert(profiles.end(), input.sqrt_s_h.begin(),
                    input.sqrt_s_h.end());
    profiles.insert(profiles.end(), input.phip_f.begin(), input.phip_f.end());
    profiles.insert(profiles.end(), input.chip_h.begin(), input.chip_h.end());
    profiles.insert(profiles.end(), input.pres_h.begin(), input.pres_h.end());
    const std::size_t geometry_bytes = input.geometry.size() * sizeof(float);
    const std::size_t base_bytes = input.base_geometry.size() * sizeof(float);
    const std::size_t profile_bytes = profiles.size() * sizeof(float);
    const std::size_t result_values = MAGNETIC_FIELD_COUNT * half_points;
    const std::size_t result_bytes = result_values * sizeof(float);
    const auto geometry_buffer =
        create_buffer(device, geometry_bytes,
                      wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                      "cuMES magnetic full geometry");
    const auto base_buffer =
        create_buffer(device, base_bytes,
                      wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                      "cuMES magnetic base geometry");
    const auto profile_buffer =
        create_buffer(device, profile_bytes,
                      wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                      "cuMES magnetic profiles");
    const auto result_buffer =
        create_buffer(device, result_bytes,
                      wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc,
                      "cuMES magnetic field");
    const auto readback_buffer =
        create_buffer(device, result_bytes,
                      wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead,
                      "cuMES magnetic field readback");
    const auto params_buffer =
        create_buffer(device, sizeof(ShaderParams),
                      wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
                      "cuMES magnetic field parameters");

    wgpu::ShaderSourceWGSL wgsl{};
    wgsl.code = shader_text.c_str();
    wgpu::ShaderModuleDescriptor shader_descriptor{};
    shader_descriptor.label = "cuMES magnetic field";
    shader_descriptor.nextInChain = &wgsl;
    const auto shader = device.CreateShaderModule(&shader_descriptor);
    wgpu::ComputePipelineDescriptor pipeline_descriptor{};
    pipeline_descriptor.label = "cuMES magnetic field pipeline";
    pipeline_descriptor.compute.module = shader;
    pipeline_descriptor.compute.entryPoint = "main";
    const auto pipeline = device.CreateComputePipeline(&pipeline_descriptor);
    const ShaderParams params{static_cast<std::uint32_t>(input.ns),
                              static_cast<std::uint32_t>(input.ntheta),
                              static_cast<std::uint32_t>(full_points),
                              static_cast<std::uint32_t>(half_points),
                              input.lamscale,
                              {0, 0, 0}};
    const auto queue = device.GetQueue();
    queue.WriteBuffer(geometry_buffer, 0, input.geometry.data(),
                      geometry_bytes);
    queue.WriteBuffer(base_buffer, 0, input.base_geometry.data(), base_bytes);
    queue.WriteBuffer(profile_buffer, 0, profiles.data(), profile_bytes);
    queue.WriteBuffer(params_buffer, 0, &params, sizeof(params));
    const auto layout = pipeline.GetBindGroupLayout(0);
    const wgpu::BindGroupEntry entries[] = {
        {nullptr, 0, geometry_buffer, 0, geometry_bytes, nullptr, nullptr},
        {nullptr, 1, base_buffer, 0, base_bytes, nullptr, nullptr},
        {nullptr, 2, profile_buffer, 0, profile_bytes, nullptr, nullptr},
        {nullptr, 3, result_buffer, 0, result_bytes, nullptr, nullptr},
        {nullptr, 4, params_buffer, 0, sizeof(params), nullptr, nullptr},
    };
    wgpu::BindGroupDescriptor bind_group_descriptor{};
    bind_group_descriptor.label = "cuMES magnetic field bindings";
    bind_group_descriptor.layout = layout;
    bind_group_descriptor.entryCount = std::size(entries);
    bind_group_descriptor.entries = entries;
    const auto bind_group = device.CreateBindGroup(&bind_group_descriptor);
    const auto encoder = device.CreateCommandEncoder();
    wgpu::ComputePassDescriptor pass_descriptor{};
    const auto pass = encoder.BeginComputePass(&pass_descriptor);
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bind_group);
    pass.DispatchWorkgroups(
        (static_cast<std::uint32_t>(half_points) + WORKGROUP_SIZE - 1) /
        WORKGROUP_SIZE);
    pass.End();
    encoder.CopyBufferToBuffer(result_buffer, 0, readback_buffer, 0,
                               result_bytes);
    const auto commands = encoder.Finish();
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
                dispatch->callback(
                    "WebGPU magnetic field mapping failed: " + detail, {});
                return;
            }
            const void* mapped = dispatch->readback_buffer.GetConstMappedRange(
                0, dispatch->result_bytes);
            if (mapped == nullptr) {
                dispatch->callback("WebGPU returned a null mapped range", {});
                return;
            }
            const auto* values = static_cast<const float*>(mapped);
            MagneticFieldResult result;
            result.fields.assign(values, values + dispatch->result_values);
            dispatch->readback_buffer.Unmap();
            dispatch->callback({}, std::move(result));
        });
}

}  // namespace cumes::webgpu
