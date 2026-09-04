#include "cumes/webgpu/axisymmetric.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <numbers>
#include <sstream>
#include <utility>

namespace cumes::webgpu {
namespace {

constexpr std::uint32_t WORKGROUP_SIZE = 256;
constexpr std::size_t RESULT_FIELD_COUNT = GEOMETRY_PARITY_FIELD_COUNT + 2;

struct ShaderParams {
    std::uint32_t ns;
    std::uint32_t mpol;
    std::uint32_t ntheta;
    std::uint32_t points;
    std::uint32_t padding[4];
};
static_assert(sizeof(ShaderParams) == 32);

std::string validate_case(const AxisymmetricInverseCase& input) {
    if (input.ns < 2 || input.mpol <= 0 || input.ntheta < 2 ||
        input.ntheta % 2 != 0) {
        return "axisymmetric inverse requires ns>=2, mpol>0, and even "
               "ntheta>=2";
    }
    const auto spectral_values = SPECTRAL_COMPONENT_COUNT *
                                 static_cast<std::size_t>(input.mpol) *
                                 static_cast<std::size_t>(input.ns);
    if (input.state.size() != spectral_values) {
        return "axisymmetric state size does not match 6*mpol*ns";
    }
    const auto points = static_cast<std::size_t>(input.ns) * input.ntheta;
    if (points > std::numeric_limits<std::uint32_t>::max() ||
        points > std::numeric_limits<std::size_t>::max() / RESULT_FIELD_COUNT) {
        return "axisymmetric output exceeds WebGPU indexing limits";
    }
    return {};
}

std::string load_shader() {
    std::ifstream stream("/shaders/axisymmetric_inverse.wgsl",
                         std::ios::binary);
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
    AxisymmetricInverseCallback callback;
    wgpu::Buffer result_buffer;
    wgpu::Buffer readback_buffer;
    std::size_t points = 0;
    std::size_t result_bytes = 0;
};

}  // namespace

AxisymmetricInverseResult axisymmetric_inverse_reference(
    const AxisymmetricInverseCase& input) {
    if (!validate_case(input).empty()) return {};

    const std::size_t points =
        static_cast<std::size_t>(input.ns) * input.ntheta;
    AxisymmetricInverseResult result;
    result.geometry.assign(GEOMETRY_PARITY_FIELD_COUNT * points, 0.0F);
    result.r_con.resize(points);
    result.z_con.resize(points);

    const auto coefficient = [&](int component, int mode, int surface) {
        return input
            .state[(static_cast<std::size_t>(component) * input.mpol + mode) *
                       input.ns +
                   surface];
    };
    const auto store = [&](int field, std::size_t point, float value) {
        result.geometry[static_cast<std::size_t>(field) * points + point] =
            value;
    };

    for (int surface = 0; surface < input.ns; ++surface) {
        const float maxsc =
            std::max(std::sqrt(static_cast<float>(surface) /
                               static_cast<float>(input.ns - 1)),
                     std::sqrt(1.0F / static_cast<float>(input.ns - 1)));
        for (int theta_index = 0; theta_index < input.ntheta; ++theta_index) {
            const std::size_t point =
                static_cast<std::size_t>(surface) * input.ntheta + theta_index;
            const float theta = 2.0F * std::numbers::pi_v<float> *
                                static_cast<float>(theta_index) /
                                static_cast<float>(input.ntheta);
            float r_e = 0.0F, z_e = 0.0F, l_e = 0.0F;
            float ru_e = 0.0F, zu_e = 0.0F, lu_e = 0.0F;
            float r_o = 0.0F, z_o = 0.0F, l_o = 0.0F;
            float ru_o = 0.0F, zu_o = 0.0F, lu_o = 0.0F;
            float r_con = 0.0F, z_con = 0.0F;
            for (int mode = 0; mode < input.mpol; ++mode) {
                const float mf = static_cast<float>(mode);
                const float cosine = std::cos(mf * theta);
                const float sine = std::sin(mf * theta);
                const float rc = coefficient(0, mode, surface);
                const float zs = coefficient(1, mode, surface);
                const float ls = coefficient(2, mode, surface);
                const bool odd = mode % 2 == 1;
                const float scale = odd ? 1.0F / maxsc : 1.0F;
                const float rv = scale * rc * cosine;
                const float zv = scale * zs * sine;
                const float lv = scale * ls * sine;
                const float ruv = scale * rc * (-mf * sine);
                const float zuv = scale * zs * (mf * cosine);
                const float luv = scale * ls * (mf * cosine);
                if (odd) {
                    r_o += rv;
                    z_o += zv;
                    l_o += lv;
                    ru_o += ruv;
                    zu_o += zuv;
                    lu_o += luv;
                } else {
                    r_e += rv;
                    z_e += zv;
                    l_e += lv;
                    ru_e += ruv;
                    zu_e += zuv;
                    lu_e += luv;
                }
                const float xmpq = mf * (mf - 1.0F);
                r_con += xmpq * rc * cosine;
                z_con += xmpq * zs * sine;
            }
            store(0, point, r_e);
            store(1, point, z_e);
            store(2, point, l_e);
            store(3, point, ru_e);
            store(4, point, zu_e);
            store(5, point, lu_e);
            store(6, point, r_o);
            store(7, point, z_o);
            store(8, point, l_o);
            store(9, point, ru_o);
            store(10, point, zu_o);
            store(11, point, lu_o);
            result.r_con[point] = r_con;
            result.z_con[point] = z_con;
        }
    }
    return result;
}

void enqueue_axisymmetric_inverse(const wgpu::Device& device,
                                  const AxisymmetricInverseCase& input,
                                  AxisymmetricInverseCallback callback) {
    const std::string validation_error = validate_case(input);
    if (!validation_error.empty()) {
        callback(validation_error, {});
        return;
    }
    const std::string shader_text = load_shader();
    if (shader_text.empty()) {
        callback("cannot load embedded /shaders/axisymmetric_inverse.wgsl", {});
        return;
    }

    const std::size_t points =
        static_cast<std::size_t>(input.ns) * input.ntheta;
    const std::size_t input_bytes = input.state.size() * sizeof(float);
    std::vector<float> basis(2 * static_cast<std::size_t>(input.mpol) *
                             input.ntheta);
    for (int mode = 0; mode < input.mpol; ++mode) {
        for (int theta_index = 0; theta_index < input.ntheta; ++theta_index) {
            const float theta = 2.0F * std::numbers::pi_v<float> *
                                static_cast<float>(theta_index) /
                                static_cast<float>(input.ntheta);
            const auto offset =
                static_cast<std::size_t>(mode) * input.ntheta + theta_index;
            basis[offset] = std::cos(static_cast<float>(mode) * theta);
            basis[static_cast<std::size_t>(input.mpol) * input.ntheta +
                  offset] = std::sin(static_cast<float>(mode) * theta);
        }
    }
    const std::size_t basis_bytes = basis.size() * sizeof(float);
    const std::size_t result_bytes =
        RESULT_FIELD_COUNT * points * sizeof(float);
    const wgpu::Buffer input_buffer =
        create_buffer(device, input_bytes,
                      wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                      "cuMES axisymmetric spectral state");
    const wgpu::Buffer basis_buffer =
        create_buffer(device, basis_bytes,
                      wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                      "cuMES axisymmetric Fourier basis");
    const wgpu::Buffer result_buffer =
        create_buffer(device, result_bytes,
                      wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc,
                      "cuMES axisymmetric inverse result");
    const wgpu::Buffer readback_buffer =
        create_buffer(device, result_bytes,
                      wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead,
                      "cuMES axisymmetric inverse readback");
    const wgpu::Buffer params_buffer =
        create_buffer(device, sizeof(ShaderParams),
                      wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
                      "cuMES axisymmetric inverse parameters");

    wgpu::ShaderSourceWGSL wgsl{};
    wgsl.code = shader_text.c_str();
    wgpu::ShaderModuleDescriptor shader_descriptor{};
    shader_descriptor.label = "cuMES axisymmetric inverse";
    shader_descriptor.nextInChain = &wgsl;
    const wgpu::ShaderModule shader =
        device.CreateShaderModule(&shader_descriptor);
    wgpu::ComputePipelineDescriptor pipeline_descriptor{};
    pipeline_descriptor.label = "cuMES axisymmetric inverse pipeline";
    pipeline_descriptor.compute.module = shader;
    pipeline_descriptor.compute.entryPoint = "main";
    const wgpu::ComputePipeline pipeline =
        device.CreateComputePipeline(&pipeline_descriptor);

    const ShaderParams params{static_cast<std::uint32_t>(input.ns),
                              static_cast<std::uint32_t>(input.mpol),
                              static_cast<std::uint32_t>(input.ntheta),
                              static_cast<std::uint32_t>(points),
                              {0, 0, 0, 0}};
    const wgpu::Queue queue = device.GetQueue();
    queue.WriteBuffer(input_buffer, 0, input.state.data(), input_bytes);
    queue.WriteBuffer(basis_buffer, 0, basis.data(), basis_bytes);
    queue.WriteBuffer(params_buffer, 0, &params, sizeof(params));

    const wgpu::BindGroupLayout layout = pipeline.GetBindGroupLayout(0);
    const wgpu::BindGroupEntry entries[] = {
        {nullptr, 0, input_buffer, 0, input_bytes, nullptr, nullptr},
        {nullptr, 1, basis_buffer, 0, basis_bytes, nullptr, nullptr},
        {nullptr, 2, result_buffer, 0, result_bytes, nullptr, nullptr},
        {nullptr, 3, params_buffer, 0, sizeof(params), nullptr, nullptr},
    };
    wgpu::BindGroupDescriptor bind_group_descriptor{};
    bind_group_descriptor.label = "cuMES axisymmetric inverse bindings";
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
        (static_cast<std::uint32_t>(points) + WORKGROUP_SIZE - 1) /
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
    dispatch->points = points;
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
                    "WebGPU axisymmetric result mapping failed: " + detail, {});
                return;
            }
            const void* mapped = dispatch->readback_buffer.GetConstMappedRange(
                0, dispatch->result_bytes);
            if (mapped == nullptr) {
                dispatch->callback("WebGPU returned a null mapped range", {});
                return;
            }
            const auto* values = static_cast<const float*>(mapped);
            const std::size_t geometry_values =
                GEOMETRY_PARITY_FIELD_COUNT * dispatch->points;
            AxisymmetricInverseResult result;
            result.geometry.assign(values, values + geometry_values);
            result.r_con.assign(values + geometry_values,
                                values + geometry_values + dispatch->points);
            result.z_con.assign(
                values + geometry_values + dispatch->points,
                values + geometry_values + 2 * dispatch->points);
            dispatch->readback_buffer.Unmap();
            dispatch->callback({}, std::move(result));
        });
}

}  // namespace cumes::webgpu
