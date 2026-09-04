#include "cumes/webgpu/prolongation.hpp"

#include "pipeline_cache.hpp"

#include <algorithm>
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

constexpr std::size_t SPECTRAL_FAMILIES = 6;
constexpr std::uint32_t WORKGROUP_SIZE = 256;

struct ShaderParams {
    std::uint32_t ns_old;
    std::uint32_t ns_new;
    std::uint32_t mnmax;
    std::uint32_t ntorp1;
    std::uint32_t interpolation;
    std::uint32_t total;
    std::uint32_t padding[2];
};
static_assert(sizeof(ShaderParams) == 32);

std::string validate_case(const ProlongationCase& input) {
    if (input.ns_old < 3 || input.ns_new <= input.ns_old) {
        return "prolongation requires ns_new > ns_old >= 3";
    }
    if (input.mnmax <= 0 || input.ntor < 0 || input.ntor + 1 > input.mnmax) {
        return "prolongation requires a positive mode count and valid ntor";
    }
    if (input.interpolation != RadialInterpolation::LINEAR &&
        input.interpolation != RadialInterpolation::CATMULL_ROM) {
        return "unsupported WebGPU radial interpolation";
    }
    const auto expected = SPECTRAL_FAMILIES *
                          static_cast<std::size_t>(input.mnmax) *
                          static_cast<std::size_t>(input.ns_old);
    if (input.state.size() != expected) {
        return "prolongation input state size does not match 6*mnmax*ns_old";
    }
    const auto total = SPECTRAL_FAMILIES *
                       static_cast<std::size_t>(input.mnmax) *
                       static_cast<std::size_t>(input.ns_new);
    if (total > std::numeric_limits<std::uint32_t>::max()) {
        return "prolongation output exceeds WebGPU's 32-bit shader indexing";
    }
    return {};
}

std::string load_shader() {
    std::ifstream stream("/shaders/prolongation.wgsl", std::ios::binary);
    if (!stream) return {};
    std::ostringstream text;
    text << stream.rdbuf();
    return text.str();
}

float scalxc(int j, int ns) {
    const float s = static_cast<float>(j) / static_cast<float>(ns - 1);
    const float sqrt_s1 = std::sqrt(1.0F / static_cast<float>(ns - 1));
    return 1.0F / std::max(std::sqrt(s), sqrt_s1);
}

float reference_value(const ProlongationCase& input, int profile, int j_new) {
    const int mode = profile % input.mnmax;
    const bool odd = ((mode / (input.ntor + 1)) % 2) == 1;
    const float s =
        static_cast<float>(j_new) / static_cast<float>(input.ns_new - 1);
    const int j0 = (j_new * (input.ns_old - 1)) / (input.ns_new - 1);
    const int j1 = std::min(j0 + 1, input.ns_old - 1);
    const float t = std::clamp(
        s * static_cast<float>(input.ns_old - 1) - static_cast<float>(j0), 0.0F,
        1.0F);
    const auto sample = [&](int j) {
        float value =
            input.state[static_cast<std::size_t>(profile) * input.ns_old + j];
        return odd ? value * scalxc(j, input.ns_old) : value;
    };
    const auto regular_sample = [&](int j) {
        return odd && j == 0 ? 2.0F * sample(1) - sample(2) : sample(j);
    };

    const float y0 = regular_sample(j0);
    const float y1 = regular_sample(j1);
    float interpolated = (1.0F - t) * y0 + t * y1;
    if (input.interpolation == RadialInterpolation::CATMULL_ROM && j0 != j1) {
        const float ym1 = j0 > 0 ? regular_sample(j0 - 1) : 2.0F * y0 - y1;
        const float yp2 =
            j1 + 1 < input.ns_old ? regular_sample(j1 + 1) : 2.0F * y1 - y0;
        interpolated = y0 + 0.5F * t *
                                (y1 - ym1 +
                                 t * (2.0F * ym1 - 5.0F * y0 + 4.0F * y1 - yp2 +
                                      t * (3.0F * (y0 - y1) + yp2 - ym1)));
    }

    if (!odd) return interpolated;
    const float sqrt_s1_new =
        std::sqrt(1.0F / static_cast<float>(input.ns_new - 1));
    const float value = interpolated * std::max(std::sqrt(s), sqrt_s1_new);
    return j_new == 0 ? 0.0F : value;
}

struct DispatchState {
    ProlongationCallback callback;
    wgpu::Buffer result_buffer;
    wgpu::Buffer readback_buffer;
    std::size_t total = 0;
    std::size_t result_bytes = 0;
};

wgpu::Buffer create_buffer(const wgpu::Device& device,
                           std::uint64_t size,
                           wgpu::BufferUsage usage,
                           const char* label) {
    return detail::cached_buffer(device, size, usage, label);
}

}  // namespace

ProlongationResult prolongation_reference(const ProlongationCase& input) {
    const std::string error = validate_case(input);
    if (!error.empty()) return {};

    ProlongationResult result;
    const std::size_t total = SPECTRAL_FAMILIES *
                              static_cast<std::size_t>(input.mnmax) *
                              static_cast<std::size_t>(input.ns_new);
    result.state.resize(total);
    result.velocity.assign(total, 0.0F);
    for (int profile = 0;
         profile < static_cast<int>(SPECTRAL_FAMILIES) * input.mnmax;
         ++profile) {
        for (int j = 0; j < input.ns_new; ++j) {
            result.state[static_cast<std::size_t>(profile) * input.ns_new + j] =
                reference_value(input, profile, j);
        }
    }
    return result;
}

void enqueue_prolongation(const wgpu::Device& device,
                          const ProlongationCase& input,
                          ProlongationCallback callback) {
    const std::string validation_error = validate_case(input);
    if (!validation_error.empty()) {
        callback(validation_error, {});
        return;
    }

    const std::string shader_text = load_shader();
    if (shader_text.empty()) {
        callback("cannot load embedded /shaders/prolongation.wgsl", {});
        return;
    }

    const std::size_t total = SPECTRAL_FAMILIES *
                              static_cast<std::size_t>(input.mnmax) *
                              static_cast<std::size_t>(input.ns_new);
    const std::size_t input_bytes = input.state.size() * sizeof(float);
    const std::size_t result_bytes = 2 * total * sizeof(float);

    const wgpu::Buffer input_buffer =
        create_buffer(device, input_bytes,
                      wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                      "cuMES prolongation input");
    const wgpu::Buffer result_buffer =
        create_buffer(device, result_bytes,
                      wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc,
                      "cuMES prolongation result");
    const wgpu::Buffer readback_buffer =
        create_buffer(device, result_bytes,
                      wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead,
                      "cuMES prolongation readback");
    const wgpu::Buffer params_buffer =
        create_buffer(device, sizeof(ShaderParams),
                      wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
                      "cuMES prolongation parameters");

    const auto& pipeline = detail::cached_compute_pipeline(
        device, "radial-prolongation", shader_text,
        "cuMES radial prolongation pipeline");

    const ShaderParams params{static_cast<std::uint32_t>(input.ns_old),
                              static_cast<std::uint32_t>(input.ns_new),
                              static_cast<std::uint32_t>(input.mnmax),
                              static_cast<std::uint32_t>(input.ntor + 1),
                              static_cast<std::uint32_t>(input.interpolation),
                              static_cast<std::uint32_t>(total),
                              {0, 0}};
    const wgpu::Queue queue = device.GetQueue();
    queue.WriteBuffer(input_buffer, 0, input.state.data(), input_bytes);
    queue.WriteBuffer(params_buffer, 0, &params, sizeof(params));

    const wgpu::BindGroupLayout layout = pipeline.GetBindGroupLayout(0);
    const wgpu::BindGroupEntry entries[] = {
        {nullptr, 0, input_buffer, 0, input_bytes, nullptr, nullptr},
        {nullptr, 1, result_buffer, 0, result_bytes, nullptr, nullptr},
        {nullptr, 2, params_buffer, 0, sizeof(params), nullptr, nullptr},
    };
    wgpu::BindGroupDescriptor bind_group_descriptor{};
    bind_group_descriptor.label = "cuMES radial prolongation bindings";
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
        (static_cast<std::uint32_t>(total) + WORKGROUP_SIZE - 1) /
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
    dispatch->total = total;
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
                dispatch->callback("WebGPU result mapping failed: " + detail,
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
            ProlongationResult result;
            result.state.assign(values, values + dispatch->total);
            result.velocity.assign(values + dispatch->total,
                                   values + 2 * dispatch->total);
            dispatch->readback_buffer.Unmap();
            dispatch->callback({}, std::move(result));
        });
}

}  // namespace cumes::webgpu
