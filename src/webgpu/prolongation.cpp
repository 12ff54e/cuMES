#include "cumes/webgpu/prolongation.hpp"

#include "cumes/webgpu/readback_batch.hpp"
#include "pipeline_cache.hpp"
#include "shader_source.hpp"

#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <span>
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

const std::string& load_shader() {
    return detail::cached_shader_source("/shaders/prolongation.wgsl");
}

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
        const int mode = profile % input.mnmax;
        const bool odd = ((mode / (input.ntor + 1)) % 2) == 1;
        const auto values =
            std::span(input.state)
                .subspan(static_cast<std::size_t>(profile) * input.ns_old,
                         input.ns_old);
        for (int j = 0; j < input.ns_new; ++j) {
            result.state[static_cast<std::size_t>(profile) * input.ns_new + j] =
                interpolate_radial_value(values, input.ns_new, j, odd,
                                         input.interpolation);
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

    const auto& shader_text = load_shader();
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
    const auto host = std::make_shared<ProlongationResult>();
    const auto batch =
        std::make_shared<ReadbackBatch>(readback_buffer, result_bytes);
    batch->append(
        encoder, result_buffer, 0, result_bytes,
        [host, total](std::span<const float> values) {
            host->state.assign(values.begin(), values.begin() + total);
            host->velocity.assign(values.begin() + total, values.end());
        });
    const auto commands = encoder.Finish();
    queue.Submit(1, &commands);
    batch->map([callback = std::move(callback), host](std::string error) {
        callback(std::move(error), std::move(*host));
    });
}

}  // namespace cumes::webgpu
