#include "cumes/webgpu/reduction.hpp"

#include "pipeline_cache.hpp"
#include "shader_source.hpp"

namespace cumes::webgpu {

void enqueue_residual_norm(
    const wgpu::Device& device,
    const ResidualNormCase& input,
    std::function<void(std::string, ResidualNormResult)> callback) {
    const auto count = input.residual.values;
    if (!input.residual || !input.readback.batch || input.ns < 2 ||
        count == 0 || count % (6 * std::size_t(input.ns)) != 0 ||
        count / 6 > (1U << 24)) {
        callback("invalid device residual norm shape or readback", {});
        return;
    }
    struct Params {
        std::uint32_t points, ns, paired, edge;
    };
    const Params params{static_cast<std::uint32_t>(count / 6),
                        static_cast<std::uint32_t>(input.ns),
                        input.paired ? 1U : 0U,
                        input.include_edge_rz ? 1U : 0U};
    const auto bytes = count * sizeof(float);
    const auto fits = [&](std::uint64_t offset) {
        return offset % sizeof(float) == 0 &&
               offset <= input.residual.buffer.GetSize() &&
               bytes <= input.residual.buffer.GetSize() - offset;
    };
    if (!fits(input.residual.high_offset) ||
        (input.paired && !fits(input.residual.low_offset))) {
        callback("device residual norm plane exceeds its buffer", {});
        return;
    }
    const auto blocks = (params.points + 255) / 256;
    const auto storage =
        wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    auto hi =
        detail::cached_buffer(device, bytes, storage, "norm residual high");
    auto lo =
        detail::cached_buffer(device, bytes, storage, "norm residual low");
    auto partials = detail::cached_buffer(device, 3 * blocks * 16, storage,
                                          "norm partials");
    auto output = detail::cached_buffer(device, 9 * sizeof(float),
                                        storage | wgpu::BufferUsage::CopySrc,
                                        "norm result");
    auto uniform = detail::cached_buffer(
        device, sizeof(params),
        wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst, "norm params");
    const auto encoder = device.CreateCommandEncoder();
    transfer_fields(device, encoder, hi, {}, input.residual);
    if (input.paired)
        transfer_fields(device, encoder, lo, {}, input.residual, true);
    else
        encoder.ClearBuffer(lo);
    device.GetQueue().WriteBuffer(uniform, 0, &params, sizeof(params));
    const auto& source =
        detail::cached_shader_source("/shaders/residual_norm.wgsl");
    for (const bool final : {false, true}) {
        const auto& pipeline = detail::cached_compute_pipeline(
            device, final ? "residual-norm-final" : "residual-norm-partial",
            source,
            final ? "cuMES paired norm final" : "cuMES paired norm partial",
            final ? "finalize" : "partial");
        // Automatic layouts omit bindings not accessed by an entry point.
        std::vector<wgpu::BindGroupEntry> entries;
        if (!final) {
            entries.push_back({nullptr, 0, hi, 0, bytes, nullptr, nullptr});
            entries.push_back({nullptr, 1, lo, 0, bytes, nullptr, nullptr});
        }
        entries.push_back(
            {nullptr, 2, partials, 0, 3 * blocks * 16, nullptr, nullptr});
        if (final)
            entries.push_back(
                {nullptr, 3, output, 0, 9 * sizeof(float), nullptr, nullptr});
        entries.push_back(
            {nullptr, 4, uniform, 0, sizeof(params), nullptr, nullptr});
        wgpu::BindGroupDescriptor descriptor{};
        descriptor.layout = pipeline.GetBindGroupLayout(0);
        descriptor.entries = entries.data();
        descriptor.entryCount = entries.size();
        const auto group = device.CreateBindGroup(&descriptor);
        auto pass = encoder.BeginComputePass();
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, group);
        pass.DispatchWorkgroups(final ? 3 : blocks, final ? 1 : 3);
        pass.End();
    }
    ResidualNormResult resident;
    resident.device_norm = {output, 3, 0, 3 * sizeof(float)};
    input.readback.batch->append(
        encoder, output, 0, 9 * sizeof(float),
        [callback = std::move(callback),
         resident](std::span<const float> values) mutable {
            for (int i = 0; i < 3; ++i) {
                resident.raw[i] = double(values[i]) + values[i + 3];
                resident.finite &= values[i + 6] == 0.0F;
            }
            callback({}, std::move(resident));
        });
    const auto commands = encoder.Finish();
    device.GetQueue().Submit(1, &commands);
    input.readback.publish_device(std::move(resident));
}

}  // namespace cumes::webgpu
