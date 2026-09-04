#include "cumes/webgpu/numerics.hpp"

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
constexpr std::uint32_t WORKGROUP_SIZE = 256;
struct Params {
    std::uint32_t ns, mpol, values_per_component, zero_m1_z;
    std::uint32_t padding[4];
};
static_assert(sizeof(Params) == 32);
std::string validate(const ResidualDecompositionCase& in) {
    if (in.ns < 2 || in.mpol < 2)
        return "residual decomposition requires ns,mpol>=2";
    const std::size_t n = static_cast<std::size_t>(in.ns) * in.mpol;
    if (n > std::numeric_limits<std::uint32_t>::max() ||
        in.residual.size() != 6 * n ||
        in.sqrt_s_f.size() != static_cast<std::size_t>(in.ns))
        return "residual decomposition input shape mismatch";
    return {};
}
std::string shader_source() {
    std::ifstream f("/shaders/residual_decompose.wgsl", std::ios::binary);
    if (!f) return {};
    std::ostringstream s;
    s << f.rdbuf();
    return s.str();
}
wgpu::Buffer make_buffer(const wgpu::Device& d,
                         std::uint64_t size,
                         wgpu::BufferUsage usage,
                         const char* label) {
    wgpu::BufferDescriptor x{};
    x.label = label;
    x.size = size;
    x.usage = usage;
    return d.CreateBuffer(&x);
}
void accumulate_norms(ResidualDecompositionResult& out,
                      int ns,
                      int mpol,
                      bool include_edge) {
    const std::size_t n = static_cast<std::size_t>(ns) * mpol;
    for (int group = 0; group < 3; ++group) {
        double sum = 0.0;
        for (int mode = 0; mode < mpol; ++mode) {
            for (int surface = 0; surface < ns; ++surface) {
                if (group < 2 && !include_edge && surface == ns - 1) continue;
                const auto index =
                    static_cast<std::size_t>(mode) * ns + surface;
                const float a = out.residual[group * n + index];
                const float b = out.residual[(group + 3) * n + index];
                sum += static_cast<double>(a * a + b * b);
            }
        }
        out.raw_norm[group] = sum / static_cast<double>(n);
    }
}
struct Dispatch {
    ResidualDecompositionCallback callback;
    wgpu::Buffer output, readback;
    std::size_t count = 0, bytes = 0;
    int ns = 0, mpol = 0;
    bool include_edge = false;
};
}  // namespace

ResidualDecompositionResult residual_decomposition_reference(
    const ResidualDecompositionCase& in) {
    if (!validate(in).empty()) return {};
    const std::size_t n = static_cast<std::size_t>(in.ns) * in.mpol;
    ResidualDecompositionResult out;
    out.residual = in.residual;
    for (int mode = 0; mode < in.mpol; ++mode) {
        for (int surface = 0; surface < in.ns; ++surface) {
            const float scale =
                mode % 2 == 0
                    ? 1.0F
                    : 1.0F / std::max(in.sqrt_s_f[surface], in.sqrt_s_f[1]);
            const auto index = static_cast<std::size_t>(mode) * in.ns + surface;
            for (int component = 0; component < 6; ++component)
                out.residual[component * n + index] *= scale;
            if (mode == 1) {
                const float old_r = out.residual[3 * n + index];
                const float old_z = out.residual[4 * n + index];
                constexpr float INV_SQRT_TWO = 0.7071067811865476F;
                out.residual[3 * n + index] = (old_r + old_z) * INV_SQRT_TWO;
                out.residual[4 * n + index] =
                    in.zero_m1_z ? 0.0F : (old_r - old_z) * INV_SQRT_TWO;
            }
        }
    }
    accumulate_norms(out, in.ns, in.mpol, in.include_edge_rz);
    return out;
}

void enqueue_residual_decomposition(const wgpu::Device& device,
                                    const ResidualDecompositionCase& in,
                                    ResidualDecompositionCallback callback) {
    const auto error = validate(in);
    if (!error.empty()) {
        callback(error, {});
        return;
    }
    const auto source = shader_source();
    if (source.empty()) {
        callback("cannot load embedded /shaders/residual_decompose.wgsl", {});
        return;
    }
    const std::size_t n = static_cast<std::size_t>(in.ns) * in.mpol;
    const auto input_bytes = in.residual.size() * sizeof(float);
    const auto radial_bytes = in.sqrt_s_f.size() * sizeof(float);
    auto input =
        make_buffer(device, input_bytes,
                    wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                    "raw residual");
    auto radial =
        make_buffer(device, radial_bytes,
                    wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                    "residual radial");
    auto output =
        make_buffer(device, input_bytes,
                    wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc,
                    "decomposed residual");
    auto readback =
        make_buffer(device, input_bytes,
                    wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead,
                    "decomposed readback");
    auto uniform =
        make_buffer(device, sizeof(Params),
                    wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
                    "decomposition params");
    wgpu::ShaderSourceWGSL wgsl{};
    wgsl.code = source.c_str();
    wgpu::ShaderModuleDescriptor sd{};
    sd.nextInChain = &wgsl;
    auto shader = device.CreateShaderModule(&sd);
    wgpu::ComputePipelineDescriptor pd{};
    pd.compute.module = shader;
    pd.compute.entryPoint = "main";
    auto pipeline = device.CreateComputePipeline(&pd);
    Params params{static_cast<std::uint32_t>(in.ns),
                  static_cast<std::uint32_t>(in.mpol),
                  static_cast<std::uint32_t>(n),
                  in.zero_m1_z ? 1U : 0U,
                  {0, 0, 0, 0}};
    auto queue = device.GetQueue();
    queue.WriteBuffer(input, 0, in.residual.data(), input_bytes);
    queue.WriteBuffer(radial, 0, in.sqrt_s_f.data(), radial_bytes);
    queue.WriteBuffer(uniform, 0, &params, sizeof(params));
    auto layout = pipeline.GetBindGroupLayout(0);
    const wgpu::BindGroupEntry entries[] = {
        {nullptr, 0, input, 0, input_bytes, nullptr, nullptr},
        {nullptr, 1, radial, 0, radial_bytes, nullptr, nullptr},
        {nullptr, 2, output, 0, input_bytes, nullptr, nullptr},
        {nullptr, 3, uniform, 0, sizeof(params), nullptr, nullptr}};
    wgpu::BindGroupDescriptor bd{};
    bd.layout = layout;
    bd.entryCount = std::size(entries);
    bd.entries = entries;
    auto group = device.CreateBindGroup(&bd);
    auto encoder = device.CreateCommandEncoder();
    wgpu::ComputePassDescriptor pdesc{};
    auto pass = encoder.BeginComputePass(&pdesc);
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, group);
    pass.DispatchWorkgroups(
        (static_cast<std::uint32_t>(n) + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE);
    pass.End();
    encoder.CopyBufferToBuffer(output, 0, readback, 0, input_bytes);
    auto commands = encoder.Finish();
    queue.Submit(1, &commands);
    auto d = std::make_shared<Dispatch>();
    d->callback = std::move(callback);
    d->output = output;
    d->readback = readback;
    d->count = in.residual.size();
    d->bytes = input_bytes;
    d->ns = in.ns;
    d->mpol = in.mpol;
    d->include_edge = in.include_edge_rz;
    readback.MapAsync(
        wgpu::MapMode::Read, 0, input_bytes,
        wgpu::CallbackMode::AllowSpontaneous,
        [d](wgpu::MapAsyncStatus status, wgpu::StringView message) {
            if (status != wgpu::MapAsyncStatus::Success) {
                d->callback("WebGPU residual mapping failed: " +
                                std::string(message.data, message.length),
                            {});
                return;
            }
            const auto* values = static_cast<const float*>(
                d->readback.GetConstMappedRange(0, d->bytes));
            if (values == nullptr) {
                d->callback("WebGPU returned a null mapped range", {});
                return;
            }
            ResidualDecompositionResult out;
            out.residual.assign(values, values + d->count);
            d->readback.Unmap();
            accumulate_norms(out, d->ns, d->mpol, d->include_edge);
            d->callback({}, std::move(out));
        });
}
}  // namespace cumes::webgpu
