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
    std::uint32_t ns, mode_count, ntor_plus_one, zero_m1_z;
    std::uint32_t padding[4];
};
static_assert(sizeof(Params) == 32);
std::string validate(const ResidualDecompositionCase& in) {
    if (in.ns < 2 || in.mpol < 2 || in.ntor < 0)
        return "residual decomposition requires ns,mpol>=2 and ntor>=0";
    const std::size_t mode_count =
        static_cast<std::size_t>(in.mpol) * (in.ntor + 1);
    const std::size_t n = static_cast<std::size_t>(in.ns) * mode_count;
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
                      int mode_count,
                      bool include_edge) {
    const std::size_t n = static_cast<std::size_t>(ns) * mode_count;
    for (int group = 0; group < 3; ++group) {
        double sum = 0.0;
        for (int mode = 0; mode < mode_count; ++mode) {
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

std::array<double, 3> accumulate_norms(const std::vector<float>& residual,
                                       int ns,
                                       int mode_count,
                                       bool include_edge) {
    ResidualDecompositionResult result;
    result.residual = residual;
    accumulate_norms(result, ns, mode_count, include_edge);
    return result.raw_norm;
}
struct Dispatch {
    ResidualDecompositionCallback callback;
    wgpu::Buffer output, readback;
    std::size_t count = 0, bytes = 0;
    int ns = 0, mode_count = 0;
    bool include_edge = false;
};
}  // namespace

ResidualDecompositionResult residual_decomposition_reference(
    const ResidualDecompositionCase& in) {
    if (!validate(in).empty()) return {};
    const int mode_count = in.mpol * (in.ntor + 1);
    const std::size_t n = static_cast<std::size_t>(in.ns) * mode_count;
    ResidualDecompositionResult out;
    out.residual = in.residual;
    for (int mode = 0; mode < mode_count; ++mode) {
        const int m = mode / (in.ntor + 1);
        for (int surface = 0; surface < in.ns; ++surface) {
            const float scale =
                m % 2 == 0
                    ? 1.0F
                    : 1.0F / std::max(in.sqrt_s_f[surface], in.sqrt_s_f[1]);
            const auto index = static_cast<std::size_t>(mode) * in.ns + surface;
            for (int component = 0; component < 6; ++component)
                out.residual[component * n + index] *= scale;
            if (m == 1) {
                const float old_r = out.residual[3 * n + index];
                const float old_z = out.residual[4 * n + index];
                constexpr float INV_SQRT_TWO = 0.7071067811865476F;
                out.residual[3 * n + index] = (old_r + old_z) * INV_SQRT_TWO;
                out.residual[4 * n + index] =
                    in.zero_m1_z ? 0.0F : (old_r - old_z) * INV_SQRT_TWO;
            }
        }
    }
    accumulate_norms(out, in.ns, mode_count, in.include_edge_rz);
    return out;
}

std::array<double, 3> residual_raw_norms(const std::vector<float>& residual,
                                         int ns,
                                         int mpol,
                                         int ntor,
                                         bool include_edge_rz) {
    const int mode_count = mpol * (ntor + 1);
    const std::size_t count = static_cast<std::size_t>(ns) * mode_count;
    if (ns < 2 || mpol < 1 || ntor < 0 || residual.size() != 6 * count)
        return {};
    return accumulate_norms(residual, ns, mode_count, include_edge_rz);
}

ForceNormalizationResult axisymmetric_force_normalization(
    const AxisymmetricForceNormalizationCase& in) {
    const std::size_t angular = static_cast<std::size_t>(in.ntheta) * in.nzeta;
    const std::size_t half = static_cast<std::size_t>(in.ns - 1) * angular;
    const int mode_count = in.mpol * (in.ntor + 1);
    const std::size_t spectral = static_cast<std::size_t>(in.ns) * mode_count;
    if (in.ns < 2 || in.mpol < 1 || in.ntor < 0 || in.ntheta < 2 ||
        in.ntheta % 2 != 0 || in.nzeta < 1 || !(in.delta_s > 0.0F) ||
        in.state.size() != 6 * spectral ||
        in.base_geometry.size() != 10 * half ||
        in.magnetic_field.size() != 5 * half ||
        in.pres_h.size() != static_cast<std::size_t>(in.ns - 1)) {
        return {};
    }
    const int ntheta_red = in.ntheta / 2 + 1;
    const float norm = 1.0F / static_cast<float>(in.nzeta * (ntheta_red - 1));
    std::vector<float> partials(4 * static_cast<std::size_t>(in.ns - 1));
    std::vector<float> dvds(in.ns - 1);
    for (int surface = 0; surface < in.ns - 1; ++surface) {
        float s_rz = 0.0F;
        float s_l = 0.0F;
        float s_mag = 0.0F;
        float s_g = 0.0F;
        for (int zeta = 0; zeta < in.nzeta; ++zeta) {
            for (int theta = 0; theta < ntheta_red; ++theta) {
                float weight = norm;
                if (theta == 0 || theta == ntheta_red - 1) weight *= 0.5F;
                const std::size_t point =
                    static_cast<std::size_t>(surface) * angular +
                    static_cast<std::size_t>(zeta) * in.ntheta + theta;
                const float gsqrt = in.base_geometry[6 * half + point];
                const float guu = in.base_geometry[7 * half + point];
                const float r12 = in.base_geometry[point];
                const float bsupu = in.magnetic_field[point];
                const float bsupv = in.magnetic_field[half + point];
                const float bsubu = in.magnetic_field[2 * half + point];
                const float bsubv = in.magnetic_field[3 * half + point];
                const float bmag2 = 0.5F * (bsupu * bsubu + bsupv * bsubv);
                s_rz += guu * r12 * r12 * weight;
                s_l += (bsubu * bsubu + bsubv * bsubv) * weight;
                s_mag += gsqrt * bmag2 * weight;
                s_g += gsqrt * weight;
            }
        }
        partials[4 * surface] = s_rz;
        partials[4 * surface + 1] = s_l;
        partials[4 * surface + 2] = s_mag;
        partials[4 * surface + 3] = s_g;
        dvds[surface] = -s_g;
    }
    ForceNormalizationResult out;
    for (int surface = 0; surface < in.ns - 1; ++surface) {
        out.raw[0] += partials[4 * surface];
        out.raw[1] += partials[4 * surface + 1];
        out.raw[2] += partials[4 * surface + 2];
        out.raw[3] += static_cast<double>(in.pres_h[surface] * dvds[surface]);
        out.raw[4] += dvds[surface];
    }
    for (int mode = 0; mode < mode_count; ++mode) {
        const int m = mode / (in.ntor + 1);
        const int n = mode % (in.ntor + 1);
        const float m_factor = m == 0 ? 1.0F : std::sqrt(2.0F);
        const float n_factor = n == 0 ? 1.0F : std::sqrt(2.0F);
        const float inverse_square =
            1.0F / (m_factor * n_factor * m_factor * n_factor);
        for (int surface = 0; surface < in.ns; ++surface) {
            if (surface == 0 && m > 0) continue;
            const std::size_t point =
                static_cast<std::size_t>(mode) * in.ns + surface;
            const float rcc = in.state[point];
            const float zsc = in.state[spectral + point];
            const float rss = in.state[3 * spectral + point];
            const float zcs = in.state[4 * spectral + point];
            if (m > 0 || n > 0) {
                out.raw[5] += static_cast<double>(rcc * rcc * inverse_square);
            }
            out.raw[5] += static_cast<double>(zsc * zsc * inverse_square);
            const float odd_pair = rss * rss + zcs * zcs;
            out.raw[5] += static_cast<double>((m == 1 ? 0.5F : 1.0F) *
                                              odd_pair * inverse_square);
        }
    }
    const double e_mag = std::abs(out.raw[2]) * in.delta_s;
    const double e_therm = out.raw[3] * in.delta_s;
    const double volume = out.raw[4] * in.delta_s;
    const double energy_density = std::max(e_mag, e_therm) / volume;
    const double denom_rz = out.raw[0] * energy_density * energy_density;
    const double denom_l =
        out.raw[1] * static_cast<double>(in.lamscale) * in.lamscale;
    out.f_norm_rz = denom_rz > 0.0 ? 1.0 / denom_rz : 1.0;
    out.f_norm_l = denom_l > 0.0 ? 1.0 / denom_l : 1.0;
    out.f_norm_1 = out.raw[5] > 0.0 ? 1.0 / out.raw[5] : 1.0;
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
    const int mode_count = in.mpol * (in.ntor + 1);
    const std::size_t n = static_cast<std::size_t>(in.ns) * mode_count;
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
                  static_cast<std::uint32_t>(mode_count),
                  static_cast<std::uint32_t>(in.ntor + 1),
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
    d->mode_count = mode_count;
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
            accumulate_norms(out, d->ns, d->mode_count, d->include_edge);
            d->callback({}, std::move(out));
        });
}
}  // namespace cumes::webgpu
