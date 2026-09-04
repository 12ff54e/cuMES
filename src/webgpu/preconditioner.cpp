#include "cumes/webgpu/preconditioner.hpp"

#include "cumes/webgpu/axisymmetric.hpp"
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

constexpr std::uint32_t WORKGROUP_SIZE = 64;

struct Params {
    std::uint32_t ns, n_z_n_t, full_points, half_points;
    float delta_s;
    std::uint32_t free_boundary, padding0, padding1;
};
static_assert(sizeof(Params) == 32);

struct HalfTerms {
    float ar[4]{};
    float az[4]{};
    float br[3]{};
    float bz[3]{};
    float cx = 0.0F;
};

struct Diagonal {
    float ard = 0.0F;
    float brd = 0.0F;
    float azd = 0.0F;
    float bzd = 0.0F;
    float cxd = 0.0F;
};

std::string validate_case(const AxisymmetricPreconditionerElementCase& in) {
    if (in.ns < 2 || in.ntheta < 2 || in.ntheta % 2 != 0 || in.nzeta < 1 ||
        !(in.delta_s > 0.0F) || !std::isfinite(in.delta_s)) {
        return "axisymmetric preconditioner elements have invalid scalars";
    }
    const std::size_t n_z_n_t = static_cast<std::size_t>(in.ntheta) * in.nzeta;
    const std::size_t full = static_cast<std::size_t>(in.ns) * n_z_n_t;
    const std::size_t half = static_cast<std::size_t>(in.ns - 1) * n_z_n_t;
    if (full > std::numeric_limits<std::uint32_t>::max() ||
        in.geometry.size() != GEOMETRY_PARITY_FIELD_COUNT * full ||
        in.base_geometry.size() != BASE_GEOMETRY_FIELD_COUNT * half ||
        in.magnetic_field.size() != 5 * half ||
        in.sqrt_s_f.size() != static_cast<std::size_t>(in.ns) ||
        in.sqrt_s_h.size() != static_cast<std::size_t>(in.ns - 1)) {
        return "axisymmetric preconditioner element input shape mismatch";
    }
    return {};
}

HalfTerms half_terms(const AxisymmetricPreconditionerElementCase& in,
                     int surface) {
    const std::size_t n_z_n_t = static_cast<std::size_t>(in.ntheta) * in.nzeta;
    const std::size_t full = static_cast<std::size_t>(in.ns) * n_z_n_t;
    const std::size_t half = static_cast<std::size_t>(in.ns - 1) * n_z_n_t;
    const auto fg = [&](int field, std::size_t point) {
        return in.geometry[static_cast<std::size_t>(field) * full + point];
    };
    const auto hg = [&](int field, std::size_t point) {
        return in.base_geometry[static_cast<std::size_t>(field) * half + point];
    };
    const auto bf = [&](int field, std::size_t point) {
        return in
            .magnetic_field[static_cast<std::size_t>(field) * half + point];
    };
    HalfTerms out;
    const float sqrt_h = in.sqrt_s_h[surface];
    const float inv_sqrt_h = 1.0F / sqrt_h;
    const float weight = 1.0F / static_cast<float>(n_z_n_t);
    for (std::size_t angular = 0; angular < n_z_n_t; ++angular) {
        const std::size_t half_point =
            static_cast<std::size_t>(surface) * n_z_n_t + angular;
        const std::size_t inner = half_point;
        const std::size_t outer = half_point + n_z_n_t;
        const float p_tau = -4.0F * hg(0, half_point) * bf(4, half_point) /
                            hg(5, half_point) * weight;
        const float r_t1a = hg(2, half_point) / in.delta_s;
        const float r_t2a =
            0.25F * (fg(4, outer) / sqrt_h + fg(10, outer)) / sqrt_h;
        const float r_t3a =
            0.25F * (fg(4, inner) / sqrt_h + fg(10, inner)) / sqrt_h;
        out.ar[0] += p_tau * r_t1a * r_t1a;
        out.ar[1] += p_tau * (r_t1a + r_t2a) * (-r_t1a + r_t3a);
        out.ar[2] += p_tau * (r_t1a + r_t2a) * (r_t1a + r_t2a);
        out.ar[3] += p_tau * (-r_t1a + r_t3a) * (-r_t1a + r_t3a);
        const float r_t1b =
            0.5F * (hg(4, half_point) + 0.5F * inv_sqrt_h * fg(7, outer));
        const float r_t2b =
            0.5F * (hg(4, half_point) + 0.5F * inv_sqrt_h * fg(7, inner));
        out.br[0] += p_tau * r_t1b * r_t2b;
        out.br[1] += p_tau * r_t1b * r_t1b;
        out.br[2] += p_tau * r_t2b * r_t2b;

        const float z_t1a = hg(1, half_point) / in.delta_s;
        const float z_t2a =
            0.25F * (fg(3, outer) / sqrt_h + fg(9, outer)) / sqrt_h;
        const float z_t3a =
            0.25F * (fg(3, inner) / sqrt_h + fg(9, inner)) / sqrt_h;
        out.az[0] += p_tau * z_t1a * z_t1a;
        out.az[1] += p_tau * (z_t1a + z_t2a) * (-z_t1a + z_t3a);
        out.az[2] += p_tau * (z_t1a + z_t2a) * (z_t1a + z_t2a);
        out.az[3] += p_tau * (-z_t1a + z_t3a) * (-z_t1a + z_t3a);
        const float z_t1b =
            0.5F * (hg(3, half_point) + 0.5F * inv_sqrt_h * fg(6, outer));
        const float z_t2b =
            0.5F * (hg(3, half_point) + 0.5F * inv_sqrt_h * fg(6, inner));
        out.bz[0] += p_tau * z_t1b * z_t2b;
        out.bz[1] += p_tau * z_t1b * z_t1b;
        out.bz[2] += p_tau * z_t2b * z_t2b;
        const float bsupv = bf(1, half_point);
        out.cx += -bsupv * bsupv * hg(6, half_point) * weight;
    }
    return out;
}

float sm(const AxisymmetricPreconditionerElementCase& in, int surface) {
    return in.sqrt_s_h[surface] / in.sqrt_s_f[surface + 1];
}

float sp(const AxisymmetricPreconditionerElementCase& in, int surface) {
    return surface == 0 ? sm(in, surface)
                        : in.sqrt_s_h[surface] / in.sqrt_s_f[surface];
}

Diagonal diagonal(const AxisymmetricPreconditionerElementCase& in,
                  int surface,
                  int parity) {
    Diagonal out;
    const bool has_inner = surface > 0;
    const bool has_outer = surface < in.ns - 1;
    if (has_inner && has_outer) {
        const auto inner = half_terms(in, surface - 1);
        const auto outer = half_terms(in, surface);
        if (parity == 0) {
            out.ard = inner.ar[0] + outer.ar[0];
            out.brd = inner.br[1] + outer.br[2];
            out.azd = inner.az[0] + outer.az[0];
            out.bzd = inner.bz[1] + outer.bz[2];
        } else {
            const float sm2 = sm(in, surface - 1) * sm(in, surface - 1);
            const float sp2 = sp(in, surface) * sp(in, surface);
            out.ard = inner.ar[2] * sm2 + outer.ar[3] * sp2;
            out.brd = inner.br[1] * sm2 + outer.br[2] * sp2;
            out.azd = inner.az[2] * sm2 + outer.az[3] * sp2;
            out.bzd = inner.bz[1] * sm2 + outer.bz[2] * sp2;
        }
        out.cxd = inner.cx + outer.cx;
    } else if (has_outer) {
        const auto outer = half_terms(in, surface);
        if (parity == 0) {
            out.ard = outer.ar[0];
            out.brd = outer.br[2];
            out.azd = outer.az[0];
            out.bzd = outer.bz[2];
        } else {
            const float sp2 = sp(in, surface) * sp(in, surface);
            out.ard = outer.ar[3] * sp2;
            out.brd = outer.br[2] * sp2;
            out.azd = outer.az[3] * sp2;
            out.bzd = outer.bz[2] * sp2;
        }
        out.cxd = outer.cx;
    } else {
        const auto inner = half_terms(in, surface - 1);
        if (parity == 0) {
            out.ard = inner.ar[0];
            out.brd = inner.br[1];
            out.azd = inner.az[0];
            out.bzd = inner.bz[1];
        } else {
            const float sm2 = sm(in, surface - 1) * sm(in, surface - 1);
            out.ard = inner.ar[2] * sm2;
            out.brd = inner.br[1] * sm2;
            out.azd = inner.az[2] * sm2;
            out.bzd = inner.bz[1] * sm2;
        }
        out.cxd = inner.cx;
    }
    if (surface == in.ns - 1 && !in.free_boundary) {
        out.ard *= 1.05F;
        out.brd *= 1.05F;
        out.azd *= 1.05F;
        out.bzd *= 1.05F;
        out.cxd *= 1.05F;
    }
    return out;
}

std::string load_shader() {
    std::ifstream stream("/shaders/axisymmetric_preconditioner_elements.wgsl",
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
    AxisymmetricPreconditionerElementCallback callback;
    wgpu::Buffer output, readback;
    std::size_t bytes = 0;
    int ns = 0;
};

}  // namespace

AxisymmetricPreconditionerElements
axisymmetric_preconditioner_element_reference(
    const AxisymmetricPreconditionerElementCase& input) {
    if (!validate_case(input).empty()) return {};
    AxisymmetricPreconditionerElements out;
    out.ard.resize(2 * input.ns);
    out.brd.resize(2 * input.ns);
    out.azd.resize(2 * input.ns);
    out.bzd.resize(2 * input.ns);
    out.cxd.resize(input.ns);
    out.arm.resize(2 * (input.ns - 1));
    out.brm.resize(2 * (input.ns - 1));
    out.azm.resize(2 * (input.ns - 1));
    out.bzm.resize(2 * (input.ns - 1));
    for (int surface = 0; surface < input.ns; ++surface) {
        const auto even = diagonal(input, surface, 0);
        const auto odd = diagonal(input, surface, 1);
        out.ard[2 * surface] = even.ard;
        out.ard[2 * surface + 1] = odd.ard;
        out.brd[2 * surface] = even.brd;
        out.brd[2 * surface + 1] = odd.brd;
        out.azd[2 * surface] = even.azd;
        out.azd[2 * surface + 1] = odd.azd;
        out.bzd[2 * surface] = even.bzd;
        out.bzd[2 * surface + 1] = odd.bzd;
        out.cxd[surface] = even.cxd;
        if (surface < input.ns - 1) {
            const auto half = half_terms(input, surface);
            const float smsp = sm(input, surface) * sp(input, surface);
            out.arm[2 * surface] = -half.ar[0];
            out.arm[2 * surface + 1] = half.ar[1] * smsp;
            out.brm[2 * surface] = half.br[0];
            out.brm[2 * surface + 1] = half.br[0] * smsp;
            out.azm[2 * surface] = -half.az[0];
            out.azm[2 * surface + 1] = half.az[1] * smsp;
            out.bzm[2 * surface] = half.bz[0];
            out.bzm[2 * surface + 1] = half.bz[0] * smsp;
        }
    }
    return out;
}

void enqueue_axisymmetric_preconditioner_elements(
    const wgpu::Device& device,
    const AxisymmetricPreconditionerElementCase& input,
    AxisymmetricPreconditionerElementCallback callback) {
    const auto error = validate_case(input);
    if (!error.empty()) {
        callback(error, {});
        return;
    }
    const auto shader_text = load_shader();
    if (shader_text.empty()) {
        callback("cannot load axisymmetric preconditioner-element shader", {});
        return;
    }
    const std::size_t n_z_n_t =
        static_cast<std::size_t>(input.ntheta) * input.nzeta;
    const std::size_t full = static_cast<std::size_t>(input.ns) * n_z_n_t;
    const std::size_t half = static_cast<std::size_t>(input.ns - 1) * n_z_n_t;
    std::vector<float> radial = input.sqrt_s_f;
    radial.insert(radial.end(), input.sqrt_s_h.begin(), input.sqrt_s_h.end());
    const auto geometry_bytes = input.geometry.size() * sizeof(float);
    const auto base_bytes = input.base_geometry.size() * sizeof(float);
    const auto magnetic_bytes = input.magnetic_field.size() * sizeof(float);
    const auto radial_bytes = radial.size() * sizeof(float);
    const auto output_values = 9 * static_cast<std::size_t>(input.ns) +
                               8 * static_cast<std::size_t>(input.ns - 1);
    const auto output_bytes = output_values * sizeof(float);
    auto geometry_buffer =
        make_buffer(device, geometry_bytes,
                    wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                    "preconditioner geometry");
    auto base_buffer =
        make_buffer(device, base_bytes,
                    wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                    "preconditioner base geometry");
    auto magnetic_buffer =
        make_buffer(device, magnetic_bytes,
                    wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                    "preconditioner magnetic field");
    auto radial_buffer =
        make_buffer(device, radial_bytes,
                    wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                    "preconditioner radial profiles");
    auto output_buffer =
        make_buffer(device, output_bytes,
                    wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc,
                    "preconditioner elements");
    auto readback =
        make_buffer(device, output_bytes,
                    wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead,
                    "preconditioner element readback");
    auto params_buffer =
        make_buffer(device, sizeof(Params),
                    wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
                    "preconditioner element params");
    wgpu::ShaderSourceWGSL wgsl{};
    wgsl.code = shader_text.c_str();
    wgpu::ShaderModuleDescriptor shader_descriptor{};
    shader_descriptor.nextInChain = &wgsl;
    auto shader = device.CreateShaderModule(&shader_descriptor);
    wgpu::ComputePipelineDescriptor pipeline_descriptor{};
    pipeline_descriptor.compute.module = shader;
    pipeline_descriptor.compute.entryPoint = "main";
    auto pipeline = device.CreateComputePipeline(&pipeline_descriptor);
    const Params params{static_cast<std::uint32_t>(input.ns),
                        static_cast<std::uint32_t>(n_z_n_t),
                        static_cast<std::uint32_t>(full),
                        static_cast<std::uint32_t>(half),
                        input.delta_s,
                        input.free_boundary ? 1U : 0U,
                        0,
                        0};
    auto queue = device.GetQueue();
    queue.WriteBuffer(geometry_buffer, 0, input.geometry.data(),
                      geometry_bytes);
    queue.WriteBuffer(base_buffer, 0, input.base_geometry.data(), base_bytes);
    queue.WriteBuffer(magnetic_buffer, 0, input.magnetic_field.data(),
                      magnetic_bytes);
    queue.WriteBuffer(radial_buffer, 0, radial.data(), radial_bytes);
    queue.WriteBuffer(params_buffer, 0, &params, sizeof(params));
    auto layout = pipeline.GetBindGroupLayout(0);
    const wgpu::BindGroupEntry entries[] = {
        {nullptr, 0, geometry_buffer, 0, geometry_bytes, nullptr, nullptr},
        {nullptr, 1, base_buffer, 0, base_bytes, nullptr, nullptr},
        {nullptr, 2, magnetic_buffer, 0, magnetic_bytes, nullptr, nullptr},
        {nullptr, 3, radial_buffer, 0, radial_bytes, nullptr, nullptr},
        {nullptr, 4, output_buffer, 0, output_bytes, nullptr, nullptr},
        {nullptr, 5, params_buffer, 0, sizeof(params), nullptr, nullptr}};
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
        (static_cast<std::uint32_t>(input.ns) + WORKGROUP_SIZE - 1) /
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
    dispatch->ns = input.ns;
    readback.MapAsync(
        wgpu::MapMode::Read, 0, output_bytes,
        wgpu::CallbackMode::AllowSpontaneous,
        [dispatch](wgpu::MapAsyncStatus status, wgpu::StringView message) {
            if (status != wgpu::MapAsyncStatus::Success) {
                const std::string detail =
                    message.length == 0
                        ? std::string{}
                        : std::string(message.data, message.length);
                dispatch->callback("preconditioner mapping failed: " + detail,
                                   {});
                return;
            }
            const auto* values = static_cast<const float*>(
                dispatch->readback.GetConstMappedRange(0, dispatch->bytes));
            if (values == nullptr) {
                dispatch->callback("preconditioner mapped range is null", {});
                return;
            }
            const std::size_t pair_count = 2 * dispatch->ns;
            AxisymmetricPreconditionerElements out;
            out.ard.assign(values, values + pair_count);
            out.brd.assign(values + pair_count, values + 2 * pair_count);
            out.azd.assign(values + 2 * pair_count, values + 3 * pair_count);
            out.bzd.assign(values + 3 * pair_count, values + 4 * pair_count);
            out.cxd.assign(values + 4 * pair_count,
                           values + 4 * pair_count + dispatch->ns);
            const std::size_t half_count = 2 * (dispatch->ns - 1);
            const auto* half = values + 4 * pair_count + dispatch->ns;
            out.arm.assign(half, half + half_count);
            out.brm.assign(half + half_count, half + 2 * half_count);
            out.azm.assign(half + 2 * half_count, half + 3 * half_count);
            out.bzm.assign(half + 3 * half_count, half + 4 * half_count);
            dispatch->readback.Unmap();
            dispatch->callback({}, std::move(out));
        });
}

}  // namespace cumes::webgpu
