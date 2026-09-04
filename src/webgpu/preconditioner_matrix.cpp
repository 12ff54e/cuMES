#include "cumes/webgpu/geometry.hpp"
#include "cumes/webgpu/preconditioner.hpp"
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

constexpr std::uint32_t WORKGROUP_SIZE = 64;

struct Params {
    std::uint32_t ns, mpol, ntor, ntheta;
    std::uint32_t nzeta, n_z_n_t, points, half_points;
    std::uint32_t nfp, free_boundary;
    float delta_s;
    std::uint32_t padding;
};
static_assert(sizeof(Params) == 48);

struct MatrixValues {
    float upper_r = 0.0F;
    float diagonal_r = 0.0F;
    float lower_r = 0.0F;
    float upper_z = 0.0F;
    float diagonal_z = 0.0F;
    float lower_z = 0.0F;
};

bool has_element_shapes(const AxisymmetricPreconditionerMatrixCase& in) {
    const std::size_t pairs = 2 * static_cast<std::size_t>(in.ns);
    const std::size_t half_pairs = 2 * static_cast<std::size_t>(in.ns - 1);
    return in.elements.ard.size() == pairs && in.elements.brd.size() == pairs &&
           in.elements.azd.size() == pairs && in.elements.bzd.size() == pairs &&
           in.elements.cxd.size() == static_cast<std::size_t>(in.ns) &&
           in.elements.arm.size() == half_pairs &&
           in.elements.brm.size() == half_pairs &&
           in.elements.azm.size() == half_pairs &&
           in.elements.bzm.size() == half_pairs;
}

std::string validate_case(const AxisymmetricPreconditionerMatrixCase& in) {
    if (in.ns < 2 || in.mpol < 1 || in.ntor < 0 || in.ntheta < 2 ||
        in.ntheta % 2 != 0 || in.nzeta < 1 || in.nfp < 1 ||
        !(in.delta_s > 0.0F) || !std::isfinite(in.delta_s)) {
        return "axisymmetric preconditioner matrix has invalid scalars";
    }
    const std::size_t half_points =
        static_cast<std::size_t>(in.ns - 1) * in.ntheta * in.nzeta;
    const std::size_t mode_count =
        static_cast<std::size_t>(in.mpol) * (in.ntor + 1);
    const std::size_t points = static_cast<std::size_t>(in.ns) * mode_count;
    if (points > std::numeric_limits<std::uint32_t>::max() ||
        !has_element_shapes(in) ||
        in.base_geometry.size() != BASE_GEOMETRY_FIELD_COUNT * half_points ||
        in.sqrt_s_f.size() != static_cast<std::size_t>(in.ns) ||
        in.phip_h.size() != static_cast<std::size_t>(in.ns - 1)) {
        return "axisymmetric preconditioner matrix input shape mismatch";
    }
    return {};
}

MatrixValues matrix_values(const AxisymmetricPreconditionerMatrixCase& in,
                           int mode,
                           int surface) {
    MatrixValues out;
    const int m = mode / (in.ntor + 1);
    const int n = mode % (in.ntor + 1);
    const int parity = m % 2;
    const float m2 = static_cast<float>(m * m);
    const float n2 = static_cast<float>(n * in.nfp * n * in.nfp);
    const auto& e = in.elements;
    if (surface < in.ns - 1) {
        out.upper_r =
            -(e.arm[2 * surface + parity] + e.brm[2 * surface + parity] * m2);
        out.upper_z =
            -(e.azm[2 * surface + parity] + e.bzm[2 * surface + parity] * m2);
    }
    out.diagonal_r = -(e.ard[2 * surface + parity] +
                       e.brd[2 * surface + parity] * m2 + e.cxd[surface] * n2);
    out.diagonal_z = -(e.azd[2 * surface + parity] +
                       e.bzd[2 * surface + parity] * m2 + e.cxd[surface] * n2);
    if (in.free_boundary && surface == in.ns - 1) {
        const float pedestal = m <= 1 ? 0.05F : 0.1F;
        out.diagonal_r *= 1.0F + pedestal;
        out.diagonal_z *= 1.0F + pedestal;
        if (m == 0 && n == 0) {
            const float mult_fact = std::min(0.25F, 0.25F * in.delta_s * 15.0F);
            out.diagonal_z *= (1.0F - mult_fact) / 1.05F;
        }
    }
    if (surface > 0) {
        out.lower_r = -(e.arm[2 * (surface - 1) + parity] +
                        e.brm[2 * (surface - 1) + parity] * m2);
        out.lower_z = -(e.azm[2 * (surface - 1) + parity] +
                        e.bzm[2 * (surface - 1) + parity] * m2);
    }
    if (surface == 1 && m == 1) {
        out.diagonal_r += out.lower_r;
        out.diagonal_z += out.lower_z;
    }
    return out;
}

float half_lambda(const AxisymmetricPreconditionerMatrixCase& in,
                  int shifted,
                  int metric_field) {
    if (shifted == 0 || shifted == in.ns) return 0.0F;
    const int half_surface = shifted - 1;
    const std::size_t half_points =
        static_cast<std::size_t>(in.ns - 1) * in.ntheta * in.nzeta;
    const std::size_t n_z_n_t = static_cast<std::size_t>(in.ntheta) * in.nzeta;
    const int ntheta_red = in.ntheta / 2 + 1;
    const float norm = 1.0F / static_cast<float>(in.nzeta * (ntheta_red - 1));
    float sum = 0.0F;
    for (int zeta = 0; zeta < in.nzeta; ++zeta) {
        for (int theta = 0; theta < ntheta_red; ++theta) {
            float weight = norm;
            if (theta == 0 || theta == ntheta_red - 1) weight *= 0.5F;
            const std::size_t point =
                static_cast<std::size_t>(half_surface) * n_z_n_t +
                zeta * in.ntheta + theta;
            const float gsqrt = in.base_geometry[6 * half_points + point];
            const float metric =
                in.base_geometry[metric_field * half_points + point];
            sum += metric / gsqrt * weight;
        }
    }
    return sum;
}

float rms_phip(const AxisymmetricPreconditionerMatrixCase& in) {
    double total = 0.0;
    for (const float value : in.phip_h) {
        total += static_cast<double>(value * value);
    }
    return static_cast<float>(total);
}

float lambda_value(const AxisymmetricPreconditionerMatrixCase& in,
                   int mode,
                   int surface,
                   float rms) {
    const int m = mode / (in.ntor + 1);
    const int n = mode % (in.ntor + 1);
    if (surface == 0 || (m == 0 && n == 0)) return 0.0F;
    const float lamscale = std::sqrt(std::max(rms * in.delta_s, 1.0e-30F));
    const float p_factor = 2.0F / (4.0F * lamscale * lamscale);
    const float pwr =
        std::min(static_cast<float>(m * m) / (16.0F * 16.0F), 8.0F);
    const auto full_metric = [&](int field) {
        return 0.5F * (half_lambda(in, surface + 1, field) +
                       half_lambda(in, surface, field));
    };
    const float b_full = full_metric(7);
    const float d_full = full_metric(8);
    const float c_full = full_metric(9);
    const float toroidal = static_cast<float>(n * in.nfp);
    float faclam = toroidal * toroidal * b_full +
                   2.0F * static_cast<float>(m) * toroidal *
                       std::copysign(d_full, b_full) +
                   static_cast<float>(m * m) * c_full;
    if (faclam == 0.0F) faclam = -1.0e-10F;
    return p_factor / faclam * std::pow(in.sqrt_s_f[surface], pwr);
}

std::vector<float> flatten_elements(
    const AxisymmetricPreconditionerElements& elements) {
    std::vector<float> values;
    const std::size_t count =
        elements.ard.size() + elements.brd.size() + elements.azd.size() +
        elements.bzd.size() + elements.cxd.size() + elements.arm.size() +
        elements.brm.size() + elements.azm.size() + elements.bzm.size();
    values.reserve(count);
    const auto append = [&values](const auto& source) {
        values.insert(values.end(), source.begin(), source.end());
    };
    append(elements.ard);
    append(elements.brd);
    append(elements.azd);
    append(elements.bzd);
    append(elements.cxd);
    append(elements.arm);
    append(elements.brm);
    append(elements.azm);
    append(elements.bzm);
    return values;
}

std::string load_shader() {
    std::ifstream stream("/shaders/axisymmetric_preconditioner_matrix.wgsl",
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
    AxisymmetricPreconditionerMatrixCallback callback;
    wgpu::Buffer output, readback;
    std::size_t bytes = 0, points = 0;
    int mpol = 0;
    std::vector<int> first_surface;
};

}  // namespace

AxisymmetricPreconditionerMatrix axisymmetric_preconditioner_matrix_reference(
    const AxisymmetricPreconditionerMatrixCase& input) {
    if (!validate_case(input).empty()) return {};
    const int mode_count = input.mpol * (input.ntor + 1);
    const std::size_t points = static_cast<std::size_t>(input.ns) * mode_count;
    AxisymmetricPreconditionerMatrix out;
    out.upper_r.resize(points);
    out.diagonal_r.resize(points);
    out.lower_r.resize(points);
    out.upper_z.resize(points);
    out.diagonal_z.resize(points);
    out.lower_z.resize(points);
    out.lambda.resize(points);
    out.scale.assign(mode_count, 0.0F);
    out.first_surface.resize(mode_count);
    const float rms = rms_phip(input);
    for (int mode = 0; mode < mode_count; ++mode) {
        const int m = mode / (input.ntor + 1);
        out.first_surface[mode] = m == 0 ? 0 : 1;
        for (int surface = 0; surface < input.ns; ++surface) {
            const std::size_t index =
                static_cast<std::size_t>(mode) * input.ns + surface;
            const auto value = matrix_values(input, mode, surface);
            out.upper_r[index] = value.upper_r;
            out.diagonal_r[index] = value.diagonal_r;
            out.lower_r[index] = value.lower_r;
            out.upper_z[index] = value.upper_z;
            out.diagonal_z[index] = value.diagonal_z;
            out.lower_z[index] = value.lower_z;
            out.lambda[index] = lambda_value(input, mode, surface, rms);
            out.scale[mode] = std::max(
                out.scale[mode],
                std::max({std::abs(value.upper_r), std::abs(value.diagonal_r),
                          std::abs(value.lower_r), std::abs(value.upper_z),
                          std::abs(value.diagonal_z),
                          std::abs(value.lower_z)}));
        }
    }
    return out;
}

void enqueue_axisymmetric_preconditioner_matrix(
    const wgpu::Device& device,
    const AxisymmetricPreconditionerMatrixCase& input,
    AxisymmetricPreconditionerMatrixCallback callback) {
    const auto error = validate_case(input);
    if (!error.empty()) {
        callback(error, {});
        return;
    }
    const auto shader_text = load_shader();
    if (shader_text.empty()) {
        callback("cannot load axisymmetric preconditioner-matrix shader", {});
        return;
    }
    const int mode_count = input.mpol * (input.ntor + 1);
    const std::size_t points = static_cast<std::size_t>(input.ns) * mode_count;
    const std::size_t half_points =
        static_cast<std::size_t>(input.ns - 1) * input.ntheta * input.nzeta;
    auto elements = flatten_elements(input.elements);
    std::vector<float> radial = input.sqrt_s_f;
    radial.insert(radial.end(), input.phip_h.begin(), input.phip_h.end());
    radial.push_back(rms_phip(input));
    const auto element_bytes = elements.size() * sizeof(float);
    const auto base_bytes = input.base_geometry.size() * sizeof(float);
    const auto radial_bytes = radial.size() * sizeof(float);
    const auto output_values = 7 * points + mode_count;
    const auto output_bytes = output_values * sizeof(float);
    auto element_buffer =
        make_buffer(device, element_bytes,
                    wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                    "preconditioner element cache");
    auto base_buffer =
        make_buffer(device, base_bytes,
                    wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                    "preconditioner lambda geometry");
    auto radial_buffer =
        make_buffer(device, radial_bytes,
                    wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                    "preconditioner matrix radial data");
    auto output_buffer =
        make_buffer(device, output_bytes,
                    wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc,
                    "preconditioner matrix");
    auto readback =
        make_buffer(device, output_bytes,
                    wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead,
                    "preconditioner matrix readback");
    auto params_buffer =
        make_buffer(device, sizeof(Params),
                    wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
                    "preconditioner matrix params");
    const auto& pipeline = detail::cached_compute_pipeline(
        device, "preconditioner-matrix", shader_text,
        "cuMES preconditioner matrix pipeline");
    const Params params{static_cast<std::uint32_t>(input.ns),
                        static_cast<std::uint32_t>(input.mpol),
                        static_cast<std::uint32_t>(input.ntor),
                        static_cast<std::uint32_t>(input.ntheta),
                        static_cast<std::uint32_t>(input.nzeta),
                        static_cast<std::uint32_t>(input.ntheta * input.nzeta),
                        static_cast<std::uint32_t>(points),
                        static_cast<std::uint32_t>(half_points),
                        static_cast<std::uint32_t>(input.nfp),
                        input.free_boundary ? 1U : 0U,
                        input.delta_s,
                        0};
    auto queue = device.GetQueue();
    queue.WriteBuffer(element_buffer, 0, elements.data(), element_bytes);
    queue.WriteBuffer(base_buffer, 0, input.base_geometry.data(), base_bytes);
    queue.WriteBuffer(radial_buffer, 0, radial.data(), radial_bytes);
    queue.WriteBuffer(params_buffer, 0, &params, sizeof(params));
    auto layout = pipeline.GetBindGroupLayout(0);
    const wgpu::BindGroupEntry entries[] = {
        {nullptr, 0, element_buffer, 0, element_bytes, nullptr, nullptr},
        {nullptr, 1, base_buffer, 0, base_bytes, nullptr, nullptr},
        {nullptr, 2, radial_buffer, 0, radial_bytes, nullptr, nullptr},
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
    dispatch->points = points;
    dispatch->mpol = mode_count;
    dispatch->first_surface.resize(mode_count);
    for (int mode = 0; mode < mode_count; ++mode) {
        const int m = mode / (input.ntor + 1);
        dispatch->first_surface[mode] = m == 0 ? 0 : 1;
    }
    readback.MapAsync(
        wgpu::MapMode::Read, 0, output_bytes,
        wgpu::CallbackMode::AllowSpontaneous,
        [dispatch](wgpu::MapAsyncStatus status, wgpu::StringView message) {
            if (status != wgpu::MapAsyncStatus::Success) {
                const std::string detail =
                    message.length == 0
                        ? std::string{}
                        : std::string(message.data, message.length);
                dispatch->callback(
                    "preconditioner matrix mapping failed: " + detail, {});
                return;
            }
            const auto* values = static_cast<const float*>(
                dispatch->readback.GetConstMappedRange(0, dispatch->bytes));
            if (values == nullptr) {
                dispatch->callback("preconditioner matrix mapped range is null",
                                   {});
                return;
            }
            const std::size_t n = dispatch->points;
            AxisymmetricPreconditionerMatrix out;
            out.upper_r.assign(values, values + n);
            out.diagonal_r.assign(values + n, values + 2 * n);
            out.lower_r.assign(values + 2 * n, values + 3 * n);
            out.upper_z.assign(values + 3 * n, values + 4 * n);
            out.diagonal_z.assign(values + 4 * n, values + 5 * n);
            out.lower_z.assign(values + 5 * n, values + 6 * n);
            out.lambda.assign(values + 6 * n, values + 7 * n);
            out.scale.assign(values + 7 * n, values + 7 * n + dispatch->mpol);
            out.first_surface = std::move(dispatch->first_surface);
            dispatch->readback.Unmap();
            dispatch->callback({}, std::move(out));
        });
}

}  // namespace cumes::webgpu
