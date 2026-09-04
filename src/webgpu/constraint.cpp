#include "cumes/webgpu/constraint.hpp"

#include "cumes/webgpu/axisymmetric.hpp"
#include "cumes/webgpu/force.hpp"
#include "cumes/webgpu/toroidal.hpp"
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

constexpr std::uint32_t WORKGROUP_SIZE = 256;

struct HeadParams {
    std::uint32_t ns, mpol, ntheta, nzeta;
    std::uint32_t n_z_n_t, points;
    std::uint32_t reset_reference, refresh_preconditioner;
    float delta_s, tcon_multiplier;
    std::uint32_t padding[2];
};
static_assert(sizeof(HeadParams) == 48);

struct TailParams {
    std::uint32_t ns, n_z_n_t, points, force_fields;
    std::uint32_t output_fields;
    std::uint32_t padding[3];
};
static_assert(sizeof(TailParams) == 32);

struct HeadResult {
    std::vector<float> g_con_eff;
    std::vector<float> r_con0;
    std::vector<float> z_con0;
    std::vector<float> tcon;
};

std::string validate_case(const AxisymmetricConstraintCase& in) {
    if (in.ns < 2 || in.mpol < 2 || in.ntor < 0 || in.ntheta < 2 ||
        in.ntheta % 2 != 0 || in.nzeta < 1 || !(in.delta_s > 0.0F) ||
        !std::isfinite(in.delta_s) || !std::isfinite(in.tcon0)) {
        return "axisymmetric constraint has invalid shape or scalars";
    }
    if ((in.ntor == 0) != (in.nzeta == 1)) {
        return "constraint requires ntor=0 exactly when nzeta=1";
    }
    const std::size_t n_z_n_t = static_cast<std::size_t>(in.ntheta) * in.nzeta;
    const std::size_t points = static_cast<std::size_t>(in.ns) * n_z_n_t;
    const std::size_t force_fields = in.ntor == 0 ? 10 : FORCE_FIELD_COUNT;
    if (points > std::numeric_limits<std::uint32_t>::max() ||
        in.geometry.size() != GEOMETRY_PARITY_FIELD_COUNT * points ||
        in.r_con.size() != points || in.z_con.size() != points ||
        in.r_con0.size() != points || in.z_con0.size() != points ||
        in.tcon.size() != static_cast<std::size_t>(in.ns) ||
        in.ard.size() != 2 * static_cast<std::size_t>(in.ns) ||
        in.azd.size() != 2 * static_cast<std::size_t>(in.ns) ||
        in.sqrt_s_f.size() != static_cast<std::size_t>(in.ns) ||
        in.force_fields.size() != force_fields * points) {
        return "constraint input shape mismatch";
    }
    return {};
}

std::string load_shader(const char* path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) return {};
    std::ostringstream text;
    text << stream.rdbuf();
    return text.str();
}

wgpu::Buffer make_buffer(const wgpu::Device& device,
                         std::uint64_t size,
                         wgpu::BufferUsage usage,
                         const char* label) {
    return detail::cached_buffer(device, size, usage, label);
}

float tcon_multiplier(const AxisymmetricConstraintCase& in) {
    const double factor =
        (1.0 + in.ns * (1.0 / 60.0 + in.ns / (200.0 * 120.0))) / 16.0;
    return in.tcon0 * static_cast<float>(factor);
}

float reference_value(const AxisymmetricConstraintCase& in,
                      const std::vector<float>& value,
                      const std::vector<float>& old_reference,
                      int surface,
                      std::size_t angular) {
    const std::size_t n_z_n_t = static_cast<std::size_t>(in.ntheta) * in.nzeta;
    const std::size_t point =
        static_cast<std::size_t>(surface) * n_z_n_t + angular;
    if (!in.reset_reference || surface == 0) return old_reference[point];
    const std::size_t lcfs =
        static_cast<std::size_t>(in.ns - 1) * n_z_n_t + angular;
    const float sqrt_s = in.sqrt_s_f[surface];
    return value[lcfs] * sqrt_s * sqrt_s;
}

float compute_tcon_base(const AxisymmetricConstraintCase& in, int surface) {
    const std::size_t n_z_n_t = static_cast<std::size_t>(in.ntheta) * in.nzeta;
    const std::size_t points = static_cast<std::size_t>(in.ns) * n_z_n_t;
    const int ntheta_red = in.ntheta / 2 + 1;
    const float norm = 1.0F / static_cast<float>(in.nzeta * (ntheta_red - 1));
    const float sqrt_s = in.sqrt_s_f[surface];
    float ar_n = 0.0F;
    float az_n = 0.0F;
    for (int zeta = 0; zeta < in.nzeta; ++zeta) {
        for (int theta = 0; theta < ntheta_red; ++theta) {
            float weight = norm;
            if (theta == 0 || theta == ntheta_red - 1) weight *= 0.5F;
            const std::size_t point =
                static_cast<std::size_t>(surface) * n_z_n_t + zeta * in.ntheta +
                theta;
            const float ru = in.geometry[3 * points + point] +
                             sqrt_s * in.geometry[9 * points + point];
            const float zu = in.geometry[4 * points + point] +
                             sqrt_s * in.geometry[10 * points + point];
            ar_n += ru * ru * weight;
            az_n += zu * zu * weight;
        }
    }
    if (ar_n == 0.0F) ar_n = 1.0e-10F;
    if (az_n == 0.0F) az_n = 1.0e-10F;
    const float base = std::min(std::abs(in.ard[2 * surface]) / ar_n,
                                std::abs(in.azd[2 * surface]) / az_n);
    return base * tcon_multiplier(in) * 32.0F * in.delta_s * 32.0F * in.delta_s;
}

HeadResult head_reference(const AxisymmetricConstraintCase& in) {
    const std::size_t n_z_n_t = static_cast<std::size_t>(in.ntheta) * in.nzeta;
    const std::size_t points = static_cast<std::size_t>(in.ns) * n_z_n_t;
    HeadResult out;
    out.g_con_eff.assign(points, 0.0F);
    out.r_con0.resize(points);
    out.z_con0.resize(points);
    out.tcon = in.tcon;
    for (int surface = 0; surface < in.ns; ++surface) {
        for (std::size_t angular = 0; angular < n_z_n_t; ++angular) {
            const std::size_t point =
                static_cast<std::size_t>(surface) * n_z_n_t + angular;
            const float r0 =
                reference_value(in, in.r_con, in.r_con0, surface, angular);
            const float z0 =
                reference_value(in, in.z_con, in.z_con0, surface, angular);
            out.r_con0[point] = r0;
            out.z_con0[point] = z0;
            if (surface == 0) continue;
            const float sqrt_s = in.sqrt_s_f[surface];
            const float ru = in.geometry[3 * points + point] +
                             sqrt_s * in.geometry[9 * points + point];
            const float zu = in.geometry[4 * points + point] +
                             sqrt_s * in.geometry[10 * points + point];
            out.g_con_eff[point] =
                (in.r_con[point] - r0) * ru + (in.z_con[point] - z0) * zu;
        }
    }
    out.tcon[0] = 0.0F;
    if (in.refresh_preconditioner) {
        for (int surface = 1; surface < in.ns - 1; ++surface) {
            out.tcon[surface] = compute_tcon_base(in, surface);
        }
        out.tcon[in.ns - 1] = 0.5F * out.tcon[in.ns - 2];
    }
    return out;
}

void apply_constraint_tail(const AxisymmetricConstraintCase& in,
                           const HeadResult& head,
                           const std::vector<float>& g_con,
                           AxisymmetricConstraintResult& out) {
    const std::size_t n_z_n_t = static_cast<std::size_t>(in.ntheta) * in.nzeta;
    const std::size_t points = static_cast<std::size_t>(in.ns) * n_z_n_t;
    const std::size_t output_fields =
        in.ntor == 0 ? FORWARD_INPUT_FIELD_COUNT : TOROIDAL_FORWARD_FIELD_COUNT;
    const std::size_t constraint_offset = output_fields - 4;
    out.fields.assign(output_fields * points, 0.0F);
    std::copy(in.force_fields.begin(), in.force_fields.end(),
              out.fields.begin());
    for (int surface = 1; surface < in.ns; ++surface) {
        const float sqrt_s = in.sqrt_s_f[surface];
        for (std::size_t angular = 0; angular < n_z_n_t; ++angular) {
            const std::size_t point =
                static_cast<std::size_t>(surface) * n_z_n_t + angular;
            const float dr = in.r_con[point] - head.r_con0[point];
            const float dz = in.z_con[point] - head.z_con0[point];
            const float gc = g_con[point];
            const float brcon = dr * gc;
            const float bzcon = dz * gc;
            out.fields[4 * points + point] += brcon;
            out.fields[5 * points + point] += brcon * sqrt_s;
            out.fields[6 * points + point] += bzcon;
            out.fields[7 * points + point] += bzcon * sqrt_s;
            const float ru = in.geometry[3 * points + point] +
                             sqrt_s * in.geometry[9 * points + point];
            const float zu = in.geometry[4 * points + point] +
                             sqrt_s * in.geometry[10 * points + point];
            out.fields[(constraint_offset + 0) * points + point] = ru * gc;
            out.fields[(constraint_offset + 1) * points + point] =
                ru * gc * sqrt_s;
            out.fields[(constraint_offset + 2) * points + point] = zu * gc;
            out.fields[(constraint_offset + 3) * points + point] =
                zu * gc * sqrt_s;
        }
    }
}

struct HeadDispatch {
    std::function<void(std::string, HeadResult)> callback;
    wgpu::Buffer output, readback;
    std::size_t points = 0, bytes = 0;
    int ns = 0;
};

void enqueue_head(const wgpu::Device& device,
                  const AxisymmetricConstraintCase& in,
                  std::function<void(std::string, HeadResult)> callback) {
    const auto shader_text =
        load_shader("/shaders/axisymmetric_constraint_head.wgsl");
    if (shader_text.empty()) {
        callback("cannot load axisymmetric constraint-head shader", {});
        return;
    }
    const std::size_t n_z_n_t = static_cast<std::size_t>(in.ntheta) * in.nzeta;
    const std::size_t points = static_cast<std::size_t>(in.ns) * n_z_n_t;
    std::vector<float> constraint;
    constraint.reserve(4 * points);
    constraint.insert(constraint.end(), in.r_con.begin(), in.r_con.end());
    constraint.insert(constraint.end(), in.z_con.begin(), in.z_con.end());
    constraint.insert(constraint.end(), in.r_con0.begin(), in.r_con0.end());
    constraint.insert(constraint.end(), in.z_con0.begin(), in.z_con0.end());
    std::vector<float> radial;
    radial.reserve(6 * static_cast<std::size_t>(in.ns));
    radial.insert(radial.end(), in.sqrt_s_f.begin(), in.sqrt_s_f.end());
    radial.insert(radial.end(), in.tcon.begin(), in.tcon.end());
    radial.insert(radial.end(), in.ard.begin(), in.ard.end());
    radial.insert(radial.end(), in.azd.begin(), in.azd.end());
    const std::size_t output_values = 3 * points + in.ns;
    const auto geometry_bytes = in.geometry.size() * sizeof(float);
    const auto constraint_bytes = constraint.size() * sizeof(float);
    const auto radial_bytes = radial.size() * sizeof(float);
    const auto output_bytes = output_values * sizeof(float);
    auto geometry_buffer =
        make_buffer(device, geometry_bytes,
                    wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                    "constraint geometry");
    auto constraint_buffer =
        make_buffer(device, constraint_bytes,
                    wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                    "constraint state");
    auto radial_buffer =
        make_buffer(device, radial_bytes,
                    wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                    "constraint radial data");
    auto output_buffer =
        make_buffer(device, output_bytes,
                    wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc,
                    "constraint head output");
    auto readback =
        make_buffer(device, output_bytes,
                    wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead,
                    "constraint head readback");
    auto params_buffer =
        make_buffer(device, sizeof(HeadParams),
                    wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
                    "constraint head params");
    const auto& pipeline =
        detail::cached_compute_pipeline(device, "constraint-head", shader_text,
                                        "cuMES constraint head pipeline");
    const HeadParams params{static_cast<std::uint32_t>(in.ns),
                            static_cast<std::uint32_t>(in.mpol),
                            static_cast<std::uint32_t>(in.ntheta),
                            static_cast<std::uint32_t>(in.nzeta),
                            static_cast<std::uint32_t>(n_z_n_t),
                            static_cast<std::uint32_t>(points),
                            in.reset_reference ? 1U : 0U,
                            in.refresh_preconditioner ? 1U : 0U,
                            in.delta_s,
                            tcon_multiplier(in),
                            {0, 0}};
    auto queue = device.GetQueue();
    queue.WriteBuffer(geometry_buffer, 0, in.geometry.data(), geometry_bytes);
    queue.WriteBuffer(constraint_buffer, 0, constraint.data(),
                      constraint_bytes);
    queue.WriteBuffer(radial_buffer, 0, radial.data(), radial_bytes);
    queue.WriteBuffer(params_buffer, 0, &params, sizeof(params));
    auto layout = pipeline.GetBindGroupLayout(0);
    const wgpu::BindGroupEntry entries[] = {
        {nullptr, 0, geometry_buffer, 0, geometry_bytes, nullptr, nullptr},
        {nullptr, 1, constraint_buffer, 0, constraint_bytes, nullptr, nullptr},
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
    auto dispatch = std::make_shared<HeadDispatch>();
    dispatch->callback = std::move(callback);
    dispatch->output = output_buffer;
    dispatch->readback = readback;
    dispatch->points = points;
    dispatch->bytes = output_bytes;
    dispatch->ns = in.ns;
    readback.MapAsync(
        wgpu::MapMode::Read, 0, output_bytes,
        wgpu::CallbackMode::AllowSpontaneous,
        [dispatch](wgpu::MapAsyncStatus status, wgpu::StringView message) {
            if (status != wgpu::MapAsyncStatus::Success) {
                const std::string detail =
                    message.length == 0
                        ? std::string{}
                        : std::string(message.data, message.length);
                dispatch->callback("constraint head mapping failed: " + detail,
                                   {});
                return;
            }
            const auto* values = static_cast<const float*>(
                dispatch->readback.GetConstMappedRange(0, dispatch->bytes));
            if (values == nullptr) {
                dispatch->callback("constraint head mapped range is null", {});
                return;
            }
            HeadResult out;
            out.g_con_eff.assign(values, values + dispatch->points);
            out.r_con0.assign(values + dispatch->points,
                              values + 2 * dispatch->points);
            out.z_con0.assign(values + 2 * dispatch->points,
                              values + 3 * dispatch->points);
            out.tcon.assign(values + 3 * dispatch->points,
                            values + 3 * dispatch->points + dispatch->ns);
            dispatch->readback.Unmap();
            dispatch->callback({}, std::move(out));
        });
}

struct TailDispatch {
    AxisymmetricConstraintCallback callback;
    AxisymmetricConstraintResult result;
    wgpu::Buffer output, readback;
    std::size_t values = 0, bytes = 0;
};

void enqueue_tail(const wgpu::Device& device,
                  const AxisymmetricConstraintCase& in,
                  HeadResult head,
                  std::vector<float> g_con,
                  AxisymmetricConstraintCallback callback) {
    const auto shader_text =
        load_shader("/shaders/axisymmetric_constraint_tail.wgsl");
    if (shader_text.empty()) {
        callback("cannot load axisymmetric constraint-tail shader", {});
        return;
    }
    const std::size_t n_z_n_t = static_cast<std::size_t>(in.ntheta) * in.nzeta;
    const std::size_t points = static_cast<std::size_t>(in.ns) * n_z_n_t;
    std::vector<float> constraint;
    constraint.reserve(5 * points);
    constraint.insert(constraint.end(), in.r_con.begin(), in.r_con.end());
    constraint.insert(constraint.end(), in.z_con.begin(), in.z_con.end());
    constraint.insert(constraint.end(), head.r_con0.begin(), head.r_con0.end());
    constraint.insert(constraint.end(), head.z_con0.begin(), head.z_con0.end());
    constraint.insert(constraint.end(), g_con.begin(), g_con.end());
    const auto force_bytes = in.force_fields.size() * sizeof(float);
    const auto geometry_bytes = in.geometry.size() * sizeof(float);
    const auto constraint_bytes = constraint.size() * sizeof(float);
    const auto radial_bytes = in.sqrt_s_f.size() * sizeof(float);
    const std::size_t force_fields = in.ntor == 0 ? 10 : FORCE_FIELD_COUNT;
    const std::size_t output_fields =
        in.ntor == 0 ? FORWARD_INPUT_FIELD_COUNT : TOROIDAL_FORWARD_FIELD_COUNT;
    const std::size_t output_values = output_fields * points;
    const auto output_bytes = output_values * sizeof(float);
    auto force_buffer =
        make_buffer(device, force_bytes,
                    wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                    "constraint force input");
    auto geometry_buffer =
        make_buffer(device, geometry_bytes,
                    wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                    "constraint tail geometry");
    auto constraint_buffer =
        make_buffer(device, constraint_bytes,
                    wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                    "dealiased constraint");
    auto radial_buffer =
        make_buffer(device, radial_bytes,
                    wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                    "constraint sqrt s");
    auto output_buffer =
        make_buffer(device, output_bytes,
                    wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc,
                    "constraint force output");
    auto readback =
        make_buffer(device, output_bytes,
                    wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead,
                    "constraint force readback");
    auto params_buffer =
        make_buffer(device, sizeof(TailParams),
                    wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
                    "constraint tail params");
    const auto& pipeline =
        detail::cached_compute_pipeline(device, "constraint-tail", shader_text,
                                        "cuMES constraint tail pipeline");
    const TailParams params{static_cast<std::uint32_t>(in.ns),
                            static_cast<std::uint32_t>(n_z_n_t),
                            static_cast<std::uint32_t>(points),
                            static_cast<std::uint32_t>(force_fields),
                            static_cast<std::uint32_t>(output_fields),
                            {0, 0, 0}};
    auto queue = device.GetQueue();
    queue.WriteBuffer(force_buffer, 0, in.force_fields.data(), force_bytes);
    queue.WriteBuffer(geometry_buffer, 0, in.geometry.data(), geometry_bytes);
    queue.WriteBuffer(constraint_buffer, 0, constraint.data(),
                      constraint_bytes);
    queue.WriteBuffer(radial_buffer, 0, in.sqrt_s_f.data(), radial_bytes);
    queue.WriteBuffer(params_buffer, 0, &params, sizeof(params));
    auto layout = pipeline.GetBindGroupLayout(0);
    const wgpu::BindGroupEntry entries[] = {
        {nullptr, 0, force_buffer, 0, force_bytes, nullptr, nullptr},
        {nullptr, 1, geometry_buffer, 0, geometry_bytes, nullptr, nullptr},
        {nullptr, 2, constraint_buffer, 0, constraint_bytes, nullptr, nullptr},
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
        (static_cast<std::uint32_t>(points) + WORKGROUP_SIZE - 1) /
        WORKGROUP_SIZE);
    pass.End();
    encoder.CopyBufferToBuffer(output_buffer, 0, readback, 0, output_bytes);
    auto commands = encoder.Finish();
    queue.Submit(1, &commands);
    auto dispatch = std::make_shared<TailDispatch>();
    dispatch->callback = std::move(callback);
    dispatch->result.r_con0 = std::move(head.r_con0);
    dispatch->result.z_con0 = std::move(head.z_con0);
    dispatch->result.tcon = std::move(head.tcon);
    dispatch->result.g_con_eff = std::move(head.g_con_eff);
    dispatch->result.g_con = std::move(g_con);
    dispatch->output = output_buffer;
    dispatch->readback = readback;
    dispatch->values = output_values;
    dispatch->bytes = output_bytes;
    readback.MapAsync(
        wgpu::MapMode::Read, 0, output_bytes,
        wgpu::CallbackMode::AllowSpontaneous,
        [dispatch](wgpu::MapAsyncStatus status, wgpu::StringView message) {
            if (status != wgpu::MapAsyncStatus::Success) {
                const std::string detail =
                    message.length == 0
                        ? std::string{}
                        : std::string(message.data, message.length);
                dispatch->callback("constraint tail mapping failed: " + detail,
                                   {});
                return;
            }
            const auto* values = static_cast<const float*>(
                dispatch->readback.GetConstMappedRange(0, dispatch->bytes));
            if (values == nullptr) {
                dispatch->callback("constraint tail mapped range is null", {});
                return;
            }
            dispatch->result.fields.assign(values, values + dispatch->values);
            dispatch->readback.Unmap();
            dispatch->callback({}, std::move(dispatch->result));
        });
}

struct ConstraintChain {
    wgpu::Device device;
    AxisymmetricConstraintCase input;
    AxisymmetricConstraintCallback callback;
};

}  // namespace

AxisymmetricConstraintResult axisymmetric_constraint_reference(
    const AxisymmetricConstraintCase& input) {
    if (!validate_case(input).empty()) return {};
    auto head = head_reference(input);
    std::vector<float> g_con;
    if (input.ntor > 0) {
        ToroidalDealiasCase dealias;
        dealias.ns = input.ns;
        dealias.mpol = input.mpol;
        dealias.ntor = input.ntor;
        dealias.ntheta = input.ntheta;
        dealias.nzeta = input.nzeta;
        dealias.g_con_eff = head.g_con_eff;
        dealias.tcon = head.tcon;
        dealias.faccon.assign(input.mpol, 0.0F);
        for (int mode = 1; mode < input.mpol; ++mode) {
            const float xmpq = static_cast<float>((mode + 1) * mode);
            dealias.faccon[mode] = 0.25F / (xmpq * xmpq);
        }
        g_con = toroidal_dealias_reference(dealias).g_con;
    } else {
        AxisymmetricDealiasCase dealias;
        dealias.ns = input.ns;
        dealias.mpol = input.mpol;
        dealias.ntheta = input.ntheta;
        dealias.g_con_eff = head.g_con_eff;
        dealias.tcon = head.tcon;
        dealias.faccon.assign(input.mpol, 0.0F);
        for (int mode = 1; mode < input.mpol; ++mode) {
            const float xmpq = static_cast<float>((mode + 1) * mode);
            dealias.faccon[mode] = 0.25F / (xmpq * xmpq);
        }
        g_con = axisymmetric_dealias_reference(dealias).g_con;
    }
    AxisymmetricConstraintResult result;
    result.r_con0 = head.r_con0;
    result.z_con0 = head.z_con0;
    result.tcon = head.tcon;
    result.g_con_eff = head.g_con_eff;
    result.g_con = g_con;
    apply_constraint_tail(input, head, g_con, result);
    return result;
}

void enqueue_axisymmetric_constraint(const wgpu::Device& device,
                                     const AxisymmetricConstraintCase& input,
                                     AxisymmetricConstraintCallback callback) {
    const auto error = validate_case(input);
    if (!error.empty()) {
        callback(error, {});
        return;
    }
    auto chain = std::make_shared<ConstraintChain>();
    chain->device = device;
    chain->input = input;
    chain->callback = std::move(callback);
    enqueue_head(
        device, chain->input, [chain](std::string error, HeadResult head) {
            if (!error.empty()) {
                chain->callback(std::move(error), {});
                return;
            }
            if (chain->input.ntor > 0) {
                ToroidalDealiasCase dealias;
                dealias.ns = chain->input.ns;
                dealias.mpol = chain->input.mpol;
                dealias.ntor = chain->input.ntor;
                dealias.ntheta = chain->input.ntheta;
                dealias.nzeta = chain->input.nzeta;
                dealias.g_con_eff = head.g_con_eff;
                dealias.tcon = head.tcon;
                dealias.faccon.assign(chain->input.mpol, 0.0F);
                for (int mode = 1; mode < chain->input.mpol; ++mode) {
                    const float xmpq = static_cast<float>((mode + 1) * mode);
                    dealias.faccon[mode] = 0.25F / (xmpq * xmpq);
                }
                enqueue_toroidal_dealias(
                    chain->device, dealias,
                    [chain, head = std::move(head)](
                        std::string filter_error,
                        ToroidalDealiasResult filtered) mutable {
                        if (!filter_error.empty()) {
                            chain->callback(std::move(filter_error), {});
                            return;
                        }
                        enqueue_tail(chain->device, chain->input,
                                     std::move(head), std::move(filtered.g_con),
                                     std::move(chain->callback));
                    });
                return;
            }
            AxisymmetricDealiasCase dealias;
            dealias.ns = chain->input.ns;
            dealias.mpol = chain->input.mpol;
            dealias.ntheta = chain->input.ntheta;
            dealias.g_con_eff = head.g_con_eff;
            dealias.tcon = head.tcon;
            dealias.faccon.assign(chain->input.mpol, 0.0F);
            for (int mode = 1; mode < chain->input.mpol; ++mode) {
                const float xmpq = static_cast<float>((mode + 1) * mode);
                dealias.faccon[mode] = 0.25F / (xmpq * xmpq);
            }
            enqueue_axisymmetric_dealias(
                chain->device, dealias,
                [chain, head = std::move(head)](
                    std::string filter_error,
                    AxisymmetricDealiasResult filtered) mutable {
                    if (!filter_error.empty()) {
                        chain->callback(std::move(filter_error), {});
                        return;
                    }
                    enqueue_tail(chain->device, chain->input, std::move(head),
                                 std::move(filtered.g_con),
                                 std::move(chain->callback));
                });
        });
}

}  // namespace cumes::webgpu
