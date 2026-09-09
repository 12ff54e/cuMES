#include "cumes/webgpu/constraint.hpp"

#include "cumes/physics/constraint_filter.hpp"
#include "cumes/webgpu/axisymmetric.hpp"
#include "cumes/webgpu/float_float.hpp"
#include "cumes/webgpu/force.hpp"
#include "cumes/webgpu/reduction.hpp"
#include "cumes/webgpu/toroidal.hpp"
#include "pipeline_cache.hpp"
#include "shader_source.hpp"

#include <algorithm>
#include <array>
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
    DeviceFields device_fields;
    DeviceFields device_tcon;
    bool finite = true;
    std::vector<float> g_con_eff;
    std::vector<float> g_con_eff_lo;
    std::vector<float> r_con0;
    std::vector<float> r_con0_lo;
    std::vector<float> z_con0;
    std::vector<float> z_con0_lo;
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
        !field_shape(in.geometry, in.device_geometry,
                     GEOMETRY_PARITY_FIELD_COUNT * points) ||
        !field_shape(in.r_con, in.device_r_con, points) ||
        !field_shape(in.z_con, in.device_z_con, points) ||
        !field_shape(in.r_con0, in.device_r_con0, points) ||
        !field_shape(in.z_con0, in.device_z_con0, points) ||
        in.tcon.size() != static_cast<std::size_t>(in.ns) ||
        (!in.device_elements &&
         (in.ard.size() != 2 * static_cast<std::size_t>(in.ns) ||
          in.azd.size() != 2 * static_cast<std::size_t>(in.ns))) ||
        in.sqrt_s_f.size() != static_cast<std::size_t>(in.ns) ||
        !field_shape(in.force_fields, in.device_force_fields,
                     force_fields * points)) {
        return "constraint input shape mismatch";
    }
    if (in.double_single &&
        (in.force_fields_lo.size() != in.force_fields.size() ||
         in.geometry_lo.size() != in.geometry.size() ||
         in.r_con_lo.size() != in.r_con.size() ||
         in.z_con_lo.size() != in.z_con.size() ||
         in.r_con0_lo.size() != in.r_con0.size() ||
         in.z_con0_lo.size() != in.z_con0.size() ||
         in.sqrt_s_f_lo.size() != in.sqrt_s_f.size()))
        return "double-single constraint force low-word shape mismatch";
    return {};
}

const std::string& load_shader(const char* path) {
    return detail::cached_shader_source(path);
}

wgpu::Buffer make_buffer(const wgpu::Device& device,
                         std::uint64_t size,
                         wgpu::BufferUsage usage,
                         const char* label) {
    return detail::cached_buffer(device, size, usage, label);
}

// Populate each plane from its actual owner. Do not upload zero placeholders
// for resident data only to overwrite them with GPU copies in the same pass.
void transfer_constraint_plane(const wgpu::Device& device,
                               const wgpu::CommandEncoder& encoder,
                               const wgpu::Buffer& target,
                               std::size_t plane,
                               std::size_t points,
                               const std::vector<float>& host,
                               const DeviceFields& source,
                               bool low) {
    const auto bytes = points * sizeof(float);
    if (source)
        encoder.CopyBufferToBuffer(source.buffer,
                                   low ? source.low_offset : source.high_offset,
                                   target, plane * bytes, bytes);
    else
        device.GetQueue().WriteBuffer(target, plane * bytes, host.data(),
                                      bytes);
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
    if (in.double_single) {
        out.fields_lo.assign(output_fields * points, 0.0F);
        std::copy(in.force_fields_lo.begin(), in.force_fields_lo.end(),
                  out.fields_lo.begin());
    }
    const auto add_force = [&](std::size_t index, double correction) {
        const double old = static_cast<double>(out.fields[index]) +
                           (in.double_single ? out.fields_lo[index] : 0.0);
        const auto pair = split(old + correction);
        out.fields[index] = pair.hi;
        if (in.double_single) out.fields_lo[index] = pair.lo;
    };
    const auto put = [&](std::size_t index, double value) {
        const auto pair = split(value);
        out.fields[index] = pair.hi;
        if (in.double_single) out.fields_lo[index] = pair.lo;
    };
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
            add_force(4 * points + point, brcon);
            add_force(5 * points + point, static_cast<double>(brcon) * sqrt_s);
            add_force(6 * points + point, bzcon);
            add_force(7 * points + point, static_cast<double>(bzcon) * sqrt_s);
            const float ru = in.geometry[3 * points + point] +
                             sqrt_s * in.geometry[9 * points + point];
            const float zu = in.geometry[4 * points + point] +
                             sqrt_s * in.geometry[10 * points + point];
            put((constraint_offset + 0) * points + point,
                static_cast<double>(ru) * gc);
            put((constraint_offset + 1) * points + point,
                static_cast<double>(ru) * gc * sqrt_s);
            put((constraint_offset + 2) * points + point,
                static_cast<double>(zu) * gc);
            put((constraint_offset + 3) * points + point,
                static_cast<double>(zu) * gc * sqrt_s);
        }
    }
}

void enqueue_head(const wgpu::Device& device,
                  const AxisymmetricConstraintCase& in,
                  std::function<void(std::string, HeadResult)> callback,
                  BatchedReadback<HeadResult> batched = {}) {
    const auto& shader_text = load_shader(
        in.double_single ? "/shaders/constraint_head_double_single.wgsl"
                         : "/shaders/axisymmetric_constraint_head.wgsl");
    if (shader_text.empty()) {
        callback("cannot load axisymmetric constraint-head shader", {});
        return;
    }
    const std::size_t n_z_n_t = static_cast<std::size_t>(in.ntheta) * in.nzeta;
    const std::size_t points = static_cast<std::size_t>(in.ns) * n_z_n_t;
    std::vector<float> radial;
    radial.reserve(6 * static_cast<std::size_t>(in.ns));
    radial.insert(radial.end(), in.sqrt_s_f.begin(), in.sqrt_s_f.end());
    radial.insert(radial.end(), in.tcon.begin(), in.tcon.end());
    radial.insert(radial.end(), in.ard.begin(), in.ard.end());
    radial.insert(radial.end(), in.azd.begin(), in.azd.end());
    if (in.device_elements) radial.resize(6 * in.ns, 0.0F);
    std::vector<float> radial_lo;
    if (in.double_single) {
        radial_lo.assign(radial.size(), 0.0F);
        std::copy(in.sqrt_s_f_lo.begin(), in.sqrt_s_f_lo.end(),
                  radial_lo.begin());
    }
    const std::size_t output_values =
        (in.double_single ? 6 : 3) * points + in.ns;
    const auto geometry_bytes =
        GEOMETRY_PARITY_FIELD_COUNT * points * sizeof(float);
    const auto constraint_bytes = 4 * points * sizeof(float);
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
    auto readback = batched.batch ? wgpu::Buffer{}
                                  : make_buffer(device, output_bytes,
                                                wgpu::BufferUsage::CopyDst |
                                                    wgpu::BufferUsage::MapRead,
                                                "constraint head readback");
    auto params_buffer =
        make_buffer(device, sizeof(HeadParams),
                    wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
                    "constraint head params");
    wgpu::Buffer geometry_low_buffer, constraint_low_buffer, radial_low_buffer;
    if (in.double_single) {
        geometry_low_buffer =
            make_buffer(device, geometry_bytes,
                        wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                        "constraint geometry low");
        constraint_low_buffer =
            make_buffer(device, constraint_bytes,
                        wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                        "constraint state low");
        radial_low_buffer =
            make_buffer(device, radial_bytes,
                        wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                        "constraint radial data low");
    }
    const auto& pipeline = detail::cached_compute_pipeline(
        device,
        in.double_single ? "constraint-head-double-single" : "constraint-head",
        shader_text,
        in.double_single ? "cuMES double-single constraint head pipeline"
                         : "cuMES constraint head pipeline");
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
    auto encoder = device.CreateCommandEncoder();
    transfer_fields(device, encoder, geometry_buffer, in.geometry,
                    in.device_geometry);
    queue.WriteBuffer(radial_buffer, 0, radial.data(), radial_bytes);
    const auto copy_constraints = [&](const wgpu::Buffer& target, bool low) {
        transfer_constraint_plane(device, encoder, target, 0, points,
                                  low ? in.r_con_lo : in.r_con, in.device_r_con,
                                  low);
        transfer_constraint_plane(device, encoder, target, 1, points,
                                  low ? in.z_con_lo : in.z_con, in.device_z_con,
                                  low);
        transfer_constraint_plane(device, encoder, target, 2, points,
                                  low ? in.r_con0_lo : in.r_con0,
                                  in.device_r_con0, low);
        transfer_constraint_plane(device, encoder, target, 3, points,
                                  low ? in.z_con0_lo : in.z_con0,
                                  in.device_z_con0, low);
    };
    copy_constraints(constraint_buffer, false);
    if (in.device_elements) {
        encoder.CopyBufferToBuffer(in.device_elements.buffer,
                                   in.device_elements.high_offset,
                                   radial_buffer, 2 * in.ns * sizeof(float),
                                   2 * in.ns * sizeof(float));
        encoder.CopyBufferToBuffer(
            in.device_elements.buffer,
            in.device_elements.high_offset + 4 * in.ns * sizeof(float),
            radial_buffer, 4 * in.ns * sizeof(float),
            2 * in.ns * sizeof(float));
    }
    if (in.double_single) {
        transfer_fields(device, encoder, geometry_low_buffer, in.geometry_lo,
                        in.device_geometry, true);
        copy_constraints(constraint_low_buffer, true);
        queue.WriteBuffer(radial_low_buffer, 0, radial_lo.data(), radial_bytes);
    }
    queue.WriteBuffer(params_buffer, 0, &params, sizeof(params));
    auto layout = pipeline.GetBindGroupLayout(0);
    std::vector<wgpu::BindGroupEntry> entries = {
        {nullptr, 0, geometry_buffer, 0, geometry_bytes, nullptr, nullptr},
        {nullptr, 1, constraint_buffer, 0, constraint_bytes, nullptr, nullptr},
        {nullptr, 2, radial_buffer, 0, radial_bytes, nullptr, nullptr},
        {nullptr, 3, output_buffer, 0, output_bytes, nullptr, nullptr},
        {nullptr, 4, params_buffer, 0, sizeof(params), nullptr, nullptr}};
    if (in.double_single) {
        entries.push_back({nullptr, 5, geometry_low_buffer, 0, geometry_bytes,
                           nullptr, nullptr});
        entries.push_back({nullptr, 6, constraint_low_buffer, 0,
                           constraint_bytes, nullptr, nullptr});
        entries.push_back(
            {nullptr, 7, radial_low_buffer, 0, radial_bytes, nullptr, nullptr});
    }
    wgpu::BindGroupDescriptor bind_descriptor{};
    bind_descriptor.layout = layout;
    bind_descriptor.entryCount = entries.size();
    bind_descriptor.entries = entries.data();
    auto bind_group = device.CreateBindGroup(&bind_descriptor);
    wgpu::ComputePassDescriptor pass_descriptor{};
    auto pass = encoder.BeginComputePass(&pass_descriptor);
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bind_group);
    pass.DispatchWorkgroups(
        (static_cast<std::uint32_t>(points) + WORKGROUP_SIZE - 1) /
        WORKGROUP_SIZE);
    pass.End();
    HeadResult resident;
    const auto tcon_offset = (in.double_single ? 6 : 3) * points;
    if (batched.batch) {
        resident.device_fields = {output_buffer, 3 * points, 0,
                                  3 * points * sizeof(float)};
        resident.device_tcon = {output_buffer, static_cast<std::size_t>(in.ns),
                                tcon_offset * sizeof(float), 0};
        if (!in.readback_intermediates) {
            auto result = std::make_shared<HeadResult>(resident);
            batched.batch->append(
                encoder, output_buffer, tcon_offset * sizeof(float),
                in.ns * sizeof(float), [result](std::span<const float> values) {
                    result->tcon.assign(values.begin(), values.end());
                });
            const auto commands = encoder.Finish();
            queue.Submit(1, &commands);
            enqueue_field_finite(
                device, {output_buffer, tcon_offset, 0, 0}, batched.batch,
                [callback = std::move(callback), result](std::string error,
                                                         bool finite) {
                    result->finite = finite;
                    callback(std::move(error), std::move(*result));
                });
            batched.publish_device(std::move(resident));
            return;
        }
    }
    const auto host = batched.batch ? nullptr : std::make_shared<HeadResult>();
    const auto batch =
        batched.batch ? batched.batch
                      : std::make_shared<ReadbackBatch>(readback, output_bytes);
    batch->append(
        encoder, output_buffer, 0, output_bytes,
        [callback = batched.batch ? std::move(callback) : decltype(callback){},
         host, resident, points, tcon_offset,
         paired = in.double_single](std::span<const float> values) mutable {
            const auto hi = values.begin();
            resident.g_con_eff.assign(hi, hi + points);
            resident.r_con0.assign(hi + points, hi + 2 * points);
            resident.z_con0.assign(hi + 2 * points, hi + 3 * points);
            if (paired) {
                resident.g_con_eff_lo.assign(hi + 3 * points, hi + 4 * points);
                resident.r_con0_lo.assign(hi + 4 * points, hi + 5 * points);
                resident.z_con0_lo.assign(hi + 5 * points, hi + 6 * points);
            }
            resident.tcon.assign(hi + tcon_offset, values.end());
            if (host)
                *host = std::move(resident);
            else
                callback({}, std::move(resident));
        });
    const auto commands = encoder.Finish();
    queue.Submit(1, &commands);
    if (batched.batch) {
        batched.publish_device(std::move(resident));
        return;
    }
    batch->map([callback = std::move(callback), host](std::string error) {
        callback(std::move(error), std::move(*host));
    });
}

void enqueue_tail(const wgpu::Device& device,
                  const AxisymmetricConstraintCase& in,
                  HeadResult head,
                  std::vector<float> g_con,
                  AxisymmetricConstraintCallback callback,
                  DeviceFields device_g_con = {}) {
    const auto& shader_text = load_shader(
        in.double_single ? "/shaders/constraint_tail_double_single.wgsl"
                         : "/shaders/axisymmetric_constraint_tail.wgsl");
    if (shader_text.empty()) {
        callback("cannot load axisymmetric constraint-tail shader", {});
        return;
    }
    const std::size_t n_z_n_t = static_cast<std::size_t>(in.ntheta) * in.nzeta;
    const std::size_t points = static_cast<std::size_t>(in.ns) * n_z_n_t;
    const auto force_bytes =
        (in.device_force_fields ? in.device_force_fields.values
                                : in.force_fields.size()) *
        sizeof(float);
    const auto geometry_bytes =
        GEOMETRY_PARITY_FIELD_COUNT * points * sizeof(float);
    const auto constraint_bytes = 5 * points * sizeof(float);
    const auto radial_bytes = in.sqrt_s_f.size() * sizeof(float);
    const std::size_t force_fields = in.ntor == 0 ? 10 : FORCE_FIELD_COUNT;
    const std::size_t output_fields =
        in.ntor == 0 ? FORWARD_INPUT_FIELD_COUNT : TOROIDAL_FORWARD_FIELD_COUNT;
    const std::size_t output_values = output_fields * points;
    const auto output_bytes =
        output_values * sizeof(float) * (in.double_single ? 2 : 1);
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
    auto readback = !in.readback ? wgpu::Buffer{}
                                 : make_buffer(device, output_bytes,
                                               wgpu::BufferUsage::CopyDst |
                                                   wgpu::BufferUsage::MapRead,
                                               "constraint force readback");
    auto params_buffer =
        make_buffer(device, sizeof(TailParams),
                    wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
                    "constraint tail params");
    wgpu::Buffer force_low_buffer, geometry_low_buffer, constraint_low_buffer,
        radial_low_buffer;
    if (in.double_single) {
        force_low_buffer =
            make_buffer(device, force_bytes,
                        wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                        "constraint force input low");
        geometry_low_buffer =
            make_buffer(device, geometry_bytes,
                        wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                        "constraint tail geometry low");
        constraint_low_buffer =
            make_buffer(device, constraint_bytes,
                        wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                        "dealiased constraint low");
        radial_low_buffer =
            make_buffer(device, radial_bytes,
                        wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                        "constraint sqrt s low");
    }
    const auto& pipeline = detail::cached_compute_pipeline(
        device,
        in.double_single ? "constraint-tail-double-single" : "constraint-tail",
        shader_text,
        in.double_single ? "cuMES double-single constraint tail pipeline"
                         : "cuMES constraint tail pipeline");
    const TailParams params{static_cast<std::uint32_t>(in.ns),
                            static_cast<std::uint32_t>(n_z_n_t),
                            static_cast<std::uint32_t>(points),
                            static_cast<std::uint32_t>(force_fields),
                            static_cast<std::uint32_t>(output_fields),
                            {0, 0, 0}};
    auto queue = device.GetQueue();
    auto encoder = device.CreateCommandEncoder();
    transfer_fields(device, encoder, force_buffer, in.force_fields,
                    in.device_force_fields);
    if (in.double_single)
        transfer_fields(device, encoder, force_low_buffer, in.force_fields_lo,
                        in.device_force_fields, true);
    transfer_fields(device, encoder, geometry_buffer, in.geometry,
                    in.device_geometry);
    const auto copy_constraint_planes = [&](const wgpu::Buffer& target,
                                            bool low) {
        transfer_constraint_plane(device, encoder, target, 0, points,
                                  low ? in.r_con_lo : in.r_con, in.device_r_con,
                                  low);
        transfer_constraint_plane(device, encoder, target, 1, points,
                                  low ? in.z_con_lo : in.z_con, in.device_z_con,
                                  low);
        transfer_constraint_plane(
            device, encoder, target, 2, points,
            low ? head.r_con0_lo : head.r_con0,
            field_slice(head.device_fields, points, points), low);
        transfer_constraint_plane(
            device, encoder, target, 3, points,
            low ? head.z_con0_lo : head.z_con0,
            field_slice(head.device_fields, 2 * points, points), low);
        // The bandpass filter is f32; its paired low plane is exactly zero.
        if (low)
            encoder.ClearBuffer(target, 4 * points * sizeof(float),
                                points * sizeof(float));
        else
            transfer_constraint_plane(device, encoder, target, 4, points, g_con,
                                      device_g_con, false);
    };
    copy_constraint_planes(constraint_buffer, false);
    queue.WriteBuffer(radial_buffer, 0, in.sqrt_s_f.data(), radial_bytes);
    if (in.double_single) {
        transfer_fields(device, encoder, geometry_low_buffer, in.geometry_lo,
                        in.device_geometry, true);
        copy_constraint_planes(constraint_low_buffer, true);
        queue.WriteBuffer(radial_low_buffer, 0, in.sqrt_s_f_lo.data(),
                          radial_bytes);
    }
    queue.WriteBuffer(params_buffer, 0, &params, sizeof(params));
    auto layout = pipeline.GetBindGroupLayout(0);
    std::vector<wgpu::BindGroupEntry> entries = {
        {nullptr, 0, force_buffer, 0, force_bytes, nullptr, nullptr},
        {nullptr, 1, geometry_buffer, 0, geometry_bytes, nullptr, nullptr},
        {nullptr, 2, constraint_buffer, 0, constraint_bytes, nullptr, nullptr},
        {nullptr, 3, radial_buffer, 0, radial_bytes, nullptr, nullptr},
        {nullptr, 4, output_buffer, 0, output_bytes, nullptr, nullptr},
        {nullptr, 5, params_buffer, 0, sizeof(params), nullptr, nullptr}};
    if (in.double_single) {
        entries.push_back(
            {nullptr, 6, force_low_buffer, 0, force_bytes, nullptr, nullptr});
        entries.push_back({nullptr, 7, geometry_low_buffer, 0, geometry_bytes,
                           nullptr, nullptr});
        entries.push_back({nullptr, 8, constraint_low_buffer, 0,
                           constraint_bytes, nullptr, nullptr});
        entries.push_back(
            {nullptr, 9, radial_low_buffer, 0, radial_bytes, nullptr, nullptr});
    }
    wgpu::BindGroupDescriptor bind_descriptor{};
    bind_descriptor.layout = layout;
    bind_descriptor.entryCount = entries.size();
    bind_descriptor.entries = entries.data();
    auto bind_group = device.CreateBindGroup(&bind_descriptor);
    wgpu::ComputePassDescriptor pass_descriptor{};
    auto pass = encoder.BeginComputePass(&pass_descriptor);
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bind_group);
    pass.DispatchWorkgroups(
        (static_cast<std::uint32_t>(points) + WORKGROUP_SIZE - 1) /
        WORKGROUP_SIZE);
    pass.End();
    auto result = std::make_shared<AxisymmetricConstraintResult>();
    result->r_con0 = std::move(head.r_con0);
    result->r_con0_lo = std::move(head.r_con0_lo);
    result->z_con0 = std::move(head.z_con0);
    result->z_con0_lo = std::move(head.z_con0_lo);
    result->tcon = std::move(head.tcon);
    result->g_con_eff = std::move(head.g_con_eff);
    result->g_con_eff_lo = std::move(head.g_con_eff_lo);
    result->g_con = std::move(g_con);
    result->device_fields = {output_buffer, output_values, 0,
                             output_values * sizeof(float)};
    if (!in.readback) {
        const auto commands = encoder.Finish();
        queue.Submit(1, &commands);
        callback({}, std::move(*result));
        return;
    }
    const auto batch = std::make_shared<ReadbackBatch>(readback, output_bytes);
    batch->append(encoder, output_buffer, 0, output_bytes,
                  [result, output_values,
                   paired = in.double_single](std::span<const float> values) {
                      result->fields.assign(values.begin(),
                                            values.begin() + output_values);
                      if (paired)
                          result->fields_lo.assign(
                              values.begin() + output_values, values.end());
                  });
    const auto commands = encoder.Finish();
    queue.Submit(1, &commands);
    batch->map([callback = std::move(callback), result](std::string error) {
        if (!error.empty())
            callback(std::move(error), {});
        else
            callback({}, std::move(*result));
    });
}

struct ConstraintChain {
    wgpu::Device device;
    AxisymmetricConstraintCase input;
    AxisymmetricConstraintCallback callback;
};

template <typename DealiasCase>
void prepare_constraint_filter(DealiasCase& dealias,
                               const AxisymmetricConstraintCase& input) {
    dealias.ns = input.ns;
    dealias.mpol = input.mpol;
    dealias.ntheta = input.ntheta;
    if constexpr (requires { dealias.ntor; }) {
        dealias.ntor = input.ntor;
        dealias.nzeta = input.nzeta;
    }
    dealias.faccon.resize(input.mpol);
    fill_constraint_filter<float>(dealias.faccon);
}

}  // namespace

AxisymmetricConstraintResult axisymmetric_constraint_reference(
    const AxisymmetricConstraintCase& input) {
    const auto points =
        static_cast<std::size_t>(input.ns) * input.ntheta * input.nzeta;
    if (input.geometry.size() != GEOMETRY_PARITY_FIELD_COUNT * points ||
        input.r_con.size() != points || input.z_con.size() != points ||
        input.ard.size() != 2 * static_cast<std::size_t>(input.ns) ||
        input.azd.size() != 2 * static_cast<std::size_t>(input.ns) ||
        input.force_fields.size() !=
            (input.ntor == 0 ? 10 : FORCE_FIELD_COUNT) * points)
        return {};
    if (!validate_case(input).empty()) return {};
    auto head = head_reference(input);
    const auto filter = [&input, &head](auto dealias, auto reference) {
        prepare_constraint_filter(dealias, input);
        dealias.g_con_eff = head.g_con_eff;
        dealias.tcon = head.tcon;
        return reference(dealias).g_con;
    };
    auto g_con =
        input.ntor > 0
            ? filter(ToroidalDealiasCase{}, toroidal_dealias_reference)
            : filter(AxisymmetricDealiasCase{}, axisymmetric_dealias_reference);
    AxisymmetricConstraintResult result;
    result.r_con0 = head.r_con0;
    result.r_con0_lo = head.r_con0_lo;
    result.z_con0 = head.z_con0;
    result.z_con0_lo = head.z_con0_lo;
    result.tcon = head.tcon;
    result.g_con_eff = head.g_con_eff;
    result.g_con_eff_lo = head.g_con_eff_lo;
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
    if (input.batched_readback.batch) {
        if (input.readback) {
            chain->callback("batched constraint requires resident output", {});
            return;
        }
        auto result = std::make_shared<AxisymmetricConstraintResult>();
        BatchedReadback<HeadResult> head_readback;
        head_readback.batch = input.batched_readback.batch;
        head_readback.device_ready = [chain, result](HeadResult head) {
            const auto& in = chain->input;
            const auto points =
                static_cast<std::size_t>(in.ns) * in.ntheta * in.nzeta;
            result->device_r_con0 =
                field_slice(head.device_fields, points, points);
            result->device_z_con0 =
                field_slice(head.device_fields, 2 * points, points);
            const auto filter = [chain, result, head, points](auto dealias,
                                                              auto enqueue) {
                const auto& in = chain->input;
                prepare_constraint_filter(dealias, in);
                if constexpr (requires { dealias.readback_values; })
                    dealias.readback_values = in.readback_intermediates;
                dealias.device_g_con_eff =
                    field_slice(head.device_fields, 0, points);
                dealias.device_tcon = head.device_tcon;
                dealias.readback.batch = in.batched_readback.batch;
                dealias.readback.device_ready = [chain, result,
                                                 head](auto filtered) {
                    enqueue_tail(
                        chain->device, chain->input, head, {},
                        [chain, result](std::string error,
                                        AxisymmetricConstraintResult tail) {
                            if (!error.empty()) {
                                chain->callback(std::move(error), {});
                                return;
                            }
                            result->device_fields = tail.device_fields;
                            chain->input.batched_readback.publish_device(
                                std::move(tail));
                        },
                        filtered.device_g_con);
                };
                enqueue(chain->device, dealias,
                        [chain, result](std::string error, auto filtered) {
                            result->g_con = std::move(filtered.g_con);
                            result->intermediates_finite &= filtered.finite;
                            chain->callback(std::move(error),
                                            std::move(*result));
                        });
            };
            if (in.ntor == 0)
                filter(AxisymmetricDealiasCase{}, enqueue_axisymmetric_dealias);
            else
                filter(ToroidalDealiasCase{}, enqueue_toroidal_dealias);
        };
        enqueue_head(
            device, input,
            [chain, result](std::string error, HeadResult head) {
                if (!error.empty()) {
                    chain->callback(std::move(error), {});
                    return;
                }
                result->r_con0 = std::move(head.r_con0);
                result->intermediates_finite = head.finite;
                result->r_con0_lo = std::move(head.r_con0_lo);
                result->z_con0 = std::move(head.z_con0);
                result->z_con0_lo = std::move(head.z_con0_lo);
                result->tcon = std::move(head.tcon);
                result->g_con_eff = std::move(head.g_con_eff);
                result->g_con_eff_lo = std::move(head.g_con_eff_lo);
            },
            std::move(head_readback));
        return;
    }
    enqueue_head(
        device, chain->input, [chain](std::string error, HeadResult head) {
            if (!error.empty()) {
                chain->callback(std::move(error), {});
                return;
            }
            const auto filter = [chain, &head](auto dealias, auto enqueue) {
                prepare_constraint_filter(dealias, chain->input);
                dealias.g_con_eff = head.g_con_eff;
                dealias.tcon = head.tcon;
                enqueue(chain->device, dealias,
                        [chain, head = std::move(head)](
                            std::string filter_error, auto filtered) mutable {
                            if (!filter_error.empty()) {
                                chain->callback(std::move(filter_error), {});
                                return;
                            }
                            enqueue_tail(chain->device, chain->input,
                                         std::move(head),
                                         std::move(filtered.g_con),
                                         std::move(chain->callback));
                        });
            };
            if (chain->input.ntor > 0)
                filter(ToroidalDealiasCase{}, enqueue_toroidal_dealias);
            else
                filter(AxisymmetricDealiasCase{}, enqueue_axisymmetric_dealias);
        });
}

}  // namespace cumes::webgpu
