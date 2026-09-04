#include "cumes/webgpu/geometry.hpp"
#include "pipeline_cache.hpp"

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
constexpr std::size_t FULL_GEOMETRY_FIELD_COUNT = 18;

struct ShaderParams {
    std::uint32_t ns;
    std::uint32_t n_z_n_t;
    std::uint32_t ntheta;
    std::uint32_t nzeta;
    std::uint32_t full_points;
    std::uint32_t half_points;
    std::uint32_t prescribed_current;
    std::uint32_t padding0;
    float lamscale;
    std::uint32_t padding[3];
};
static_assert(sizeof(ShaderParams) == 48);

std::string validate_case(const MagneticFieldCase& input) {
    if (input.ns < 2 || input.ntheta < 2 || input.ntheta % 2 != 0 ||
        input.nzeta < 1 || !std::isfinite(input.lamscale)) {
        return "magnetic field requires ns>=2, even ntheta>=2, nzeta>=1, and "
               "finite lamscale";
    }
    const std::size_t n_z_n_t =
        static_cast<std::size_t>(input.ntheta) * input.nzeta;
    const std::size_t full_points =
        static_cast<std::size_t>(input.ns) * n_z_n_t;
    const std::size_t half_points =
        static_cast<std::size_t>(input.ns - 1) * n_z_n_t;
    if (full_points > std::numeric_limits<std::uint32_t>::max() ||
        half_points > std::numeric_limits<std::uint32_t>::max()) {
        return "magnetic field exceeds WebGPU indexing limits";
    }
    const auto half_surfaces = static_cast<std::size_t>(input.ns - 1);
    if (input.geometry.size() != FULL_GEOMETRY_FIELD_COUNT * full_points ||
        input.base_geometry.size() != BASE_GEOMETRY_FIELD_COUNT * half_points ||
        input.sqrt_s_h.size() != half_surfaces ||
        input.phip_f.size() != static_cast<std::size_t>(input.ns) ||
        input.chip_h.size() != half_surfaces ||
        input.pres_h.size() != half_surfaces ||
        input.iota_h.size() != half_surfaces ||
        (input.prescribed_current && (input.curr_h.size() != half_surfaces ||
                                      input.phip_h.size() != half_surfaces))) {
        return "magnetic field input shape mismatch";
    }
    return {};
}

std::string load_shader() {
    std::ifstream stream("/shaders/magnetic_field.wgsl", std::ios::binary);
    if (!stream) return {};
    std::ostringstream text;
    text << stream.rdbuf();
    return text.str();
}

wgpu::Buffer create_buffer(const wgpu::Device& device,
                           std::uint64_t size,
                           wgpu::BufferUsage usage,
                           const char* label) {
    return detail::cached_buffer(device, size, usage, label);
}

struct DispatchState {
    MagneticFieldCallback callback;
    wgpu::Buffer result_buffer;
    wgpu::Buffer readback_buffer;
    std::size_t result_values = 0;
    std::size_t result_bytes = 0;
    std::size_t half_surfaces = 0;
};

}  // namespace

MagneticFieldResult magnetic_field_reference(const MagneticFieldCase& input) {
    if (!validate_case(input).empty()) return {};
    const std::size_t n_z_n_t =
        static_cast<std::size_t>(input.ntheta) * input.nzeta;
    const std::size_t full_points =
        static_cast<std::size_t>(input.ns) * n_z_n_t;
    const std::size_t half_points =
        static_cast<std::size_t>(input.ns - 1) * n_z_n_t;
    MagneticFieldResult result;
    result.fields.resize(MAGNETIC_FIELD_COUNT * half_points);
    result.chip_h = input.chip_h;
    result.iota_h = input.iota_h;
    const auto full = [&](std::size_t field, std::size_t point) {
        return input.geometry[field * full_points + point];
    };
    const auto half = [&](std::size_t field, std::size_t point) {
        return input.base_geometry[field * half_points + point];
    };
    const auto store = [&](std::size_t field, std::size_t point, float value) {
        result.fields[field * half_points + point] = value;
    };
    for (int surface = 0; surface < input.ns - 1; ++surface) {
        const float sqrt_h = input.sqrt_s_h[surface];
        const float phip_average =
            0.5F * (input.phip_f[surface] + input.phip_f[surface + 1]);
        for (std::size_t angular = 0; angular < n_z_n_t; ++angular) {
            const std::size_t point =
                static_cast<std::size_t>(surface) * n_z_n_t + angular;
            const std::size_t inside = point;
            const std::size_t outside = point + n_z_n_t;
            const float lu_h =
                0.5F * ((full(5, inside) + full(5, outside)) +
                        sqrt_h * (full(11, inside) + full(11, outside)));
            const float lv_h =
                0.5F * ((full(14, inside) + full(14, outside)) +
                        sqrt_h * (full(17, inside) + full(17, outside)));
            const float gsqrt = half(6, point);
            float bsupu = 0.0F;
            float bsupv = 0.0F;
            if (std::isfinite(gsqrt) && std::abs(gsqrt) > 1.0e-30F) {
                bsupv = (input.lamscale * lu_h + phip_average) / gsqrt;
                bsupu = input.lamscale * lv_h / gsqrt;
                if (!input.prescribed_current) {
                    bsupu += input.chip_h[surface] / gsqrt;
                }
            }
            store(0, point, bsupu);
            store(1, point, bsupv);
            if (!input.prescribed_current) {
                const float bsubu =
                    half(7, point) * bsupu + half(8, point) * bsupv;
                const float bsubv =
                    half(8, point) * bsupu + half(9, point) * bsupv;
                const float magnetic_pressure =
                    0.5F * (bsupu * bsubu + bsupv * bsubv);
                store(2, point, bsubu);
                store(3, point, bsubv);
                store(4, point, magnetic_pressure + input.pres_h[surface]);
            }
        }
    }
    if (input.prescribed_current) {
        const int reduced_ntheta = input.ntheta / 2 + 1;
        const float normalization =
            1.0F / static_cast<float>(input.nzeta * (reduced_ntheta - 1));
        for (int surface = 0; surface < input.ns - 1; ++surface) {
            float jv = 0.0F;
            float average = 0.0F;
            const std::size_t base_point =
                static_cast<std::size_t>(surface) * n_z_n_t;
            for (int izeta = 0; izeta < input.nzeta; ++izeta) {
                for (int itheta = 0; itheta < reduced_ntheta; ++itheta) {
                    const std::size_t point =
                        base_point +
                        static_cast<std::size_t>(izeta) * input.ntheta + itheta;
                    const float weight =
                        normalization *
                        ((itheta == 0 || itheta == reduced_ntheta - 1) ? 0.5F
                                                                       : 1.0F);
                    const float gsqrt = half(6, point);
                    jv +=
                        (half(7, point) * result.fields[point] +
                         half(8, point) * result.fields[half_points + point]) *
                        weight;
                    if (std::isfinite(gsqrt) && std::abs(gsqrt) > 1.0e-30F) {
                        average += half(7, point) / gsqrt * weight;
                    }
                }
            }
            const float chip =
                average == 0.0F ? 0.0F : (input.curr_h[surface] - jv) / average;
            result.chip_h[surface] = chip;
            if (input.phip_h[surface] != 0.0F) {
                result.iota_h[surface] = chip / input.phip_h[surface];
            }
            for (std::size_t angular = 0; angular < n_z_n_t; ++angular) {
                const std::size_t point = base_point + angular;
                const float gsqrt = half(6, point);
                float bsupu = result.fields[point];
                const float bsupv = result.fields[half_points + point];
                if (std::isfinite(gsqrt) && std::abs(gsqrt) > 1.0e-30F) {
                    bsupu += chip / gsqrt;
                }
                const float bsubu =
                    half(7, point) * bsupu + half(8, point) * bsupv;
                const float bsubv =
                    half(8, point) * bsupu + half(9, point) * bsupv;
                result.fields[point] = bsupu;
                store(2, point, bsubu);
                store(3, point, bsubv);
                store(4, point,
                      0.5F * (bsupu * bsubu + bsupv * bsubv) +
                          input.pres_h[surface]);
            }
        }
    }
    return result;
}

void enqueue_magnetic_field(const wgpu::Device& device,
                            const MagneticFieldCase& input,
                            MagneticFieldCallback callback) {
    const std::string validation_error = validate_case(input);
    if (!validation_error.empty()) {
        callback(validation_error, {});
        return;
    }
    const std::string shader_text = load_shader();
    if (shader_text.empty()) {
        callback("cannot load embedded /shaders/magnetic_field.wgsl", {});
        return;
    }
    const std::size_t n_z_n_t =
        static_cast<std::size_t>(input.ntheta) * input.nzeta;
    const std::size_t full_points =
        static_cast<std::size_t>(input.ns) * n_z_n_t;
    const std::size_t half_points =
        static_cast<std::size_t>(input.ns - 1) * n_z_n_t;
    std::vector<float> profiles;
    const std::size_t half_surfaces = static_cast<std::size_t>(input.ns - 1);
    profiles.reserve(input.phip_f.size() + 6 * half_surfaces);
    profiles.insert(profiles.end(), input.sqrt_s_h.begin(),
                    input.sqrt_s_h.end());
    profiles.insert(profiles.end(), input.phip_f.begin(), input.phip_f.end());
    profiles.insert(profiles.end(), input.chip_h.begin(), input.chip_h.end());
    profiles.insert(profiles.end(), input.pres_h.begin(), input.pres_h.end());
    if (input.curr_h.empty()) {
        profiles.insert(profiles.end(), half_surfaces, 0.0F);
    } else {
        profiles.insert(profiles.end(), input.curr_h.begin(),
                        input.curr_h.end());
    }
    if (input.phip_h.empty()) {
        profiles.insert(profiles.end(), half_surfaces, 0.0F);
    } else {
        profiles.insert(profiles.end(), input.phip_h.begin(),
                        input.phip_h.end());
    }
    profiles.insert(profiles.end(), input.iota_h.begin(), input.iota_h.end());
    const std::size_t geometry_bytes = input.geometry.size() * sizeof(float);
    const std::size_t base_bytes = input.base_geometry.size() * sizeof(float);
    const std::size_t profile_bytes = profiles.size() * sizeof(float);
    const std::size_t field_values = MAGNETIC_FIELD_COUNT * half_points;
    const std::size_t result_values = field_values + 2 * half_surfaces;
    const std::size_t result_bytes = result_values * sizeof(float);
    const auto geometry_buffer =
        create_buffer(device, geometry_bytes,
                      wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                      "cuMES magnetic full geometry");
    const auto base_buffer =
        create_buffer(device, base_bytes,
                      wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                      "cuMES magnetic base geometry");
    const auto profile_buffer =
        create_buffer(device, profile_bytes,
                      wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                      "cuMES magnetic profiles");
    const auto result_buffer =
        create_buffer(device, result_bytes,
                      wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc,
                      "cuMES magnetic field");
    const auto readback_buffer =
        create_buffer(device, result_bytes,
                      wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead,
                      "cuMES magnetic field readback");
    const auto params_buffer =
        create_buffer(device, sizeof(ShaderParams),
                      wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
                      "cuMES magnetic field parameters");

    const auto& pipeline = detail::cached_compute_pipeline(
        device, "magnetic-field", shader_text, "cuMES magnetic field pipeline");
    const auto& finalize_pipeline = detail::cached_compute_pipeline(
        device, "magnetic-field-finalize-current", shader_text,
        "cuMES prescribed-current finalize pipeline", "finalize_current");
    const ShaderParams params{static_cast<std::uint32_t>(input.ns),
                              static_cast<std::uint32_t>(n_z_n_t),
                              static_cast<std::uint32_t>(input.ntheta),
                              static_cast<std::uint32_t>(input.nzeta),
                              static_cast<std::uint32_t>(full_points),
                              static_cast<std::uint32_t>(half_points),
                              input.prescribed_current ? 1U : 0U,
                              0U,
                              input.lamscale,
                              {0, 0, 0}};
    const auto queue = device.GetQueue();
    queue.WriteBuffer(geometry_buffer, 0, input.geometry.data(),
                      geometry_bytes);
    queue.WriteBuffer(base_buffer, 0, input.base_geometry.data(), base_bytes);
    queue.WriteBuffer(profile_buffer, 0, profiles.data(), profile_bytes);
    queue.WriteBuffer(params_buffer, 0, &params, sizeof(params));
    const auto layout = pipeline.GetBindGroupLayout(0);
    const wgpu::BindGroupEntry entries[] = {
        {nullptr, 0, geometry_buffer, 0, geometry_bytes, nullptr, nullptr},
        {nullptr, 1, base_buffer, 0, base_bytes, nullptr, nullptr},
        {nullptr, 2, profile_buffer, 0, profile_bytes, nullptr, nullptr},
        {nullptr, 3, result_buffer, 0, result_bytes, nullptr, nullptr},
        {nullptr, 4, params_buffer, 0, sizeof(params), nullptr, nullptr},
    };
    wgpu::BindGroupDescriptor bind_group_descriptor{};
    bind_group_descriptor.label = "cuMES magnetic field bindings";
    bind_group_descriptor.layout = layout;
    bind_group_descriptor.entryCount = std::size(entries);
    bind_group_descriptor.entries = entries;
    const auto bind_group = device.CreateBindGroup(&bind_group_descriptor);
    const auto finalize_layout = finalize_pipeline.GetBindGroupLayout(0);
    const wgpu::BindGroupEntry finalize_entries[] = {
        {nullptr, 1, base_buffer, 0, base_bytes, nullptr, nullptr},
        {nullptr, 2, profile_buffer, 0, profile_bytes, nullptr, nullptr},
        {nullptr, 3, result_buffer, 0, result_bytes, nullptr, nullptr},
        {nullptr, 4, params_buffer, 0, sizeof(params), nullptr, nullptr},
    };
    bind_group_descriptor.label = "cuMES prescribed-current finalize bindings";
    bind_group_descriptor.layout = finalize_layout;
    bind_group_descriptor.entryCount = std::size(finalize_entries);
    bind_group_descriptor.entries = finalize_entries;
    const auto finalize_bind_group =
        device.CreateBindGroup(&bind_group_descriptor);
    const auto encoder = device.CreateCommandEncoder();
    wgpu::ComputePassDescriptor pass_descriptor{};
    const auto pass = encoder.BeginComputePass(&pass_descriptor);
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bind_group);
    pass.DispatchWorkgroups(
        (static_cast<std::uint32_t>(half_points) + WORKGROUP_SIZE - 1) /
        WORKGROUP_SIZE);
    pass.SetPipeline(finalize_pipeline);
    pass.SetBindGroup(0, finalize_bind_group);
    pass.DispatchWorkgroups((static_cast<std::uint32_t>(half_surfaces) + 63U) /
                            64U);
    pass.End();
    encoder.CopyBufferToBuffer(result_buffer, 0, readback_buffer, 0,
                               result_bytes);
    const auto commands = encoder.Finish();
    queue.Submit(1, &commands);
    auto dispatch = std::make_shared<DispatchState>();
    dispatch->callback = std::move(callback);
    dispatch->result_buffer = result_buffer;
    dispatch->readback_buffer = readback_buffer;
    dispatch->result_values = result_values;
    dispatch->result_bytes = result_bytes;
    dispatch->half_surfaces = half_surfaces;
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
                    "WebGPU magnetic field mapping failed: " + detail, {});
                return;
            }
            const void* mapped = dispatch->readback_buffer.GetConstMappedRange(
                0, dispatch->result_bytes);
            if (mapped == nullptr) {
                dispatch->callback("WebGPU returned a null mapped range", {});
                return;
            }
            const auto* values = static_cast<const float*>(mapped);
            MagneticFieldResult result;
            const std::size_t half_surfaces = dispatch->half_surfaces;
            const std::size_t profile_values = 2 * half_surfaces;
            const std::size_t field_values =
                dispatch->result_values - profile_values;
            result.fields.assign(values, values + field_values);
            result.chip_h.assign(values + field_values,
                                 values + field_values + half_surfaces);
            result.iota_h.assign(values + field_values + half_surfaces,
                                 values + dispatch->result_values);
            dispatch->readback_buffer.Unmap();
            dispatch->callback({}, std::move(result));
        });
}

}  // namespace cumes::webgpu
