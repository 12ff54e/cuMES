#include "cumes/webgpu/geometry.hpp"

#include "cumes/webgpu/float_float.hpp"
#include "pipeline_cache.hpp"

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
constexpr std::size_t GEOMETRY_INPUT_FIELD_COUNT = 18;

struct ShaderParams {
    std::uint32_t ns;
    std::uint32_t n_z_n_t;
    std::uint32_t full_points;
    std::uint32_t half_points;
    float delta_s;
    float delta_s_lo;
    std::uint32_t padding[2];
};
static_assert(sizeof(ShaderParams) == 32);

std::string validate_case(const BaseGeometryCase& input) {
    if (input.ns < 2 || input.ntheta < 2 || input.ntheta % 2 != 0 ||
        input.nzeta < 1 || !std::isfinite(input.delta_s) ||
        input.delta_s <= 0.0F) {
        return "base geometry requires ns>=2, even ntheta>=2, nzeta>=1, and "
               "positive finite delta_s";
    }
    const std::size_t n_z_n_t =
        static_cast<std::size_t>(input.ntheta) * input.nzeta;
    const std::size_t full_points =
        static_cast<std::size_t>(input.ns) * n_z_n_t;
    const std::size_t half_points =
        static_cast<std::size_t>(input.ns - 1) * n_z_n_t;
    if (full_points > std::numeric_limits<std::uint32_t>::max() ||
        half_points > std::numeric_limits<std::uint32_t>::max()) {
        return "base geometry exceeds WebGPU indexing limits";
    }
    if (input.geometry.size() != GEOMETRY_INPUT_FIELD_COUNT * full_points ||
        (input.double_single &&
         input.geometry_lo.size() != input.geometry.size()) ||
        input.sqrt_s_f.size() != static_cast<std::size_t>(input.ns) ||
        input.sqrt_s_h.size() != static_cast<std::size_t>(input.ns - 1)) {
        return "base geometry input shape mismatch";
    }
    return {};
}

std::string read_shader(const char* path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) return {};
    std::ostringstream text;
    text << stream.rdbuf();
    return text.str();
}

std::string load_shader(bool double_single) {
    if (!double_single) return read_shader("/shaders/base_geometry.wgsl");
    const std::string prelude = read_shader("/shaders/float_float.wgsl");
    const std::string kernel =
        read_shader("/shaders/base_geometry_double_single.wgsl");
    if (prelude.empty() || kernel.empty()) return {};
    return prelude + '\n' + kernel;
}

wgpu::Buffer create_buffer(const wgpu::Device& device,
                           std::uint64_t size,
                           wgpu::BufferUsage usage,
                           const char* label) {
    return detail::cached_buffer(device, size, usage, label);
}

struct DispatchState {
    BaseGeometryCallback callback;
    wgpu::Buffer result_buffer;
    wgpu::Buffer readback_buffer;
    std::size_t result_values = 0;
    std::size_t result_bytes = 0;
    bool double_single = false;
};

}  // namespace

namespace {

BaseGeometryResult base_geometry_double_single_reference(
    const BaseGeometryCase& input) {
    const std::size_t angular_points =
        static_cast<std::size_t>(input.ntheta) * input.nzeta;
    const std::size_t full_points =
        static_cast<std::size_t>(input.ns) * angular_points;
    const std::size_t half_points =
        static_cast<std::size_t>(input.ns - 1) * angular_points;
    std::vector<double> values(BASE_GEOMETRY_FIELD_COUNT * half_points);
    const auto full = [&](std::size_t field, std::size_t point) {
        const std::size_t index = field * full_points + point;
        return static_cast<double>(input.geometry[index]) +
               input.geometry_lo[index];
    };
    const double delta_s = 1.0 / static_cast<double>(input.ns - 1);
    for (int surface = 0; surface < input.ns - 1; ++surface) {
        const double sqrt_i =
            std::sqrt(static_cast<double>(surface) * delta_s + 1.0e-12);
        const double sqrt_o =
            std::sqrt(static_cast<double>(surface + 1) * delta_s + 1.0e-12);
        const double sqrt_h =
            std::sqrt((static_cast<double>(surface) + 0.5) * delta_s);
        for (std::size_t angular = 0; angular < angular_points; ++angular) {
            const std::size_t point =
                static_cast<std::size_t>(surface) * angular_points + angular;
            const std::size_t inside = point;
            const std::size_t outside = point + angular_points;
            const double r12 =
                0.5 * ((full(0, inside) + full(0, outside)) +
                       sqrt_h * (full(6, inside) + full(6, outside)));
            const double ru12 =
                0.5 * ((full(3, inside) + full(3, outside)) +
                       sqrt_h * (full(9, inside) + full(9, outside)));
            const double zu12 =
                0.5 * ((full(4, inside) + full(4, outside)) +
                       sqrt_h * (full(10, inside) + full(10, outside)));
            const double rs = ((full(0, outside) - full(0, inside)) +
                               sqrt_h * (full(6, outside) - full(6, inside))) /
                              delta_s;
            const double zs = ((full(1, outside) - full(1, inside)) +
                               sqrt_h * (full(7, outside) - full(7, inside))) /
                              delta_s;
            const double tau1 = ru12 * zs - rs * zu12;
            const double tau2 = full(9, outside) * full(7, outside) +
                                full(9, inside) * full(7, inside) -
                                full(10, outside) * full(6, outside) -
                                full(10, inside) * full(6, inside) +
                                (full(3, outside) * full(7, outside) +
                                 full(3, inside) * full(7, inside) -
                                 full(4, outside) * full(6, outside) -
                                 full(4, inside) * full(6, inside)) /
                                    sqrt_h;
            const double tau = tau1 + 0.25 * tau2;
            const double gsqrt = tau * r12;
            const double sqrt_i_squared = sqrt_i * sqrt_i;
            const double sqrt_o_squared = sqrt_o * sqrt_o;
            const double guu =
                0.5 *
                    (full(3, inside) * full(3, inside) +
                     full(4, inside) * full(4, inside) +
                     full(3, outside) * full(3, outside) +
                     full(4, outside) * full(4, outside) +
                     sqrt_i_squared * (full(9, inside) * full(9, inside) +
                                       full(10, inside) * full(10, inside)) +
                     sqrt_o_squared * (full(9, outside) * full(9, outside) +
                                       full(10, outside) * full(10, outside))) +
                sqrt_h * (full(3, inside) * full(9, inside) +
                          full(4, inside) * full(10, inside) +
                          full(3, outside) * full(9, outside) +
                          full(4, outside) * full(10, outside));
            double gvv =
                0.5 * (full(0, inside) * full(0, inside) +
                       full(0, outside) * full(0, outside) +
                       sqrt_i_squared * full(6, inside) * full(6, inside) +
                       sqrt_o_squared * full(6, outside) * full(6, outside)) +
                sqrt_h * (full(0, inside) * full(6, inside) +
                          full(0, outside) * full(6, outside));
            const double guv =
                0.5 *
                (full(3, inside) * full(12, inside) +
                 full(4, inside) * full(13, inside) +
                 full(3, outside) * full(12, outside) +
                 full(4, outside) * full(13, outside) +
                 sqrt_i_squared * (full(9, inside) * full(15, inside) +
                                   full(10, inside) * full(16, inside)) +
                 sqrt_o_squared * (full(9, outside) * full(15, outside) +
                                   full(10, outside) * full(16, outside)) +
                 sqrt_h * (full(3, inside) * full(15, inside) +
                           full(4, inside) * full(16, inside) +
                           full(3, outside) * full(15, outside) +
                           full(4, outside) * full(16, outside) +
                           full(12, inside) * full(9, inside) +
                           full(13, inside) * full(10, inside) +
                           full(12, outside) * full(9, outside) +
                           full(13, outside) * full(10, outside)));
            gvv +=
                0.5 *
                    (full(12, inside) * full(12, inside) +
                     full(13, inside) * full(13, inside) +
                     full(12, outside) * full(12, outside) +
                     full(13, outside) * full(13, outside) +
                     sqrt_i_squared * (full(15, inside) * full(15, inside) +
                                       full(16, inside) * full(16, inside)) +
                     sqrt_o_squared * (full(15, outside) * full(15, outside) +
                                       full(16, outside) * full(16, outside))) +
                sqrt_h * (full(12, inside) * full(15, inside) +
                          full(13, inside) * full(16, inside) +
                          full(12, outside) * full(15, outside) +
                          full(13, outside) * full(16, outside));
            const std::array output{r12, ru12,  zu12, rs,  zs,
                                    tau, gsqrt, guu,  guv, gvv};
            for (std::size_t field = 0; field < output.size(); ++field) {
                values[field * half_points + point] = output[field];
            }
        }
    }
    BaseGeometryResult result;
    result.fields.resize(values.size());
    result.fields_lo.resize(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        const FloatFloat pair = split(values[i]);
        result.fields[i] = pair.hi;
        result.fields_lo[i] = pair.lo;
    }
    return result;
}

}  // namespace

BaseGeometryResult base_geometry_reference(const BaseGeometryCase& input) {
    if (!validate_case(input).empty()) return {};
    if (input.double_single) {
        return base_geometry_double_single_reference(input);
    }
    const std::size_t n_z_n_t =
        static_cast<std::size_t>(input.ntheta) * input.nzeta;
    const std::size_t full_points =
        static_cast<std::size_t>(input.ns) * n_z_n_t;
    const std::size_t half_points =
        static_cast<std::size_t>(input.ns - 1) * n_z_n_t;
    BaseGeometryResult result;
    result.fields.resize(BASE_GEOMETRY_FIELD_COUNT * half_points);
    const auto full = [&](std::size_t field, std::size_t point) {
        return input.geometry[field * full_points + point];
    };
    const auto store = [&](std::size_t field, std::size_t point, float value) {
        result.fields[field * half_points + point] = value;
    };

    for (int surface = 0; surface < input.ns - 1; ++surface) {
        const float sqrt_h = input.sqrt_s_h[surface];
        const float sqrt_i = input.sqrt_s_f[surface];
        const float sqrt_o = input.sqrt_s_f[surface + 1];
        const float sqrt_i_squared = sqrt_i * sqrt_i;
        const float sqrt_o_squared = sqrt_o * sqrt_o;
        for (std::size_t angular = 0; angular < n_z_n_t; ++angular) {
            const std::size_t point =
                static_cast<std::size_t>(surface) * n_z_n_t + angular;
            const std::size_t inside = point;
            const std::size_t outside = point + n_z_n_t;
            const float r12 =
                0.5F * ((full(0, inside) + full(0, outside)) +
                        sqrt_h * (full(6, inside) + full(6, outside)));
            const float ru12 =
                0.5F * ((full(3, inside) + full(3, outside)) +
                        sqrt_h * (full(9, inside) + full(9, outside)));
            const float zu12 =
                0.5F * ((full(4, inside) + full(4, outside)) +
                        sqrt_h * (full(10, inside) + full(10, outside)));
            const float rs = ((full(0, outside) - full(0, inside)) +
                              sqrt_h * (full(6, outside) - full(6, inside))) /
                             input.delta_s;
            const float zs = ((full(1, outside) - full(1, inside)) +
                              sqrt_h * (full(7, outside) - full(7, inside))) /
                             input.delta_s;
            const float tau1 = ru12 * zs - rs * zu12;
            const float tau2 = full(9, outside) * full(7, outside) +
                               full(9, inside) * full(7, inside) -
                               full(10, outside) * full(6, outside) -
                               full(10, inside) * full(6, inside) +
                               (full(3, outside) * full(7, outside) +
                                full(3, inside) * full(7, inside) -
                                full(4, outside) * full(6, outside) -
                                full(4, inside) * full(6, inside)) /
                                   sqrt_h;
            const float tau = tau1 + 0.25F * tau2;
            const float gsqrt = tau * r12;
            const float guu =
                0.5F *
                    (full(3, inside) * full(3, inside) +
                     full(4, inside) * full(4, inside) +
                     full(3, outside) * full(3, outside) +
                     full(4, outside) * full(4, outside) +
                     sqrt_i_squared * (full(9, inside) * full(9, inside) +
                                       full(10, inside) * full(10, inside)) +
                     sqrt_o_squared * (full(9, outside) * full(9, outside) +
                                       full(10, outside) * full(10, outside))) +
                sqrt_h * (full(3, inside) * full(9, inside) +
                          full(4, inside) * full(10, inside) +
                          full(3, outside) * full(9, outside) +
                          full(4, outside) * full(10, outside));
            float gvv =
                0.5F * (full(0, inside) * full(0, inside) +
                        full(0, outside) * full(0, outside) +
                        sqrt_i_squared * full(6, inside) * full(6, inside) +
                        sqrt_o_squared * full(6, outside) * full(6, outside)) +
                sqrt_h * (full(0, inside) * full(6, inside) +
                          full(0, outside) * full(6, outside));
            const float guv =
                0.5F *
                (full(3, inside) * full(12, inside) +
                 full(4, inside) * full(13, inside) +
                 full(3, outside) * full(12, outside) +
                 full(4, outside) * full(13, outside) +
                 sqrt_i_squared * (full(9, inside) * full(15, inside) +
                                   full(10, inside) * full(16, inside)) +
                 sqrt_o_squared * (full(9, outside) * full(15, outside) +
                                   full(10, outside) * full(16, outside)) +
                 sqrt_h * (full(3, inside) * full(15, inside) +
                           full(4, inside) * full(16, inside) +
                           full(3, outside) * full(15, outside) +
                           full(4, outside) * full(16, outside) +
                           full(12, inside) * full(9, inside) +
                           full(13, inside) * full(10, inside) +
                           full(12, outside) * full(9, outside) +
                           full(13, outside) * full(10, outside)));
            gvv +=
                0.5F *
                    (full(12, inside) * full(12, inside) +
                     full(13, inside) * full(13, inside) +
                     full(12, outside) * full(12, outside) +
                     full(13, outside) * full(13, outside) +
                     sqrt_i_squared * (full(15, inside) * full(15, inside) +
                                       full(16, inside) * full(16, inside)) +
                     sqrt_o_squared * (full(15, outside) * full(15, outside) +
                                       full(16, outside) * full(16, outside))) +
                sqrt_h * (full(12, inside) * full(15, inside) +
                          full(13, inside) * full(16, inside) +
                          full(12, outside) * full(15, outside) +
                          full(13, outside) * full(16, outside));
            store(0, point, r12);
            store(1, point, ru12);
            store(2, point, zu12);
            store(3, point, rs);
            store(4, point, zs);
            store(5, point, tau);
            store(6, point, gsqrt);
            store(7, point, guu);
            store(8, point, guv);
            store(9, point, gvv);
        }
    }
    return result;
}

void enqueue_base_geometry(const wgpu::Device& device,
                           const BaseGeometryCase& input,
                           BaseGeometryCallback callback) {
    const std::string validation_error = validate_case(input);
    if (!validation_error.empty()) {
        callback(validation_error, {});
        return;
    }
    const std::string shader_text = load_shader(input.double_single);
    if (shader_text.empty()) {
        callback("cannot load embedded /shaders/base_geometry.wgsl", {});
        return;
    }
    const std::size_t n_z_n_t =
        static_cast<std::size_t>(input.ntheta) * input.nzeta;
    const std::size_t full_points =
        static_cast<std::size_t>(input.ns) * n_z_n_t;
    const std::size_t half_points =
        static_cast<std::size_t>(input.ns - 1) * n_z_n_t;
    std::vector<float> radial;
    std::vector<float> radial_lo;
    if (input.double_single) {
        radial.resize(3 * static_cast<std::size_t>(input.ns) - 2);
        radial_lo.resize(radial.size());
        const double delta_s = 1.0 / static_cast<double>(input.ns - 1);
        const auto put_radial = [&](std::size_t index, double value) {
            const FloatFloat pair = split(value);
            radial[index] = pair.hi;
            radial_lo[index] = pair.lo;
        };
        for (int surface = 0; surface < input.ns; ++surface) {
            put_radial(
                surface,
                std::sqrt(static_cast<double>(surface) * delta_s + 1.0e-12));
        }
        for (int surface = 0; surface < input.ns - 1; ++surface) {
            const double sqrt_h =
                std::sqrt((static_cast<double>(surface) + 0.5) * delta_s);
            put_radial(input.ns + surface, sqrt_h);
            put_radial(2 * input.ns - 1 + surface, 1.0 / sqrt_h);
        }
    } else {
        radial.reserve(input.sqrt_s_f.size() + input.sqrt_s_h.size());
        radial.insert(radial.end(), input.sqrt_s_f.begin(),
                      input.sqrt_s_f.end());
        radial.insert(radial.end(), input.sqrt_s_h.begin(),
                      input.sqrt_s_h.end());
    }
    const std::size_t input_bytes = input.geometry.size() * sizeof(float);
    const std::size_t radial_bytes = radial.size() * sizeof(float);
    const std::size_t result_values = BASE_GEOMETRY_FIELD_COUNT * half_points;
    const std::size_t result_bytes =
        result_values * (input.double_single ? 2 : 1) * sizeof(float);
    const wgpu::Buffer input_buffer =
        create_buffer(device, input_bytes,
                      wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                      "cuMES full-grid geometry");
    const wgpu::Buffer radial_buffer =
        create_buffer(device, radial_bytes,
                      wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                      "cuMES geometry radial profiles");
    wgpu::Buffer input_lo_buffer;
    wgpu::Buffer radial_lo_buffer;
    wgpu::Buffer rounding_buffer;
    const std::size_t rounding_bytes = half_points * sizeof(std::uint32_t);
    if (input.double_single) {
        input_lo_buffer = create_buffer(
            device, input_bytes,
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
            "cuMES full-grid geometry low");
        radial_lo_buffer = create_buffer(
            device, radial_bytes,
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
            "cuMES geometry radial profiles low");
        rounding_buffer =
            create_buffer(device, rounding_bytes, wgpu::BufferUsage::Storage,
                          "cuMES double-single geometry rounding barriers");
    }
    const wgpu::Buffer result_buffer =
        create_buffer(device, result_bytes,
                      wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc,
                      "cuMES half-grid base geometry");
    const wgpu::Buffer readback_buffer =
        create_buffer(device, result_bytes,
                      wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead,
                      "cuMES half-grid geometry readback");
    const wgpu::Buffer params_buffer =
        create_buffer(device, sizeof(ShaderParams),
                      wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
                      "cuMES base geometry parameters");

    const char* pipeline_key =
        input.double_single ? "base-geometry-double-single" : "base-geometry";
    const char* pipeline_label =
        input.double_single ? "cuMES double-single base geometry pipeline"
                            : "cuMES base geometry pipeline";
    const auto& pipeline = detail::cached_compute_pipeline(
        device, pipeline_key, shader_text, pipeline_label);

    const FloatFloat delta_s =
        input.double_single ? split(1.0 / static_cast<double>(input.ns - 1))
                            : FloatFloat{input.delta_s, 0.0F};
    const ShaderParams params{static_cast<std::uint32_t>(input.ns),
                              static_cast<std::uint32_t>(n_z_n_t),
                              static_cast<std::uint32_t>(full_points),
                              static_cast<std::uint32_t>(half_points),
                              delta_s.hi,
                              delta_s.lo,
                              {0, 0}};
    const wgpu::Queue queue = device.GetQueue();
    queue.WriteBuffer(input_buffer, 0, input.geometry.data(), input_bytes);
    queue.WriteBuffer(radial_buffer, 0, radial.data(), radial_bytes);
    if (input.double_single) {
        queue.WriteBuffer(input_lo_buffer, 0, input.geometry_lo.data(),
                          input_bytes);
        queue.WriteBuffer(radial_lo_buffer, 0, radial_lo.data(), radial_bytes);
    }
    queue.WriteBuffer(params_buffer, 0, &params, sizeof(params));
    const wgpu::BindGroupLayout layout = pipeline.GetBindGroupLayout(0);
    std::vector<wgpu::BindGroupEntry> entries = {
        {nullptr, 0, input_buffer, 0, input_bytes, nullptr, nullptr},
        {nullptr, 1, radial_buffer, 0, radial_bytes, nullptr, nullptr},
        {nullptr, 2, result_buffer, 0, result_bytes, nullptr, nullptr},
        {nullptr, 3, params_buffer, 0, sizeof(params), nullptr, nullptr},
    };
    if (input.double_single) {
        entries.push_back(
            {nullptr, 4, input_lo_buffer, 0, input_bytes, nullptr, nullptr});
        entries.push_back(
            {nullptr, 5, radial_lo_buffer, 0, radial_bytes, nullptr, nullptr});
        entries.push_back(
            {nullptr, 6, rounding_buffer, 0, rounding_bytes, nullptr, nullptr});
    }
    wgpu::BindGroupDescriptor bind_group_descriptor{};
    bind_group_descriptor.label = "cuMES base geometry bindings";
    bind_group_descriptor.layout = layout;
    bind_group_descriptor.entryCount = entries.size();
    bind_group_descriptor.entries = entries.data();
    const wgpu::BindGroup bind_group =
        device.CreateBindGroup(&bind_group_descriptor);
    const wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    wgpu::ComputePassDescriptor pass_descriptor{};
    const wgpu::ComputePassEncoder pass =
        encoder.BeginComputePass(&pass_descriptor);
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bind_group);
    pass.DispatchWorkgroups(
        (static_cast<std::uint32_t>(half_points) + WORKGROUP_SIZE - 1) /
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
    dispatch->result_values = result_values;
    dispatch->result_bytes = result_bytes;
    dispatch->double_single = input.double_single;
    readback_buffer.MapAsync(
        wgpu::MapMode::Read, 0, result_bytes,
        wgpu::CallbackMode::AllowSpontaneous,
        [dispatch](wgpu::MapAsyncStatus status, wgpu::StringView message) {
            if (status != wgpu::MapAsyncStatus::Success) {
                const std::string detail =
                    message.length == 0
                        ? std::string{}
                        : std::string(message.data, message.length);
                dispatch->callback("WebGPU geometry mapping failed: " + detail,
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
            BaseGeometryResult result;
            result.fields.assign(values, values + dispatch->result_values);
            if (dispatch->double_single) {
                result.fields_lo.assign(values + dispatch->result_values,
                                        values + 2 * dispatch->result_values);
            }
            dispatch->readback_buffer.Unmap();
            dispatch->callback({}, std::move(result));
        });
}

}  // namespace cumes::webgpu
