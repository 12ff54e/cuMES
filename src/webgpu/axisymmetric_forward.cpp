#include "cumes/webgpu/axisymmetric.hpp"
#include "cumes/webgpu/toroidal.hpp"
#include "pipeline_cache.hpp"
#include "shader_source.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <numbers>
#include <utility>

namespace cumes::webgpu {
namespace {

constexpr std::uint32_t WORKGROUP_SIZE = 256;

struct ShaderParams {
    std::uint32_t ns;
    std::uint32_t mpol;
    std::uint32_t ntheta;
    std::uint32_t points;
    std::uint32_t flag;
    std::uint32_t padding[3];
};
static_assert(sizeof(ShaderParams) == 32);

std::string validate_forward_case(const AxisymmetricForwardCase& input) {
    if (input.ns < 2 || input.mpol <= 0 || input.ntheta < 2 ||
        input.ntheta % 2 != 0) {
        return "axisymmetric forward requires ns>=2, mpol>0, and even "
               "ntheta>=2";
    }
    const auto points = static_cast<std::size_t>(input.ns) * input.ntheta;
    if (points > std::numeric_limits<std::uint32_t>::max() ||
        points > std::numeric_limits<std::size_t>::max() /
                     FORWARD_INPUT_FIELD_COUNT) {
        return "axisymmetric forward input exceeds WebGPU indexing limits";
    }
    if (input.fields.size() != FORWARD_INPUT_FIELD_COUNT * points) {
        return "axisymmetric forward field size does not match 14*ns*ntheta";
    }
    return {};
}

std::string validate_dealias_case(const AxisymmetricDealiasCase& input) {
    if (input.ns < 2 || input.mpol <= 0 || input.ntheta < 2 ||
        input.ntheta % 2 != 0) {
        return "axisymmetric dealias requires ns>=2, mpol>0, and even "
               "ntheta>=2";
    }
    const auto points = static_cast<std::size_t>(input.ns) * input.ntheta;
    if (points > std::numeric_limits<std::uint32_t>::max()) {
        return "axisymmetric dealias input exceeds WebGPU indexing limits";
    }
    if (!field_shape(input.g_con_eff, input.device_g_con_eff, points) ||
        !field_shape(input.tcon, input.device_tcon,
                     static_cast<std::size_t>(input.ns)) ||
        input.faccon.size() != static_cast<std::size_t>(input.mpol)) {
        return "axisymmetric dealias input shape mismatch";
    }
    return {};
}

const std::string& load_shader(const char* path) {
    return detail::cached_shader_source(path);
}

wgpu::Buffer create_buffer(const wgpu::Device& device,
                           std::uint64_t size,
                           wgpu::BufferUsage usage,
                           const char* label) {
    return detail::cached_buffer(device, size, usage, label);
}

std::vector<float> make_forward_basis(int mpol, int ntheta) {
    std::vector<float> basis(4 * static_cast<std::size_t>(mpol) * ntheta);
    const auto table_size = static_cast<std::size_t>(mpol) * ntheta;
    for (int mode = 0; mode < mpol; ++mode) {
        for (int theta_index = 0; theta_index < ntheta; ++theta_index) {
            const float theta = 2.0F * std::numbers::pi_v<float> *
                                static_cast<float>(theta_index) /
                                static_cast<float>(ntheta);
            const float mf = static_cast<float>(mode);
            const float cosine = std::cos(mf * theta);
            const float sine = std::sin(mf * theta);
            const auto offset =
                static_cast<std::size_t>(mode) * ntheta + theta_index;
            basis[offset] = cosine;
            basis[table_size + offset] = sine;
            basis[2 * table_size + offset] = mf * cosine;
            basis[3 * table_size + offset] = -mf * sine;
        }
    }
    return basis;
}

std::vector<float> make_forward_weights(int ntheta) {
    const int reduced_theta = ntheta / 2 + 1;
    const float norm = 1.0F / static_cast<float>(reduced_theta - 1);
    std::vector<float> weights(reduced_theta, norm);
    weights.front() *= 0.5F;
    weights.back() *= 0.5F;
    return weights;
}

}  // namespace

AxisymmetricForwardResult axisymmetric_forward_reference(
    const AxisymmetricForwardCase& input) {
    if (!validate_forward_case(input).empty()) return {};

    const std::size_t points =
        static_cast<std::size_t>(input.ns) * input.ntheta;
    AxisymmetricForwardResult result;
    result.residual.assign(SPECTRAL_COMPONENT_COUNT *
                               static_cast<std::size_t>(input.mpol) * input.ns,
                           0.0F);
    const auto basis = make_forward_basis(input.mpol, input.ntheta);
    const auto weights = make_forward_weights(input.ntheta);
    const auto table_size = static_cast<std::size_t>(input.mpol) * input.ntheta;
    const auto field = [&](std::size_t index, int surface, int theta) {
        return input
            .fields[index * points +
                    static_cast<std::size_t>(surface) * input.ntheta + theta];
    };
    const auto store = [&](int component, int mode, int surface, float value) {
        result.residual[(static_cast<std::size_t>(component) * input.mpol +
                         mode) *
                            input.ns +
                        surface] = value;
    };

    for (int surface = 0; surface < input.ns; ++surface) {
        for (int mode = 0; mode < input.mpol; ++mode) {
            const std::size_t parity = mode % 2 == 0 ? 0 : 1;
            const float mf = static_cast<float>(mode);
            const float xmpq = mf * (mf - 1.0F);
            float r_value = 0.0F;
            float z_value = 0.0F;
            float l_value = 0.0F;
            for (int theta = 0; theta <= input.ntheta / 2; ++theta) {
                const auto basis_offset =
                    static_cast<std::size_t>(mode) * input.ntheta + theta;
                const float weight = weights[theta];
                const float cosine = weight * basis[basis_offset];
                const float sine = weight * basis[table_size + basis_offset];
                const float mcosine =
                    weight * basis[2 * table_size + basis_offset];
                const float msine =
                    weight * basis[3 * table_size + basis_offset];
                const float force_r = field(parity, surface, theta) +
                                      xmpq * field(10 + parity, surface, theta);
                const float force_z = field(2 + parity, surface, theta) +
                                      xmpq * field(12 + parity, surface, theta);
                r_value += force_r * cosine +
                           field(4 + parity, surface, theta) * msine;
                z_value += force_z * sine +
                           field(6 + parity, surface, theta) * mcosine;
                l_value += field(8 + parity, surface, theta) * mcosine;
            }
            const float mode_scale = mode == 0 ? 1.0F : std::sqrt(2.0F);
            float r_output = mode_scale * r_value;
            float z_output = mode_scale * z_value;
            float l_output = mode_scale * l_value;
            if (surface == 0) {
                r_output = mode == 0 ? r_output : 0.0F;
                z_output = 0.0F;
                l_output = 0.0F;
            } else if (surface == input.ns - 1 && !input.include_lcfs) {
                r_output = 0.0F;
                z_output = 0.0F;
            }
            store(0, mode, surface, r_output);
            store(1, mode, surface, z_output);
            store(2, mode, surface, l_output);
        }
    }
    return result;
}

void enqueue_axisymmetric_forward(const wgpu::Device& device,
                                  const AxisymmetricForwardCase& input,
                                  AxisymmetricForwardCallback callback) {
    const std::string validation_error = validate_forward_case(input);
    if (!validation_error.empty()) {
        callback(validation_error, {});
        return;
    }
    ToroidalForwardCase shared;
    shared.ns = input.ns;
    shared.mpol = input.mpol;
    shared.ntheta = input.ntheta;
    shared.nzeta = 1;
    shared.nfp = 1;
    shared.include_lcfs = input.include_lcfs;
    const auto points = static_cast<std::size_t>(input.ns) * input.ntheta;
    shared.fields.resize(TOROIDAL_FORWARD_FIELD_COUNT * points, 0.0F);
    std::copy_n(input.fields.begin(), 10 * points, shared.fields.begin());
    std::copy_n(input.fields.begin() + 10 * points, 4 * points,
                shared.fields.begin() + 16 * points);
    enqueue_toroidal_forward(
        device, shared,
        [callback = std::move(callback)](std::string error,
                                         ToroidalForwardResult result) {
            callback(std::move(error), {std::move(result.residual)});
        });
}

AxisymmetricDealiasResult axisymmetric_dealias_reference(
    const AxisymmetricDealiasCase& input) {
    if (!validate_dealias_case(input).empty()) return {};

    const std::size_t points =
        static_cast<std::size_t>(input.ns) * input.ntheta;
    AxisymmetricDealiasResult result;
    result.g_con.assign(points, 0.0F);
    const auto basis = make_forward_basis(input.mpol, input.ntheta);
    const std::size_t table_size =
        static_cast<std::size_t>(input.mpol) * input.ntheta;
    const float norm = 2.0F / static_cast<float>(input.ntheta);
    for (int surface = 1; surface < input.ns; ++surface) {
        for (int theta = 0; theta < input.ntheta; ++theta) {
            float value = 0.0F;
            for (int mode = 1; mode <= input.mpol - 2; ++mode) {
                float coefficient = 0.0F;
                for (int source_theta = 0; source_theta < input.ntheta;
                     ++source_theta) {
                    coefficient +=
                        input.g_con_eff[static_cast<std::size_t>(surface) *
                                            input.ntheta +
                                        source_theta] *
                        basis[table_size +
                              static_cast<std::size_t>(mode) * input.ntheta +
                              source_theta];
                }
                value += norm * input.tcon[surface] * input.faccon[mode] *
                         coefficient *
                         basis[table_size +
                               static_cast<std::size_t>(mode) * input.ntheta +
                               theta];
            }
            result.g_con[static_cast<std::size_t>(surface) * input.ntheta +
                         theta] = value;
        }
    }
    return result;
}

void enqueue_axisymmetric_dealias(const wgpu::Device& device,
                                  const AxisymmetricDealiasCase& input,
                                  AxisymmetricDealiasCallback callback) {
    const std::string validation_error = validate_dealias_case(input);
    if (!validation_error.empty()) {
        callback(validation_error, {});
        return;
    }
    const auto& shader_text = load_shader("/shaders/axisymmetric_dealias.wgsl");
    if (shader_text.empty()) {
        callback("cannot load embedded /shaders/axisymmetric_dealias.wgsl", {});
        return;
    }

    const std::size_t points =
        static_cast<std::size_t>(input.ns) * input.ntheta;
    const auto forward_basis = make_forward_basis(input.mpol, input.ntheta);
    const std::size_t table_size =
        static_cast<std::size_t>(input.mpol) * input.ntheta;
    std::vector<float> sine_basis(forward_basis.begin() + table_size,
                                  forward_basis.begin() + 2 * table_size);
    std::vector<float> profiles(input.ns, 0.0F);
    if (!input.device_tcon)
        std::copy(input.tcon.begin(), input.tcon.end(), profiles.begin());
    profiles.insert(profiles.end(), input.faccon.begin(), input.faccon.end());
    const std::size_t input_bytes = points * sizeof(float);
    const std::size_t profile_bytes = profiles.size() * sizeof(float);
    const std::size_t basis_bytes = sine_basis.size() * sizeof(float);
    const std::size_t result_bytes = points * sizeof(float);
    const wgpu::Buffer input_buffer =
        create_buffer(device, input_bytes,
                      wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                      "cuMES axisymmetric effective constraint");
    const wgpu::Buffer profile_buffer =
        create_buffer(device, profile_bytes,
                      wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                      "cuMES axisymmetric constraint profiles");
    const wgpu::Buffer basis_buffer =
        create_buffer(device, basis_bytes,
                      wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                      "cuMES axisymmetric sine basis");
    const wgpu::Buffer result_buffer =
        create_buffer(device, result_bytes,
                      wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc,
                      "cuMES axisymmetric dealiased constraint");
    const wgpu::Buffer readback_buffer =
        input.readback.batch
            ? wgpu::Buffer{}
            : create_buffer(
                  device, result_bytes,
                  wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead,
                  "cuMES axisymmetric dealiased constraint readback");
    const wgpu::Buffer params_buffer =
        create_buffer(device, sizeof(ShaderParams),
                      wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
                      "cuMES axisymmetric dealias parameters");

    const auto& pipeline = detail::cached_compute_pipeline(
        device, "axisymmetric-dealias", shader_text,
        "cuMES axisymmetric dealias pipeline");

    const ShaderParams params{static_cast<std::uint32_t>(input.ns),
                              static_cast<std::uint32_t>(input.mpol),
                              static_cast<std::uint32_t>(input.ntheta),
                              static_cast<std::uint32_t>(points),
                              0,
                              {0, 0, 0}};
    const wgpu::Queue queue = device.GetQueue();
    const wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    transfer_fields(device, encoder, input_buffer, input.g_con_eff,
                    input.device_g_con_eff);
    queue.WriteBuffer(profile_buffer, 0, profiles.data(), profile_bytes);
    if (input.device_tcon)
        encoder.CopyBufferToBuffer(input.device_tcon.buffer,
                                   input.device_tcon.high_offset,
                                   profile_buffer, 0, input.ns * sizeof(float));
    queue.WriteBuffer(basis_buffer, 0, sine_basis.data(), basis_bytes);
    queue.WriteBuffer(params_buffer, 0, &params, sizeof(params));

    const wgpu::BindGroupLayout layout = pipeline.GetBindGroupLayout(0);
    const wgpu::BindGroupEntry entries[] = {
        {nullptr, 0, input_buffer, 0, input_bytes, nullptr, nullptr},
        {nullptr, 1, profile_buffer, 0, profile_bytes, nullptr, nullptr},
        {nullptr, 2, basis_buffer, 0, basis_bytes, nullptr, nullptr},
        {nullptr, 3, result_buffer, 0, result_bytes, nullptr, nullptr},
        {nullptr, 4, params_buffer, 0, sizeof(params), nullptr, nullptr},
    };
    wgpu::BindGroupDescriptor bind_group_descriptor{};
    bind_group_descriptor.label = "cuMES axisymmetric dealias bindings";
    bind_group_descriptor.layout = layout;
    bind_group_descriptor.entryCount = std::size(entries);
    bind_group_descriptor.entries = entries;
    const wgpu::BindGroup bind_group =
        device.CreateBindGroup(&bind_group_descriptor);

    wgpu::ComputePassDescriptor pass_descriptor{};
    const wgpu::ComputePassEncoder pass =
        encoder.BeginComputePass(&pass_descriptor);
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bind_group);
    pass.DispatchWorkgroups(
        (static_cast<std::uint32_t>(points) + WORKGROUP_SIZE - 1) /
        WORKGROUP_SIZE);
    pass.End();
    const auto host = input.readback.batch
                          ? nullptr
                          : std::make_shared<AxisymmetricDealiasResult>();
    const auto batch =
        input.readback.batch
            ? input.readback.batch
            : std::make_shared<ReadbackBatch>(readback_buffer, result_bytes);
    AxisymmetricDealiasResult ready;
    ready.device_g_con = {result_buffer, points, 0, 0};
    batch->append(
        encoder, result_buffer, 0, result_bytes,
        [callback = input.readback.batch ? std::move(callback)
                                         : AxisymmetricDealiasCallback{},
         host, ready](std::span<const float> values) mutable {
            ready.g_con.assign(values.begin(), values.end());
            ready.finite =
                std::all_of(values.begin(), values.end(),
                            [](float value) { return std::isfinite(value); });
            if (host)
                *host = std::move(ready);
            else
                callback({}, std::move(ready));
        });
    const auto commands = encoder.Finish();
    queue.Submit(1, &commands);
    if (input.readback.batch) {
        input.readback.publish_device(std::move(ready));
        return;
    }
    batch->map([callback = std::move(callback), host](std::string error) {
        callback(std::move(error), std::move(*host));
    });
}

}  // namespace cumes::webgpu
