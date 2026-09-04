#include "cumes/webgpu/toroidal.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <numbers>
#include <sstream>
#include <utility>

namespace cumes::webgpu {
namespace {

constexpr std::uint32_t WORKGROUP_SIZE = 128;
constexpr std::size_t RESULT_FIELD_COUNT = GEOMETRY_PARITY_FIELD_COUNT + 2;

struct ShaderParams {
    std::uint32_t ns;
    std::uint32_t mpol;
    std::uint32_t ntor;
    std::uint32_t ntheta;
    std::uint32_t nzeta;
    std::uint32_t nfp;
    std::uint32_t n_z_n_t;
    std::uint32_t total_points;
};
static_assert(sizeof(ShaderParams) == 32);

std::string validate_case(const ToroidalInverseCase& input) {
    if (input.ns < 2 || input.mpol <= 0 || input.ntor < 1 || input.ntheta < 2 ||
        input.ntheta % 2 != 0 || input.nzeta < 2 || input.nfp < 1) {
        return "toroidal inverse requires ns>=2, mpol>0, ntor>=1, even "
               "ntheta>=2, nzeta>=2, and nfp>=1";
    }
    const std::size_t mnmax =
        static_cast<std::size_t>(input.mpol) * (input.ntor + 1);
    const std::size_t n_z_n_t =
        static_cast<std::size_t>(input.ntheta) * input.nzeta;
    const std::size_t total_points =
        static_cast<std::size_t>(input.ns) * n_z_n_t;
    if (input.state.size() != SPECTRAL_COMPONENT_COUNT * mnmax * input.ns) {
        return "toroidal state size does not match 6*mnmax*ns";
    }
    if (total_points > std::numeric_limits<std::uint32_t>::max() ||
        RESULT_FIELD_COUNT * total_points >
            std::numeric_limits<std::uint32_t>::max()) {
        return "toroidal inverse exceeds WebGPU indexing limits";
    }
    return {};
}

std::vector<float> make_basis(const ToroidalInverseCase& input) {
    const std::size_t mnmax =
        static_cast<std::size_t>(input.mpol) * (input.ntor + 1);
    const std::size_t n_z_n_t =
        static_cast<std::size_t>(input.ntheta) * input.nzeta;
    std::vector<float> basis(4 * mnmax * n_z_n_t);
    for (int n = 0; n <= input.ntor; ++n) {
        for (int m = 0; m < input.mpol; ++m) {
            const std::size_t mode =
                static_cast<std::size_t>(m) * (input.ntor + 1) + n;
            for (int zeta_index = 0; zeta_index < input.nzeta; ++zeta_index) {
                const float zeta = 2.0F * std::numbers::pi_v<float> *
                                   static_cast<float>(zeta_index) /
                                   static_cast<float>(input.nzeta);
                for (int theta_index = 0; theta_index < input.ntheta;
                     ++theta_index) {
                    const float theta = 2.0F * std::numbers::pi_v<float> *
                                        static_cast<float>(theta_index) /
                                        static_cast<float>(input.ntheta);
                    const std::size_t angular =
                        static_cast<std::size_t>(zeta_index) * input.ntheta +
                        theta_index;
                    const float cm = std::cos(static_cast<float>(m) * theta);
                    const float sm = std::sin(static_cast<float>(m) * theta);
                    const float cn = std::cos(static_cast<float>(n) * zeta);
                    const float sn = std::sin(static_cast<float>(n) * zeta);
                    const std::size_t offset = mode * n_z_n_t + angular;
                    basis[0 * mnmax * n_z_n_t + offset] = cm * cn;
                    basis[1 * mnmax * n_z_n_t + offset] = sm * sn;
                    basis[2 * mnmax * n_z_n_t + offset] = sm * cn;
                    basis[3 * mnmax * n_z_n_t + offset] = cm * sn;
                }
            }
        }
    }
    return basis;
}

std::string validate_case(const ToroidalForwardCase& input) {
    if (input.ns < 2 || input.mpol <= 0 || input.ntor < 1 || input.ntheta < 2 ||
        input.ntheta % 2 != 0 || input.nzeta < 2 || input.nfp < 1) {
        return "toroidal forward requires ns>=2, mpol>0, ntor>=1, even "
               "ntheta>=2, nzeta>=2, and nfp>=1";
    }
    const std::size_t mnmax =
        static_cast<std::size_t>(input.mpol) * (input.ntor + 1);
    const std::size_t n_z_n_t =
        static_cast<std::size_t>(input.ntheta) * input.nzeta;
    if (input.fields.size() !=
        TOROIDAL_FORWARD_FIELD_COUNT * input.ns * n_z_n_t) {
        return "toroidal force size does not match 20*ns*ntheta*nzeta";
    }
    if (static_cast<std::size_t>(input.ns) * mnmax >
        std::numeric_limits<std::uint32_t>::max()) {
        return "toroidal forward exceeds WebGPU indexing limits";
    }
    return {};
}

std::vector<float> make_basis(const ToroidalForwardCase& input) {
    ToroidalInverseCase shape;
    shape.ns = input.ns;
    shape.mpol = input.mpol;
    shape.ntor = input.ntor;
    shape.ntheta = input.ntheta;
    shape.nzeta = input.nzeta;
    shape.nfp = input.nfp;
    return make_basis(shape);
}

std::string load_shader() {
    std::ifstream stream("/shaders/toroidal_inverse.wgsl", std::ios::binary);
    if (!stream) return {};
    std::ostringstream text;
    text << stream.rdbuf();
    return text.str();
}

std::string load_forward_shader() {
    std::ifstream stream("/shaders/toroidal_forward.wgsl", std::ios::binary);
    if (!stream) return {};
    std::ostringstream text;
    text << stream.rdbuf();
    return text.str();
}

wgpu::Buffer create_buffer(const wgpu::Device& device,
                           std::uint64_t size,
                           wgpu::BufferUsage usage,
                           const char* label) {
    wgpu::BufferDescriptor descriptor{};
    descriptor.label = label;
    descriptor.size = size;
    descriptor.usage = usage;
    return device.CreateBuffer(&descriptor);
}

struct DispatchState {
    ToroidalInverseCallback callback;
    wgpu::Buffer result_buffer;
    wgpu::Buffer readback_buffer;
    std::size_t total_points = 0;
    std::size_t result_bytes = 0;
};

struct ForwardDispatchState {
    ToroidalForwardCallback callback;
    wgpu::Buffer result_buffer;
    wgpu::Buffer readback_buffer;
    std::size_t result_values = 0;
    std::size_t result_bytes = 0;
};

}  // namespace

ToroidalInverseResult toroidal_inverse_reference(
    const ToroidalInverseCase& input) {
    if (!validate_case(input).empty()) return {};
    const int mnmax = input.mpol * (input.ntor + 1);
    const int n_z_n_t = input.ntheta * input.nzeta;
    const std::size_t total_points =
        static_cast<std::size_t>(input.ns) * n_z_n_t;
    const auto basis = make_basis(input);
    ToroidalInverseResult result;
    result.geometry.assign(GEOMETRY_PARITY_FIELD_COUNT * total_points, 0.0F);
    result.r_con.assign(total_points, 0.0F);
    result.z_con.assign(total_points, 0.0F);
    const auto coeff = [&](int component, int mode, int surface) {
        return input
            .state[(static_cast<std::size_t>(component) * mnmax + mode) *
                       input.ns +
                   surface];
    };
    const auto table = [&](int field, int mode, int angular) {
        return basis[(static_cast<std::size_t>(field) * mnmax + mode) *
                         n_z_n_t +
                     angular];
    };
    for (int surface = 0; surface < input.ns; ++surface) {
        const float maxsc =
            std::max(std::sqrt(static_cast<float>(surface) /
                               static_cast<float>(input.ns - 1)),
                     std::sqrt(1.0F / static_cast<float>(input.ns - 1)));
        for (int angular = 0; angular < n_z_n_t; ++angular) {
            const std::size_t point =
                static_cast<std::size_t>(surface) * n_z_n_t + angular;
            for (int mode = 0; mode < mnmax; ++mode) {
                const int m = mode / (input.ntor + 1);
                const int n = mode % (input.ntor + 1);
                const float mf = static_cast<float>(m);
                const float nf = static_cast<float>(n * input.nfp);
                const float cc = table(0, mode, angular);
                const float ss = table(1, mode, angular);
                const float sc = table(2, mode, angular);
                const float cs = table(3, mode, angular);
                const float scale = m % 2 == 1 ? 1.0F / maxsc : 1.0F;
                const int parity = m % 2 == 1 ? 6 : 0;
                const float rc = coeff(0, mode, surface);
                const float zs = coeff(1, mode, surface);
                const float ls = coeff(2, mode, surface);
                const float rs = coeff(3, mode, surface);
                const float zc = coeff(4, mode, surface);
                const float lc = coeff(5, mode, surface);
                auto add = [&](int field, float value) {
                    result.geometry[static_cast<std::size_t>(field) *
                                        total_points +
                                    point] += scale * value;
                };
                add(parity + 0, rc * cc + rs * ss);
                add(parity + 1, zs * sc + zc * cs);
                add(parity + 2, ls * sc + lc * cs);
                add(parity + 3, -mf * rc * sc + mf * rs * cs);
                add(parity + 4, mf * zs * cc - mf * zc * ss);
                add(parity + 5, mf * ls * cc - mf * lc * ss);
                add(12 + (m % 2 == 1 ? 3 : 0), -nf * rc * cs + nf * rs * sc);
                add(13 + (m % 2 == 1 ? 3 : 0), -nf * zs * ss + nf * zc * cc);
                add(14 + (m % 2 == 1 ? 3 : 0), nf * ls * ss - nf * lc * cc);
                const float xmpq = mf * (mf - 1.0F);
                result.r_con[point] += xmpq * (rc * cc + rs * ss);
                result.z_con[point] += xmpq * (zs * sc + zc * cs);
            }
        }
    }
    return result;
}

void enqueue_toroidal_inverse(const wgpu::Device& device,
                              const ToroidalInverseCase& input,
                              ToroidalInverseCallback callback) {
    const std::string validation_error = validate_case(input);
    if (!validation_error.empty()) {
        callback(validation_error, {});
        return;
    }
    const std::string shader_text = load_shader();
    if (shader_text.empty()) {
        callback("cannot load embedded /shaders/toroidal_inverse.wgsl", {});
        return;
    }
    const std::vector<float> basis = make_basis(input);
    const std::size_t n_z_n_t =
        static_cast<std::size_t>(input.ntheta) * input.nzeta;
    const std::size_t total_points =
        static_cast<std::size_t>(input.ns) * n_z_n_t;
    const std::size_t result_values = RESULT_FIELD_COUNT * total_points;
    const std::size_t state_bytes = input.state.size() * sizeof(float);
    const std::size_t basis_bytes = basis.size() * sizeof(float);
    const std::size_t result_bytes = result_values * sizeof(float);
    const auto state_buffer =
        create_buffer(device, state_bytes,
                      wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                      "cuMES toroidal state");
    const auto basis_buffer =
        create_buffer(device, basis_bytes,
                      wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                      "cuMES toroidal basis");
    const auto result_buffer =
        create_buffer(device, result_bytes,
                      wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc,
                      "cuMES toroidal inverse result");
    const auto readback_buffer =
        create_buffer(device, result_bytes,
                      wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead,
                      "cuMES toroidal inverse readback");
    const auto params_buffer =
        create_buffer(device, sizeof(ShaderParams),
                      wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
                      "cuMES toroidal inverse parameters");

    wgpu::ShaderSourceWGSL wgsl{};
    wgsl.code = shader_text.c_str();
    wgpu::ShaderModuleDescriptor shader_descriptor{};
    shader_descriptor.label = "cuMES direct toroidal inverse";
    shader_descriptor.nextInChain = &wgsl;
    const auto shader = device.CreateShaderModule(&shader_descriptor);
    wgpu::ComputePipelineDescriptor pipeline_descriptor{};
    pipeline_descriptor.label = "cuMES direct toroidal inverse pipeline";
    pipeline_descriptor.compute.module = shader;
    pipeline_descriptor.compute.entryPoint = "main";
    const auto pipeline = device.CreateComputePipeline(&pipeline_descriptor);

    const ShaderParams params{static_cast<std::uint32_t>(input.ns),
                              static_cast<std::uint32_t>(input.mpol),
                              static_cast<std::uint32_t>(input.ntor),
                              static_cast<std::uint32_t>(input.ntheta),
                              static_cast<std::uint32_t>(input.nzeta),
                              static_cast<std::uint32_t>(input.nfp),
                              static_cast<std::uint32_t>(n_z_n_t),
                              static_cast<std::uint32_t>(total_points)};
    const auto queue = device.GetQueue();
    queue.WriteBuffer(state_buffer, 0, input.state.data(), state_bytes);
    queue.WriteBuffer(basis_buffer, 0, basis.data(), basis_bytes);
    queue.WriteBuffer(params_buffer, 0, &params, sizeof(params));

    const auto layout = pipeline.GetBindGroupLayout(0);
    const wgpu::BindGroupEntry entries[] = {
        {nullptr, 0, state_buffer, 0, state_bytes, nullptr, nullptr},
        {nullptr, 1, basis_buffer, 0, basis_bytes, nullptr, nullptr},
        {nullptr, 2, result_buffer, 0, result_bytes, nullptr, nullptr},
        {nullptr, 3, params_buffer, 0, sizeof(params), nullptr, nullptr},
    };
    wgpu::BindGroupDescriptor bind_descriptor{};
    bind_descriptor.label = "cuMES toroidal inverse bindings";
    bind_descriptor.layout = layout;
    bind_descriptor.entryCount = std::size(entries);
    bind_descriptor.entries = entries;
    const auto bind_group = device.CreateBindGroup(&bind_descriptor);
    const auto encoder = device.CreateCommandEncoder();
    wgpu::ComputePassDescriptor pass_descriptor{};
    const auto pass = encoder.BeginComputePass(&pass_descriptor);
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bind_group);
    pass.DispatchWorkgroups(
        (static_cast<std::uint32_t>(total_points) + WORKGROUP_SIZE - 1) /
        WORKGROUP_SIZE);
    pass.End();
    encoder.CopyBufferToBuffer(result_buffer, 0, readback_buffer, 0,
                               result_bytes);
    const auto commands = encoder.Finish();
    queue.Submit(1, &commands);

    auto dispatch = std::make_shared<DispatchState>();
    dispatch->callback = std::move(callback);
    dispatch->result_buffer = result_buffer;
    dispatch->readback_buffer = readback_buffer;
    dispatch->total_points = total_points;
    dispatch->result_bytes = result_bytes;
    readback_buffer.MapAsync(
        wgpu::MapMode::Read, 0, result_bytes,
        wgpu::CallbackMode::AllowSpontaneous,
        [dispatch](wgpu::MapAsyncStatus status, wgpu::StringView message) {
            if (status != wgpu::MapAsyncStatus::Success) {
                dispatch->callback(
                    "WebGPU toroidal inverse mapping failed: " +
                        std::string(message.data, message.length),
                    {});
                return;
            }
            const auto* values = static_cast<const float*>(
                dispatch->readback_buffer.GetConstMappedRange(
                    0, dispatch->result_bytes));
            if (values == nullptr) {
                dispatch->callback(
                    "WebGPU toroidal inverse returned a null mapped range", {});
                return;
            }
            ToroidalInverseResult result;
            const std::size_t geometry_values =
                GEOMETRY_PARITY_FIELD_COUNT * dispatch->total_points;
            result.geometry.assign(values, values + geometry_values);
            result.r_con.assign(
                values + geometry_values,
                values + geometry_values + dispatch->total_points);
            result.z_con.assign(
                values + geometry_values + dispatch->total_points,
                values + geometry_values + 2 * dispatch->total_points);
            dispatch->readback_buffer.Unmap();
            dispatch->callback({}, std::move(result));
        });
}

ToroidalForwardResult toroidal_forward_reference(
    const ToroidalForwardCase& input) {
    if (!validate_case(input).empty()) return {};
    const int mnmax = input.mpol * (input.ntor + 1);
    const int n_z_n_t = input.ntheta * input.nzeta;
    const int theta_reduced = input.ntheta / 2 + 1;
    const float norm =
        1.0F / static_cast<float>(input.nzeta * (theta_reduced - 1));
    const auto basis = make_basis(input);
    ToroidalForwardResult result;
    result.residual.assign(SPECTRAL_COMPONENT_COUNT * mnmax * input.ns, 0.0F);
    const auto field = [&](int component, int surface, int angular) {
        return input
            .fields[(static_cast<std::size_t>(component) * input.ns + surface) *
                        n_z_n_t +
                    angular];
    };
    const auto table = [&](int component, int mode, int angular) {
        return basis[(static_cast<std::size_t>(component) * mnmax + mode) *
                         n_z_n_t +
                     angular];
    };
    for (int mode = 0; mode < mnmax; ++mode) {
        const int m = mode / (input.ntor + 1);
        const int n = mode % (input.ntor + 1);
        const float mf = static_cast<float>(m);
        const float nf = static_cast<float>(n * input.nfp);
        const int parity = m % 2;
        const float xmpq = mf * (mf - 1.0F);
        const float scale = (m == 0 ? 1.0F : std::sqrt(2.0F)) *
                            (n == 0 ? 1.0F : std::sqrt(2.0F));
        for (int surface = 0; surface < input.ns; ++surface) {
            std::array<float, SPECTRAL_COMPONENT_COUNT> sums{};
            for (int zeta = 0; zeta < input.nzeta; ++zeta) {
                for (int theta = 0; theta < theta_reduced; ++theta) {
                    const int angular = zeta * input.ntheta + theta;
                    float weight = norm;
                    if (theta == 0 || theta + 1 == theta_reduced) {
                        weight *= 0.5F;
                    }
                    const float cc = weight * table(0, mode, angular);
                    const float ss = weight * table(1, mode, angular);
                    const float sc = weight * table(2, mode, angular);
                    const float cs = weight * table(3, mode, angular);
                    const float temp_r =
                        field(parity, surface, angular) +
                        xmpq * field(16 + parity, surface, angular);
                    const float temp_z =
                        field(2 + parity, surface, angular) +
                        xmpq * field(18 + parity, surface, angular);
                    const float br = field(4 + parity, surface, angular);
                    const float bz = field(6 + parity, surface, angular);
                    const float bl = field(8 + parity, surface, angular);
                    const float cr = field(10 + parity, surface, angular);
                    const float cz = field(12 + parity, surface, angular);
                    const float cl = field(14 + parity, surface, angular);
                    sums[0] += temp_r * cc - mf * br * sc + nf * cr * cs;
                    sums[3] += temp_r * ss + mf * br * cs - nf * cr * sc;
                    sums[1] += temp_z * sc + mf * bz * cc + nf * cz * ss;
                    sums[4] += temp_z * cs - mf * bz * ss - nf * cz * cc;
                    sums[2] += mf * bl * cc + nf * cl * ss;
                    sums[5] += -mf * bl * ss - nf * cl * cc;
                }
            }
            for (int component = 0;
                 component < static_cast<int>(SPECTRAL_COMPONENT_COUNT);
                 ++component) {
                float value = scale * sums[component];
                if (surface == 0 &&
                    !(m == 0 && (component == 0 || component == 4))) {
                    value = 0.0F;
                } else if (surface == input.ns - 1 && !input.include_lcfs &&
                           component != 2 && component != 5) {
                    value = 0.0F;
                }
                result.residual[(static_cast<std::size_t>(component) * mnmax +
                                 mode) *
                                    input.ns +
                                surface] = value;
            }
        }
    }
    return result;
}

void enqueue_toroidal_forward(const wgpu::Device& device,
                              const ToroidalForwardCase& input,
                              ToroidalForwardCallback callback) {
    const std::string validation_error = validate_case(input);
    if (!validation_error.empty()) {
        callback(validation_error, {});
        return;
    }
    const std::string shader_text = load_forward_shader();
    if (shader_text.empty()) {
        callback("cannot load embedded /shaders/toroidal_forward.wgsl", {});
        return;
    }
    const std::vector<float> basis = make_basis(input);
    const std::size_t mnmax =
        static_cast<std::size_t>(input.mpol) * (input.ntor + 1);
    const std::size_t n_z_n_t =
        static_cast<std::size_t>(input.ntheta) * input.nzeta;
    const std::size_t result_values =
        SPECTRAL_COMPONENT_COUNT * mnmax * input.ns;
    const std::size_t fields_bytes = input.fields.size() * sizeof(float);
    const std::size_t basis_bytes = basis.size() * sizeof(float);
    const std::size_t result_bytes = result_values * sizeof(float);
    const auto fields_buffer =
        create_buffer(device, fields_bytes,
                      wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                      "cuMES toroidal forces");
    const auto basis_buffer =
        create_buffer(device, basis_bytes,
                      wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                      "cuMES toroidal forward basis");
    const auto result_buffer =
        create_buffer(device, result_bytes,
                      wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc,
                      "cuMES toroidal residual");
    const auto readback_buffer =
        create_buffer(device, result_bytes,
                      wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead,
                      "cuMES toroidal residual readback");
    const auto params_buffer =
        create_buffer(device, sizeof(ShaderParams),
                      wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
                      "cuMES toroidal forward parameters");

    wgpu::ShaderSourceWGSL wgsl{};
    wgsl.code = shader_text.c_str();
    wgpu::ShaderModuleDescriptor shader_descriptor{};
    shader_descriptor.label = "cuMES direct toroidal forward";
    shader_descriptor.nextInChain = &wgsl;
    const auto shader = device.CreateShaderModule(&shader_descriptor);
    wgpu::ComputePipelineDescriptor pipeline_descriptor{};
    pipeline_descriptor.label = "cuMES direct toroidal forward pipeline";
    pipeline_descriptor.compute.module = shader;
    pipeline_descriptor.compute.entryPoint = "main";
    const auto pipeline = device.CreateComputePipeline(&pipeline_descriptor);
    const ShaderParams params{static_cast<std::uint32_t>(input.ns),
                              static_cast<std::uint32_t>(input.mpol),
                              static_cast<std::uint32_t>(input.ntor),
                              static_cast<std::uint32_t>(input.ntheta),
                              static_cast<std::uint32_t>(input.nzeta),
                              static_cast<std::uint32_t>(input.nfp),
                              static_cast<std::uint32_t>(n_z_n_t),
                              static_cast<std::uint32_t>(input.include_lcfs)};
    const auto queue = device.GetQueue();
    queue.WriteBuffer(fields_buffer, 0, input.fields.data(), fields_bytes);
    queue.WriteBuffer(basis_buffer, 0, basis.data(), basis_bytes);
    queue.WriteBuffer(params_buffer, 0, &params, sizeof(params));
    const auto layout = pipeline.GetBindGroupLayout(0);
    const wgpu::BindGroupEntry entries[] = {
        {nullptr, 0, fields_buffer, 0, fields_bytes, nullptr, nullptr},
        {nullptr, 1, basis_buffer, 0, basis_bytes, nullptr, nullptr},
        {nullptr, 2, result_buffer, 0, result_bytes, nullptr, nullptr},
        {nullptr, 3, params_buffer, 0, sizeof(params), nullptr, nullptr},
    };
    wgpu::BindGroupDescriptor bind_descriptor{};
    bind_descriptor.label = "cuMES toroidal forward bindings";
    bind_descriptor.layout = layout;
    bind_descriptor.entryCount = std::size(entries);
    bind_descriptor.entries = entries;
    const auto bind_group = device.CreateBindGroup(&bind_descriptor);
    const auto encoder = device.CreateCommandEncoder();
    wgpu::ComputePassDescriptor pass_descriptor{};
    const auto pass = encoder.BeginComputePass(&pass_descriptor);
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, bind_group);
    pass.DispatchWorkgroups(
        (static_cast<std::uint32_t>(input.ns * mnmax) + WORKGROUP_SIZE - 1) /
        WORKGROUP_SIZE);
    pass.End();
    encoder.CopyBufferToBuffer(result_buffer, 0, readback_buffer, 0,
                               result_bytes);
    const auto commands = encoder.Finish();
    queue.Submit(1, &commands);

    auto dispatch = std::make_shared<ForwardDispatchState>();
    dispatch->callback = std::move(callback);
    dispatch->result_buffer = result_buffer;
    dispatch->readback_buffer = readback_buffer;
    dispatch->result_values = result_values;
    dispatch->result_bytes = result_bytes;
    readback_buffer.MapAsync(
        wgpu::MapMode::Read, 0, result_bytes,
        wgpu::CallbackMode::AllowSpontaneous,
        [dispatch](wgpu::MapAsyncStatus status, wgpu::StringView message) {
            if (status != wgpu::MapAsyncStatus::Success) {
                dispatch->callback(
                    "WebGPU toroidal forward mapping failed: " +
                        std::string(message.data, message.length),
                    {});
                return;
            }
            const auto* values = static_cast<const float*>(
                dispatch->readback_buffer.GetConstMappedRange(
                    0, dispatch->result_bytes));
            if (values == nullptr) {
                dispatch->callback(
                    "WebGPU toroidal forward returned a null mapped range", {});
                return;
            }
            ToroidalForwardResult result;
            result.residual.assign(values, values + dispatch->result_values);
            dispatch->readback_buffer.Unmap();
            dispatch->callback({}, std::move(result));
        });
}

}  // namespace cumes::webgpu
