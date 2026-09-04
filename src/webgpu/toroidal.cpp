#include "cumes/webgpu/toroidal.hpp"

#include "pipeline_cache.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <numbers>
#include <sstream>
#include <utility>

namespace cumes::webgpu {
namespace {

constexpr std::uint32_t WORKGROUP_SIZE = 128;
constexpr std::size_t RESULT_FIELD_COUNT = GEOMETRY_PARITY_FIELD_COUNT + 2;
using BasisKey = std::array<int, 4>;

void compensated_add(float& sum, float& correction, float term) {
    const float adjusted = term - correction;
    const float next = sum + adjusted;
    correction = (next - sum) - adjusted;
    sum = next;
}

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

struct DealiasParams {
    std::uint32_t ns;
    std::uint32_t mpol;
    std::uint32_t ntor;
    std::uint32_t ntheta;
    std::uint32_t nzeta;
    std::uint32_t n_z_n_t;
    std::uint32_t band_modes;
    std::uint32_t points;
};
static_assert(sizeof(DealiasParams) == 32);

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

std::vector<float> generate_basis(const ToroidalInverseCase& input) {
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

const std::vector<float>& cached_basis(int mpol,
                                       int ntor,
                                       int ntheta,
                                       int nzeta) {
    static std::map<BasisKey, std::vector<float>> cache;
    const BasisKey key{mpol, ntor, ntheta, nzeta};
    auto [position, inserted] = cache.try_emplace(key);
    if (inserted) {
        ToroidalInverseCase shape;
        shape.ns = 2;
        shape.mpol = mpol;
        shape.ntor = ntor;
        shape.ntheta = ntheta;
        shape.nzeta = nzeta;
        shape.nfp = 1;
        position->second = generate_basis(shape);
    }
    return position->second;
}

const std::vector<float>& make_basis(const ToroidalInverseCase& input) {
    return cached_basis(input.mpol, input.ntor, input.ntheta, input.nzeta);
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

const std::vector<float>& make_basis(const ToroidalForwardCase& input) {
    return cached_basis(input.mpol, input.ntor, input.ntheta, input.nzeta);
}

std::string validate_case(const ToroidalDealiasCase& input) {
    if (input.ns < 2 || input.mpol < 3 || input.ntor < 1 || input.ntheta < 2 ||
        input.ntheta % 2 != 0 || input.nzeta < 2) {
        return "toroidal dealias requires ns>=2, mpol>=3, ntor>=1, even "
               "ntheta>=2, and nzeta>=2";
    }
    const std::size_t points =
        static_cast<std::size_t>(input.ns) * input.ntheta * input.nzeta;
    if (points > std::numeric_limits<std::uint32_t>::max() ||
        input.g_con_eff.size() != points ||
        input.tcon.size() != static_cast<std::size_t>(input.ns) ||
        input.faccon.size() != static_cast<std::size_t>(input.mpol)) {
        return "toroidal dealias input shape mismatch";
    }
    return {};
}

const std::vector<float>& make_basis(const ToroidalDealiasCase& input) {
    return cached_basis(input.mpol, input.ntor, input.ntheta, input.nzeta);
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

std::string load_dealias_shader() {
    std::ifstream stream("/shaders/toroidal_dealias.wgsl", std::ios::binary);
    if (!stream) return {};
    std::ostringstream text;
    text << stream.rdbuf();
    return text.str();
}

wgpu::Buffer create_uncached_buffer(const wgpu::Device& device,
                                    std::uint64_t size,
                                    wgpu::BufferUsage usage,
                                    const char* label) {
    wgpu::BufferDescriptor descriptor{};
    descriptor.label = label;
    descriptor.size = size;
    descriptor.usage = usage;
    return device.CreateBuffer(&descriptor);
}

wgpu::Buffer create_buffer(const wgpu::Device& device,
                           std::uint64_t size,
                           wgpu::BufferUsage usage,
                           const char* label) {
    return detail::cached_buffer(device, size, usage, label);
}

struct GpuBasis {
    wgpu::Buffer buffer;
    std::size_t bytes = 0;
};

const GpuBasis& cached_gpu_basis(const wgpu::Device& device,
                                 int mpol,
                                 int ntor,
                                 int ntheta,
                                 int nzeta) {
    static std::map<BasisKey, GpuBasis> cache;
    const BasisKey key{mpol, ntor, ntheta, nzeta};
    auto [position, inserted] = cache.try_emplace(key);
    if (inserted) {
        const auto& basis = cached_basis(mpol, ntor, ntheta, nzeta);
        position->second.bytes = basis.size() * sizeof(float);
        position->second.buffer = create_uncached_buffer(
            device, position->second.bytes,
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
            "cuMES cached toroidal basis");
        device.GetQueue().WriteBuffer(position->second.buffer, 0, basis.data(),
                                      position->second.bytes);
    }
    return position->second;
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

struct DealiasDispatchState {
    ToroidalDealiasCallback callback;
    wgpu::Buffer coefficients_buffer;
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
    const auto& basis = make_basis(input);
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
            std::array<float, GEOMETRY_PARITY_FIELD_COUNT> corrections{};
            float r_con_correction = 0.0F;
            float z_con_correction = 0.0F;
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
                    auto& sum =
                        result.geometry[static_cast<std::size_t>(field) *
                                            total_points +
                                        point];
                    compensated_add(sum, corrections[field], scale * value);
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
                compensated_add(result.r_con[point], r_con_correction,
                                xmpq * (rc * cc + rs * ss));
                compensated_add(result.z_con[point], z_con_correction,
                                xmpq * (zs * sc + zc * cs));
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
    const auto& gpu_basis = cached_gpu_basis(device, input.mpol, input.ntor,
                                             input.ntheta, input.nzeta);
    const std::size_t n_z_n_t =
        static_cast<std::size_t>(input.ntheta) * input.nzeta;
    const std::size_t total_points =
        static_cast<std::size_t>(input.ns) * n_z_n_t;
    const std::size_t result_values = RESULT_FIELD_COUNT * total_points;
    const std::size_t state_bytes = input.state.size() * sizeof(float);
    const std::size_t basis_bytes = gpu_basis.bytes;
    const std::size_t result_bytes = result_values * sizeof(float);
    const auto state_buffer =
        create_buffer(device, state_bytes,
                      wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                      "cuMES toroidal state");
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

    const auto& pipeline = detail::cached_compute_pipeline(
        device, "toroidal-inverse", shader_text,
        "cuMES direct toroidal inverse pipeline", "main");

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
    queue.WriteBuffer(params_buffer, 0, &params, sizeof(params));

    const auto layout = pipeline.GetBindGroupLayout(0);
    const wgpu::BindGroupEntry entries[] = {
        {nullptr, 0, state_buffer, 0, state_bytes, nullptr, nullptr},
        {nullptr, 1, gpu_basis.buffer, 0, basis_bytes, nullptr, nullptr},
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
    const auto& basis = make_basis(input);
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
            std::array<float, SPECTRAL_COMPONENT_COUNT> corrections{};
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
                    compensated_add(sums[0], corrections[0],
                                    temp_r * cc - mf * br * sc + nf * cr * cs);
                    compensated_add(sums[3], corrections[3],
                                    temp_r * ss + mf * br * cs - nf * cr * sc);
                    compensated_add(sums[1], corrections[1],
                                    temp_z * sc + mf * bz * cc + nf * cz * ss);
                    compensated_add(sums[4], corrections[4],
                                    temp_z * cs - mf * bz * ss - nf * cz * cc);
                    compensated_add(sums[2], corrections[2],
                                    mf * bl * cc + nf * cl * ss);
                    compensated_add(sums[5], corrections[5],
                                    -mf * bl * ss - nf * cl * cc);
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
    const auto& gpu_basis = cached_gpu_basis(device, input.mpol, input.ntor,
                                             input.ntheta, input.nzeta);
    const std::size_t mnmax =
        static_cast<std::size_t>(input.mpol) * (input.ntor + 1);
    const std::size_t n_z_n_t =
        static_cast<std::size_t>(input.ntheta) * input.nzeta;
    const std::size_t result_values =
        SPECTRAL_COMPONENT_COUNT * mnmax * input.ns;
    const std::size_t fields_bytes = input.fields.size() * sizeof(float);
    const std::size_t basis_bytes = gpu_basis.bytes;
    const std::size_t result_bytes = result_values * sizeof(float);
    const auto fields_buffer =
        create_buffer(device, fields_bytes,
                      wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                      "cuMES toroidal forces");
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

    const auto& pipeline = detail::cached_compute_pipeline(
        device, "toroidal-forward", shader_text,
        "cuMES direct toroidal forward pipeline", "main");
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
    queue.WriteBuffer(params_buffer, 0, &params, sizeof(params));
    const auto layout = pipeline.GetBindGroupLayout(0);
    const wgpu::BindGroupEntry entries[] = {
        {nullptr, 0, fields_buffer, 0, fields_bytes, nullptr, nullptr},
        {nullptr, 1, gpu_basis.buffer, 0, basis_bytes, nullptr, nullptr},
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

ToroidalDealiasResult toroidal_dealias_reference(
    const ToroidalDealiasCase& input) {
    if (!validate_case(input).empty()) return {};
    const int band_modes = input.mpol - 2;
    const int n_z_n_t = input.ntheta * input.nzeta;
    const int mnmax = input.mpol * (input.ntor + 1);
    const auto& basis = make_basis(input);
    const auto table = [&](int family, int mode, int angular) {
        return basis[(static_cast<std::size_t>(family) * mnmax + mode) *
                         n_z_n_t +
                     angular];
    };
    ToroidalDealiasResult result;
    result.g_con.assign(static_cast<std::size_t>(input.ns) * n_z_n_t, 0.0F);
    for (int surface = 1; surface < input.ns; ++surface) {
        for (int angular = 0; angular < n_z_n_t; ++angular) {
            float value = 0.0F;
            float correction = 0.0F;
            for (int m = 1; m <= band_modes; ++m) {
                for (int n = 0; n <= input.ntor; ++n) {
                    const int mode = m * (input.ntor + 1) + n;
                    float sum_sc = 0.0F;
                    float sum_cs = 0.0F;
                    float correction_sc = 0.0F;
                    float correction_cs = 0.0F;
                    for (int source = 0; source < n_z_n_t; ++source) {
                        const float g =
                            input.g_con_eff[static_cast<std::size_t>(surface) *
                                                n_z_n_t +
                                            source];
                        compensated_add(sum_sc, correction_sc,
                                        g * table(2, mode, source));
                        compensated_add(sum_cs, correction_cs,
                                        g * table(3, mode, source));
                    }
                    const float norm = n == 0
                                           ? 2.0F / static_cast<float>(n_z_n_t)
                                           : 4.0F / static_cast<float>(n_z_n_t);
                    const float scale =
                        norm * input.tcon[surface] * input.faccon[m];
                    compensated_add(value, correction,
                                    scale * (sum_sc * table(2, mode, angular) +
                                             sum_cs * table(3, mode, angular)));
                }
            }
            result
                .g_con[static_cast<std::size_t>(surface) * n_z_n_t + angular] =
                value;
        }
    }
    return result;
}

void enqueue_toroidal_dealias(const wgpu::Device& device,
                              const ToroidalDealiasCase& input,
                              ToroidalDealiasCallback callback) {
    const std::string validation_error = validate_case(input);
    if (!validation_error.empty()) {
        callback(validation_error, {});
        return;
    }
    const std::string shader_text = load_dealias_shader();
    if (shader_text.empty()) {
        callback("cannot load embedded /shaders/toroidal_dealias.wgsl", {});
        return;
    }
    const auto& gpu_basis = cached_gpu_basis(device, input.mpol, input.ntor,
                                             input.ntheta, input.nzeta);
    std::vector<float> profiles;
    profiles.reserve(input.tcon.size() + input.faccon.size());
    profiles.insert(profiles.end(), input.tcon.begin(), input.tcon.end());
    profiles.insert(profiles.end(), input.faccon.begin(), input.faccon.end());
    const std::size_t n_z_n_t =
        static_cast<std::size_t>(input.ntheta) * input.nzeta;
    const std::size_t points = static_cast<std::size_t>(input.ns) * n_z_n_t;
    const std::size_t band_modes = static_cast<std::size_t>(input.mpol - 2);
    const std::size_t coefficient_values =
        2 * input.ns * band_modes * (input.ntor + 1);
    const std::size_t input_bytes = input.g_con_eff.size() * sizeof(float);
    const std::size_t profile_bytes = profiles.size() * sizeof(float);
    const std::size_t basis_bytes = gpu_basis.bytes;
    const std::size_t coefficient_bytes = coefficient_values * sizeof(float);
    const std::size_t result_bytes = points * sizeof(float);
    const auto input_buffer =
        create_buffer(device, input_bytes,
                      wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                      "cuMES toroidal effective constraint");
    const auto profile_buffer =
        create_buffer(device, profile_bytes,
                      wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                      "cuMES toroidal constraint profiles");
    const auto coefficient_buffer =
        create_buffer(device, coefficient_bytes, wgpu::BufferUsage::Storage,
                      "cuMES toroidal constraint coefficients");
    const auto result_buffer =
        create_buffer(device, result_bytes,
                      wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc,
                      "cuMES toroidal dealiased constraint");
    const auto readback_buffer =
        create_buffer(device, result_bytes,
                      wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead,
                      "cuMES toroidal constraint readback");
    const auto params_buffer =
        create_buffer(device, sizeof(DealiasParams),
                      wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
                      "cuMES toroidal dealias parameters");

    const auto& analyze_pipeline = detail::cached_compute_pipeline(
        device, "toroidal-dealias-analyze", shader_text,
        "cuMES toroidal dealias analysis pipeline", "analyze");
    const auto& synthesize_pipeline = detail::cached_compute_pipeline(
        device, "toroidal-dealias-synthesize", shader_text,
        "cuMES toroidal dealias synthesis pipeline", "synthesize");
    const DealiasParams params{static_cast<std::uint32_t>(input.ns),
                               static_cast<std::uint32_t>(input.mpol),
                               static_cast<std::uint32_t>(input.ntor),
                               static_cast<std::uint32_t>(input.ntheta),
                               static_cast<std::uint32_t>(input.nzeta),
                               static_cast<std::uint32_t>(n_z_n_t),
                               static_cast<std::uint32_t>(band_modes),
                               static_cast<std::uint32_t>(points)};
    const auto queue = device.GetQueue();
    queue.WriteBuffer(input_buffer, 0, input.g_con_eff.data(), input_bytes);
    queue.WriteBuffer(profile_buffer, 0, profiles.data(), profile_bytes);
    queue.WriteBuffer(params_buffer, 0, &params, sizeof(params));
    const auto analyze_layout = analyze_pipeline.GetBindGroupLayout(0);
    const wgpu::BindGroupEntry analyze_entries[] = {
        {nullptr, 0, input_buffer, 0, input_bytes, nullptr, nullptr},
        {nullptr, 1, profile_buffer, 0, profile_bytes, nullptr, nullptr},
        {nullptr, 2, gpu_basis.buffer, 0, basis_bytes, nullptr, nullptr},
        {nullptr, 3, coefficient_buffer, 0, coefficient_bytes, nullptr,
         nullptr},
        {nullptr, 5, params_buffer, 0, sizeof(params), nullptr, nullptr},
    };
    wgpu::BindGroupDescriptor analyze_bind_descriptor{};
    analyze_bind_descriptor.label = "cuMES toroidal dealias analysis bindings";
    analyze_bind_descriptor.layout = analyze_layout;
    analyze_bind_descriptor.entryCount = std::size(analyze_entries);
    analyze_bind_descriptor.entries = analyze_entries;
    const auto analyze_bind_group =
        device.CreateBindGroup(&analyze_bind_descriptor);
    const auto synthesize_layout = synthesize_pipeline.GetBindGroupLayout(0);
    const wgpu::BindGroupEntry synthesize_entries[] = {
        {nullptr, 2, gpu_basis.buffer, 0, basis_bytes, nullptr, nullptr},
        {nullptr, 3, coefficient_buffer, 0, coefficient_bytes, nullptr,
         nullptr},
        {nullptr, 4, result_buffer, 0, result_bytes, nullptr, nullptr},
        {nullptr, 5, params_buffer, 0, sizeof(params), nullptr, nullptr},
    };
    wgpu::BindGroupDescriptor synthesize_bind_descriptor{};
    synthesize_bind_descriptor.label =
        "cuMES toroidal dealias synthesis bindings";
    synthesize_bind_descriptor.layout = synthesize_layout;
    synthesize_bind_descriptor.entryCount = std::size(synthesize_entries);
    synthesize_bind_descriptor.entries = synthesize_entries;
    const auto synthesize_bind_group =
        device.CreateBindGroup(&synthesize_bind_descriptor);
    const auto encoder = device.CreateCommandEncoder();
    wgpu::ComputePassDescriptor pass_descriptor{};
    const auto pass = encoder.BeginComputePass(&pass_descriptor);
    pass.SetPipeline(analyze_pipeline);
    pass.SetBindGroup(0, analyze_bind_group);
    pass.DispatchWorkgroups(
        (static_cast<std::uint32_t>(coefficient_values / 2) + WORKGROUP_SIZE -
         1) /
        WORKGROUP_SIZE);
    pass.SetPipeline(synthesize_pipeline);
    pass.SetBindGroup(0, synthesize_bind_group);
    pass.DispatchWorkgroups(
        (static_cast<std::uint32_t>(points) + WORKGROUP_SIZE - 1) /
        WORKGROUP_SIZE);
    pass.End();
    encoder.CopyBufferToBuffer(result_buffer, 0, readback_buffer, 0,
                               result_bytes);
    const auto commands = encoder.Finish();
    queue.Submit(1, &commands);

    auto dispatch = std::make_shared<DealiasDispatchState>();
    dispatch->callback = std::move(callback);
    dispatch->coefficients_buffer = coefficient_buffer;
    dispatch->result_buffer = result_buffer;
    dispatch->readback_buffer = readback_buffer;
    dispatch->result_values = points;
    dispatch->result_bytes = result_bytes;
    readback_buffer.MapAsync(
        wgpu::MapMode::Read, 0, result_bytes,
        wgpu::CallbackMode::AllowSpontaneous,
        [dispatch](wgpu::MapAsyncStatus status, wgpu::StringView message) {
            if (status != wgpu::MapAsyncStatus::Success) {
                dispatch->callback(
                    "WebGPU toroidal dealias mapping failed: " +
                        std::string(message.data, message.length),
                    {});
                return;
            }
            const auto* values = static_cast<const float*>(
                dispatch->readback_buffer.GetConstMappedRange(
                    0, dispatch->result_bytes));
            if (values == nullptr) {
                dispatch->callback(
                    "WebGPU toroidal dealias returned a null mapped range", {});
                return;
            }
            ToroidalDealiasResult result;
            result.g_con.assign(values, values + dispatch->result_values);
            dispatch->readback_buffer.Unmap();
            dispatch->callback({}, std::move(result));
        });
}

}  // namespace cumes::webgpu
