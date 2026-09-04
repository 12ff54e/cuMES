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

constexpr int MAX_SURFACES = 512;

struct Params {
    std::uint32_t ns, mode_count, ntor, points;
    std::uint32_t last_surface, padding[3];
};
static_assert(sizeof(Params) == 32);

int mode_count(const AxisymmetricPreconditionerApplyCase& in) {
    return in.mpol * (in.ntor + 1);
}

bool valid_elements(const AxisymmetricPreconditionerApplyCase& in) {
    const std::size_t pairs = 2 * static_cast<std::size_t>(in.ns);
    return in.elements.ard.size() == pairs && in.elements.brd.size() == pairs &&
           in.elements.azd.size() == pairs && in.elements.bzd.size() == pairs;
}

bool valid_matrix(const AxisymmetricPreconditionerApplyCase& in) {
    const int modes = mode_count(in);
    const std::size_t points = static_cast<std::size_t>(in.ns) * modes;
    const auto has_points = [points](const auto& values) {
        return values.size() == points;
    };
    return has_points(in.matrix.upper_r) && has_points(in.matrix.diagonal_r) &&
           has_points(in.matrix.lower_r) && has_points(in.matrix.upper_z) &&
           has_points(in.matrix.diagonal_z) && has_points(in.matrix.lower_z) &&
           has_points(in.matrix.lambda) &&
           in.matrix.scale.size() == static_cast<std::size_t>(modes) &&
           in.matrix.first_surface.size() == static_cast<std::size_t>(modes);
}

std::string validate_case(const AxisymmetricPreconditionerApplyCase& in) {
    if (in.ns < 2 || in.ns > MAX_SURFACES || in.mpol < 2 || in.ntor < 0) {
        return "axisymmetric preconditioner apply has invalid dimensions";
    }
    const std::size_t points = static_cast<std::size_t>(in.ns) * mode_count(in);
    if (!valid_elements(in) || !valid_matrix(in) ||
        in.residual.size() != 6 * points) {
        return "axisymmetric preconditioner apply input shape mismatch";
    }
    return {};
}

std::vector<float> flatten_matrix(
    const AxisymmetricPreconditionerMatrix& matrix) {
    std::vector<float> values;
    const auto append = [&values](const auto& source) {
        values.insert(values.end(), source.begin(), source.end());
    };
    append(matrix.upper_r);
    append(matrix.diagonal_r);
    append(matrix.lower_r);
    append(matrix.upper_z);
    append(matrix.diagonal_z);
    append(matrix.lower_z);
    append(matrix.lambda);
    append(matrix.scale);
    return values;
}

std::vector<float> flatten_elements(
    const AxisymmetricPreconditionerElements& elements) {
    std::vector<float> values;
    const auto append = [&values](const auto& source) {
        values.insert(values.end(), source.begin(), source.end());
    };
    append(elements.ard);
    append(elements.brd);
    append(elements.azd);
    append(elements.bzd);
    return values;
}

float guarded_pivot(float value, float floor, bool& broke) {
    if (std::abs(value) >= floor) return value;
    broke = true;
    return std::copysign(floor, value);
}

bool solve_pair(const AxisymmetricPreconditionerApplyCase& in,
                bool z_system,
                int mode,
                int last_surface,
                float floor,
                std::vector<float>& residual) {
    const int first = in.matrix.first_surface[mode];
    const int count = last_surface - first;
    if (count <= 0) return false;
    const std::size_t points = static_cast<std::size_t>(in.ns) * mode_count(in);
    const auto& lower = z_system ? in.matrix.lower_z : in.matrix.lower_r;
    const auto& diagonal =
        z_system ? in.matrix.diagonal_z : in.matrix.diagonal_r;
    const auto& upper = z_system ? in.matrix.upper_z : in.matrix.upper_r;
    const int component0 = z_system ? 1 : 0;
    const int component1 = component0 + 3;
    const auto index = [&](int component, int surface) {
        return static_cast<std::size_t>(component) * points +
               static_cast<std::size_t>(mode) * in.ns + surface;
    };
    std::vector<float> cprime(count);
    std::vector<float> dprime0(count);
    std::vector<float> dprime1(count);
    bool broke = false;
    const std::size_t first_index =
        static_cast<std::size_t>(mode) * in.ns + first;
    float denominator = guarded_pivot(diagonal[first_index], floor, broke);
    cprime[0] = upper[first_index] / denominator;
    dprime0[0] = residual[index(component0, first)] / denominator;
    dprime1[0] = residual[index(component1, first)] / denominator;
    for (int i = 1; i < count; ++i) {
        const int surface = first + i;
        const std::size_t matrix_index =
            static_cast<std::size_t>(mode) * in.ns + surface;
        const float lower_value = lower[matrix_index];
        float pivot = diagonal[matrix_index] - lower_value * cprime[i - 1];
        pivot = guarded_pivot(pivot, floor, broke);
        cprime[i] = upper[matrix_index] / pivot;
        dprime0[i] = (residual[index(component0, surface)] -
                      lower_value * dprime0[i - 1]) /
                     pivot;
        dprime1[i] = (residual[index(component1, surface)] -
                      lower_value * dprime1[i - 1]) /
                     pivot;
    }
    residual[index(component0, last_surface - 1)] = dprime0[count - 1];
    residual[index(component1, last_surface - 1)] = dprime1[count - 1];
    for (int i = count - 2; i >= 0; --i) {
        const int surface = first + i;
        residual[index(component0, surface)] =
            dprime0[i] - cprime[i] * residual[index(component0, surface + 1)];
        residual[index(component1, surface)] =
            dprime1[i] - cprime[i] * residual[index(component1, surface + 1)];
    }
    return broke;
}

std::string load_shader() {
    std::ifstream stream("/shaders/axisymmetric_preconditioner_apply.wgsl",
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
    return detail::cached_buffer(device, size, usage, label);
}

struct Dispatch {
    AxisymmetricPreconditionerApplyCallback callback;
    wgpu::Buffer output, readback;
    std::size_t bytes = 0, points = 0;
    int mpol = 0;
};

}  // namespace

AxisymmetricPreconditionerApplyResult
axisymmetric_preconditioner_apply_reference(
    const AxisymmetricPreconditionerApplyCase& input) {
    if (!validate_case(input).empty()) return {};
    const int modes = mode_count(input);
    const std::size_t points = static_cast<std::size_t>(input.ns) * modes;
    AxisymmetricPreconditionerApplyResult out;
    out.residual = input.residual;
    if (input.mpol > 1) {
        for (int n = 0; n <= input.ntor; ++n) {
            const int mode = input.ntor + 1 + n;
            for (int surface = 0; surface < input.ns; ++surface) {
                const int odd = 2 * surface + 1;
                const float rsum =
                    input.elements.ard[odd] + input.elements.brd[odd];
                const float zsum =
                    input.elements.azd[odd] + input.elements.bzd[odd];
                const float denominator = rsum + zsum;
                if (std::abs(denominator) >= 1.0e-30F) {
                    const std::size_t base =
                        static_cast<std::size_t>(mode) * input.ns + surface;
                    out.residual[3 * points + base] *= rsum / denominator;
                    out.residual[4 * points + base] *= zsum / denominator;
                }
            }
        }
    }
    const int last_surface = input.include_lcfs ? input.ns : input.ns - 1;
    for (int mode = 0; mode < modes; ++mode) {
        const float relative_floor = std::numeric_limits<float>::epsilon();
        const float scale = input.matrix.scale[mode];
        const float floor =
            scale > 0.0F ? relative_floor * scale : relative_floor;
        bool broke =
            solve_pair(input, false, mode, last_surface, floor, out.residual);
        broke =
            solve_pair(input, true, mode, last_surface, floor, out.residual) ||
            broke;
        out.breakdown_count += broke ? 1 : 0;
        const int first = input.matrix.first_surface[mode];
        for (int surface = 0; surface < first; ++surface) {
            for (int component = 0; component < 5; ++component) {
                out.residual[static_cast<std::size_t>(component) * points +
                             static_cast<std::size_t>(mode) * input.ns +
                             surface] = 0.0F;
            }
        }
        for (int surface = 0; surface < input.ns; ++surface) {
            const std::size_t matrix_index =
                static_cast<std::size_t>(mode) * input.ns + surface;
            out.residual[2 * points + matrix_index] *=
                input.matrix.lambda[matrix_index];
            out.residual[5 * points + matrix_index] *=
                input.matrix.lambda[matrix_index];
        }
    }
    return out;
}

void enqueue_axisymmetric_preconditioner_apply(
    const wgpu::Device& device,
    const AxisymmetricPreconditionerApplyCase& input,
    AxisymmetricPreconditionerApplyCallback callback) {
    const auto error = validate_case(input);
    if (!error.empty()) {
        callback(error, {});
        return;
    }
    const auto shader_text = load_shader();
    if (shader_text.empty()) {
        callback("cannot load axisymmetric preconditioner-apply shader", {});
        return;
    }
    const int modes = mode_count(input);
    const std::size_t points = static_cast<std::size_t>(input.ns) * modes;
    auto matrix = flatten_matrix(input.matrix);
    auto elements = flatten_elements(input.elements);
    const auto matrix_bytes = matrix.size() * sizeof(float);
    const auto element_bytes = elements.size() * sizeof(float);
    const auto input_bytes = input.residual.size() * sizeof(float);
    const auto output_values = 6 * points + modes;
    const auto output_bytes = output_values * sizeof(float);
    auto matrix_buffer =
        make_buffer(device, matrix_bytes,
                    wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                    "preconditioner apply matrix");
    auto element_buffer =
        make_buffer(device, element_bytes,
                    wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                    "preconditioner apply elements");
    auto input_buffer =
        make_buffer(device, input_bytes,
                    wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                    "preconditioner residual input");
    auto output_buffer =
        make_buffer(device, output_bytes,
                    wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc,
                    "preconditioned residual");
    auto readback =
        make_buffer(device, output_bytes,
                    wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead,
                    "preconditioned residual readback");
    auto params_buffer =
        make_buffer(device, sizeof(Params),
                    wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst,
                    "preconditioner apply params");
    const auto& pipeline = detail::cached_compute_pipeline(
        device, "preconditioner-apply", shader_text,
        "cuMES preconditioner apply pipeline");
    const Params params{static_cast<std::uint32_t>(input.ns),
                        static_cast<std::uint32_t>(modes),
                        static_cast<std::uint32_t>(input.ntor),
                        static_cast<std::uint32_t>(points),
                        static_cast<std::uint32_t>(
                            input.include_lcfs ? input.ns : input.ns - 1),
                        {0, 0, 0}};
    auto queue = device.GetQueue();
    queue.WriteBuffer(matrix_buffer, 0, matrix.data(), matrix_bytes);
    queue.WriteBuffer(element_buffer, 0, elements.data(), element_bytes);
    queue.WriteBuffer(input_buffer, 0, input.residual.data(), input_bytes);
    queue.WriteBuffer(params_buffer, 0, &params, sizeof(params));
    auto layout = pipeline.GetBindGroupLayout(0);
    const wgpu::BindGroupEntry entries[] = {
        {nullptr, 0, matrix_buffer, 0, matrix_bytes, nullptr, nullptr},
        {nullptr, 1, element_buffer, 0, element_bytes, nullptr, nullptr},
        {nullptr, 2, input_buffer, 0, input_bytes, nullptr, nullptr},
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
    pass.DispatchWorkgroups(static_cast<std::uint32_t>(modes));
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
    dispatch->mpol = modes;
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
                    "preconditioner apply mapping failed: " + detail, {});
                return;
            }
            const auto* values = static_cast<const float*>(
                dispatch->readback.GetConstMappedRange(0, dispatch->bytes));
            if (values == nullptr) {
                dispatch->callback("preconditioner apply mapped range is null",
                                   {});
                return;
            }
            AxisymmetricPreconditionerApplyResult out;
            out.residual.assign(values, values + 6 * dispatch->points);
            for (int mode = 0; mode < dispatch->mpol; ++mode) {
                out.breakdown_count +=
                    values[6 * dispatch->points + mode] != 0.0F ? 1 : 0;
            }
            dispatch->readback.Unmap();
            dispatch->callback({}, std::move(out));
        });
}

}  // namespace cumes::webgpu
