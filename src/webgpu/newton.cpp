#include "cumes/webgpu/newton.hpp"

#include "pipeline_cache.hpp"
#include "shader_source.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace cumes::webgpu {
namespace {

wgpu::Buffer make_buffer(const wgpu::Device& device,
                         std::uint64_t bytes,
                         wgpu::BufferUsage usage,
                         const char* label) {
    wgpu::BufferDescriptor descriptor{};
    descriptor.label = label;
    descriptor.size = bytes;
    descriptor.usage = usage;
    return device.CreateBuffer(&descriptor);
}

constexpr auto STORAGE = wgpu::BufferUsage::Storage |
                         wgpu::BufferUsage::CopySrc |
                         wgpu::BufferUsage::CopyDst;
constexpr auto UNIFORM =
    wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;

void check_field(const DeviceFields& fields,
                 std::size_t count,
                 bool paired = false) {
    const auto bytes = std::uint64_t(count) * sizeof(float);
    const auto size = fields ? fields.buffer.GetSize() : 0;
    const auto fits = [&](std::uint64_t offset) {
        return offset % sizeof(float) == 0 && offset <= size &&
               bytes <= size - offset;
    };
    if (!fields || fields.values != count || !fits(fields.high_offset) ||
        (paired && !fits(fields.low_offset)) ||
        size / sizeof(float) > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("invalid Newton device field range");
}

void copy(const wgpu::Device& device,
          const DeviceFields& source,
          const wgpu::Buffer& output,
          std::uint32_t offset,
          bool low = false) {
    const auto encoder = device.CreateCommandEncoder();
    encoder.CopyBufferToBuffer(
        source.buffer, low ? source.low_offset : source.high_offset, output,
        std::uint64_t(offset) * sizeof(float), source.values * sizeof(float));
    const auto commands = encoder.Finish();
    device.GetQueue().Submit(1, &commands);
}

template <class Params>
void dispatch(const wgpu::Device& device,
              const wgpu::Buffer& uniform,
              const Params& params,
              std::vector<wgpu::BindGroupEntry> entries,
              const char* shader,
              const char* entry,
              std::uint32_t groups = 1) {
    device.GetQueue().WriteBuffer(uniform, 0, &params, sizeof(params));
    const auto path = std::string("/shaders/") + shader;
    const auto& pipeline = detail::cached_compute_pipeline(
        device, std::string(shader) + "/" + entry,
        detail::cached_shader_source(path.c_str()), entry, entry);
    entries.push_back({nullptr, static_cast<std::uint32_t>(entries.size()),
                       uniform, 0, sizeof(params), nullptr, nullptr});
    wgpu::BindGroupDescriptor descriptor{};
    descriptor.layout = pipeline.GetBindGroupLayout(0);
    descriptor.entryCount = entries.size();
    descriptor.entries = entries.data();
    const auto group = device.CreateBindGroup(&descriptor);
    const auto encoder = device.CreateCommandEncoder();
    auto pass = encoder.BeginComputePass();
    pass.SetPipeline(pipeline);
    pass.SetBindGroup(0, group);
    pass.DispatchWorkgroups(groups);
    pass.End();
    const auto commands = encoder.Finish();
    device.GetQueue().Submit(1, &commands);
}

}  // namespace

struct DeviceGmres::Impl {
    struct Params {
        std::uint32_t size, capacity, column = 0, pad = 0;
        std::uint32_t basis = 0, work, residual, x, rhs, h, projection;
        std::uint32_t cosine, sine, g, y, control;
        float tolerance = 0;
        std::uint32_t pad1 = 0, pad2 = 0, pad3 = 0;
    } params;
    wgpu::Device device;
    wgpu::Buffer workspace, uniform;
    std::shared_ptr<ReadbackBatch> readback;
    bool busy = false;

    Impl(const wgpu::Device& device, int size, int capacity) : device(device) {
        if (size < 1 || capacity < 1 || capacity > 256 ||
            std::uint64_t(size) * (capacity + 6) > (1U << 26))
            throw std::invalid_argument("invalid WebGPU GMRES dimensions");
        params.size = static_cast<std::uint32_t>(size);
        params.capacity = static_cast<std::uint32_t>(capacity);
        params.work = params.size * (params.capacity + 1);
        params.residual = params.work + params.size;
        params.x = params.residual + params.size;
        params.rhs = params.x + params.size;
        params.h = params.rhs + params.size;
        params.projection = params.h + (params.capacity + 1) * params.capacity;
        params.cosine = params.projection + params.capacity;
        params.sine = params.cosine + params.capacity;
        params.g = params.sine + params.capacity;
        params.y = params.g + params.capacity + 1;
        params.control = params.y + params.capacity;
        workspace =
            make_buffer(device, (params.control + 16ULL) * sizeof(float),
                        STORAGE, "Newton GMRES workspace");
        uniform = make_buffer(device, sizeof(params), UNIFORM,
                              "Newton GMRES parameters");
        readback = std::make_shared<ReadbackBatch>(device, 16 * sizeof(float));
    }

    DeviceFields view(std::uint32_t offset) const {
        return {workspace, params.size, std::uint64_t(offset) * sizeof(float),
                0};
    }

    void run(const char* entry, std::uint32_t groups = 1) {
        dispatch(
            device, uniform, params,
            {{nullptr, 0, workspace, 0, workspace.GetSize(), nullptr, nullptr}},
            "newton_gmres.wgsl", entry, groups);
    }
};

DeviceGmres::DeviceGmres(const wgpu::Device& device, int size, int max_basis)
    : impl_(std::make_shared<Impl>(device, size, max_basis)) {}

DeviceFields DeviceGmres::correction() const {
    return impl_->view(impl_->params.x);
}

void DeviceGmres::enqueue(const DeviceFields& rhs,
                          Apply apply,
                          int iterations,
                          float tolerance,
                          NewtonCallback callback) {
    auto self = impl_;
    if (self->busy || iterations < 1 || iterations > 4096 ||
        !(tolerance > 0 && tolerance < 1) || !std::isfinite(tolerance) ||
        !apply)
        throw std::invalid_argument("invalid or overlapping GMRES solve");
    check_field(rhs, self->params.size);
    self->busy = true;
    self->params.tolerance = tolerance;
    const auto blocks = (self->params.size + 255) / 256;
    int evaluations = 0;
    try {
        copy(self->device, rhs, self->workspace, self->params.rhs);
        self->run("initialize");
        for (int completed = 0; completed < iterations;) {
            const auto count =
                std::min<int>(self->params.capacity, iterations - completed);
            self->run("begin_cycle");
            for (int j = 0; j < count; ++j) {
                const auto output = apply(self->view(j * self->params.size));
                ++evaluations;
                check_field(output, self->params.size);
                copy(self->device, output, self->workspace, self->params.work);
                self->params.column = static_cast<std::uint32_t>(j);
                for (int sweep = 0; sweep < 2; ++sweep) {
                    self->run("project", j + 1);
                    self->run("subtract_projection", blocks);
                }
                self->run("arnoldi");
            }
            self->run("backsolve");
            self->run("update", blocks);
            const auto output = apply(self->view(self->params.x));
            ++evaluations;
            check_field(output, self->params.size);
            copy(self->device, output, self->workspace, self->params.work);
            self->run("residual", blocks);
            self->run("finish_cycle");
            completed += count;
        }
        auto result = std::make_shared<NewtonControl>();
        const auto encoder = self->device.CreateCommandEncoder();
        self->readback->append(
            encoder, self->workspace,
            std::uint64_t(self->params.control) * sizeof(float),
            16 * sizeof(float),
            [result, evaluations](std::span<const float> values) {
                result->rhs_norm = values[0];
                result->residual_norm = values[1];
                result->target_norm = values[2];
                result->steps = static_cast<int>(values[3]);
                result->cycles = static_cast<int>(values[4]);
                result->converged = values[6] != 0;
                result->breakdown = static_cast<int>(values[7]);
                result->evaluations = evaluations;
            });
        const auto commands = encoder.Finish();
        self->device.GetQueue().Submit(1, &commands);
        self->readback->map(
            [self, result, callback = std::move(callback)](std::string error) {
                self->busy = false;
                callback(std::move(error), *result);
            });
    } catch (...) {
        self->busy = false;
        throw;
    }
}

struct NewtonCorrection::Impl {
    struct Params {
        std::uint32_t ns, mpol, size, source = 0;
        std::uint32_t base = 0, trial, rhs, positive, negative, result, control;
        std::uint32_t destination = 0, count = 0, low = 0, paired = 0;
        std::uint32_t normalized = 0;
        float scale = 0;
        std::uint32_t central = 0, pad0 = 0, pad1 = 0;
    } params;
    wgpu::Device device;
    wgpu::Buffer workspace, uniform, probe_control, direction_copy;
    DeviceGmres krylov;
    IterationCase frozen;
    bool prepared = false, busy = false;

    Impl(const wgpu::Device& device, int ns, int mpol)
        : device(device), krylov(device, checked_size(ns, mpol)) {
        params.ns = static_cast<std::uint32_t>(ns);
        params.mpol = static_cast<std::uint32_t>(mpol);
        params.size = static_cast<std::uint32_t>(checked_size(ns, mpol));
        params.trial = 2 * params.size;
        params.rhs = params.trial + 2 * params.size;
        params.positive = params.rhs + params.size;
        params.negative = params.positive + params.size;
        params.result = params.negative + params.size;
        params.control = params.result + params.size;
        workspace = make_buffer(device, (params.control + 4ULL) * sizeof(float),
                                STORAGE, "Newton paired state workspace");
        uniform = make_buffer(device, sizeof(params), UNIFORM,
                              "Newton coordinate parameters");
        probe_control = make_buffer(device, 35 * sizeof(float), STORAGE,
                                    "Newton probe validity records");
        direction_copy = make_buffer(device, params.size * sizeof(float),
                                     STORAGE, "Newton diagnostic direction");
    }

    static int checked_size(int ns, int mpol) {
        if (ns < 2 || ns > 512 || mpol < 2 || mpol > 256)
            throw std::invalid_argument("invalid axisymmetric Newton shape");
        return 6 * ns * mpol;
    }

    DeviceFields view(std::uint32_t offset, bool paired = false) const {
        const auto high = std::uint64_t(offset) * sizeof(float);
        return {workspace, params.size, high,
                paired ? high + params.size * sizeof(float) : 0};
    }

    void run(const char* entry,
             const DeviceFields& source,
             std::uint32_t groups = 1) {
        params.source = static_cast<std::uint32_t>(source.high_offset / 4);
        params.low = static_cast<std::uint32_t>(source.low_offset / 4);
        // difference reads only the owned workspace; its automatic layout
        // omits the external source binding but keeps uniform binding 2.
        if (std::string_view(entry) == "difference") {
            device.GetQueue().WriteBuffer(uniform, 0, &params, sizeof(params));
            const auto& pipeline = detail::cached_compute_pipeline(
                device, "newton_coordinates.wgsl/difference",
                detail::cached_shader_source(
                    "/shaders/newton_coordinates.wgsl"),
                "Newton difference", "difference");
            const wgpu::BindGroupEntry entries[] = {
                {nullptr, 0, workspace, 0, workspace.GetSize(), nullptr,
                 nullptr},
                {nullptr, 2, uniform, 0, sizeof(params), nullptr, nullptr}};
            wgpu::BindGroupDescriptor descriptor{};
            descriptor.layout = pipeline.GetBindGroupLayout(0);
            descriptor.entryCount = std::size(entries);
            descriptor.entries = entries;
            const auto group = device.CreateBindGroup(&descriptor);
            const auto encoder = device.CreateCommandEncoder();
            auto pass = encoder.BeginComputePass();
            pass.SetPipeline(pipeline);
            pass.SetBindGroup(0, group);
            pass.DispatchWorkgroups(groups);
            pass.End();
            const auto commands = encoder.Finish();
            device.GetQueue().Submit(1, &commands);
            return;
        }
        dispatch(
            device, uniform, params,
            {{nullptr, 0, workspace, 0, workspace.GetSize(), nullptr, nullptr},
             {nullptr, 1, source.buffer, 0, source.buffer.GetSize(), nullptr,
              nullptr}},
            "newton_coordinates.wgsl", entry, groups);
    }

    void pack(const DeviceFields& source, std::uint32_t destination) {
        check_field(source, params.size);
        params.destination = destination;
        run("pack", source, (params.size + 255) / 256);
    }

    DeviceFields trial(const DeviceFields& direction,
                       float scale,
                       bool normalized) {
        check_field(direction, params.size);
        const auto input = external_direction(direction);
        params.scale = scale;
        params.normalized = normalized ? 1 : 0;
        run("trial", input, (params.size + 255) / 256);
        return view(params.trial, true);
    }

    DeviceFields external_direction(const DeviceFields& direction) {
        // A diagnostic may use rhs(), which shares the writable state buffer.
        // WebGPU forbids binding overlapping writable/read-only buffer ranges.
        if (direction.buffer.Get() != workspace.Get()) return direction;
        copy(device, direction, direction_copy, 0);
        return {direction_copy, params.size, 0, 0};
    }

    DeviceFields evaluate(const DeviceFields& state) {
        frozen.device_state = state;
        const auto value =
            enqueue_iteration_probe(device, frozen, {probe_control, 35, 0, 0});
        run("check_control", value.control);
        for (const auto& field : {value.inverse, value.magnetic}) {
            check_field(field, field.values, true);
            params.count = static_cast<std::uint32_t>(field.values);
            params.paired = 1;
            run("check_values", field);
        }
        run("check_preconditioner", value.preconditioned);
        return value.preconditioned;
    }

    DeviceFields jvp(const DeviceFields& direction, float step, bool central) {
        if (!prepared || !(step > 0) || !std::isfinite(step))
            throw std::invalid_argument("invalid Newton JVP setup or step");
        check_field(direction, params.size);
        const auto input = external_direction(direction);
        params.scale = step;
        run("difference_scale", input);
        pack(evaluate(trial(input, 1, true)), params.positive);
        if (central) pack(evaluate(trial(input, -1, true)), params.negative);
        params.central = central ? 1 : 0;
        run("difference", {}, (params.size + 255) / 256);
        return view(params.result);
    }
};

NewtonCorrection::NewtonCorrection(const wgpu::Device& device, int ns, int mpol)
    : impl_(std::make_shared<Impl>(device, ns, mpol)) {}

void NewtonCorrection::prepare(IterationCase frozen,
                               const DeviceFields& base,
                               const DeviceFields& preconditioned) {
    auto& state = *impl_;
    if (state.busy || frozen.stage.ns != static_cast<int>(state.params.ns) ||
        frozen.stage.mpol != static_cast<int>(state.params.mpol) ||
        !frozen.double_single || frozen.stage.free_boundary ||
        frozen.stage.ntor != 0 || frozen.stage.nzeta != 1)
        throw std::invalid_argument("invalid Newton base state");
    check_field(base, state.params.size, true);
    check_field(preconditioned, state.params.size);
    frozen.refresh_preconditioner = false;
    frozen.reset_reference = false;
    frozen.stage.state.clear();
    frozen.stage.state_lo.clear();
    state.frozen = std::move(frozen);
    copy(state.device, base, state.workspace, state.params.base);
    copy(state.device, base, state.workspace,
         state.params.base + state.params.size, true);
    state.pack(preconditioned, state.params.rhs);
    state.prepared = true;
}

void NewtonCorrection::solve(const NewtonOptions& options,
                             NewtonCallback callback) {
    auto self = impl_;
    if (!self->prepared || self->busy)
        throw std::invalid_argument(
            "Newton solve requires an idle frozen base");
    self->busy = true;
    try {
        self->krylov.enqueue(
            self->view(self->params.rhs),
            [self, options](const DeviceFields& direction) {
                return self->jvp(direction, options.difference_step,
                                 options.central_difference);
            },
            options.iterations, options.tolerance,
            [self, options, callback = std::move(callback)](
                std::string error, NewtonControl control) {
                self->busy = false;
                if (options.central_difference) control.evaluations *= 2;
                callback(std::move(error), control);
            });
    } catch (...) {
        self->busy = false;
        throw;
    }
}

DeviceFields NewtonCorrection::enqueue_trial(float scale) {
    return enqueue_trial(impl_->krylov.correction(), scale);
}

DeviceFields NewtonCorrection::enqueue_trial(const DeviceFields& direction,
                                             float scale) {
    if (!impl_->prepared || impl_->busy || !std::isfinite(scale))
        throw std::invalid_argument("invalid Newton trial");
    return impl_->trial(direction, scale, false);
}

DeviceFields NewtonCorrection::base_state() const {
    return impl_->view(impl_->params.base, true);
}
DeviceFields NewtonCorrection::rhs() const {
    return impl_->view(impl_->params.rhs);
}
DeviceFields NewtonCorrection::correction() const {
    return impl_->krylov.correction();
}
DeviceFields NewtonCorrection::enqueue_jvp(const DeviceFields& direction,
                                           float step,
                                           bool central) {
    if (impl_->busy) throw std::invalid_argument("Newton solve is active");
    return impl_->jvp(direction, step, central);
}

}  // namespace cumes::webgpu
