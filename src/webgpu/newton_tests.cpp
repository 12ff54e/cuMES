#include "newton_tests.hpp"

#include "cumes/webgpu/newton.hpp"
#include "pipeline_cache.hpp"
#include "shader_source.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>

namespace cumes::webgpu {
namespace {

wgpu::Buffer buffer(const wgpu::Device& device,
                    std::size_t bytes,
                    wgpu::BufferUsage usage) {
    wgpu::BufferDescriptor descriptor{};
    descriptor.label = "Newton conformance fixture";
    descriptor.size = bytes;
    descriptor.usage = usage;
    return device.CreateBuffer(&descriptor);
}

DeviceFields upload(const wgpu::Device& device,
                    const std::vector<float>& values,
                    bool paired = false) {
    auto result =
        buffer(device, values.size() * sizeof(float),
               wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst |
                   wgpu::BufferUsage::CopySrc);
    device.GetQueue().WriteBuffer(result, 0, values.data(),
                                  values.size() * sizeof(float));
    const auto count = paired ? values.size() / 2 : values.size();
    return {result, count, 0, paired ? count * sizeof(float) : 0};
}

class Tests : public std::enable_shared_from_this<Tests> {
   public:
    wgpu::Device device;
    std::function<void(std::string)> complete;

    void coordinates() {
        constexpr int NS = 5, MPOL = 4, COUNT = 6 * NS * MPOL;
        const float amount = 1.0e-4F;
        std::vector<float> base(2 * COUNT), direction(COUNT);
        for (int i = 0; i < COUNT; ++i) {
            base[i] = 2.0F + static_cast<float>(i) * 0.01F;
            base[COUNT + i] = 1.0e-8F;
            direction[i] = 0.01F * static_cast<float>(1 + i % 9);
        }
        base[0] = 1000.0F;
        base[NS - 1] = -0.0F;
        base[COUNT + NS - 1] = -0.0F;
        const auto initial = upload(device, base, true);
        const auto q = upload(device, direction);
        auto correction = std::make_shared<NewtonCorrection>(device, NS, MPOL);
        IterationCase frozen;
        frozen.double_single = true;
        frozen.stage.ns = NS;
        frozen.stage.mpol = MPOL;
        frozen.stage.ntor = 0;
        frozen.stage.nzeta = 1;
        correction->prepare(std::move(frozen), initial, q);
        const auto trial = correction->enqueue_trial(correction->rhs(), amount);
        auto batch =
            std::make_shared<ReadbackBatch>(device, 4 * COUNT * sizeof(float));
        const auto encoder = device.CreateCommandEncoder();
        auto error = std::make_shared<std::string>();
        batch->append(
            encoder, trial.buffer, trial.high_offset, 2 * COUNT * sizeof(float),
            [error, base, direction, amount](std::span<const float> values) {
                for (int i = 0; i < COUNT; ++i) {
                    const int c = i / (NS * MPOL), point = i % (NS * MPOL);
                    const int m = point / NS, j = point % NS;
                    const int source = j == 0 && m == 1 ? i + 1 : i;
                    const int radial = source % NS;
                    const bool movable =
                        (c == 2 ? m > 0 && radial > 0
                                : radial < NS - 1 && !(radial == 0 && m > 0) &&
                                      (c == 0 || (c == 1 && m > 0)));
                    const double increment =
                        movable
                            ? double(direction[source]) * amount *
                                  (m == 0 ? 1.0F
                                          : static_cast<float>(std::sqrt(2.0)))
                            : 0;
                    const double expected =
                        double(base[source]) + base[COUNT + source] + increment;
                    const double actual = double(values[i]) + values[COUNT + i];
                    if (!std::isfinite(actual) ||
                        std::abs(expected - actual) >
                            2.0e-12 * std::max(1.0, std::abs(expected))) {
                        *error =
                            "Newton active-coordinate/paired trial mismatch";
                        return;
                    }
                    if (!movable &&
                        (std::bit_cast<std::uint32_t>(values[i]) !=
                             std::bit_cast<std::uint32_t>(base[source]) ||
                         std::bit_cast<std::uint32_t>(values[COUNT + i]) !=
                             std::bit_cast<std::uint32_t>(
                                 base[COUNT + source]))) {
                        *error = "Newton trial changed an inactive coefficient";
                        return;
                    }
                }
                if (values[COUNT] == base[COUNT])
                    *error =
                        "Newton paired trial lost its sub-ULP displacement";
            });
        batch->append(
            encoder, correction->base_state().buffer,
            correction->base_state().high_offset, 2 * COUNT * sizeof(float),
            [error, base](std::span<const float> values) {
                for (std::size_t i = 0; i < base.size(); ++i) {
                    if (std::bit_cast<std::uint32_t>(values[i]) !=
                        std::bit_cast<std::uint32_t>(base[i])) {
                        *error = "Newton trial overwrote its frozen base";
                        return;
                    }
                }
            });
        const auto commands = encoder.Finish();
        device.GetQueue().Submit(1, &commands);
        batch->map([self = shared_from_this(), error,
                    correction](std::string message) {
            if (message.empty()) message = *error;
            if (!message.empty()) {
                self->complete(std::move(message));
                return;
            }
            std::printf(
                "  Newton active coordinates, paired trials and frozen base: "
                "PASS\n");
            self->gmres(0);
        });
    }

   private:
    void gmres(int index) {
        constexpr int COUNT = 37;
        if (index == 6) {
            std::printf(
                "  Newton GMRES analytic, restarted, tiny/zero RHS and "
                "breakdown cases: PASS\n");
            complete({});
            return;
        }
        // Identity, restarted nonsymmetric, tiny RHS, zero RHS, singular,
        // nonfinite. Known solutions establish a true residual independently.
        const int kind = index == 1 ? 1 : index == 4 ? 2 : index == 5 ? 3 : 0;
        const int capacity = index == 1 ? 3 : 8;
        const int iterations = index == 1 ? 30 : 8;
        const float magnitude = index == 2   ? 1.0e-30F
                                : index == 3 ? 0.0F
                                             : 1.0F;
        std::vector<float> expected(COUNT), rhs(COUNT);
        for (int i = 0; i < COUNT; ++i)
            expected[i] =
                magnitude * static_cast<float>(0.25 + std::sin(0.7 * i));
        for (int i = 0; i < COUNT; ++i) {
            rhs[i] = kind == 1 ? (1.0F + 0.125F * static_cast<float>(i % 7)) *
                                         expected[i] +
                                     0.08F * expected[(i + 1) % COUNT]
                               : expected[i];
        }
        const auto source = upload(device, rhs);
        const auto output =
            buffer(device, COUNT * sizeof(float),
                   wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc);
        struct Params {
            std::uint32_t count, offset, kind, pad = 0;
        };
        const auto uniform =
            buffer(device, sizeof(Params),
                   wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
        auto solver = std::make_shared<DeviceGmres>(device, COUNT, capacity);
        auto calls = std::make_shared<int>(0);
        const auto apply = [self = shared_from_this(), output, uniform, kind,
                            calls](const DeviceFields& input) {
            ++*calls;
            const Params params{
                COUNT, static_cast<std::uint32_t>(input.high_offset / 4),
                static_cast<std::uint32_t>(kind)};
            self->device.GetQueue().WriteBuffer(uniform, 0, &params,
                                                sizeof(params));
            const auto& pipeline = detail::cached_compute_pipeline(
                self->device, "newton-test-map",
                detail::cached_shader_source("/shaders/newton_test_map.wgsl"),
                "Newton analytic linear map");
            const wgpu::BindGroupEntry entries[] = {
                {nullptr, 0, input.buffer, 0, input.buffer.GetSize(), nullptr,
                 nullptr},
                {nullptr, 1, output, 0, output.GetSize(), nullptr, nullptr},
                {nullptr, 2, uniform, 0, sizeof(params), nullptr, nullptr}};
            wgpu::BindGroupDescriptor descriptor{};
            descriptor.layout = pipeline.GetBindGroupLayout(0);
            descriptor.entryCount = std::size(entries);
            descriptor.entries = entries;
            const auto group = self->device.CreateBindGroup(&descriptor);
            const auto encoder = self->device.CreateCommandEncoder();
            auto pass = encoder.BeginComputePass();
            pass.SetPipeline(pipeline);
            pass.SetBindGroup(0, group);
            pass.DispatchWorkgroups(1);
            pass.End();
            const auto commands = encoder.Finish();
            self->device.GetQueue().Submit(1, &commands);
            return DeviceFields{output, COUNT, 0, 0};
        };
        solver->enqueue(
            source, apply, iterations, 2.0e-5F,
            [self = shared_from_this(), solver, index, capacity, iterations,
             expected, rhs, kind, magnitude,
             calls](std::string message, NewtonControl control) {
                if (!message.empty()) {
                    self->complete(std::move(message));
                    return;
                }
                if (*calls !=
                        iterations + (iterations + capacity - 1) / capacity ||
                    control.evaluations != *calls ||
                    (index < 4 && (!control.converged || control.breakdown)) ||
                    (index == 3 && control.steps != 0) ||
                    (index == 4 && control.breakdown != 1) ||
                    (index == 5 && control.breakdown != 2)) {
                    self->complete("Newton GMRES control mismatch in case " +
                                   std::to_string(index));
                    return;
                }
                if (index >= 4) {
                    self->gmres(index + 1);
                    return;
                }
                auto error = std::make_shared<std::string>();
                auto batch = std::make_shared<ReadbackBatch>(
                    self->device, COUNT * sizeof(float));
                const auto encoder = self->device.CreateCommandEncoder();
                const auto result = solver->correction();
                batch->append(
                    encoder, result.buffer, result.high_offset,
                    COUNT * sizeof(float),
                    [error, expected, rhs, kind,
                     magnitude](std::span<const float> values) {
                        double residual = 0, norm = 0, error_norm = 0;
                        for (int i = 0; i < COUNT; ++i) {
                            const double ax =
                                kind == 1 ? (1 + 0.125 * (i % 7)) * values[i] +
                                                double(0.08F) *
                                                    values[(i + 1) % COUNT]
                                          : values[i];
                            residual +=
                                (double(rhs[i]) - ax) * (double(rhs[i]) - ax);
                            norm += double(rhs[i]) * rhs[i];
                            error_norm = std::max(
                                error_norm,
                                std::abs(double(values[i]) - expected[i]));
                        }
                        if (residual > 2.5e-9 * norm ||
                            error_norm > 2.0e-4 * magnitude)
                            *error =
                                "Newton GMRES failed independent "
                                "true-residual/solution check";
                    });
                const auto commands = encoder.Finish();
                self->device.GetQueue().Submit(1, &commands);
                batch->map([self, solver, error, index](std::string message) {
                    if (message.empty()) message = *error;
                    if (!message.empty()) {
                        self->complete(std::move(message));
                        return;
                    }
                    self->gmres(index + 1);
                });
            });
    }
};

}  // namespace

void run_newton_tests(const wgpu::Device& device,
                      std::function<void(std::string)> callback) {
    auto tests = std::make_shared<Tests>();
    tests->device = device;
    tests->complete = std::move(callback);
    try {
        tests->coordinates();
    } catch (const std::exception& error) { tests->complete(error.what()); }
}

}  // namespace cumes::webgpu
