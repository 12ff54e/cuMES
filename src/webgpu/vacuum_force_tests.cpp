#include "vacuum_force_tests.hpp"

#include "cumes/webgpu/axisymmetric.hpp"
#include "cumes/webgpu/float_float.hpp"
#include "cumes/webgpu/force.hpp"
#include "cumes/webgpu/geometry.hpp"
#include "cumes/webgpu/vacuum_force.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace cumes::webgpu {
namespace {

struct TestFields {
    DeviceFields device;
    std::vector<float> words;
    std::size_t low = 0;

    explicit TestFields(std::size_t count)
        : words(2 * count + 12, -1234.5F), low(count + 7) {
        device.values = count;
        device.high_offset = 4 * sizeof(float);
        device.low_offset = low * sizeof(float);
    }
    void put(std::size_t index, double value) {
        const auto pair = split(value);
        words[4 + index] = pair.hi;
        words[low + index] = pair.lo;
    }
    double get(std::size_t index, bool paired) const {
        return double(words[4 + index]) + (paired ? words[low + index] : 0.0);
    }
    void upload(const wgpu::Device& gpu) {
        wgpu::BufferDescriptor descriptor{};
        descriptor.size = words.size() * sizeof(float);
        descriptor.usage = wgpu::BufferUsage::Storage |
                           wgpu::BufferUsage::CopySrc |
                           wgpu::BufferUsage::CopyDst;
        device.buffer = gpu.CreateBuffer(&descriptor);
        gpu.GetQueue().WriteBuffer(device.buffer, 0, words.data(),
                                   descriptor.size);
    }
};

class VacuumForceTests : public std::enable_shared_from_this<VacuumForceTests> {
   public:
    wgpu::Device device;
    std::function<void(std::string)> callback;
    double max_force_error = 0.0, max_pressure_error = 0.0;

    void run(int variant = 0) {
        if (variant == 8) {
            std::printf(
                "  resident vacuum force (f32/paired, axisymmetric/3D), "
                "interior preservation, malformed ranges, nonfinite gates: "
                "PASS (scaled force %.3e, pressure %.3e)\n",
                max_force_error, max_pressure_error);
            callback({});
            return;
        }
        VacuumForceCase input;
        input.ns = 5;
        input.ntheta = variant < 2 ? 8 : 18;
        input.nzeta = variant < 2 ? 1 : 18;
        input.paired = variant % 2 != 0 || variant >= 4;
        input.delta_s = 1.0 / 7.0;
        input.edge_pressure = 0.03765432123456;
        const auto angular = std::size_t(input.ntheta) * input.nzeta;
        const auto full = input.ns * angular;
        const auto half = (input.ns - 1) * angular;
        TestFields geometry(GEOMETRY_PARITY_FIELD_COUNT * full);
        TestFields magnetic(MAGNETIC_FIELD_COUNT * half);
        TestFields force(FORCE_FIELD_COUNT * full);
        for (std::size_t i = 0; i < geometry.device.values; ++i)
            geometry.put(i, 0.05 * double(int(i % 37) - 18) + 1.23e-9);
        for (std::size_t i = 0; i < magnetic.device.values; ++i)
            magnetic.put(i, 0.3 + 0.015 * double(i % 23) + 7.65e-10);
        for (std::size_t i = 0; i < force.device.values; ++i)
            force.put(i, 0.0625 * double(int(i % 29) - 14) + 3.21e-9);
        for (std::size_t point = 0; point < angular; ++point) {
            geometry.put(full - angular + point,
                         5.0 + 0.001 * double(point) + 1.09e-8);
            // Include cancellation in a parity sum and in the final force.
            const auto even_z = 5 * full - angular + point;
            const auto odd_z = 11 * full - angular + point;
            if (point % 5 == 0) {
                geometry.put(even_z, 1.23456789123);
                geometry.put(odd_z, -1.23456780123);
            }
        }
        const auto vacuum_count =
            std::size_t(input.ntheta / 2 + 1) * input.nzeta;
        std::vector<float> vacuum(2 * vacuum_count + 4, -456.0F);
        for (std::size_t i = 0; i < vacuum_count; ++i) {
            const auto value = split(0.4 + 0.013 * double(i % 19) + 4.56e-9);
            vacuum[2 + 2 * i] = value.hi;
            vacuum[3 + 2 * i] = value.lo;
        }
        if (variant == 4)
            force.words[force.low + 4 * full - 1] =
                std::numeric_limits<float>::quiet_NaN();
        if (variant == 5) vacuum[3] = std::numeric_limits<float>::infinity();
        if (variant == 6) geometry.put(full - angular, 2.0e38);
        if (variant == 7)
            magnetic.words[magnetic.low + 5 * half - 1] =
                std::numeric_limits<float>::infinity();
        geometry.upload(device);
        magnetic.upload(device);
        force.upload(device);
        input.geometry = geometry.device;
        input.magnetic_field = magnetic.device;
        input.force = force.device;
        wgpu::BufferDescriptor descriptor{};
        descriptor.size = vacuum.size() * sizeof(float);
        descriptor.usage =
            wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
        input.vacuum_pressure = {device.CreateBuffer(&descriptor),
                                 2 * sizeof(float), vacuum_count};
        device.GetQueue().WriteBuffer(input.vacuum_pressure.buffer, 0,
                                      vacuum.data(), descriptor.size);
        const auto force_bytes = force.words.size() * sizeof(float);
        input.readback.batch = std::make_shared<ReadbackBatch>(
            device, 3 * angular * sizeof(float) + force_bytes + 16);
        if (variant == 0) {
            for (int malformed = 0; malformed < 5; ++malformed) {
                auto bad = input;
                if (malformed == 0)
                    bad.force.high_offset = force.device.buffer.GetSize();
                else if (malformed == 1)
                    --bad.geometry.values;
                else if (malformed == 2)
                    bad.vacuum_pressure.byte_offset += sizeof(float);
                else if (malformed == 3)
                    --bad.vacuum_pressure.count;
                else
                    bad.delta_s = 0.0;
                bool rejected = false;
                enqueue_vacuum_force(
                    device, bad,
                    [&rejected](std::string error, VacuumForceResult) {
                        rejected = !error.empty();
                    });
                if (!rejected) {
                    callback("vacuum force accepted malformed input");
                    return;
                }
            }
        }
        auto result = std::make_shared<VacuumForceResult>();
        auto error = std::make_shared<std::string>();
        bool published = false;
        input.readback.device_ready = [&](VacuumForceResult ready) {
            published =
                ready.device_fields.buffer.Get() == input.force.buffer.Get();
        };
        enqueue_vacuum_force(
            device, input,
            [result, error](std::string message, VacuumForceResult actual) {
                *error = std::move(message);
                *result = std::move(actual);
            });
        if (!published || !error->empty()) {
            callback("vacuum force did not publish its resident output: " +
                     *error);
            return;
        }
        input.readback.device_ready = {};
        auto actual = std::make_shared<std::vector<float>>();
        const auto encoder = device.CreateCommandEncoder();
        input.readback.batch->append(
            encoder, input.force.buffer, 0, force_bytes,
            [actual](std::span<const float> values) {
                actual->assign(values.begin(), values.end());
            });
        const auto commands = encoder.Finish();
        device.GetQueue().Submit(1, &commands);
        const auto self = shared_from_this();
        input.readback.batch->map([self, input, geometry = std::move(geometry),
                                   magnetic = std::move(magnetic),
                                   force = std::move(force),
                                   vacuum = std::move(vacuum), result, error,
                                   actual, variant, angular, full,
                                   half](std::string message) {
            if (!message.empty() || !error->empty() ||
                actual->size() != force.words.size() ||
                result->finite != (variant < 4)) {
                self->callback("vacuum force status mismatch variant=" +
                               std::to_string(variant) + ": " + message +
                               *error);
                return;
            }
            std::vector<bool> corrected(force.words.size(), false);
            double delbsq = 0.0, pressure_scale = 0.0;
            for (std::size_t point = 0; point < angular; ++point) {
                // Independent scalar-double expression of the shared
                // rBSq/edge-force contract, including stellarator symmetry.
                const int theta = point % input.ntheta;
                const int zeta = point / input.ntheta;
                const bool reflected = theta > input.ntheta / 2;
                const int l = reflected ? input.ntheta - theta : theta;
                const int k =
                    reflected ? (input.nzeta - zeta) % input.nzeta : zeta;
                const auto index = 2 + 2 * (l * input.nzeta + k);
                const double outside = double(vacuum[index]) +
                                       vacuum[index + 1] + input.edge_pressure;
                const double inside =
                    1.5 *
                        magnetic.get(5 * half - angular + point, input.paired) -
                    0.5 * magnetic.get(5 * half - 2 * angular + point,
                                       input.paired);
                delbsq += std::abs(outside - inside) / double(angular);
                pressure_scale +=
                    (std::abs(outside) + std::abs(inside)) / double(angular);
                const auto g = [&](int field) {
                    return geometry.get((field + 1) * full - angular + point,
                                        input.paired);
                };
                const double rbsq = outside * (g(0) + g(6)) / input.delta_s;
                const double radial = (g(4) + g(10)) * rbsq;
                const double vertical = -(g(3) + g(9)) * rbsq;
                for (int field = 0; field < 4; ++field) {
                    const auto force_index =
                        (field + 1) * full - angular + point;
                    const auto hi_index = 4 + force_index;
                    const auto lo_index = force.low + force_index;
                    corrected[hi_index] = true;
                    if (input.paired) corrected[lo_index] = true;
                    if (variant >= 4) continue;
                    const double previous =
                        force.get(force_index, input.paired);
                    const double increment = field < 2 ? radial : vertical;
                    const double expected = previous + increment;
                    const double observed =
                        double((*actual)[hi_index]) +
                        (input.paired ? (*actual)[lo_index] : 0.0F);
                    const double scale =
                        std::abs(previous) + std::abs(increment);
                    const double tolerance = input.paired ? 4.0e-12 : 1.3e-7;
                    const double scaled_error =
                        std::abs(observed - expected) / (1.0 + scale);
                    self->max_force_error =
                        std::max(self->max_force_error, scaled_error);
                    if (!std::isfinite(observed) || scaled_error > tolerance) {
                        self->callback(
                            "vacuum force scalar reference mismatch variant=" +
                            std::to_string(variant) +
                            " field=" + std::to_string(field));
                        return;
                    }
                }
            }
            if (variant < 4) {
                const double error_value =
                    std::abs(result->delbsq - delbsq) / (1.0 + pressure_scale);
                self->max_pressure_error =
                    std::max(self->max_pressure_error, error_value);
                if (error_value > 4.0e-12) {
                    self->callback(
                        "vacuum force ordered pressure-error sum mismatch");
                    return;
                }
            }
            for (std::size_t i = 0; i < force.words.size(); ++i) {
                if (!corrected[i] &&
                    std::bit_cast<std::uint32_t>((*actual)[i]) !=
                        std::bit_cast<std::uint32_t>(force.words[i])) {
                    self->callback(
                        "vacuum force changed an interior, unused, or padding "
                        "word");
                    return;
                }
            }
            self->run(variant + 1);
        });
    }
};
}  // namespace

void run_vacuum_force_tests(const wgpu::Device& device,
                            std::function<void(std::string)> callback) {
    auto tests = std::make_shared<VacuumForceTests>();
    tests->device = device;
    tests->callback = std::move(callback);
    tests->run();
}

}  // namespace cumes::webgpu
