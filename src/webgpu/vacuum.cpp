// Compile the existing vacuum coupling and bridge kernels for Wasm memory.
#include "../free_boundary_impl.cuh"

template class cumes::FreeBoundaryOperator<double>;

#include "cumes/config/profile_functions.hpp"
#include "cumes/config/validated_problem.hpp"
#include "cumes/webgpu/float_float.hpp"
#include "cumes/webgpu/vacuum.hpp"
#include "cumes/webgpu/vacuum_force.hpp"
#include "pipeline_cache.hpp"

#include <algorithm>
#include <array>
#include <cstdio>

namespace cumes::webgpu {
namespace {
std::vector<double> reconstruct(std::span<const float> high,
                                std::span<const float> low,
                                std::size_t offset,
                                std::size_t count) {
    const auto selected = high.subspan(offset, count);
    std::vector<double> values(selected.begin(), selected.end());
    if (!low.empty()) {
        if (low.size() != high.size())
            throw CumesError("vacuum precision shape mismatch");
        for (std::size_t i = 0; i < values.size(); ++i)
            values[i] += low[offset + i];
    }
    return values;
}
}  // namespace

std::unique_ptr<FreeBoundaryOperator<double>> create_vacuum(
    const ValidatedProblem& problem,
    const AxisymmetricStageData& stage,
    const wgpu::Device& device,
    bool use_webgpu,
    bool device_lu) {
    if (device_lu && !use_webgpu)
        throw CumesError("WebGPU LU requires vacuum=webgpu");
    const auto& free = problem.spec().free_boundary;
    FreeBoundaryOperator<double>::HostParams host;
    host.coils_file = free.coils_file;
    host.makegrid_parameters_file = free.makegrid_parameters_file;
    host.embedded_makegrid_parameters = free.embedded_makegrid_parameters;
    host.extcur = free.extcur;
    host.nvacskip = free.nvacskip;
    host.use_process_environment = false;
    DeviceParams<double> params{};
    params.ns = stage.ns;
    params.mpol = stage.mpol;
    params.ntor = stage.ntor;
    params.mnmax = stage.mpol * (stage.ntor + 1);
    params.ntheta = stage.ntheta;
    params.nzeta = stage.nzeta;
    params.nZnT = stage.ntheta * stage.nzeta;
    params.nfp = stage.nfp;
    auto vacuum = std::make_unique<FreeBoundaryOperator<double>>(host, params);
    if (use_webgpu) vacuum->enable_webgpu(device, device_lu);
    std::printf("  Vacuum backend: %s\n",
                !use_webgpu ? "HOST/Wasm double"
                : device_lu ? "WebGPU paired-f32 kernels and LU"
                            : "WebGPU paired-f32 kernels, Wasm double LU");
    return vacuum;
}

void prepare_vacuum_stage(FreeBoundaryOperator<double>& vacuum,
                          const ValidatedProblem& problem,
                          const AxisymmetricStageData& stage) {
    const double mass_edge = eval_mass_profile<double>(problem.spec(), 1.0);
    double pressure = eval_mass_profile<double>(
        problem.spec(), (stage.ns - 1.5) / (stage.ns - 1.0));
    if (pressure != 0.0) {
        pressure = mass_edge / pressure *
                   (static_cast<double>(stage.profiles.pres_h.back()) +
                    stage.profiles.pres_h_lo.back());
    }
    vacuum.set_edge_pressure(pressure);
}

void update_vacuum(FreeBoundaryOperator<double>& vacuum,
                   const AxisymmetricStageData& stage,
                   std::span<const float> state_lo,
                   const AxisymmetricForceCase& fields) {
    const int angular = stage.ntheta * stage.nzeta;
    const int points = stage.ns * angular;
    const int half = (stage.ns - 1) * angular;
    const int modes = stage.mpol * (stage.ntor + 1);
    const int family = modes * stage.ns;
    // NESTOR consumes only the outer two half-grid averages, the LCFS
    // coefficients and the magnetic axis. Reconstruct those slices only.
    const auto field = [&](int component) {
        const int count = 2 * angular;
        const int offset = fields.magnetic_field_is_vacuum
                               ? (component - 2) * count
                               : (component + 1) * half - count;
        return reconstruct(fields.magnetic_field, fields.magnetic_field_lo,
                           offset, count);
    };
    const auto bu = field(2);
    const auto bv = field(3);
    std::array<double, 4> edge_averages{};
    vacuum.enqueue_surface_averages(bu.data(), bv.data(), edge_averages.data(),
                                    3, stage.ntheta, stage.nzeta, nullptr);
    std::vector<double> averages(2 * (stage.ns - 1));
    std::copy_n(edge_averages.begin(), 2, averages.end() - stage.ns - 1);
    std::copy_n(edge_averages.begin() + 2, 2, averages.end() - 2);
    std::vector<double> boundary(4 * modes), lcfs(4 * modes);
    constexpr std::array<int, 4> FAMILIES{0, 3, 1, 4};
    for (int field_index = 0; field_index < 4; ++field_index) {
        for (int mode = 0; mode < modes; ++mode) {
            const auto index =
                FAMILIES[field_index] * family + (mode + 1) * stage.ns - 1;
            double value = stage.state[index];
            if (!state_lo.empty()) value += state_lo[index];
            boundary[field_index * modes + mode] = value;
        }
    }
    vacuum.enqueue_lcfs_repack(boundary.data(), boundary.data() + modes,
                               boundary.data() + 2 * modes,
                               boundary.data() + 3 * modes, lcfs.data(), 1,
                               modes, stage.mpol, stage.ntor, nullptr);
    const auto geometry = [&](int component) {
        const int offset =
            component * (fields.geometry_is_vacuum ? angular : points);
        return reconstruct(fields.geometry, fields.geometry_lo, offset,
                           angular);
    };
    const auto r_axis = geometry(0), z_axis = geometry(1);
    std::vector<double> axis(2 * stage.nzeta);
    vacuum.enqueue_axis_extract(r_axis.data(), z_axis.data(), axis.data(),
                                stage.ntheta, stage.nzeta, nullptr);
    vacuum.run_host_update(stage.ns, averages.data(),
                           averages.data() + stage.ns - 1, lcfs.data(),
                           axis.data(), axis.data() + stage.nzeta, nullptr);
}

void apply_vacuum_force(const wgpu::Device& device,
                        FreeBoundaryOperator<double>& vacuum,
                        const AxisymmetricStageData& stage,
                        const AxisymmetricForceCase& fields,
                        AxisymmetricForceResult& force) {
    const int angular = stage.ntheta * stage.nzeta;
    const int points = stage.ns * angular;
    const int half = (stage.ns - 1) * angular;
    if (force.lcfs_only &&
        (!device || !force.device_fields ||
         force.fields.size() != static_cast<std::size_t>(4 * angular) ||
         (!force.fields_lo.empty() &&
          force.fields_lo.size() != force.fields.size())))
        throw CumesError("compact vacuum force requires resident full fields");
    const auto geometry = [&](int field, int surfaces) {
        constexpr std::array<int, 12> PACKED_ROW{2, -1, -1, 10, 8, -1,
                                                 5, -1, -1, 11, 9, -1};
        const int offset = fields.geometry_is_vacuum
                               ? PACKED_ROW.at(field) * angular
                               : (field + 1) * points - surfaces * angular;
        return reconstruct(fields.geometry, fields.geometry_lo, offset,
                           surfaces * angular);
    };
    // The shared rBSq kernel needs the last three full-grid rows and two
    // half-grid rows. The edge-force kernel needs only the LCFS row.
    const auto r_e = geometry(0, 3), r_o = geometry(6, 3);
    const auto pressure = reconstruct(
        fields.magnetic_field, fields.magnetic_field_lo,
        fields.magnetic_field_is_vacuum ? 4 * angular : 5 * half - 2 * angular,
        2 * angular);
    std::vector<double> values(4 * angular), rbsq(angular);
    for (int field = 0; field < 4; ++field) {
        for (int point = 0; point < angular; ++point) {
            const int index = force.lcfs_only
                                  ? field * angular + point
                                  : (field + 1) * points - angular + point;
            double value = force.fields[index];
            if (!force.fields_lo.empty()) value += force.fields_lo[index];
            values[field * angular + point] = value;
        }
    }
    double delbsq = 0;
    vacuum.enqueue_rbsq(r_e.data(), r_o.data(), pressure.data(), rbsq.data(),
                        &delbsq, 3, stage.ntheta, stage.nzeta, angular,
                        1.0 / (stage.ns - 1), nullptr);
    const auto zu_e = geometry(4, 1), zu_o = geometry(10, 1);
    const auto ru_e = geometry(3, 1), ru_o = geometry(9, 1);
    vacuum.enqueue_edge_force(
        values.data(), values.data() + angular, values.data() + 2 * angular,
        values.data() + 3 * angular, zu_e.data(), zu_o.data(), ru_e.data(),
        ru_o.data(), rbsq.data(), 1, stage.ntheta, stage.nzeta, nullptr);
    // Only the LCFS force changes; preserve every interior GPU result word.
    for (int field = 0; field < 4; ++field) {
        for (int point = 0; point < angular; ++point) {
            const int index = force.lcfs_only
                                  ? field * angular + point
                                  : (field + 1) * points - angular + point;
            const auto pair = split(values[field * angular + point]);
            force.fields[index] = pair.hi;
            if (!force.fields_lo.empty()) force.fields_lo[index] = pair.lo;
        }
    }
    if (device && force.device_fields) {
        const auto queue = device.GetQueue();
        const auto& fields = force.device_fields;
        for (int field = 0; field < 4; ++field) {
            const std::size_t first = (field + 1) * points - angular;
            const auto host_first = force.lcfs_only ? field * angular : first;
            const auto offset = first * sizeof(float);
            queue.WriteBuffer(fields.buffer, fields.high_offset + offset,
                              force.fields.data() + host_first,
                              angular * sizeof(float));
            if (!force.fields_lo.empty())
                queue.WriteBuffer(fields.buffer, fields.low_offset + offset,
                                  force.fields_lo.data() + host_first,
                                  angular * sizeof(float));
        }
    } else {
        force.device_fields = {};
    }
    vacuum.set_delbsq(delbsq);
}

std::function<void()> enqueue_resident_vacuum_force(
    const wgpu::Device& device,
    FreeBoundaryOperator<double>& vacuum,
    const AxisymmetricStageData& stage,
    const AxisymmetricForceCase& fields,
    const AxisymmetricForceResult& force,
    const std::shared_ptr<ReadbackBatch>& batch) {
    struct Completion {
        std::array<float, 6> summary{};
        std::string error;
        VacuumForceResult force;
    };
    const auto result = std::make_shared<Completion>();
    const bool pending = vacuum.webgpu_result_pending();
    if (pending) {
        // These checks precede controller acceptance, but need not interrupt
        // device force/projection work. Preserve raw bits in the two flags.
        const auto integrals = vacuum.webgpu_output("driver.surface_integrals");
        const auto encoder = device.CreateCommandEncoder();
        batch->append(encoder, integrals.buffer, integrals.byte_offset, 16,
                      [result](std::span<const float> values) {
                          std::copy(values.begin(), values.end(),
                                    result->summary.begin());
                      });
        batch->append(encoder, vacuum.webgpu_result_flags(), 0, 8,
                      [result](std::span<const float> values) {
                          std::copy(values.begin(), values.end(),
                                    result->summary.begin() + 4);
                      });
        const auto commands = encoder.Finish();
        device.GetQueue().Submit(1, &commands);
    }
    const bool apply = vacuum.apply_edge_force();
    if (apply) {
        VacuumForceCase input;
        input.ns = stage.ns;
        input.ntheta = stage.ntheta;
        input.nzeta = stage.nzeta;
        input.paired = fields.double_single;
        input.delta_s = 1.0 / (stage.ns - 1);
        input.edge_pressure = vacuum.edge_pressure();
        input.geometry = fields.device_geometry;
        input.magnetic_field = fields.device_magnetic_field;
        input.force = force.device_fields;
        const auto pressure = vacuum.webgpu_output("driver.b_sq_vac");
        if (pressure.buffer) {
            input.vacuum_pressure = {pressure.buffer, pressure.byte_offset,
                                     pressure.count};
        } else {
            const auto host = vacuum.host_vacuum_pressure();
            std::vector<FloatFloat> words(host.size());
            std::transform(host.begin(), host.end(), words.begin(), split);
            const auto bytes = words.size() * sizeof(FloatFloat);
            auto buffer = detail::cached_buffer(
                device, bytes,
                wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst,
                "HOST vacuum pressure for device force");
            device.GetQueue().WriteBuffer(buffer, 0, words.data(), bytes);
            input.vacuum_pressure = {buffer, 0, words.size()};
        }
        input.readback.batch = batch;
        enqueue_vacuum_force(
            device, input,
            [result](std::string error, VacuumForceResult force) {
                result->error = std::move(error);
                result->force = std::move(force);
            });
    }
    return [&vacuum, result, pending, apply] {
        if (pending) vacuum.finish_webgpu_update(result->summary);
        if (!result->error.empty()) throw CumesError(result->error);
        if (!result->force.finite)
            throw CumesError("WebGPU vacuum force produced a nonfinite value");
        if (apply) vacuum.set_delbsq(result->force.delbsq);
    };
}

void decay_vacuum_reference(std::vector<float>& high, std::vector<float>& low) {
    for (std::size_t i = 0; i < high.size(); ++i) {
        const auto value = split(
            (static_cast<double>(high[i]) + (low.empty() ? 0.0 : low[i])) *
            control_policy::VACUUM_CONSTRAINT_DECAY_FACTOR);
        high[i] = value.hi;
        if (!low.empty()) low[i] = value.lo;
    }
}
}  // namespace cumes::webgpu
