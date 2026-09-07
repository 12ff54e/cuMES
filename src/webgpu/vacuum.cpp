// Compile the existing vacuum coupling and bridge kernels for Wasm memory.
#include "../free_boundary_impl.cuh"

template class cumes::FreeBoundaryOperator<double>;

#include "cumes/config/profile_functions.hpp"
#include "cumes/config/validated_problem.hpp"
#include "cumes/webgpu/float_float.hpp"
#include "cumes/webgpu/vacuum.hpp"

namespace cumes::webgpu {
namespace {
std::vector<double> reconstruct(std::span<const float> high,
                                std::span<const float> low) {
    std::vector<double> values(high.begin(), high.end());
    if (!low.empty()) {
        if (low.size() != high.size())
            throw CumesError("vacuum precision shape mismatch");
        for (std::size_t i = 0; i < values.size(); ++i) values[i] += low[i];
    }
    return values;
}
}  // namespace

std::unique_ptr<FreeBoundaryOperator<double>> create_vacuum(
    const ValidatedProblem& problem,
    const AxisymmetricStageData& stage) {
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
    return std::make_unique<FreeBoundaryOperator<double>>(host, params);
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
    auto state = reconstruct(stage.state, state_lo);
    const auto geometry = reconstruct(fields.geometry, fields.geometry_lo);
    const auto magnetic =
        reconstruct(fields.magnetic_field, fields.magnetic_field_lo);
    const int angular = stage.ntheta * stage.nzeta;
    const int points = stage.ns * angular;
    const int half = (stage.ns - 1) * angular;
    const int modes = stage.mpol * (stage.ntor + 1);
    const int family = modes * stage.ns;
    std::vector<double> averages(2 * (stage.ns - 1));
    std::vector<double> lcfs(4 * modes);
    std::vector<double> axis(2 * stage.nzeta);
    vacuum.enqueue_surface_averages(
        magnetic.data() + 2 * half, magnetic.data() + 3 * half, averages.data(),
        stage.ns, stage.ntheta, stage.nzeta, nullptr);
    vacuum.enqueue_lcfs_repack(state.data(), state.data() + 3 * family,
                               state.data() + family, state.data() + 4 * family,
                               lcfs.data(), stage.ns, modes, stage.mpol,
                               stage.ntor, nullptr);
    vacuum.enqueue_axis_extract(geometry.data(), geometry.data() + points,
                                axis.data(), stage.ntheta, stage.nzeta,
                                nullptr);
    vacuum.run_host_update(stage.ns, averages.data(),
                           averages.data() + stage.ns - 1, lcfs.data(),
                           axis.data(), axis.data() + stage.nzeta, nullptr);
}

void apply_vacuum_force(FreeBoundaryOperator<double>& vacuum,
                        const AxisymmetricStageData& stage,
                        const AxisymmetricForceCase& fields,
                        AxisymmetricForceResult& force) {
    const auto geometry = reconstruct(fields.geometry, fields.geometry_lo);
    const auto magnetic =
        reconstruct(fields.magnetic_field, fields.magnetic_field_lo);
    auto values = reconstruct(force.fields, force.fields_lo);
    const int angular = stage.ntheta * stage.nzeta;
    const int points = stage.ns * angular;
    const int half = (stage.ns - 1) * angular;
    std::vector<double> rbsq(angular);
    double delbsq = 0;
    vacuum.enqueue_rbsq(geometry.data(), geometry.data() + 6 * points,
                        magnetic.data() + 4 * half, rbsq.data(), &delbsq,
                        stage.ns, stage.ntheta, stage.nzeta, angular,
                        1.0 / (stage.ns - 1), nullptr);
    vacuum.enqueue_edge_force(
        values.data(), values.data() + points, values.data() + 2 * points,
        values.data() + 3 * points, geometry.data() + 4 * points,
        geometry.data() + 10 * points, geometry.data() + 3 * points,
        geometry.data() + 9 * points, rbsq.data(), stage.ns, stage.ntheta,
        stage.nzeta, nullptr);
    // Only the LCFS force changes; preserve every interior GPU result word.
    for (int field = 0; field < 4; ++field) {
        for (int point = points - angular; point < points; ++point) {
            const int index = field * points + point;
            const auto pair = split(values[index]);
            force.fields[index] = pair.hi;
            if (!force.fields_lo.empty()) force.fields_lo[index] = pair.lo;
        }
    }
    force.device_fields = {};
    vacuum.set_delbsq(delbsq);
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
