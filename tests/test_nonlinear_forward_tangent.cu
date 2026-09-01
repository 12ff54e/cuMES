#include "cumes/numerics/dual_spectral_operator.hpp"
#include "cumes/numerics/equilibrium_residual_jvp.hpp"
#include "cumes/physics/force_operator.hpp"
#include "cumes/physics/geometry_operator.hpp"
#include "cumes/physics/magnetic_field_operator.hpp"
#include "cumes/physics/profiles.hpp"
#include "cumes/solver/equilibrium_linearization.hpp"
#include "cumes/state/mode_table.cuh"
#include "cumes/state/spectral_storage.hpp"
#include "cumes/transforms/toroidal_fft_operator.hpp"
#include "cumes_test_cuda_helper.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

using Dual = cumes::ForwardDualDouble;
using cumes::test::cc;

template <class T>
std::array<const T*, 16> force_fields(const cumes::RealSpaceStorage<T>& rs) {
    return {rs.d_armn_e, rs.d_armn_o, rs.d_azmn_e, rs.d_azmn_o,
            rs.d_brmn_e, rs.d_brmn_o, rs.d_bzmn_e, rs.d_bzmn_o,
            rs.d_blmn_e, rs.d_blmn_o, rs.d_clmn_e, rs.d_clmn_o,
            rs.d_crmn_e, rs.d_crmn_o, rs.d_czmn_e, rs.d_czmn_o};
}

void require_finite(double value, const char* label) {
    if (!std::isfinite(value)) {
        std::cerr << "FAIL: non-finite " << label << '\n';
        std::exit(1);
    }
}

}  // namespace

int main() {
    cumes::ValidatedProblem vp = cumes::test::load_validated("inputs/w7x.json");
    const auto& spec = vp.spec();
    DeviceParams<double> p{};
    p.ns = 9;
    p.mpol = spec.mpol;
    p.ntor = spec.ntor;
    p.mnmax = p.mpol * (p.ntor + 1);
    p.ntheta = spec.angular.ntheta;
    p.nzeta = spec.angular.nzeta;
    p.nfp = spec.nfp;
    p.nZnT = p.ntheta * p.nzeta;
    p.ncurr = 1;
    p.delt = spec.delt;
    p.ftol = 1e-12;
    p.max_iter = 1;

    const std::size_t family_count = static_cast<std::size_t>(p.ns) * p.mnmax;
    const std::size_t state_count = 6 * family_count;
    std::array<std::vector<double>, 6> primal_families;
    cumes::test::manufactured_state<double>(
        cumes::test::ManufacturedShape::W7X_GENERIC, p.ns, p.mnmax, p.ntor,
        primal_families[0], primal_families[3], primal_families[1],
        primal_families[4], primal_families[2], primal_families[5]);

    constexpr double epsilon = 2e-6;
    std::vector<Dual> dual_state(state_count);
    std::vector<double> plus_state(state_count);
    std::vector<double> minus_state(state_count);
    for (std::size_t family = 0; family < primal_families.size(); ++family) {
        for (std::size_t offset = 0; offset < family_count; ++offset) {
            const int surface = static_cast<int>(offset % p.ns);
            const int mode = static_cast<int>(offset / p.ns);
            const int m = mode / (p.ntor + 1);
            const double s = static_cast<double>(surface) / (p.ns - 1);
            const double envelope = m == 0 ? 1.0 : std::pow(std::sqrt(s), m);
            const double scale = family == 2 || family == 5 ? 2e-4 : 1e-3;
            const double direction =
                scale * envelope * std::sin(0.13 * (1 + offset + 17 * family));
            const double primal = primal_families[family][offset];
            const std::size_t index = family * family_count + offset;
            dual_state[index] = {primal, direction};
            plus_state[index] = primal + epsilon * direction;
            minus_state[index] = primal - epsilon * direction;
        }
    }
    for (int mode = 0; mode < p.mnmax; ++mode) {
        const int m = mode / (p.ntor + 1);
        for (std::size_t family = 0; family < primal_families.size();
             ++family) {
            if (m != 1 && !(m == 0 && family == 5)) continue;
            const std::size_t axis = family * family_count + mode * p.ns;
            const std::size_t first = axis + 1;
            dual_state[axis] = dual_state[first];
            plus_state[axis] = plus_state[first];
            minus_state[axis] = minus_state[first];
        }
    }

    cumes::Profiles<double> profiles(p, vp, std::nullopt, false);
    DeviceParams<Dual> dual_p = cumes::make_forward_dual_params(p);
    cumes::Profiles<Dual> dual_profiles(dual_p, vp, std::nullopt, false);

    cumes::SpectralStorage<Dual> dual_storage(p.ns, p.mnmax);
    cumes::SpectralStorage<double> plus_storage(p.ns, p.mnmax);
    cumes::SpectralStorage<double> minus_storage(p.ns, p.mnmax);
    cc(cudaMemcpy(dual_storage.state_slab(), dual_state.data(),
                  state_count * sizeof(Dual), cudaMemcpyHostToDevice),
       "upload dual state");
    cc(cudaMemcpy(plus_storage.state_slab(), plus_state.data(),
                  state_count * sizeof(double), cudaMemcpyHostToDevice),
       "upload plus state");
    cc(cudaMemcpy(minus_storage.state_slab(), minus_state.data(),
                  state_count * sizeof(double), cudaMemcpyHostToDevice),
       "upload minus state");

    auto mode_table = cumes::mode_table_create(p);
    auto dual_rs = real_space_create(dual_p);
    auto plus_rs = real_space_create(p);
    auto minus_rs = real_space_create(p);
    cumes::DualSpectralOperator dual_transform(p, mode_table);
    cumes::ToroidalFftOperator<double> plus_transform(p, plus_rs, mode_table);
    cumes::ToroidalFftOperator<double> minus_transform(p, minus_rs, mode_table);
    cumes::GeometryOperator<Dual> dual_geometry(dual_p, std::nullopt);
    cumes::GeometryOperator<double> plus_geometry(p, std::nullopt);
    cumes::GeometryOperator<double> minus_geometry(p, std::nullopt);
    cumes::DeviceBuffer<Dual> dual_rcon(static_cast<std::size_t>(p.ns) *
                                        p.nZnT);
    cumes::DeviceBuffer<Dual> dual_zcon(static_cast<std::size_t>(p.ns) *
                                        p.nZnT);
    cumes::DeviceBuffer<double> plus_rcon(static_cast<std::size_t>(p.ns) *
                                          p.nZnT);
    cumes::DeviceBuffer<double> plus_zcon(static_cast<std::size_t>(p.ns) *
                                          p.nZnT);
    cumes::DeviceBuffer<double> minus_rcon(static_cast<std::size_t>(p.ns) *
                                           p.nZnT);
    cumes::DeviceBuffer<double> minus_zcon(static_cast<std::size_t>(p.ns) *
                                           p.nZnT);

    dual_transform.enqueue_inverse(
        dual_storage.physical_const(),
        cumes::geometry_parity_views(dual_rs, dual_p),
        {dual_rcon.data(), p.ns, p.ntheta, p.nzeta},
        {dual_zcon.data(), p.ns, p.ntheta, p.nzeta}, 0);
    dual_geometry.enqueue(dual_rs, dual_p, dual_profiles.profile_views(), 0);
    cumes::MagneticFieldOperator<Dual>{}.enqueue(
        dual_rs, dual_p, dual_profiles.profile_views(),
        dual_geometry.base_geometry_views(dual_p),
        dual_geometry.magnetic_field_views(dual_p), nullptr, 0, true);
    cumes::ForceOperator<Dual>{}.enqueue(
        dual_rs, dual_p, dual_profiles.profile_views(),
        dual_geometry.base_geometry_views(dual_p),
        dual_geometry.magnetic_field_views(dual_p), nullptr, 0);

    auto run_double = [&](const cumes::SpectralStorage<double>& storage,
                          cumes::RealSpaceStorage<double>& rs,
                          cumes::ToroidalFftOperator<double>& transform,
                          cumes::GeometryOperator<double>& geometry,
                          cumes::DeviceBuffer<double>& rcon,
                          cumes::DeviceBuffer<double>& zcon) {
        transform.enqueue_inverse(storage.physical_const(),
                                  cumes::geometry_parity_views(rs, p),
                                  {rcon.data(), p.ns, p.ntheta, p.nzeta},
                                  {zcon.data(), p.ns, p.ntheta, p.nzeta}, 0);
        geometry.enqueue(rs, p, profiles.profile_views(), 0);
        cumes::MagneticFieldOperator<double>{}.enqueue(
            rs, p, profiles.profile_views(), geometry.base_geometry_views(p),
            geometry.magnetic_field_views(p), nullptr, 0, true);
        cumes::ForceOperator<double>{}.enqueue(
            rs, p, profiles.profile_views(), geometry.base_geometry_views(p),
            geometry.magnetic_field_views(p), nullptr, 0);
    };
    run_double(plus_storage, plus_rs, plus_transform, plus_geometry, plus_rcon,
               plus_zcon);
    run_double(minus_storage, minus_rs, minus_transform, minus_geometry,
               minus_rcon, minus_zcon);
    cc(cudaDeviceSynchronize(), "nonlinear tangent evaluations");

    const auto dual_forces = force_fields(dual_rs);
    const auto plus_forces = force_fields(plus_rs);
    const auto minus_forces = force_fields(minus_rs);
    const std::size_t real_count = static_cast<std::size_t>(p.ns) * p.nZnT;
    double maximum_error = 0.0;
    double maximum_reference = 0.0;
    for (std::size_t field = 0; field < dual_forces.size(); ++field) {
        std::vector<Dual> actual(real_count);
        std::vector<double> plus(real_count);
        std::vector<double> minus(real_count);
        cc(cudaMemcpy(actual.data(), dual_forces[field],
                      real_count * sizeof(Dual), cudaMemcpyDeviceToHost),
           "download dual force");
        cc(cudaMemcpy(plus.data(), plus_forces[field],
                      real_count * sizeof(double), cudaMemcpyDeviceToHost),
           "download plus force");
        cc(cudaMemcpy(minus.data(), minus_forces[field],
                      real_count * sizeof(double), cudaMemcpyDeviceToHost),
           "download minus force");
        for (std::size_t index = 0; index < real_count; ++index) {
            const double reference =
                (plus[index] - minus[index]) / (2.0 * epsilon);
            require_finite(actual[index].value, "force primal");
            require_finite(actual[index].tangent, "force tangent");
            require_finite(reference, "finite-difference tangent");
            maximum_error = std::max(
                maximum_error, std::abs(actual[index].tangent - reference));
            maximum_reference =
                std::max(maximum_reference, std::abs(reference));
        }
    }
    const double relative_error = maximum_error / maximum_reference;
    std::cout << "nonlinear force JVP max reference=" << maximum_reference
              << " max error=" << maximum_error
              << " relative error=" << relative_error << '\n';
    if (!(relative_error < 2e-6)) {
        std::cerr << "FAIL: nonlinear forward tangent disagrees with centered "
                     "finite difference\n";
        return 1;
    }

    cumes::EquilibriumResidualJvpOperator residual_jvp(p, vp, mode_table);
    cumes::DeviceBuffer<Dual> dual_residual(state_count);
    residual_jvp.enqueue(dual_storage.physical_const(),
                         {dual_residual.data(), p.ns, p.mnmax}, 0);
    std::array<cumes::DeviceBuffer<double>, 4> zero_constraint;
    const std::size_t real_count_for_constraint =
        static_cast<std::size_t>(p.ns) * p.nZnT;
    for (auto& field : zero_constraint) {
        field.allocate(real_count_for_constraint);
        field.zero();
    }
    const auto zero_constraint_views =
        cumes::ConstraintForceViews<const double>{
            {zero_constraint[0].data(), p.ns, p.ntheta, p.nzeta},
            {zero_constraint[1].data(), p.ns, p.ntheta, p.nzeta},
            {zero_constraint[2].data(), p.ns, p.ntheta, p.nzeta},
            {zero_constraint[3].data(), p.ns, p.ntheta, p.nzeta}};
    cumes::DeviceBuffer<double> plus_residual(state_count);
    cumes::DeviceBuffer<double> minus_residual(state_count);
    plus_transform.enqueue_forward(
        cumes::ForceParityViews<const double>{
            {plus_rs.d_armn_e, p.ns, p.ntheta, p.nzeta},
            {plus_rs.d_armn_o, p.ns, p.ntheta, p.nzeta},
            {plus_rs.d_azmn_e, p.ns, p.ntheta, p.nzeta},
            {plus_rs.d_azmn_o, p.ns, p.ntheta, p.nzeta},
            {plus_rs.d_brmn_e, p.ns, p.ntheta, p.nzeta},
            {plus_rs.d_brmn_o, p.ns, p.ntheta, p.nzeta},
            {plus_rs.d_bzmn_e, p.ns, p.ntheta, p.nzeta},
            {plus_rs.d_bzmn_o, p.ns, p.ntheta, p.nzeta},
            {plus_rs.d_blmn_e, p.ns, p.ntheta, p.nzeta},
            {plus_rs.d_blmn_o, p.ns, p.ntheta, p.nzeta},
            {plus_rs.d_clmn_e, p.ns, p.ntheta, p.nzeta},
            {plus_rs.d_clmn_o, p.ns, p.ntheta, p.nzeta},
            {plus_rs.d_crmn_e, p.ns, p.ntheta, p.nzeta},
            {plus_rs.d_crmn_o, p.ns, p.ntheta, p.nzeta},
            {plus_rs.d_czmn_e, p.ns, p.ntheta, p.nzeta},
            {plus_rs.d_czmn_o, p.ns, p.ntheta, p.nzeta}},
        zero_constraint_views, {plus_residual.data(), p.ns, p.mnmax}, 0);
    minus_transform.enqueue_forward(
        cumes::ForceParityViews<const double>{
            {minus_rs.d_armn_e, p.ns, p.ntheta, p.nzeta},
            {minus_rs.d_armn_o, p.ns, p.ntheta, p.nzeta},
            {minus_rs.d_azmn_e, p.ns, p.ntheta, p.nzeta},
            {minus_rs.d_azmn_o, p.ns, p.ntheta, p.nzeta},
            {minus_rs.d_brmn_e, p.ns, p.ntheta, p.nzeta},
            {minus_rs.d_brmn_o, p.ns, p.ntheta, p.nzeta},
            {minus_rs.d_bzmn_e, p.ns, p.ntheta, p.nzeta},
            {minus_rs.d_bzmn_o, p.ns, p.ntheta, p.nzeta},
            {minus_rs.d_blmn_e, p.ns, p.ntheta, p.nzeta},
            {minus_rs.d_blmn_o, p.ns, p.ntheta, p.nzeta},
            {minus_rs.d_clmn_e, p.ns, p.ntheta, p.nzeta},
            {minus_rs.d_clmn_o, p.ns, p.ntheta, p.nzeta},
            {minus_rs.d_crmn_e, p.ns, p.ntheta, p.nzeta},
            {minus_rs.d_crmn_o, p.ns, p.ntheta, p.nzeta},
            {minus_rs.d_czmn_e, p.ns, p.ntheta, p.nzeta},
            {minus_rs.d_czmn_o, p.ns, p.ntheta, p.nzeta}},
        zero_constraint_views, {minus_residual.data(), p.ns, p.mnmax}, 0);
    cc(cudaDeviceSynchronize(), "residual JVP evaluation");

    std::vector<Dual> actual_residual(state_count);
    std::vector<double> plus_residual_host(state_count);
    std::vector<double> minus_residual_host(state_count);
    cc(cudaMemcpy(actual_residual.data(), dual_residual.data(),
                  state_count * sizeof(Dual), cudaMemcpyDeviceToHost),
       "download residual JVP");
    cc(cudaMemcpy(plus_residual_host.data(), plus_residual.data(),
                  state_count * sizeof(double), cudaMemcpyDeviceToHost),
       "download plus residual");
    cc(cudaMemcpy(minus_residual_host.data(), minus_residual.data(),
                  state_count * sizeof(double), cudaMemcpyDeviceToHost),
       "download minus residual");
    auto postprocess = [&](std::vector<double>& residual) {
        const double sqrt_s1 = std::sqrt(1.0 / (p.ns - 1.0));
        for (int mode = 0; mode < p.mnmax; ++mode) {
            const int m = mode / (p.ntor + 1);
            if (m % 2 == 0) continue;
            for (int surface = 0; surface < p.ns; ++surface) {
                const double sqrt_s = std::sqrt(surface / (p.ns - 1.0) + 1e-12);
                const double scale = 1.0 / std::max(sqrt_s, sqrt_s1);
                for (std::size_t family = 0; family < 6; ++family) {
                    residual[family * family_count + mode * p.ns + surface] *=
                        scale;
                }
            }
        }
        const double gauge_scale = 1.0 / std::sqrt(2.0);
        for (int n = 0; n < p.ntor + 1; ++n) {
            const int mode = p.ntor + 1 + n;
            for (int surface = 0; surface < p.ns; ++surface) {
                const std::size_t rss =
                    3 * family_count + mode * p.ns + surface;
                const std::size_t zcs =
                    4 * family_count + mode * p.ns + surface;
                residual[rss] = (residual[rss] + residual[zcs]) * gauge_scale;
                residual[zcs] = 0.0;
            }
        }
    };
    postprocess(plus_residual_host);
    postprocess(minus_residual_host);
    double residual_error = 0.0;
    double residual_reference = 0.0;
    for (std::size_t index = 0; index < state_count; ++index) {
        const double reference =
            (plus_residual_host[index] - minus_residual_host[index]) /
            (2.0 * epsilon);
        require_finite(actual_residual[index].tangent, "residual tangent");
        require_finite(reference, "finite-difference residual tangent");
        residual_error =
            std::max(residual_error,
                     std::abs(actual_residual[index].tangent - reference));
        residual_reference = std::max(residual_reference, std::abs(reference));
    }
    const double residual_relative_error = residual_error / residual_reference;
    std::cout << "spectral residual JVP max reference=" << residual_reference
              << " max error=" << residual_error
              << " relative error=" << residual_relative_error << '\n';
    if (!(residual_relative_error < 3e-6)) {
        std::cerr << "FAIL: spectral residual JVP disagrees with centered "
                     "finite difference\n";
        return 1;
    }

    cumes::EquilibriumSnapshot snapshot;
    snapshot.ns = p.ns;
    snapshot.mnmax = p.mnmax;
    snapshot.ntheta = p.ntheta;
    snapshot.nzeta = p.nzeta;
    std::vector<double> host_state_direction(state_count);
    for (std::size_t family = 0; family < 6; ++family) {
        snapshot.families[family].resize(family_count);
        for (std::size_t offset = 0; offset < family_count; ++offset) {
            const std::size_t index = family * family_count + offset;
            snapshot.families[family][offset] = dual_state[index].value;
            host_state_direction[index] = dual_state[index].tangent;
        }
    }
    cumes::EquilibriumLinearization linearization(vp, snapshot);
    const cumes::ResidualJvp host_jvp =
        linearization.residual_jvp(host_state_direction);
    for (std::size_t index = 0; index < state_count; ++index) {
        const double scale =
            std::max(1.0, std::abs(actual_residual[index].tangent));
        if (std::abs(host_jvp.tangent[index] - actual_residual[index].tangent) >
            2e-13 * scale) {
            std::cerr << "FAIL: host linearization facade changed the JVP\n";
            return 1;
        }
    }
    cumes::BoundaryTangent boundary = cumes::BoundaryTangent::zero(vp);
    boundary.rbcc[1] = 0.03;
    boundary.zbcs[p.ntor + 2] = -0.02;
    std::vector<double> boundary_state_direction(state_count, 0.0);
    for (int mode = 0; mode < p.mnmax; ++mode) {
        const std::size_t offset =
            static_cast<std::size_t>(mode) * p.ns + p.ns - 1;
        boundary_state_direction[0 * family_count + offset] =
            boundary.rbcc[mode];
        boundary_state_direction[1 * family_count + offset] =
            boundary.zbsc[mode];
        boundary_state_direction[3 * family_count + offset] =
            boundary.rbss[mode];
        boundary_state_direction[4 * family_count + offset] =
            boundary.zbcs[mode];
    }
    const cumes::ResidualJvp boundary_jvp =
        linearization.boundary_residual_jvp(boundary);
    const cumes::ResidualJvp explicit_boundary_jvp =
        linearization.residual_jvp(boundary_state_direction);
    for (std::size_t index = 0; index < state_count; ++index) {
        if (boundary_jvp.tangent[index] !=
            explicit_boundary_jvp.tangent[index]) {
            std::cerr << "FAIL: folded boundary tangent mapping changed JVP\n";
            return 1;
        }
    }
    const cumes::SpectralTangentSolve zero_solve =
        linearization.solve_boundary_tangent(cumes::BoundaryTangent::zero(vp));
    if (!zero_solve.converged || zero_solve.iterations != 0 ||
        zero_solve.final_residual != 0.0) {
        std::cerr << "FAIL: zero boundary tangent was not an exact solve\n";
        return 1;
    }
    cumes::TangentLinearOptions trial_options;
    trial_options.max_iterations = 40;
    trial_options.restart = 12;
    trial_options.relative_tolerance = 1e-4;
    const cumes::SpectralTangentSolve trial_solve =
        linearization.solve_boundary_tangent(boundary, trial_options);
    if (trial_solve.state_tangent.size() != state_count ||
        !std::isfinite(trial_solve.initial_residual) ||
        !std::isfinite(trial_solve.final_residual) ||
        !(trial_solve.final_residual < trial_solve.initial_residual)) {
        std::cerr << "FAIL: matrix-free tangent solve returned invalid data\n";
        return 1;
    }
    std::cout << "trial GMRES initial=" << trial_solve.initial_residual
              << " final=" << trial_solve.final_residual
              << " iterations=" << trial_solve.iterations
              << " converged=" << trial_solve.converged << '\n';

    real_space_free(dual_rs);
    real_space_free(plus_rs);
    real_space_free(minus_rs);
    cumes::mode_table_free(mode_table);
    std::cout << "nonlinear forward tangent test passed\n";
    return 0;
}
