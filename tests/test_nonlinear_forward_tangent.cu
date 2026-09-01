#include "cumes/numerics/dual_spectral_operator.hpp"
#include "cumes/physics/force_operator.hpp"
#include "cumes/physics/geometry_operator.hpp"
#include "cumes/physics/magnetic_field_operator.hpp"
#include "cumes/physics/profiles.hpp"
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

    real_space_free(dual_rs);
    real_space_free(plus_rs);
    real_space_free(minus_rs);
    cumes::mode_table_free(mode_table);
    std::cout << "nonlinear forward tangent test passed\n";
    return 0;
}
