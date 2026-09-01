#include "cumes/numerics/dual_spectral_operator.hpp"
#include "cumes/runtime/cuda_status.hpp"
#include "cumes/state/mode_table.cuh"
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
std::array<T*, 18> geometry_fields(cumes::RealSpaceStorage<T>& rs) {
    return {rs.d_r_e,  rs.d_z_e,  rs.d_l_e,  rs.d_ru_e, rs.d_zu_e, rs.d_lu_e,
            rs.d_r_o,  rs.d_z_o,  rs.d_l_o,  rs.d_ru_o, rs.d_zu_o, rs.d_lu_o,
            rs.d_rv_e, rs.d_zv_e, rs.d_lv_e, rs.d_rv_o, rs.d_zv_o, rs.d_lv_o};
}

template <class T>
std::array<T*, 16> force_fields(cumes::RealSpaceStorage<T>& rs) {
    return {rs.d_armn_e, rs.d_armn_o, rs.d_azmn_e, rs.d_azmn_o,
            rs.d_brmn_e, rs.d_brmn_o, rs.d_bzmn_e, rs.d_bzmn_o,
            rs.d_blmn_e, rs.d_blmn_o, rs.d_clmn_e, rs.d_clmn_o,
            rs.d_crmn_e, rs.d_crmn_o, rs.d_czmn_e, rs.d_czmn_o};
}

template <class T>
cumes::ForceParityViews<const T> const_force_views(
    const cumes::RealSpaceStorage<T>& rs,
    const DeviceParams<double>& p) {
    auto field = [&](const T* data) {
        return cumes::RealFieldView<const T>(data, p.ns, p.ntheta, p.nzeta);
    };
    cumes::ForceParityViews<const T> result;
    result.armn_e = field(rs.d_armn_e);
    result.armn_o = field(rs.d_armn_o);
    result.azmn_e = field(rs.d_azmn_e);
    result.azmn_o = field(rs.d_azmn_o);
    result.brmn_e = field(rs.d_brmn_e);
    result.brmn_o = field(rs.d_brmn_o);
    result.bzmn_e = field(rs.d_bzmn_e);
    result.bzmn_o = field(rs.d_bzmn_o);
    result.blmn_e = field(rs.d_blmn_e);
    result.blmn_o = field(rs.d_blmn_o);
    result.clmn_e = field(rs.d_clmn_e);
    result.clmn_o = field(rs.d_clmn_o);
    result.crmn_e = field(rs.d_crmn_e);
    result.crmn_o = field(rs.d_crmn_o);
    result.czmn_e = field(rs.d_czmn_e);
    result.czmn_o = field(rs.d_czmn_o);
    return result;
}

void require_close(double actual,
                   double expected,
                   double tolerance,
                   const char* message) {
    if (std::abs(actual - expected) >
        tolerance * std::max(1.0, std::abs(expected))) {
        std::cerr << "FAIL: " << message << " actual=" << actual
                  << " expected=" << expected << '\n';
        std::exit(1);
    }
}

}  // namespace

int main() {
    DeviceParams<double> p{};
    p.ns = 4;
    p.mpol = 3;
    p.ntor = 2;
    p.mnmax = p.mpol * (p.ntor + 1);
    p.ntheta = 12;
    p.nzeta = 8;
    p.nfp = 5;
    p.nZnT = p.ntheta * p.nzeta;
    p.ncurr = 0;

    const std::size_t spectral_count =
        static_cast<std::size_t>(6) * p.ns * p.mnmax;
    std::vector<Dual> host_dual_state(spectral_count);
    std::vector<double> host_primal_state(spectral_count);
    std::vector<double> host_tangent_state(spectral_count);
    for (std::size_t index = 0; index < spectral_count; ++index) {
        const double primal = 0.2 * std::sin(0.17 * (index + 1));
        const double tangent = 0.03 * std::cos(0.11 * (index + 2));
        host_dual_state[index] = {primal, tangent};
        host_primal_state[index] = primal;
        host_tangent_state[index] = tangent;
    }

    cumes::SpectralStorage<Dual> dual_state(p.ns, p.mnmax);
    cumes::SpectralStorage<double> primal_state(p.ns, p.mnmax);
    cumes::SpectralStorage<double> tangent_state(p.ns, p.mnmax);
    cc(cudaMemcpy(dual_state.state_slab(), host_dual_state.data(),
                  spectral_count * sizeof(Dual), cudaMemcpyHostToDevice),
       "dual state upload");
    cc(cudaMemcpy(primal_state.state_slab(), host_primal_state.data(),
                  spectral_count * sizeof(double), cudaMemcpyHostToDevice),
       "primal state upload");
    cc(cudaMemcpy(tangent_state.state_slab(), host_tangent_state.data(),
                  spectral_count * sizeof(double), cudaMemcpyHostToDevice),
       "tangent state upload");

    auto mode_table = cumes::mode_table_create(p);
    auto dual_rs = real_space_create(cumes::make_forward_dual_params(p));
    auto primal_rs = real_space_create(p);
    auto tangent_rs = real_space_create(p);
    cumes::DualSpectralOperator dual_transform(p, mode_table);
    cumes::ToroidalFftOperator<double> reference_transform(p, primal_rs,
                                                           mode_table);
    cumes::DeviceBuffer<Dual> dual_rcon(static_cast<std::size_t>(p.ns) *
                                        p.nZnT);
    cumes::DeviceBuffer<Dual> dual_zcon(static_cast<std::size_t>(p.ns) *
                                        p.nZnT);
    cumes::DeviceBuffer<double> primal_rcon(static_cast<std::size_t>(p.ns) *
                                            p.nZnT);
    cumes::DeviceBuffer<double> primal_zcon(static_cast<std::size_t>(p.ns) *
                                            p.nZnT);
    cumes::DeviceBuffer<double> tangent_rcon(static_cast<std::size_t>(p.ns) *
                                             p.nZnT);
    cumes::DeviceBuffer<double> tangent_zcon(static_cast<std::size_t>(p.ns) *
                                             p.nZnT);

    dual_transform.enqueue_inverse(
        dual_state.physical_const(),
        cumes::geometry_parity_views(dual_rs,
                                     cumes::make_forward_dual_params(p)),
        {dual_rcon.data(), p.ns, p.ntheta, p.nzeta},
        {dual_zcon.data(), p.ns, p.ntheta, p.nzeta}, 0);
    reference_transform.enqueue_inverse(
        primal_state.physical_const(),
        cumes::geometry_parity_views(primal_rs, p),
        {primal_rcon.data(), p.ns, p.ntheta, p.nzeta},
        {primal_zcon.data(), p.ns, p.ntheta, p.nzeta}, 0);
    reference_transform.enqueue_inverse(
        tangent_state.physical_const(),
        cumes::geometry_parity_views(tangent_rs, p),
        {tangent_rcon.data(), p.ns, p.ntheta, p.nzeta},
        {tangent_zcon.data(), p.ns, p.ntheta, p.nzeta}, 0);
    cc(cudaDeviceSynchronize(), "inverse transforms");

    const std::size_t real_count = static_cast<std::size_t>(p.ns) * p.nZnT;
    const auto dual_geometry = geometry_fields(dual_rs);
    const auto primal_geometry = geometry_fields(primal_rs);
    const auto tangent_geometry = geometry_fields(tangent_rs);
    for (std::size_t field = 0; field < dual_geometry.size(); ++field) {
        std::vector<Dual> actual(real_count);
        std::vector<double> expected_primal(real_count);
        std::vector<double> expected_tangent(real_count);
        cc(cudaMemcpy(actual.data(), dual_geometry[field],
                      real_count * sizeof(Dual), cudaMemcpyDeviceToHost),
           "dual geometry download");
        cc(cudaMemcpy(expected_primal.data(), primal_geometry[field],
                      real_count * sizeof(double), cudaMemcpyDeviceToHost),
           "primal geometry download");
        cc(cudaMemcpy(expected_tangent.data(), tangent_geometry[field],
                      real_count * sizeof(double), cudaMemcpyDeviceToHost),
           "tangent geometry download");
        for (std::size_t index = 0; index < real_count; ++index) {
            require_close(actual[index].value, expected_primal[index], 2e-13,
                          "inverse primal lane");
            require_close(actual[index].tangent, expected_tangent[index], 2e-13,
                          "inverse tangent lane");
        }
    }

    const auto dual_forces = force_fields(dual_rs);
    const auto primal_forces = force_fields(primal_rs);
    const auto tangent_forces = force_fields(tangent_rs);
    for (std::size_t field = 0; field < dual_forces.size(); ++field) {
        std::vector<Dual> dual_values(real_count);
        std::vector<double> primal_values(real_count);
        std::vector<double> tangent_values(real_count);
        for (std::size_t index = 0; index < real_count; ++index) {
            primal_values[index] =
                0.4 * std::sin(0.03 * (1 + index + 7 * field));
            tangent_values[index] =
                0.05 * std::cos(0.05 * (2 + index + 5 * field));
            dual_values[index] = {primal_values[index], tangent_values[index]};
        }
        cc(cudaMemcpy(dual_forces[field], dual_values.data(),
                      real_count * sizeof(Dual), cudaMemcpyHostToDevice),
           "dual force upload");
        cc(cudaMemcpy(primal_forces[field], primal_values.data(),
                      real_count * sizeof(double), cudaMemcpyHostToDevice),
           "primal force upload");
        cc(cudaMemcpy(tangent_forces[field], tangent_values.data(),
                      real_count * sizeof(double), cudaMemcpyHostToDevice),
           "tangent force upload");
    }

    std::array<cumes::DeviceBuffer<Dual>, 4> dual_constraint;
    std::array<cumes::DeviceBuffer<double>, 4> primal_constraint;
    std::array<cumes::DeviceBuffer<double>, 4> tangent_constraint;
    for (std::size_t field = 0; field < 4; ++field) {
        dual_constraint[field].allocate(real_count);
        primal_constraint[field].allocate(real_count);
        tangent_constraint[field].allocate(real_count);
        std::vector<Dual> dual_values(real_count);
        std::vector<double> primal_values(real_count);
        std::vector<double> tangent_values(real_count);
        for (std::size_t index = 0; index < real_count; ++index) {
            primal_values[index] = 0.01 * (field + 1) * std::sin(0.02 * index);
            tangent_values[index] =
                0.002 * (field + 1) * std::cos(0.04 * index);
            dual_values[index] = {primal_values[index], tangent_values[index]};
        }
        cc(cudaMemcpy(dual_constraint[field].data(), dual_values.data(),
                      real_count * sizeof(Dual), cudaMemcpyHostToDevice),
           "dual constraint upload");
        cc(cudaMemcpy(primal_constraint[field].data(), primal_values.data(),
                      real_count * sizeof(double), cudaMemcpyHostToDevice),
           "primal constraint upload");
        cc(cudaMemcpy(tangent_constraint[field].data(), tangent_values.data(),
                      real_count * sizeof(double), cudaMemcpyHostToDevice),
           "tangent constraint upload");
    }
    auto dual_constraint_views = cumes::ConstraintForceViews<const Dual>{
        {dual_constraint[0].data(), p.ns, p.ntheta, p.nzeta},
        {dual_constraint[1].data(), p.ns, p.ntheta, p.nzeta},
        {dual_constraint[2].data(), p.ns, p.ntheta, p.nzeta},
        {dual_constraint[3].data(), p.ns, p.ntheta, p.nzeta}};
    auto primal_constraint_views = cumes::ConstraintForceViews<const double>{
        {primal_constraint[0].data(), p.ns, p.ntheta, p.nzeta},
        {primal_constraint[1].data(), p.ns, p.ntheta, p.nzeta},
        {primal_constraint[2].data(), p.ns, p.ntheta, p.nzeta},
        {primal_constraint[3].data(), p.ns, p.ntheta, p.nzeta}};
    auto tangent_constraint_views = cumes::ConstraintForceViews<const double>{
        {tangent_constraint[0].data(), p.ns, p.ntheta, p.nzeta},
        {tangent_constraint[1].data(), p.ns, p.ntheta, p.nzeta},
        {tangent_constraint[2].data(), p.ns, p.ntheta, p.nzeta},
        {tangent_constraint[3].data(), p.ns, p.ntheta, p.nzeta}};

    cumes::DeviceBuffer<Dual> dual_residual(spectral_count);
    cumes::DeviceBuffer<double> primal_residual(spectral_count);
    cumes::DeviceBuffer<double> tangent_residual(spectral_count);
    dual_transform.enqueue_forward(const_force_views(dual_rs, p),
                                   dual_constraint_views,
                                   {dual_residual.data(), p.ns, p.mnmax}, 0);
    reference_transform.enqueue_forward(
        const_force_views(primal_rs, p), primal_constraint_views,
        {primal_residual.data(), p.ns, p.mnmax}, 0);
    reference_transform.enqueue_forward(
        const_force_views(tangent_rs, p), tangent_constraint_views,
        {tangent_residual.data(), p.ns, p.mnmax}, 0);
    cc(cudaDeviceSynchronize(), "forward transforms");

    std::vector<Dual> actual_residual(spectral_count);
    std::vector<double> expected_primal_residual(spectral_count);
    std::vector<double> expected_tangent_residual(spectral_count);
    cc(cudaMemcpy(actual_residual.data(), dual_residual.data(),
                  spectral_count * sizeof(Dual), cudaMemcpyDeviceToHost),
       "dual residual download");
    cc(cudaMemcpy(expected_primal_residual.data(), primal_residual.data(),
                  spectral_count * sizeof(double), cudaMemcpyDeviceToHost),
       "primal residual download");
    cc(cudaMemcpy(expected_tangent_residual.data(), tangent_residual.data(),
                  spectral_count * sizeof(double), cudaMemcpyDeviceToHost),
       "tangent residual download");
    for (std::size_t index = 0; index < spectral_count; ++index) {
        require_close(actual_residual[index].value,
                      expected_primal_residual[index], 2e-13,
                      "forward primal lane");
        require_close(actual_residual[index].tangent,
                      expected_tangent_residual[index], 2e-13,
                      "forward tangent lane");
    }

    std::vector<Dual> host_dual_gcon_eff(real_count);
    std::vector<Dual> host_dual_tcon(p.ns);
    std::vector<Dual> host_dual_faccon(p.mnmax);
    std::vector<double> host_primal_gcon_eff(real_count);
    std::vector<double> host_tangent_gcon_eff(real_count);
    std::vector<double> host_primal_tcon(p.ns);
    std::vector<double> host_tangent_tcon(p.ns);
    std::vector<double> host_primal_faccon(p.mnmax);
    std::vector<double> host_tangent_faccon(p.mnmax);
    for (std::size_t index = 0; index < real_count; ++index) {
        host_primal_gcon_eff[index] = 0.3 * std::sin(0.07 * (index + 1));
        host_tangent_gcon_eff[index] = 0.04 * std::cos(0.09 * (index + 2));
        host_dual_gcon_eff[index] = {host_primal_gcon_eff[index],
                                     host_tangent_gcon_eff[index]};
    }
    for (int index = 0; index < p.ns; ++index) {
        host_primal_tcon[index] = 0.7 + 0.03 * index;
        host_tangent_tcon[index] = -0.02 + 0.01 * index;
        host_dual_tcon[index] = {host_primal_tcon[index],
                                 host_tangent_tcon[index]};
    }
    for (int index = 0; index < p.mnmax; ++index) {
        host_primal_faccon[index] = 0.8 + 0.01 * index;
        host_tangent_faccon[index] = 0.015 * std::sin(0.2 * index);
        host_dual_faccon[index] = {host_primal_faccon[index],
                                   host_tangent_faccon[index]};
    }
    cumes::DeviceBuffer<Dual> dual_gcon_eff(real_count);
    cumes::DeviceBuffer<Dual> dual_tcon(p.ns);
    cumes::DeviceBuffer<Dual> dual_faccon(p.mnmax);
    cumes::DeviceBuffer<Dual> dual_gcon(real_count);
    cumes::DeviceBuffer<double> primal_gcon_eff(real_count);
    cumes::DeviceBuffer<double> tangent_gcon_eff(real_count);
    cumes::DeviceBuffer<double> primal_tcon(p.ns);
    cumes::DeviceBuffer<double> tangent_tcon(p.ns);
    cumes::DeviceBuffer<double> primal_faccon(p.mnmax);
    cumes::DeviceBuffer<double> tangent_faccon(p.mnmax);
    cumes::DeviceBuffer<double> primal_gcon(real_count);
    cumes::DeviceBuffer<double> tangent_gcon0(real_count);
    cumes::DeviceBuffer<double> tangent_gcon1(real_count);
    cumes::DeviceBuffer<double> tangent_gcon2(real_count);
#define CUMES_UPLOAD(buffer, values, label)              \
    cc(cudaMemcpy((buffer).data(), (values).data(),      \
                  (values).size() * sizeof((values)[0]), \
                  cudaMemcpyHostToDevice),               \
       label)
    CUMES_UPLOAD(dual_gcon_eff, host_dual_gcon_eff, "dual gcon upload");
    CUMES_UPLOAD(dual_tcon, host_dual_tcon, "dual tcon upload");
    CUMES_UPLOAD(dual_faccon, host_dual_faccon, "dual faccon upload");
    CUMES_UPLOAD(primal_gcon_eff, host_primal_gcon_eff, "primal gcon upload");
    CUMES_UPLOAD(tangent_gcon_eff, host_tangent_gcon_eff,
                 "tangent gcon upload");
    CUMES_UPLOAD(primal_tcon, host_primal_tcon, "primal tcon upload");
    CUMES_UPLOAD(tangent_tcon, host_tangent_tcon, "tangent tcon upload");
    CUMES_UPLOAD(primal_faccon, host_primal_faccon, "primal faccon upload");
    CUMES_UPLOAD(tangent_faccon, host_tangent_faccon, "tangent faccon upload");
#undef CUMES_UPLOAD
    dual_transform.enqueue_dealias(
        {dual_gcon_eff.data(), p.ns, p.ntheta, p.nzeta}, dual_tcon.data(),
        dual_faccon.data(), {dual_gcon.data(), p.ns, p.ntheta, p.nzeta}, 0);
    reference_transform.dealias_bandpass(
        primal_gcon_eff.data(), primal_tcon.data(), primal_faccon.data(),
        primal_gcon.data(), 0);
    reference_transform.dealias_bandpass(
        tangent_gcon_eff.data(), primal_tcon.data(), primal_faccon.data(),
        tangent_gcon0.data(), 0);
    reference_transform.dealias_bandpass(
        primal_gcon_eff.data(), tangent_tcon.data(), primal_faccon.data(),
        tangent_gcon1.data(), 0);
    reference_transform.dealias_bandpass(
        primal_gcon_eff.data(), primal_tcon.data(), tangent_faccon.data(),
        tangent_gcon2.data(), 0);
    cc(cudaDeviceSynchronize(), "de-alias transforms");
    std::vector<Dual> actual_gcon(real_count);
    std::vector<double> expected_primal_gcon(real_count);
    std::vector<double> expected_tangent_gcon0(real_count);
    std::vector<double> expected_tangent_gcon1(real_count);
    std::vector<double> expected_tangent_gcon2(real_count);
#define CUMES_DOWNLOAD(values, buffer, label)            \
    cc(cudaMemcpy((values).data(), (buffer).data(),      \
                  (values).size() * sizeof((values)[0]), \
                  cudaMemcpyDeviceToHost),               \
       label)
    CUMES_DOWNLOAD(actual_gcon, dual_gcon, "dual gcon download");
    CUMES_DOWNLOAD(expected_primal_gcon, primal_gcon, "primal gcon download");
    CUMES_DOWNLOAD(expected_tangent_gcon0, tangent_gcon0,
                   "tangent gcon 0 download");
    CUMES_DOWNLOAD(expected_tangent_gcon1, tangent_gcon1,
                   "tangent gcon 1 download");
    CUMES_DOWNLOAD(expected_tangent_gcon2, tangent_gcon2,
                   "tangent gcon 2 download");
#undef CUMES_DOWNLOAD
    for (std::size_t index = 0; index < real_count; ++index) {
        require_close(actual_gcon[index].value, expected_primal_gcon[index],
                      2e-13, "de-alias primal lane");
        const double expected_tangent = expected_tangent_gcon0[index] +
                                        expected_tangent_gcon1[index] +
                                        expected_tangent_gcon2[index];
        require_close(actual_gcon[index].tangent, expected_tangent, 2e-13,
                      "de-alias tangent lane");
    }

    real_space_free(dual_rs);
    real_space_free(primal_rs);
    real_space_free(tangent_rs);
    cumes::mode_table_free(mode_table);
    std::cout << "dual spectral operator tests passed\n";
    return 0;
}
