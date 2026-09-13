// Analytic contracts for the bridge shared by CUDA and WebAssembly.
#include "../src/free_boundary_impl.cuh"

#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {
void require_close(double actual, double expected) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > 1e-12)
        throw std::runtime_error("vacuum bridge analytic contract failed");
}
vfield::DeviceBuffer<double> upload(std::span<const double> values) {
    vfield::DeviceBuffer<double> buffer(values.size());
    buffer.upload(values.data(), values.size());
    return buffer;
}

void check_activation_consistency() {
    // An axis filament near the circular LCFS is deliberately under-resolved
    // by this coarse quadrature. Its spurious loop-current mismatch must not
    // abort the preceding fixed-boundary relaxation, but must reject coupling.
    DeviceParams<double> p{};
    p.ns = 3;
    p.mpol = p.mnmax = 5;
    p.ntheta = p.nZnT = 16;
    p.nzeta = p.nfp = 1;
    cumes::FreeBoundaryOperator<double>::HostParams params;
    params.coils_file = "deps/vacuum-field/tests/data/coils.solovev";
    params.extcur.assign(13, 0.0);
    params.extcur[0] = 1e6;
    params.use_process_environment = false;
    params.embedded_makegrid_parameters = cumes::MakegridParametersSpec{
        false, false, 1, 2.0, 6.0, 9, -2.0, 2.0, 9, 1};
    cumes::FreeBoundaryOperator<double> vacuum(params, p);
    std::array<double, 20> lcfs{};
    lcfs[0] = 4.0;
    lcfs[1] = lcfs[11] = 1.0 / std::sqrt(2.0);
    const std::array<double, 2> buco{0.1, 0.1}, bvco{1.0, 1.0};
    const std::array<double, 1> axis_r{4.99}, axis_z{0.0};
    auto d_lcfs = upload(lcfs), d_axis_r = upload(axis_r),
         d_axis_z = upload(axis_z);
    const auto update = [&] {
        vacuum.run_host_update(p.ns, buco.data(), bvco.data(), d_lcfs.data(),
                               d_axis_r.data(), d_axis_z.data(), nullptr);
    };
    vacuum.advance(2, 1, 1.0, 1.0);
    update();
    if (vacuum.state() != cumes::VacuumState::OFF ||
        vacuum.apply_edge_force() ||
        std::abs((vacuum.ctor() - vacuum.bsubu_vac()) / vacuum.rbtor()) <= 0.01)
        throw std::runtime_error("inactive vacuum fixture lost its mismatch");
    vacuum.advance(3, 1, 0.0, 0.0);
    bool rejected = false;
    try {
        update();
    } catch (const cumes::CumesError& error) {
        rejected = std::string_view(error.what()).find("I_TOR MISMATCH") !=
                   std::string_view::npos;
    }
    if (!rejected)
        throw std::runtime_error("active vacuum accepted a current mismatch");
}
}  // namespace

int main() try {
    check_activation_consistency();
    constexpr int NS = 3, NTHETA = 8, NZETA = 4, ANGULAR = NTHETA * NZETA;
    constexpr int POINTS = NS * ANGULAR;
    const double pi = std::acos(-1.0);
    for (const bool lasym : {false, true}) {
        // A stellarator-symmetric scalar with nontrivial theta AND zeta
        // dependence. Reconstructing the second poloidal half must reverse both
        // angles.
        const auto pressure = [pi, lasym](int theta, int zeta) {
            return 2.0 +
                   std::sin(2 * pi * theta / NTHETA) *
                       std::sin(2 * pi * zeta / NZETA) +
                   (lasym ? .4 * std::sin(2 * pi * theta / NTHETA) : 0.0);
        };
        const int rows = lasym ? NTHETA : NTHETA / 2 + 1;
        std::vector<double> reduced(rows * NZETA);
        for (int l = 0; l < rows; ++l)
            for (int k = 0; k < NZETA; ++k)
                reduced[l * NZETA + k] = pressure(l, k);
        std::vector<double> r(POINTS, 3.0), zero(POINTS),
            total((NS - 1) * ANGULAR, 2.0);
        std::vector<double> rbsq(ANGULAR);
        auto d_reduced = upload(reduced), d_r = upload(r),
             d_zero = upload(zero), d_total = upload(total);
        vfield::DeviceBuffer<double> d_rbsq(ANGULAR), d_mismatch(1);
        d_mismatch.zero();
        cumes::launch_rbsq(d_reduced.data(), d_r.data(), d_zero.data(),
                           d_total.data(), d_rbsq.data(), d_mismatch.data(), NS,
                           NTHETA, NZETA, ANGULAR, 0.0, 0.5, nullptr, lasym);
        d_rbsq.download(rbsq.data(), rbsq.size());
        for (int i = 0; i < ANGULAR; ++i)
            require_close(rbsq[i], 6.0 * pressure(i % NTHETA, i / NTHETA));
        // Constant covariant fields retain their surface averages exactly.
        std::vector<double> bu((NS - 1) * ANGULAR, 5.0),
            bv((NS - 1) * ANGULAR, -7.0);
        std::array<double, 2 * (NS - 1)> averages{};
        if (lasym)
            for (int i = 0; i < (NS - 1) * ANGULAR; ++i) {
                bu[i] += std::sin(2 * pi * (i % NTHETA) / NTHETA);
                bv[i] += std::sin(4 * pi * (i % NTHETA) / NTHETA);
            }
        auto d_bu = upload(bu), d_bv = upload(bv);
        vfield::DeviceBuffer<double> d_averages(averages.size());
        cumes::launch_surface_averages(d_bu.data(), d_bv.data(),
                                       d_averages.data(), NS, NTHETA, NZETA,
                                       nullptr, lasym);
        d_averages.download(averages.data(), averages.size());
        for (int j = 0; j < NS - 1; ++j) {
            require_close(averages[j], 5.0);
            require_close(averages[NS - 1 + j], -7.0);
        }
        // The outer pressure changes only R/Z at the boundary, with matching
        // parities.
        std::vector<double> zu(POINTS, 2.0), ru(POINTS, 4.0);
        auto d_zu = upload(zu), d_ru = upload(ru);
        std::array<vfield::DeviceBuffer<double>, 4> d_force;
        for (auto& field : d_force) {
            field.allocate(POINTS);
            field.zero();
        }
        cumes::launch_edge_force(
            d_force[0].data(), d_force[1].data(), d_force[2].data(),
            d_force[3].data(), d_zu.data(), d_zero.data(), d_ru.data(),
            d_zero.data(), d_rbsq.data(), NS, ANGULAR, nullptr);
        const std::array<double, 4> factors{2.0, 2.0, -4.0, -4.0};
        std::vector<double> result(POINTS);
        for (int field = 0; field < 4; ++field) {
            d_force[field].download(result.data(), result.size());
            for (int i = 0; i < POINTS; ++i) {
                const double outside =
                    i < POINTS - ANGULAR ? 0.0 : rbsq[i % ANGULAR];
                require_close(result[i], factors[field] * outside);
            }
        }
    }
    // All eight R/Z boundary families use the same n-major vacuum convention.
    constexpr int MPOL = 3, NTOR = 2, MODES = MPOL * (NTOR + 1);
    std::array<vfield::DeviceBuffer<double>, 8> coefficients;
    for (int c = 0; c < 8; ++c) {
        std::vector<double> values(MODES * NS);
        for (int mode = 0; mode < MODES; ++mode)
            values[mode * NS + NS - 1] = c + .1 * mode;
        coefficients[c] = upload(values);
    }
    vfield::DeviceBuffer<double> packed(8 * MODES);
    cumes::launch_lcfs_repack(coefficients[0].data(), coefficients[1].data(),
                              coefficients[2].data(), coefficients[3].data(),
                              packed.data(), NS, MPOL, NTOR, nullptr,
                              coefficients[4].data(), coefficients[5].data(),
                              coefficients[6].data(), coefficients[7].data());
    std::vector<double> values(8 * MODES);
    packed.download(values.data(), values.size());
    for (int c = 0; c < 8; ++c)
        for (int m = 0; m < MPOL; ++m)
            for (int n = 0; n <= NTOR; ++n)
                require_close(
                    values[c * MODES + n * MPOL + m],
                    (c + .1 * (m * (NTOR + 1) + n)) /
                        ((m ? std::sqrt(2.0) : 1) * (n ? std::sqrt(2.0) : 1)));
    std::puts(
        "PASS: vacuum pressure symmetry, averages, and moving-boundary force");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
