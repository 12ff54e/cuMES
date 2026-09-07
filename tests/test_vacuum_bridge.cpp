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
}  // namespace

int main() try {
    constexpr int NS = 3, NTHETA = 8, NZETA = 4, ANGULAR = NTHETA * NZETA;
    constexpr int POINTS = NS * ANGULAR;
    const double pi = std::acos(-1.0);
    // A stellarator-symmetric scalar with nontrivial theta AND zeta dependence.
    // Reconstructing the second poloidal half must reverse both angles.
    const auto pressure = [pi](int theta, int zeta) {
        return 2.0 + std::sin(2 * pi * theta / NTHETA) *
                         std::sin(2 * pi * zeta / NZETA);
    };
    std::vector<double> reduced((NTHETA / 2 + 1) * NZETA);
    for (int l = 0; l <= NTHETA / 2; ++l)
        for (int k = 0; k < NZETA; ++k) reduced[l * NZETA + k] = pressure(l, k);
    std::vector<double> r(POINTS, 3.0), zero(POINTS),
        total((NS - 1) * ANGULAR, 2.0);
    std::vector<double> rbsq(ANGULAR);
    auto d_reduced = upload(reduced), d_r = upload(r), d_zero = upload(zero),
         d_total = upload(total);
    vfield::DeviceBuffer<double> d_rbsq(ANGULAR), d_mismatch(1);
    d_mismatch.zero();
    cumes::launch_rbsq(d_reduced.data(), d_r.data(), d_zero.data(),
                       d_total.data(), d_rbsq.data(), d_mismatch.data(), NS,
                       NTHETA, NZETA, ANGULAR, 0.0, 0.5, nullptr);
    d_rbsq.download(rbsq.data(), rbsq.size());
    for (int i = 0; i < ANGULAR; ++i)
        require_close(rbsq[i], 6.0 * pressure(i % NTHETA, i / NTHETA));
    // Constant covariant fields retain their surface averages exactly.
    std::vector<double> bu((NS - 1) * ANGULAR, 5.0),
        bv((NS - 1) * ANGULAR, -7.0);
    std::array<double, 2 * (NS - 1)> averages{};
    auto d_bu = upload(bu), d_bv = upload(bv);
    vfield::DeviceBuffer<double> d_averages(averages.size());
    cumes::launch_surface_averages(d_bu.data(), d_bv.data(), d_averages.data(),
                                   NS, NTHETA, NZETA, nullptr);
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
    cumes::launch_edge_force(d_force[0].data(), d_force[1].data(),
                             d_force[2].data(), d_force[3].data(), d_zu.data(),
                             d_zero.data(), d_ru.data(), d_zero.data(),
                             d_rbsq.data(), NS, ANGULAR, nullptr);
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
    std::puts(
        "PASS: vacuum pressure symmetry, averages, and moving-boundary force");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
