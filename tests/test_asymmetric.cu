// Full-period Fourier and weak-form checks for all asymmetric families.
#include "cumes/config/json_reader.hpp"
#include "cumes/state/seed_state.hpp"
#include "cumes/state/spectral_storage.hpp"
#include "cumes/transforms/toroidal_fft_operator.hpp"
#include "cumes_test_cuda_helper.cuh"

#include <array>
#include <cmath>
#include <vector>

using namespace cumes;

namespace {
constexpr std::array<bool, ASYMMETRIC_COMPONENT_COUNT> THETA_SINE = {
    false, true,  true,  true,  false, false,
    true,  false, false, false, true,  true};

std::array<double, 3> basis(int c,
                            int m,
                            int n,
                            int nfp,
                            double theta,
                            double zeta) {
    const bool ts = THETA_SINE[c];
    const bool zs = (c / 3) % 2 == 1;
    const double t = ts ? std::sin(m * theta) : std::cos(m * theta);
    const double z = zs ? std::sin(n * zeta) : std::cos(n * zeta);
    const double dt = m * (ts ? std::cos(m * theta) : -std::sin(m * theta));
    const double dz = n * nfp * (zs ? std::cos(n * zeta) : -std::sin(n * zeta));
    return {t * z, dt * z, t * dz};
}

void close(double actual,
           double expected,
           double tolerance,
           const char* context) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        std::cerr << context << ": " << actual << " != " << expected << '\n';
        ++test::failures();
    }
}

template <typename T>
std::vector<T> download(const T* d, std::size_t count) {
    std::vector<T> values(count);
    test::cc(
        cudaMemcpy(values.data(), d, count * sizeof(T), cudaMemcpyDeviceToHost),
        "download");
    return values;
}

template <typename T>
void run(int ntor) {
    DeviceParams<T> p{};
    p.ns = 5;
    p.mpol = 5;
    p.ntor = ntor;
    p.ntheta = 40;
    p.nzeta = ntor == 0 ? 1 : 10;
    p.nfp = 3;
    p.ncurr = 0;
    p.mnmax = p.mpol * (ntor + 1);
    p.nZnT = p.ntheta * p.nzeta;
    p.lasym = true;
    const auto one = std::size_t(p.mnmax) * p.ns;
    const auto points = std::size_t(p.nZnT) * p.ns;
    const double tolerance = sizeof(T) == sizeof(float) ? 2e-4 : 3e-12;
    auto mt = mode_table_create<T>(p);
    auto rs = real_space_create(p);
    SpectralStorage<T> state(p.ns, p.mnmax, {}, true);
    ToroidalFftOperator<T> op(p, rs, mt);
    std::vector<T> coefficients(state.components() * one);
    for (std::size_t i = 0; i < coefficients.size(); ++i)
        coefficients[i] = T(0.015 * std::sin(double(i + 1)));
    test::cc(
        cudaMemcpy(state.state_slab(), coefficients.data(),
                   coefficients.size() * sizeof(T), cudaMemcpyHostToDevice),
        "state");
    DeviceBuffer<T> rcon(points), zcon(points);
    op.inverse_fused(state.physical_const(), false, rcon.data(), zcon.data());
    std::array<T*, 18> device_geometry = {
        rs.d_r_e,  rs.d_z_e,  rs.d_l_e,  rs.d_ru_e, rs.d_zu_e, rs.d_lu_e,
        rs.d_rv_e, rs.d_zv_e, rs.d_lv_e, rs.d_r_o,  rs.d_z_o,  rs.d_l_o,
        rs.d_ru_o, rs.d_zu_o, rs.d_lu_o, rs.d_rv_o, rs.d_zv_o, rs.d_lv_o};
    std::array<std::vector<T>, 18> geometry;
    for (int f = 0; f < 18; ++f)
        geometry[f] = download(device_geometry[f], points);
    const auto rconstraint = download(rcon.data(), points);
    const auto zconstraint = download(zcon.data(), points);
    for (int j = 0; j < p.ns; ++j) {
        const double scale =
            1.0 / std::sqrt(double(std::max(j, 1)) / (p.ns - 1));
        for (int k = 0; k < p.nZnT; ++k) {
            const double theta = 2 * M_PI * (k % p.ntheta) / p.ntheta;
            const double zeta = 2 * M_PI * (k / p.ntheta) / p.nzeta;
            std::array<double, 18> expected{};
            std::array<double, 2> con{};
            for (int c = 0; c < state.components(); ++c) {
                for (int mode = 0; mode < p.mnmax; ++mode) {
                    const int m = mode / (ntor + 1), n = mode % (ntor + 1);
                    const auto b = basis(c, m, n, p.nfp, theta, zeta);
                    const double value =
                        coefficients[c * one + mode * p.ns + j];
                    for (int derivative = 0; derivative < 3; ++derivative) {
                        const int f = 9 * (m % 2) + 3 * derivative + c % 3;
                        const double sign =
                            derivative == 2 && c % 3 == 2 ? -1.0 : 1.0;
                        expected[f] += sign * value * b[derivative] *
                                       (m % 2 ? scale : 1.0);
                    }
                    if (c % 3 < 2) con[c % 3] += m * (m - 1) * value * b[0];
                }
            }
            const auto i = j * p.nZnT + k;
            for (int f = 0; f < 18; ++f)
                close(geometry[f][i], expected[f], tolerance, "inverse");
            close(rconstraint[i], con[0], tolerance, "R constraint");
            close(zconstraint[i], con[1], tolerance, "Z constraint");
        }
    }

    // Arbitrary full-period forces, with both symmetry sectors present.
    std::array<T*, 16> force_device = {
        rs.d_armn_e, rs.d_azmn_e, rs.d_brmn_e, rs.d_bzmn_e,
        rs.d_crmn_e, rs.d_czmn_e, rs.d_blmn_e, rs.d_clmn_e,
        rs.d_armn_o, rs.d_azmn_o, rs.d_brmn_o, rs.d_bzmn_o,
        rs.d_crmn_o, rs.d_czmn_o, rs.d_blmn_o, rs.d_clmn_o};
    std::array<std::vector<T>, 16> forces;
    for (int f = 0; f < 16; ++f) {
        forces[f].resize(points);
        for (std::size_t i = 0; i < points; ++i)
            forces[f][i] = T(std::sin((i + 1) * (f + 1) * .017));
        test::cc(cudaMemcpy(force_device[f], forces[f].data(),
                            points * sizeof(T), cudaMemcpyHostToDevice),
                 "force");
    }
    DeviceBuffer<T> residual(state.components() * one), zero(points);
    zero.zero();
    op.forward(SpectralView<T, DecomposedResidualDomain>(
                   residual.data(), p.ns, p.mnmax, nullptr, 0, true),
               zero.data(), zero.data(), zero.data(), zero.data());
    const auto projected = download(residual.data(), residual.size());
    for (int c = 0; c < state.components(); ++c) {
        const int field = c % 3;
        for (int mode = 0; mode < p.mnmax; ++mode) {
            const int m = mode / (ntor + 1), n = mode % (ntor + 1);
            const int parity = (m % 2) * 8;
            const double norm =
                std::sqrt(double((m ? 2 : 1) * (n ? 2 : 1))) / p.nZnT;
            for (int j = 0; j < p.ns; ++j) {
                double expected = 0;
                if ((j != 0 || (m == 0 && field < 2)) &&
                    (j != p.ns - 1 || field == 2)) {
                    for (int k = 0; k < p.nZnT; ++k) {
                        const auto i = j * p.nZnT + k;
                        const auto b =
                            basis(c, m, n, p.nfp,
                                  2 * M_PI * (k % p.ntheta) / p.ntheta,
                                  2 * M_PI * (k / p.ntheta) / p.nzeta);
                        const double a =
                            field < 2 ? forces[parity + field][i] : 0;
                        const double bu =
                            forces[parity + (field < 2 ? 2 + field : 6)][i];
                        const double bv =
                            forces[parity + (field < 2 ? 4 + field : 7)][i];
                        expected += norm * (a * b[0] + bu * b[1] - bv * b[2]);
                    }
                }
                close(projected[c * one + mode * p.ns + j], expected, tolerance,
                      "forward weak form");
            }
        }
    }
    // The constraint filter must retain both sine/cosine toroidal parts.
    std::vector<T> input(points), tcon(p.ns, T(1)), faccon(p.mnmax, T(1));
    for (int j = 0; j < p.ns; ++j) {
        for (int k = 0; k < p.nZnT; ++k) {
            const double theta = 2 * M_PI * (k % p.ntheta) / p.ntheta;
            const double zeta = 2 * M_PI * (k / p.ntheta) / p.nzeta;
            input[j * p.nZnT + k] =
                T(std::cos(2 * theta) * std::cos(ntor * zeta) +
                  .3 * std::sin(3 * theta) * std::sin(ntor * zeta));
        }
    }
    DeviceBuffer<T> d_input(points), d_tcon(p.ns), d_faccon(p.mnmax);
    test::cc(cudaMemcpy(d_input.data(), input.data(), points * sizeof(T),
                        cudaMemcpyHostToDevice),
             "constraint input");
    test::cc(cudaMemcpy(d_tcon.data(), tcon.data(), tcon.size() * sizeof(T),
                        cudaMemcpyHostToDevice),
             "tcon");
    test::cc(cudaMemcpy(d_faccon.data(), faccon.data(),
                        faccon.size() * sizeof(T), cudaMemcpyHostToDevice),
             "faccon");
    op.dealias_bandpass(d_input.data(), d_tcon.data(), d_faccon.data(),
                        zero.data());
    const auto filtered = download(zero.data(), points);
    for (std::size_t i = p.nZnT; i < points; ++i)
        close(filtered[i], input[i], tolerance,
              "asymmetric constraint bandpass");
    real_space_free(rs);
    mode_table_free(mt);
}
}  // namespace

int main() {
    auto parsed = read_problem_spec("inputs/asymmetric_tokamak.json", {});
    auto spec = parsed.spec;
    spec.ntor = 1;
    spec.raxis_c = {4, .01};
    spec.raxis_s = {0, .023};
    spec.zaxis_c = {.12, -.035};
    spec.zaxis_s = {0, .015};
    const auto problem = validate(spec, {}).value();
    const auto params = init_params<double>(problem);
    auto seeded = init_state(params, problem, false, false);
    const auto one = std::size_t(params.ns) * params.mnmax;
    const auto state = download(seeded.state_slab(), 12 * one);
    close(state[9 * one + params.ns], -.023, 1e-15, "VMEC R axis sine phase");
    close(state[7 * one + params.ns], -.035, 1e-15, "VMEC Z axis cosine phase");
    run<double>(0);
    run<double>(2);
    run<float>(0);
    run<float>(2);
    return test::summary();
}
