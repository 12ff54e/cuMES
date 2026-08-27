// test_derived_fields.cpp — self-contained staggered curl(B) verification.
#include "cumes/io/derived_fields.hpp"
#include "cumes_test.h"

#include <cmath>
#include <cstddef>

using namespace cumes::test;

namespace {

cumes::DerivedFieldInputs manufactured_input() {
    cumes::DerivedFieldInputs in;
    in.ns = 6;
    in.ntheta = 8;
    in.nzeta = 7;
    in.nfp = 3;
    in.delta_s = 1.0 / (in.ns - 1);
    in.mu0 = 1.0;
    const std::size_t points = static_cast<std::size_t>(in.ntheta) * in.nzeta;
    const std::size_t full = static_cast<std::size_t>(in.ns) * points;
    const std::size_t half = static_cast<std::size_t>(in.ns - 1) * points;

    in.sqrt_s_full.resize(in.ns);
    in.sqrt_s_half.resize(in.ns - 1);
    for (int j = 0; j < in.ns; ++j)
        in.sqrt_s_full[j] = std::sqrt(j * in.delta_s);
    for (int j = 0; j < in.ns - 1; ++j)
        in.sqrt_s_half[j] = std::sqrt((j + 0.5) * in.delta_s);

    auto full_zero = [&]() { return std::vector<double>(full, 0.0); };
    in.r_e.assign(full, 2.0);
    in.r_o = full_zero();
    in.z_e = full_zero();
    in.z_o = full_zero();
    in.ru_e = full_zero();
    in.ru_o = full_zero();
    in.zu_e = full_zero();
    in.zu_o = full_zero();
    in.rv_e = full_zero();
    in.rv_o = full_zero();
    in.zv_e = full_zero();
    in.zv_o = full_zero();

    auto half_zero = [&]() { return std::vector<double>(half, 0.0); };
    in.rs = half_zero();
    in.zs = half_zero();
    in.ru12 = half_zero();
    in.zu12 = half_zero();
    in.sqrtg.assign(half, 1.0);
    in.bsupu = half_zero();
    in.bsupv = half_zero();
    in.bsubu.resize(half);
    in.bsubv.resize(half);

    const double two_pi = 2.0 * std::acos(-1.0);
    for (int jh = 0; jh < in.ns - 1; ++jh) {
        const double s = (jh + 0.5) * in.delta_s;
        for (int k = 0; k < in.nzeta; ++k) {
            const double alpha = two_pi * k / in.nzeta;
            for (int l = 0; l < in.ntheta; ++l) {
                const double theta = two_pi * l / in.ntheta;
                const std::size_t index =
                    static_cast<std::size_t>(jh) * points + k * in.ntheta + l;
                in.bsupu[index] = 10.0 + s + std::cos(theta);
                in.bsupv[index] = -2.0 + std::sin(alpha);
                in.bsubu[index] = s * std::cos(alpha);
                in.bsubv[index] = s * s * std::sin(theta);
            }
        }
    }
    return in;
}

void test_manufactured_curl() {
    const auto in = manufactured_input();
    cumes::EquilibriumSnapshot snapshot;
    const cumes::Status status = cumes::populate_derived_fields(in, snapshot);
    check(status.has_value(), "derived fields: manufactured input accepted");
    if (!status) return;
    check(snapshot.has_derived_fields(), "derived fields: complete shapes");
    check(snapshot.half_fields[cumes::EquilibriumSnapshot::BSUPU] == in.bsupu,
          "derived fields: B^theta copied exactly");
    check(snapshot.half_fields[cumes::EquilibriumSnapshot::BSUBV] == in.bsubv,
          "derived fields: B_zeta copied exactly");

    const std::size_t points = static_cast<std::size_t>(in.ntheta) * in.nzeta;
    const double two_pi = 2.0 * std::acos(-1.0);
    double max_js = 0.0;
    double max_ju = 0.0;
    double max_jv = 0.0;
    double max_jsubv = 0.0;
    double max_bsubs = 0.0;
    for (double value : snapshot.half_fields[cumes::EquilibriumSnapshot::BSUBS])
        max_bsubs = std::max(max_bsubs, std::abs(value));
    for (int jf = 1; jf < in.ns - 1; ++jf) {
        const double s = jf * in.delta_s;
        const double s_inside = (jf - 0.5) * in.delta_s;
        const double s_outside = (jf + 0.5) * in.delta_s;
        const double mean_s_squared =
            0.5 * (s_inside * s_inside + s_outside * s_outside);
        for (int k = 0; k < in.nzeta; ++k) {
            const double alpha = two_pi * k / in.nzeta;
            for (int l = 0; l < in.ntheta; ++l) {
                const double theta = two_pi * l / in.ntheta;
                const std::size_t index =
                    static_cast<std::size_t>(jf) * points + k * in.ntheta + l;
                const double want_js = mean_s_squared * std::cos(theta) +
                                       in.nfp * s * std::sin(alpha);
                const double want_ju = -2.0 * s * std::sin(theta);
                const double want_jv = std::cos(alpha);
                max_js = std::max(
                    max_js,
                    std::abs(
                        snapshot.full_fields[cumes::EquilibriumSnapshot::JSUPS]
                                            [index] -
                        want_js));
                max_ju = std::max(
                    max_ju,
                    std::abs(
                        snapshot.full_fields[cumes::EquilibriumSnapshot::JSUPU]
                                            [index] -
                        want_ju));
                max_jv = std::max(
                    max_jv,
                    std::abs(
                        snapshot.full_fields[cumes::EquilibriumSnapshot::JSUPV]
                                            [index] -
                        want_jv));
                max_jsubv = std::max(
                    max_jsubv,
                    std::abs(
                        snapshot.full_fields[cumes::EquilibriumSnapshot::JSUBV]
                                            [index] -
                        4.0 * want_jv));
            }
        }
    }
    expect_near(max_bsubs, 0.0, 1.0e-14, "derived fields: B_s metric lowering");
    expect_near(max_js, 0.0, 2.0e-14,
                "derived fields: spectral angular curl J^s");
    expect_near(max_ju, 0.0, 2.0e-14,
                "derived fields: staggered radial curl J^theta");
    expect_near(max_jv, 0.0, 2.0e-14,
                "derived fields: staggered radial curl J^zeta");
    expect_near(max_jsubv, 0.0, 8.0e-14,
                "derived fields: lower J index with cylindrical metric");
}

void test_shape_rejection() {
    auto in = manufactured_input();
    in.sqrtg.pop_back();
    cumes::EquilibriumSnapshot snapshot;
    const auto status = cumes::populate_derived_fields(in, snapshot);
    check(!status.has_value(), "derived fields: malformed half-grid rejected");
}

}  // namespace

int main() {
    test_manufactured_curl();
    test_shape_rejection();
    return summary();
}
