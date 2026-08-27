// derived_fields.cpp — stagger-aware B/J scientific output post-processing.
#include "cumes/io/derived_fields.hpp"

#include "cumes/core/checked_size.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace cumes {
namespace {

struct Counts {
    std::size_t points = 0;
    std::size_t full = 0;
    std::size_t half = 0;
};

Result<Counts> checked_counts(const DerivedFieldInputs& in) {
    if (in.ns < 3 || in.ntheta < 1 || in.nzeta < 1 || in.nfp < 1) {
        return Result<Counts>("derived fields: invalid grid dimensions");
    }
    if (!(in.delta_s > 0.0) || !(in.mu0 > 0.0)) {
        return Result<Counts>("derived fields: invalid physical scale");
    }
    const auto points = checked_mul(static_cast<std::size_t>(in.ntheta),
                                    static_cast<std::size_t>(in.nzeta));
    if (!points)
        return Result<Counts>("derived fields: angular extent overflows");
    const auto full = checked_mul(static_cast<std::size_t>(in.ns), *points);
    const auto half = checked_mul(static_cast<std::size_t>(in.ns - 1), *points);
    if (!full || !half)
        return Result<Counts>("derived fields: radial extent overflows");
    return Counts{*points, *full, *half};
}

bool has_size(const std::vector<double>& values, std::size_t expected) {
    return values.size() == expected;
}

// Periodic Fourier-collocation derivative. The input layout is
// [surface][zeta][theta]. direction_theta differentiates with respect to
// theta; otherwise it differentiates with respect to the physical toroidal
// angle zeta, so the one-field-period grid receives the nfp multiplier.
std::vector<double> periodic_derivative(const std::vector<double>& values,
                                        int surfaces,
                                        int ntheta,
                                        int nzeta,
                                        int nfp,
                                        bool direction_theta) {
    const int length = direction_theta ? ntheta : nzeta;
    std::vector<double> result(values.size(), 0.0);
    if (length == 1) return result;

    const double pi = std::acos(-1.0);
    const double direction_scale = direction_theta ? 1.0 : nfp;
    const int lines_per_surface = direction_theta ? nzeta : ntheta;
    for (int surface = 0; surface < surfaces; ++surface) {
        for (int line = 0; line < lines_per_surface; ++line) {
            for (int target = 0; target < length; ++target) {
                double sum = 0.0;
                for (int source = 0; source < length; ++source) {
                    if (source == target) continue;
                    const int difference = target - source;
                    const double angle =
                        pi * static_cast<double>(difference) / length;
                    const double parity =
                        (std::abs(difference) % 2 == 0) ? 1.0 : -1.0;
                    const double weight = (length % 2 == 0)
                                              ? 0.5 * parity / std::tan(angle)
                                              : 0.5 * parity / std::sin(angle);
                    const std::size_t index =
                        direction_theta
                            ? static_cast<std::size_t>(surface) * ntheta *
                                      nzeta +
                                  static_cast<std::size_t>(line) * ntheta +
                                  source
                            : static_cast<std::size_t>(surface) * ntheta *
                                      nzeta +
                                  static_cast<std::size_t>(source) * ntheta +
                                  line;
                    sum += weight * values[index];
                }
                const std::size_t index =
                    direction_theta
                        ? static_cast<std::size_t>(surface) * ntheta * nzeta +
                              static_cast<std::size_t>(line) * ntheta + target
                        : static_cast<std::size_t>(surface) * ntheta * nzeta +
                              static_cast<std::size_t>(target) * ntheta + line;
                result[index] = direction_scale * sum;
            }
        }
    }
    return result;
}

void extrapolate_endpoints(std::vector<double>& values,
                           int ns,
                           std::size_t points) {
    for (std::size_t point = 0; point < points; ++point) {
        if (ns >= 4) {
            values[point] =
                2.0 * values[points + point] - values[2 * points + point];
            values[static_cast<std::size_t>(ns - 1) * points + point] =
                2.0 *
                    values[static_cast<std::size_t>(ns - 2) * points + point] -
                values[static_cast<std::size_t>(ns - 3) * points + point];
        } else {
            values[point] = values[points + point];
            values[2 * points + point] = values[points + point];
        }
    }
}

double safe_divide(double numerator, double denominator) {
    return std::isfinite(denominator) && std::abs(denominator) > 1.0e-30
               ? numerator / denominator
               : 0.0;
}

}  // namespace

Status populate_derived_fields(const DerivedFieldInputs& in,
                               EquilibriumSnapshot& snapshot) {
    const auto counts_result = checked_counts(in);
    if (!counts_result) return Status(counts_result.error());
    const Counts counts = counts_result.value();

    const std::vector<const std::vector<double>*> full_arrays = {
        &in.r_e,  &in.r_o,  &in.z_e,  &in.z_o,  &in.ru_e, &in.ru_o,
        &in.zu_e, &in.zu_o, &in.rv_e, &in.rv_o, &in.zv_e, &in.zv_o};
    const std::vector<const std::vector<double>*> half_arrays = {
        &in.rs,    &in.zs,    &in.ru12,  &in.zu12, &in.sqrtg,
        &in.bsupu, &in.bsupv, &in.bsubu, &in.bsubv};
    if (!has_size(in.sqrt_s_full, static_cast<std::size_t>(in.ns)) ||
        !has_size(in.sqrt_s_half, static_cast<std::size_t>(in.ns - 1))) {
        return Status("derived fields: radial profile shape mismatch");
    }
    for (const auto* values : full_arrays)
        if (!has_size(*values, counts.full))
            return Status("derived fields: full-grid shape mismatch");
    for (const auto* values : half_arrays)
        if (!has_size(*values, counts.half))
            return Status("derived fields: half-grid shape mismatch");

    if (snapshot.ns != 0 && snapshot.ns != in.ns)
        return Status("derived fields: snapshot radial dimension mismatch");
    snapshot.ns = in.ns;
    snapshot.ntheta = in.ntheta;
    snapshot.nzeta = in.nzeta;
    for (auto& field : snapshot.half_fields) field.assign(counts.half, 0.0);
    for (auto& field : snapshot.full_fields) field.assign(counts.full, 0.0);

    auto& sqrtg = snapshot.half_fields[EquilibriumSnapshot::SQRTG];
    auto& bsupu = snapshot.half_fields[EquilibriumSnapshot::BSUPU];
    auto& bsupv = snapshot.half_fields[EquilibriumSnapshot::BSUPV];
    auto& bsubs = snapshot.half_fields[EquilibriumSnapshot::BSUBS];
    auto& bsubu = snapshot.half_fields[EquilibriumSnapshot::BSUBU];
    auto& bsubv = snapshot.half_fields[EquilibriumSnapshot::BSUBV];
    sqrtg = in.sqrtg;
    bsupu = in.bsupu;
    bsupv = in.bsupv;
    bsubu = in.bsubu;
    bsubv = in.bsubv;

    // Physical half-grid radial basis vectors, including the derivative of
    // the sqrt(s)-scaled odd-m representation. These also provide B_s and the
    // full-grid radial metric used to lower the current-density index.
    std::vector<double> rs_physical(counts.half, 0.0);
    std::vector<double> zs_physical(counts.half, 0.0);
    for (int jh = 0; jh < in.ns - 1; ++jh) {
        const double sqrt_s = in.sqrt_s_half[jh];
        const std::size_t inner = static_cast<std::size_t>(jh) * counts.points;
        const std::size_t outer = inner + counts.points;
        const std::size_t half = inner;
        for (std::size_t point = 0; point < counts.points; ++point) {
            const std::size_t hi = half + point;
            const std::size_t fi = inner + point;
            const std::size_t fo = outer + point;
            const double r_odd_half = 0.5 * (in.r_o[fi] + in.r_o[fo]);
            const double z_odd_half = 0.5 * (in.z_o[fi] + in.z_o[fo]);
            const double rs_value =
                in.rs[hi] + safe_divide(r_odd_half, 2.0 * sqrt_s);
            const double zs_value =
                in.zs[hi] + safe_divide(z_odd_half, 2.0 * sqrt_s);
            const double rv_half = 0.5 * ((in.rv_e[fi] + in.rv_e[fo]) +
                                          sqrt_s * (in.rv_o[fi] + in.rv_o[fo]));
            const double zv_half = 0.5 * ((in.zv_e[fi] + in.zv_e[fo]) +
                                          sqrt_s * (in.zv_o[fi] + in.zv_o[fo]));
            rs_physical[hi] = rs_value;
            zs_physical[hi] = zs_value;
            const double gsu = rs_value * in.ru12[hi] + zs_value * in.zu12[hi];
            const double gsv = rs_value * rv_half + zs_value * zv_half;
            bsubs[hi] = gsu * bsupu[hi] + gsv * bsupv[hi];
        }
    }

    // Curl(B) in flux coordinates. J^s is naturally a half-grid quantity;
    // J^u/J^v are naturally full-grid quantities because their radial
    // differences straddle an integer surface.
    const auto dtheta_bsubv = periodic_derivative(bsubv, in.ns - 1, in.ntheta,
                                                  in.nzeta, in.nfp, true);
    const auto dzeta_bsubu = periodic_derivative(bsubu, in.ns - 1, in.ntheta,
                                                 in.nzeta, in.nfp, false);
    std::vector<double> jsups_half(counts.half, 0.0);
    for (std::size_t i = 0; i < counts.half; ++i) {
        jsups_half[i] =
            safe_divide(dtheta_bsubv[i] - dzeta_bsubu[i], in.mu0 * sqrtg[i]);
    }

    std::vector<double> bsubs_full(counts.full, 0.0);
    for (int jf = 1; jf < in.ns - 1; ++jf) {
        const std::size_t full = static_cast<std::size_t>(jf) * counts.points;
        const std::size_t inside =
            static_cast<std::size_t>(jf - 1) * counts.points;
        const std::size_t outside =
            static_cast<std::size_t>(jf) * counts.points;
        for (std::size_t point = 0; point < counts.points; ++point) {
            bsubs_full[full + point] =
                0.5 * (bsubs[inside + point] + bsubs[outside + point]);
        }
    }
    extrapolate_endpoints(bsubs_full, in.ns, counts.points);
    const auto dtheta_bsubs = periodic_derivative(bsubs_full, in.ns, in.ntheta,
                                                  in.nzeta, in.nfp, true);
    const auto dzeta_bsubs = periodic_derivative(bsubs_full, in.ns, in.ntheta,
                                                 in.nzeta, in.nfp, false);

    auto& jsups = snapshot.full_fields[EquilibriumSnapshot::JSUPS];
    auto& jsupu = snapshot.full_fields[EquilibriumSnapshot::JSUPU];
    auto& jsupv = snapshot.full_fields[EquilibriumSnapshot::JSUPV];
    auto& jsubs = snapshot.full_fields[EquilibriumSnapshot::JSUBS];
    auto& jsubu = snapshot.full_fields[EquilibriumSnapshot::JSUBU];
    auto& jsubv = snapshot.full_fields[EquilibriumSnapshot::JSUBV];

    for (int jf = 1; jf < in.ns - 1; ++jf) {
        const std::size_t full = static_cast<std::size_t>(jf) * counts.points;
        const std::size_t inside =
            static_cast<std::size_t>(jf - 1) * counts.points;
        const std::size_t outside =
            static_cast<std::size_t>(jf) * counts.points;
        const double sqrt_s = in.sqrt_s_full[jf];
        for (std::size_t point = 0; point < counts.points; ++point) {
            const std::size_t f = full + point;
            const std::size_t hi = inside + point;
            const std::size_t ho = outside + point;
            const double sqrtg_full = 0.5 * (sqrtg[hi] + sqrtg[ho]);
            jsups[f] = safe_divide(
                sqrtg[hi] * jsups_half[hi] + sqrtg[ho] * jsups_half[ho],
                2.0 * sqrtg_full);
            jsupu[f] = safe_divide(
                dzeta_bsubs[f] - (bsubv[ho] - bsubv[hi]) / in.delta_s,
                in.mu0 * sqrtg_full);
            jsupv[f] = safe_divide(
                (bsubu[ho] - bsubu[hi]) / in.delta_s - dtheta_bsubs[f],
                in.mu0 * sqrtg_full);

            const double r = in.r_e[f] + sqrt_s * in.r_o[f];
            const double ru = in.ru_e[f] + sqrt_s * in.ru_o[f];
            const double zu = in.zu_e[f] + sqrt_s * in.zu_o[f];
            const double rv = in.rv_e[f] + sqrt_s * in.rv_o[f];
            const double zv = in.zv_e[f] + sqrt_s * in.zv_o[f];
            const double rs_full = 0.5 * (rs_physical[hi] + rs_physical[ho]);
            const double zs_full = 0.5 * (zs_physical[hi] + zs_physical[ho]);
            const double gss = rs_full * rs_full + zs_full * zs_full;
            const double gsu = rs_full * ru + zs_full * zu;
            const double gsv = rs_full * rv + zs_full * zv;
            const double guu = ru * ru + zu * zu;
            const double guv = ru * rv + zu * zv;
            const double gvv = rv * rv + zv * zv + r * r;
            jsubs[f] = gss * jsups[f] + gsu * jsupu[f] + gsv * jsupv[f];
            jsubu[f] = gsu * jsups[f] + guu * jsupu[f] + guv * jsupv[f];
            jsubv[f] = gsv * jsups[f] + guv * jsupu[f] + gvv * jsupv[f];
        }
    }

    for (auto& field : snapshot.full_fields)
        extrapolate_endpoints(field, in.ns, counts.points);
    return Status();
}

}  // namespace cumes
