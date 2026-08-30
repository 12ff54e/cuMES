// axisymmetric_lambda_seed.hpp — straight-field-line lambda cold predictor.
#ifndef CUMES_INCLUDE_CUMES_STATE_AXISYMMETRIC_LAMBDA_SEED_HPP_
#define CUMES_INCLUDE_CUMES_STATE_AXISYMMETRIC_LAMBDA_SEED_HPP_

#include "cumes/state/seed_envelope.hpp"

#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

namespace cumes {

inline constexpr double DEFAULT_AXISYMMETRIC_LAMBDA_SEED_SCALE = 0.65;
inline constexpr double DEFAULT_FREE_AXISYMMETRIC_LAMBDA_SEED_SCALE = 1.0;

constexpr double default_axisymmetric_lambda_seed(int ntor,
                                                  bool free_boundary) noexcept {
    if (ntor != 0) return 0.0;
    return free_boundary ? DEFAULT_FREE_AXISYMMETRIC_LAMBDA_SEED_SCALE
                         : DEFAULT_AXISYMMETRIC_LAMBDA_SEED_SCALE;
}

// For axisymmetry, B^zeta and the cylindrical toroidal-field relation imply
//
//   lambda_theta = 1 - q/<q>,  q = sqrt(g)/R^2.
//
// With sqrt(g) = R (R_s Z_theta - R_theta Z_s), q is the cross product divided
// by R. Projecting lambda_theta onto cos(m theta), then dividing by m, gives
// the sine-family lambda coefficients. The predictor is scaled because R/Z
// will still move during the coupled solve. All output is staged so invalid
// input geometry leaves the caller's zero-lambda seed untouched.
template <typename T>
bool seed_axisymmetric_lambda(int ns,
                              int mpol,
                              std::span<const T> rmncc,
                              std::span<const T> zmnsc,
                              std::span<T> lmnsc,
                              std::span<const double> boundary_r,
                              std::span<const double> boundary_z,
                              double raxis_c,
                              T envelope_correction,
                              T predictor_scale) {
    const std::size_t size = static_cast<std::size_t>(ns) * mpol;
    if (ns < 2 || mpol < 1 || rmncc.size() != size || zmnsc.size() != size ||
        lmnsc.size() != size || boundary_r.size() < std::size_t(mpol) ||
        boundary_z.size() < std::size_t(mpol) ||
        !std::isfinite(static_cast<double>(predictor_scale))) {
        return false;
    }
    if (predictor_scale == T(0)) return true;

    const int nquad = std::max(64, 8 * mpol);
    std::vector<double> cosine(static_cast<std::size_t>(mpol) * nquad);
    std::vector<double> sine(static_cast<std::size_t>(mpol) * nquad);
    for (int m = 0; m < mpol; ++m) {
        for (int k = 0; k < nquad; ++k) {
            const double theta = 6.2831853071795864769 * k / nquad;
            cosine[static_cast<std::size_t>(m) * nquad + k] =
                std::cos(m * theta);
            sine[static_cast<std::size_t>(m) * nquad + k] = std::sin(m * theta);
        }
    }

    std::vector<T> seeded(lmnsc.begin(), lmnsc.end());
    std::vector<double> q(nquad);
    for (int j = 1; j < ns; ++j) {
        const double s = static_cast<double>(j) / (ns - 1);
        double q_mean = 0.0;
        double q_scale = 0.0;
        for (int k = 0; k < nquad; ++k) {
            double r = 0.0;
            double r_s = 0.0;
            double r_theta = 0.0;
            double z_s = 0.0;
            double z_theta = 0.0;
            for (int m = 0; m < mpol; ++m) {
                const std::size_t index = static_cast<std::size_t>(m) * ns + j;
                const double c =
                    cosine[static_cast<std::size_t>(m) * nquad + k];
                const double sn = sine[static_cast<std::size_t>(m) * nquad + k];
                const double r_m = static_cast<double>(rmncc[index]);
                const double z_m = static_cast<double>(zmnsc[index]);
                r += r_m * c;
                r_theta -= m * r_m * sn;
                z_theta += m * z_m * c;
                if (m == 0) {
                    r_s += boundary_r[0] - raxis_c;
                } else {
                    const double dw = seed_radial_weight_derivative(
                        m, s, static_cast<double>(envelope_correction));
                    r_s += dw * boundary_r[m] * c;
                    z_s += dw * boundary_z[m] * sn;
                }
            }
            if (!std::isfinite(r) || std::abs(r) <= 1.0e-12) return false;
            q[k] = (r_s * z_theta - r_theta * z_s) / r;
            if (!std::isfinite(q[k])) return false;
            q_mean += q[k];
            q_scale = std::max(q_scale, std::abs(q[k]));
        }
        q_mean /= nquad;
        if (!std::isfinite(q_mean) || q_scale == 0.0 ||
            std::abs(q_mean) <= 1.0e-12 * q_scale) {
            return false;
        }

        for (int m = 1; m < mpol; ++m) {
            double coefficient = 0.0;
            for (int k = 0; k < nquad; ++k) {
                coefficient += (1.0 - q[k] / q_mean) *
                               cosine[static_cast<std::size_t>(m) * nquad + k];
            }
            const double value = static_cast<double>(predictor_scale) * 2.0 *
                                 coefficient / (nquad * m);
            if (!std::isfinite(value)) return false;
            seeded[static_cast<std::size_t>(m) * ns + j] = T(value);
        }
    }

    std::copy(seeded.begin(), seeded.end(), lmnsc.begin());
    return true;
}

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_STATE_AXISYMMETRIC_LAMBDA_SEED_HPP_
