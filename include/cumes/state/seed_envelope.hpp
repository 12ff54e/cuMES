// seed_envelope.hpp — pure host radial-envelope policy for cold starts.
#ifndef CUMES_INCLUDE_CUMES_STATE_SEED_ENVELOPE_HPP_
#define CUMES_INCLUDE_CUMES_STATE_SEED_ENVELOPE_HPP_

#include <cmath>

namespace cumes {

inline constexpr double DEFAULT_3D_SEED_ENVELOPE = 0.12;
inline constexpr double DEFAULT_FREE_3D_FINE_SEED_ENVELOPE = 0.03;
inline constexpr int FREE_3D_COARSE_SEED_MAX_NS = 25;
inline constexpr double DEFAULT_AXISYMMETRIC_COARSE_SEED_ENVELOPE = -0.07;

constexpr double default_seed_envelope(int ntor,
                                       bool free_boundary,
                                       int initial_ns) noexcept {
    if (free_boundary) {
        if (ntor == 0) return 0.0;
        return initial_ns <= FREE_3D_COARSE_SEED_MAX_NS
                   ? DEFAULT_3D_SEED_ENVELOPE
                   : DEFAULT_FREE_3D_FINE_SEED_ENVELOPE;
    }
    if (ntor > 0) return DEFAULT_3D_SEED_ENVELOPE;
    return initial_ns <= 11 ? DEFAULT_AXISYMMETRIC_COARSE_SEED_ENVELOPE : 0.0;
}

template <typename T>
T seed_radial_weight(int m, T s, T envelope_correction) {
    const T sqrt_s = std::sqrt(s);
    const T regular_weight = m == 1 ? sqrt_s : std::pow(sqrt_s, m);
    return regular_weight * (T(1) + envelope_correction * (T(1) - s));
}

template <typename T>
T seed_radial_weight_derivative(int m, T s, T envelope_correction) {
    const T half_m = T(m) / T(2);
    const T regular_weight = std::pow(s, half_m);
    const T regular_derivative = half_m * std::pow(s, half_m - T(1));
    return regular_derivative * (T(1) + envelope_correction * (T(1) - s)) -
           envelope_correction * regular_weight;
}

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_STATE_SEED_ENVELOPE_HPP_
