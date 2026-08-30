// seed_envelope.hpp — pure host radial-envelope policy for cold starts.
#ifndef CUMES_INCLUDE_CUMES_STATE_SEED_ENVELOPE_HPP_
#define CUMES_INCLUDE_CUMES_STATE_SEED_ENVELOPE_HPP_

#include <cmath>

namespace cumes {

inline constexpr double DEFAULT_3D_SEED_ENVELOPE = 0.12;
inline constexpr double DEFAULT_AXISYMMETRIC_COARSE_SEED_ENVELOPE = -0.07;

constexpr double default_seed_envelope(int ntor,
                                       bool free_boundary,
                                       int initial_ns) noexcept {
    if (free_boundary) return 0.0;
    if (ntor > 0) return DEFAULT_3D_SEED_ENVELOPE;
    return initial_ns <= 11 ? DEFAULT_AXISYMMETRIC_COARSE_SEED_ENVELOPE : 0.0;
}

template <typename T>
T seed_radial_weight(int m, T s, T envelope_correction) {
    const T sqrt_s = std::sqrt(s);
    const T regular_weight = m == 1 ? sqrt_s : std::pow(sqrt_s, m);
    return regular_weight * (T(1) + envelope_correction * (T(1) - s));
}

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_STATE_SEED_ENVELOPE_HPP_
