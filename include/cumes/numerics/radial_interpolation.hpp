// Scalar host reference for the radial transfer contracts.
#ifndef CUMES_INCLUDE_CUMES_NUMERICS_RADIAL_INTERPOLATION_HPP_
#define CUMES_INCLUDE_CUMES_NUMERICS_RADIAL_INTERPOLATION_HPP_

#include <algorithm>
#include <cmath>
#include <span>

namespace cumes {

enum class RadialInterpolation {
    LINEAR = 0,
    CATMULL_ROM = 1,
    BSPLINE = 2,
};

// Interpolate one physical spectral profile in s. Odd-m profiles use the
// scalxc-decomposed coordinate, an extrapolated old axis, and a zero new axis.
// Preconditions: values.size() >= 3, ns_new > values.size(), 0 <= j_new <
// ns_new. BSPLINE requires one row of weights with values.size() entries.
template <typename T>
T interpolate_radial_value(std::span<const T> values,
                           int ns_new,
                           int j_new,
                           bool odd,
                           RadialInterpolation interpolation,
                           std::span<const double> bspline_weights = {}) {
    const int ns_old = static_cast<int>(values.size());
    const T s = T(j_new) / T(ns_new - 1);
    const int j0 = (j_new * (ns_old - 1)) / (ns_new - 1);
    const int j1 = std::min(j0 + 1, ns_old - 1);
    const T t = std::clamp(s * T(ns_old - 1) - T(j0), T(0), T(1));
    const auto sample = [&](int j) {
        if (!odd) return values[j];
        const T s_old = T(j) / T(ns_old - 1);
        const T sqrt_s1 = std::sqrt(T(1) / T(ns_old - 1));
        const T scalxc = T(1) / std::max(std::sqrt(s_old), sqrt_s1);
        return values[j] * scalxc;
    };
    const auto regular_sample = [&](int j) {
        return odd && j == 0 ? T(2) * sample(1) - sample(2) : sample(j);
    };
    const T y0 = regular_sample(j0);
    const T y1 = regular_sample(j1);
    T interpolated = (T(1) - t) * y0 + t * y1;
    if (interpolation == RadialInterpolation::CATMULL_ROM && j0 != j1) {
        const T ym1 = j0 > 0 ? regular_sample(j0 - 1) : T(2) * y0 - y1;
        const T yp2 = j1 + 1 < ns_old ? regular_sample(j1 + 1) : T(2) * y1 - y0;
        interpolated = y0 + T(0.5) * t *
                                (y1 - ym1 +
                                 t * (T(2) * ym1 - T(5) * y0 + T(4) * y1 - yp2 +
                                      t * (T(3) * (y0 - y1) + yp2 - ym1)));
    } else if (interpolation == RadialInterpolation::BSPLINE) {
        if (j_new == ns_new - 1) return values.back();
        interpolated = T(0);
        for (int j_old = 0; j_old < ns_old; ++j_old)
            interpolated += T(bspline_weights[j_old]) * regular_sample(j_old);
    }
    if (!odd) return interpolated;
    const T value =
        interpolated * std::max(std::sqrt(s), std::sqrt(T(1) / T(ns_new - 1)));
    return j_new == 0 ? T(0) : value;
}

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_NUMERICS_RADIAL_INTERPOLATION_HPP_
