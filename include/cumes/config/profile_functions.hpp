// profile_functions.hpp — the radial profile power-series evaluators, shared
// by the host-side validation (ValidatedProblem::validate) and the device
// Profiles upload step (src/profiles_impl.cuh).
//
// Single source of truth: both consumers evaluate the identical Horner loops
// over the ProblemSpec coefficient vectors, so the host-validated
// normalization scalars (T_edge = torflux(1), C_edge = current integral at
// min(|bloat|, 1)) are bit-identical to the values the Profiles constructor
// divides by. The functions are plain host/device C++ (no CUDA includes) and
// templated on the scalar type T exactly as the legacy file-static copies.
#pragma once

#include "cumes/config/device_params.hpp"
#include "cumes/config/problem_spec.hpp"

#include <cmath>

namespace cumes {

// Horner evaluation of a power series; with integrate=true gives the
// integral ∫₀ˣ Σ c_i t^i dt = x·Σ c_i·x^i/(i+1) (vmecpp evalPowerSeries).
template <typename T>
inline T evalPowerSeries(const double* c, int n, T x, bool integrate) {
    T ret = T(0.0);
    for (int i = n - 1; i >= 0; --i) {
        if (integrate) {
            ret = x * ret + T(c[i]) / T(i + 1);
        } else {
            ret = x * ret + T(c[i]);
        }
    }
    if (integrate) ret *= x;
    return ret;
}

// The power-series coefficients live on the ProblemSpec; the helpers read them
// by (data, length) exactly as the legacy fixed-capacity InputParams did.
template <typename T>
inline T torflux(const ProblemSpec& sp, T x) {
    const auto& c = sp.toroidal_flux.coefficients;
    return x * evalPowerSeries<T>(c.data(), (int)c.size(), x, false);
}

template <typename T>
inline T torfluxDeriv(const ProblemSpec& sp, T x) {
    T ret = T(0.0);
    const auto& c = sp.toroidal_flux.coefficients;
    for (int i = 0; i < (int)c.size(); ++i) {
        ret += T(i + 1) * T(c[i]) * pow(x, i);
    }
    return ret;
}

template <typename T>
inline T evalIotaProfile(const ProblemSpec& sp, T x) {
    const auto& c = sp.iota.coefficients;
    return evalPowerSeries<T>(c.data(), (int)c.size(), x, false);
}

template <typename T>
inline T evalMassProfile(const ProblemSpec& sp, T x) {
    T normX = fmin(fabs(x * T(sp.physical.bloat)), T(1.0));
    const auto& c = sp.mass.coefficients;
    return evalPowerSeries<T>(c.data(), (int)c.size(), normX, false) *
           (DeviceParams<T>::kMu0 * T(sp.physical.pres_scale));
}

template <typename T>
inline T evalCurrProfile(const ProblemSpec& sp, T x) {
    T normX = fmin(fabs(x * T(sp.physical.bloat)), T(1.0));
    const auto& c = sp.current.coefficients;
    return evalPowerSeries<T>(c.data(), (int)c.size(), normX, true);
}

}  // namespace cumes
