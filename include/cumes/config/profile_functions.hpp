// profile_functions.hpp — the radial profile power-series evaluators, shared
// by the host-side validation (ValidatedProblem::validate) and the device
// Profiles upload step (src/kernels/profiles_impl.cuh).
//
// Single source of truth: both consumers evaluate the identical Horner loops
// over the ProblemSpec coefficient vectors, so the host-validated
// normalization scalars (T_edge = torflux(1), C_edge = current integral at
// min(|bloat|, 1)) are bit-identical to the values the Profiles constructor
// divides by. The functions are plain host/device C++ (no CUDA includes) and
// templated on the scalar type T exactly as the legacy file-static copies.
#ifndef CUMES_INCLUDE_CUMES_CONFIG_PROFILE_FUNCTIONS_HPP_
#define CUMES_INCLUDE_CUMES_CONFIG_PROFILE_FUNCTIONS_HPP_

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

// vmecpp's "two_power" profile: f(x) = c[0]·(1 − x^c[1])^c[2] (radial_profiles
// evalTwoPower). With integrate=true the same form is integrated from 0 to x
// with the 10-point Gauss-Legendre quadrature vmecpp uses (the current
// profile is specified as I-prime and must be integrated); the node/weight
// table is copied verbatim so the host-validated edge integral C_edge and
// the value the Profiles upload step divides by are bit-identical to vmecpp's
// double evaluation (same constants, same left-to-right arithmetic order).
template <typename T>
inline T evalTwoPower(const double* c, int n, T x, bool integrate) {
    if (n < 3) return T(0.0);  // validation rejects this case up front
    constexpr double GLX[10] = {0.01304673574141414, 0.06746831665550774,
                                0.1602952158504878,  0.2833023029353764,
                                0.4255628305091844,  0.5744371694908156,
                                0.7166976970646236,  0.8397047841495122,
                                0.9325316833444923,  0.9869532642585859};
    constexpr double GLW[10] = {0.03333567215434407, 0.0747256745752903,
                                0.1095431812579910,  0.1346333596549982,
                                0.1477621123573764,  0.1477621123573764,
                                0.1346333596549982,  0.1095431812579910,
                                0.0747256745752903,  0.03333567215434407};
    if (!integrate) { return T(c[0]) * pow(T(1.0) - pow(x, T(c[1])), T(c[2])); }
    T ret = T(0.0);
    for (int i = 0; i < 10; ++i) {
        const T xp = x * T(GLX[i]);
        ret += T(GLW[i]) * T(c[0]) * pow(T(1.0) - pow(xp, T(c[1])), T(c[2]));
    }
    return ret * x;
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
    const auto& prof = sp.mass;
    const auto& c = prof.coefficients;
    T p = (prof.type == ProfileType::TWO_POWER)
              ? evalTwoPower<T>(c.data(), (int)c.size(), normX, false)
              : evalPowerSeries<T>(c.data(), (int)c.size(), normX, false);
    return p * (DeviceParams<T>::MU_0 * T(sp.physical.pres_scale));
}

template <typename T>
inline T evalCurrProfile(const ProblemSpec& sp, T x) {
    T normX = fmin(fabs(x * T(sp.physical.bloat)), T(1.0));
    const auto& prof = sp.current;
    const auto& c = prof.coefficients;
    if (prof.type == ProfileType::TWO_POWER) {
        return evalTwoPower<T>(c.data(), (int)c.size(), normX, true);
    }
    return evalPowerSeries<T>(c.data(), (int)c.size(), normX, true);
}

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_CONFIG_PROFILE_FUNCTIONS_HPP_
