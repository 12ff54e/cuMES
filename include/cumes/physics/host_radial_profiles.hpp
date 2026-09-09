#ifndef CUMES_INCLUDE_CUMES_PHYSICS_HOST_RADIAL_PROFILES_HPP_
#define CUMES_INCLUDE_CUMES_PHYSICS_HOST_RADIAL_PROFILES_HPP_

#include "cumes/config/profile_functions.hpp"
#include "cumes/core/error.hpp"

#include <cmath>
#include <vector>

namespace cumes {

// Prescribed radial data, evaluated before backend allocation. For gamma=0,
// pressure is the mass profile and dVds remains its unit placeholder.
template <typename T>
struct HostRadialProfiles {
    using val_type = T;

    std::vector<T> iota_f;
    std::vector<T> phip_f;
    std::vector<T> chi_f;
    std::vector<T> sqrt_s_f;
    std::vector<T> iota_h;
    std::vector<T> pres_h;
    std::vector<T> phip_h;
    std::vector<T> dvds_h;
    std::vector<T> sqrt_s_h;
    std::vector<T> curr_h;
    std::vector<T> chip_h;
    T delta_s = T(0);
    T lamscale = T(0);
    T max_toroidal_flux = T(0);
};

// Requires a validated profile specification and ns >= 2.
// Keep each backend's existing exceptional-value semantics: native fmin
// discards a NaN operand while browser std::min retains a NaN first operand.
// Native zero current skips evaluation and produces +0; browser multiplication
// retains a negative zero (or nonfinite profile value) for its paired words.
template <typename T, typename Minimum>
HostRadialProfiles<T> evaluate_radial_profiles(const ProblemSpec& spec,
                                               int ns,
                                               Minimum minimum,
                                               bool skip_zero_current) {
    using std::sqrt;

    HostRadialProfiles<T> profiles;
    profiles.delta_s = T(1.0) / T(ns - 1);
    profiles.max_toroidal_flux =
        T(DeviceParams<T>::SIGN_JACOBIAN * spec.physical.phiedge) /
        T(2.0 * M_PI);
    const T torflux_edge = torflux<T>(spec, T(1.0));
    if (torflux_edge != T(0.0)) profiles.max_toroidal_flux /= torflux_edge;

    T current_scale = T(0.0);
    if (spec.current_model == CurrentModel::PRESCRIBED_CURRENT &&
        spec.physical.curtor != 0.0) {
        const T edge_current = eval_curr_profile<T>(spec, T(1.0));
        if (edge_current == T(0.0)) {
            throw CumesError(
                "profiles: ncurr=1 with a zero edge current integral "
                "(ac profile integrates to 0 at s=1)");
        }
        current_scale = T(DeviceParams<T>::SIGN_JACOBIAN) *
                        DeviceParams<T>::MU_0 * T(spec.physical.curtor) /
                        (T(2.0 * M_PI) * edge_current);
    }

    profiles.iota_f.resize(ns);
    profiles.phip_f.resize(ns);
    profiles.chi_f.resize(ns);
    profiles.sqrt_s_f.resize(ns);
    for (int surface = 0; surface < ns; ++surface) {
        const T s = profiles.delta_s * T(surface);
        const T flux = minimum(torflux<T>(spec, s), T(1.0));
        const T derivative = torflux_deriv<T>(spec, s);
        const T iota = eval_iota_profile<T>(spec, flux);
        profiles.iota_f[surface] = iota;
        profiles.phip_f[surface] = profiles.max_toroidal_flux * derivative;
        profiles.chi_f[surface] =
            profiles.max_toroidal_flux * iota * derivative;
        profiles.sqrt_s_f[surface] = sqrt(s + T(1.0e-12));
    }

    const int half_surfaces = ns - 1;
    profiles.iota_h.resize(half_surfaces);
    profiles.pres_h.resize(half_surfaces);
    profiles.phip_h.resize(half_surfaces);
    profiles.dvds_h.assign(half_surfaces, T(1.0));
    profiles.sqrt_s_h.resize(half_surfaces);
    profiles.curr_h.resize(half_surfaces);
    profiles.chip_h.resize(half_surfaces);
    for (int surface = 0; surface < half_surfaces; ++surface) {
        const T s = profiles.delta_s * (T(surface) + T(0.5));
        const T flux = minimum(torflux<T>(spec, s), T(1.0));
        const T pressure_flux = minimum(
            torflux<T>(spec, minimum(s, T(spec.physical.spres_ped))), T(1.0));
        const T derivative = torflux_deriv<T>(spec, s);
        const T iota = eval_iota_profile<T>(spec, flux);
        profiles.iota_h[surface] = iota;
        profiles.pres_h[surface] = eval_mass_profile<T>(spec, pressure_flux);
        profiles.phip_h[surface] = profiles.max_toroidal_flux * derivative;
        profiles.sqrt_s_h[surface] = sqrt(s);
        profiles.curr_h[surface] =
            skip_zero_current && spec.physical.curtor == 0.0
                ? T(0.0)
                : current_scale * eval_curr_profile<T>(spec, flux);
        profiles.chip_h[surface] =
            profiles.max_toroidal_flux * iota * derivative;
    }
    T phip_square_sum = T(0.0);
    for (const T phip : profiles.phip_h) phip_square_sum += phip * phip;
    profiles.lamscale = sqrt(phip_square_sum * profiles.delta_s);
    return profiles;
}

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_PHYSICS_HOST_RADIAL_PROFILES_HPP_
