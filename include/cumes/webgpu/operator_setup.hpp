#ifndef CUMES_INCLUDE_CUMES_WEBGPU_OPERATOR_SETUP_HPP_
#define CUMES_INCLUDE_CUMES_WEBGPU_OPERATOR_SETUP_HPP_

#include "cumes/webgpu/initialization.hpp"

namespace cumes::webgpu {

// Matches the force/constraint slots in IterationResult.
enum class ResidualPhase { FORCE = 0, CONSTRAINT = 1 };

template <typename Case>
void assign_stage_shape(Case& value, const AxisymmetricStageData& stage) {
    value.ns = stage.ns;
    if constexpr (requires { value.mpol; }) value.mpol = stage.mpol;
    if constexpr (requires { value.ntor; }) value.ntor = stage.ntor;
    if constexpr (requires { value.ntheta; }) value.ntheta = stage.ntheta;
    if constexpr (requires { value.nzeta; }) value.nzeta = stage.nzeta;
    if constexpr (requires { value.nfp; }) value.nfp = stage.nfp;
}

// Copy only the radial arrays consumed by the operator. Scalars and controls
// stay at the call site; current-closure profiles are copied when it runs.
template <typename Case>
void assign_radial_profiles(Case& value, const RadialProfiles& profiles) {
    if constexpr (requires { value.sqrt_s_f; })
        value.sqrt_s_f = profiles.sqrt_s_f;
    if constexpr (requires { value.sqrt_s_f_lo; })
        value.sqrt_s_f_lo = profiles.sqrt_s_f_lo;
    if constexpr (requires { value.sqrt_s_h; })
        value.sqrt_s_h = profiles.sqrt_s_h;
    if constexpr (requires { value.sqrt_s_h_lo; })
        value.sqrt_s_h_lo = profiles.sqrt_s_h_lo;
    if constexpr (requires { value.phip_f; }) value.phip_f = profiles.phip_f;
    if constexpr (requires { value.phip_f_lo; })
        value.phip_f_lo = profiles.phip_f_lo;
    if constexpr (requires { value.chip_h; }) value.chip_h = profiles.chip_h;
    if constexpr (requires { value.chip_h_lo; })
        value.chip_h_lo = profiles.chip_h_lo;
    if constexpr (requires { value.pres_h; }) value.pres_h = profiles.pres_h;
    if constexpr (requires { value.pres_h_lo; })
        value.pres_h_lo = profiles.pres_h_lo;
    if constexpr (requires { value.curr_h; }) value.curr_h = profiles.curr_h;
    if constexpr (requires { value.curr_h_lo; })
        value.curr_h_lo = profiles.curr_h_lo;
    if constexpr (requires { value.phip_h; }) value.phip_h = profiles.phip_h;
    if constexpr (requires { value.phip_h_lo; })
        value.phip_h_lo = profiles.phip_h_lo;
    if constexpr (requires { value.iota_h; }) value.iota_h = profiles.iota_h;
    if constexpr (requires { value.iota_h_lo; })
        value.iota_h_lo = profiles.iota_h_lo;
}

}  // namespace cumes::webgpu

#endif  // CUMES_INCLUDE_CUMES_WEBGPU_OPERATOR_SETUP_HPP_
