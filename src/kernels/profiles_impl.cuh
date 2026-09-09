// kernels/profiles_impl.cuh — template definitions for the cumes::Profiles
// operator. Included once per scalar type by profiles_double.cu /
// profiles_float.cu; see the explicit-instantiation split (cumes_cuda_double /
// cumes_cuda_float).
#ifndef CUMES_SRC_PROFILES_IMPL_CUH_
#define CUMES_SRC_PROFILES_IMPL_CUH_
// profiles.cu — evaluate radial profiles on host from the validated problem
// and upload to GPU. Matches vmecpp's evalRadialProfiles (radial_profiles.cc
// lines 1149-1220) and computeMagneticFluxes (440-455):
//   maxToroidalFlux = signJ*phiedge/(2π) / torflux(1)
//   phip = maxToroidalFlux * torflux_deriv(s)
//   torflux(x) = x*Σ aphi_i*x^i, torflux_deriv(x) = Σ (i+1)*aphi_i*x^i
//   mass = μ0*pres_scale * Σ am_i*tf^i   (tf = toroidal flux coordinate)
//   pres = mass / dVds^gamma  (gamma = adiabatic_index; dVds placeholder
//          for gamma != 0 — the geometry-dependent dVds is not implemented)
//   iota = Σ ai_i*tf^i  (ncurr=0; for ncurr=1 iotaH/chipH are recomputed
//          every iteration from the prescribed current, see geometry.cu)
//   curr = Itor * Σ ac_i*tf^(i+1)/(i+1), Itor = signJ*μ0*curtor/(2π*I(1))
// lamscale = sqrt(deltaS * Σ_j phipH[j]^2), the vmecpp constants_.lamscale.
//
// All computation is templated on the scalar type T (double or float); the
// ValidatedProblem profile coefficients stay double (host config) and are
// converted at the point of use.
#include "cumes/config/validated_problem.hpp"
#include "cumes/physics/host_radial_profiles.hpp"
#include "cumes/physics/profiles.hpp"
#include "cumes/runtime/cuda_status.hpp"
#include "cumes/runtime/device_arena.cuh"

#include <cmath>
#include <cstdio>
#include <functional>
#include <optional>

// The power-series evaluators (torflux/torflux_deriv/eval_iota_profile/
// eval_mass_profile/eval_curr_profile) live in the shared header above: the
// host validator and this upload step must divide by bit-identical
// normalizations.

template <typename T>
cumes::Profiles<T>::Profiles(
    DeviceParams<T>& p,
    const cumes::ValidatedProblem& vp,
    const std::optional<std::reference_wrapper<DeviceArena>>& arena,
    bool verbose) {
    const cumes::ProblemSpec& sp = vp.spec();
    // Validate normalizations and evaluate all prescribed data before device
    // allocation; the shared evaluator retains the native typed error boundary.
    const auto profiles = cumes::evaluate_radial_profiles<T>(
        sp, p.ns, [](T a, T b) { return fmin(a, b); },
        /*skip_zero_current=*/true);
    delta_s_ = profiles.delta_s;

    auto alloc = [&](T*& dst, size_t count, const char* name) {
        if (arena)
            dst = arena->get().alloc_span<T>(name, count);
        else
            cumes::check_cuda(cudaMalloc(&dst, count * sizeof(T)), name);
    };
    alloc(d_iota_F_, p.ns, "profiles/iota_F");
    alloc(d_phip_F_, p.ns, "profiles/phip_F");
    alloc(d_chi_F_, p.ns, "profiles/chi_F");
    alloc(d_sqrtS_F_, p.ns, "profiles/sqrtS_F");
    alloc(d_iota_H_, p.ns - 1, "profiles/iota_H");
    alloc(d_pres_H_, p.ns - 1, "profiles/pres_H");
    alloc(d_phip_H_, p.ns - 1, "profiles/phip_H");
    alloc(d_dVds_H_, p.ns - 1, "profiles/dVds_H");
    alloc(d_sqrtS_H_, p.ns - 1, "profiles/sqrtS_H");
    alloc(d_curr_H_, p.ns - 1, "profiles/curr_H");
    alloc(d_chip_H_, p.ns - 1, "profiles/chip_H");
    arena_backed_ = arena.has_value();

    const auto upload = [](T* d_destination, const std::vector<T>& values,
                           const char* name) {
        cumes::check_cuda(
            cudaMemcpy(d_destination, values.data(), values.size() * sizeof(T),
                       cudaMemcpyHostToDevice),
            name);
    };
    upload(d_iota_F_, profiles.iota_f, "iota_F cpy");
    upload(d_phip_F_, profiles.phip_f, "phip_F cpy");
    upload(d_chi_F_, profiles.chi_f, "chi_F cpy");
    upload(d_sqrtS_F_, profiles.sqrt_s_f, "sqrtS_F cpy");
    upload(d_iota_H_, profiles.iota_h, "iota_H cpy");
    upload(d_pres_H_, profiles.pres_h, "pres_H cpy");
    upload(d_phip_H_, profiles.phip_h, "phip_H cpy");
    upload(d_chip_H_, profiles.chip_h, "chip_H cpy");
    upload(d_dVds_H_, profiles.dvds_h, "dVds_H cpy");
    upload(d_curr_H_, profiles.curr_h, "curr_H cpy");
    upload(d_sqrtS_H_, profiles.sqrt_s_h, "sqrtS_H cpy");
    p.lamscale = profiles.lamscale;
    if (verbose) {
        printf(
            "  profiles: ns=%d phip=%.6e lamscale=%.6e "
            "maxToroidalFlux=%.6e\n",
            p.ns,
            (double)(profiles.max_toroidal_flux * torflux_deriv<T>(sp, T(0.5))),
            (double)p.lamscale, (double)profiles.max_toroidal_flux);
    }
}

template <typename T>
cumes::Profiles<T>::~Profiles() {
    if (!arena_backed_) {
        cudaFree(d_iota_F_);
        cudaFree(d_phip_F_);
        cudaFree(d_chi_F_);
        cudaFree(d_sqrtS_F_);
        cudaFree(d_iota_H_);
        cudaFree(d_pres_H_);
        cudaFree(d_phip_H_);
        cudaFree(d_dVds_H_);
        cudaFree(d_sqrtS_H_);
        cudaFree(d_curr_H_);
        cudaFree(d_chip_H_);
    }
}

#endif  // CUMES_SRC_PROFILES_IMPL_CUH_
