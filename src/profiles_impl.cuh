// profiles_impl.cuh — template definitions for the cumes::Profiles operator.
// Included once per scalar type by profiles_double.cu / profiles_float.cu; see the
// explicit-instantiation split (cumes_cuda_double / cumes_cuda_float).
#pragma once
// profiles.cu — evaluate radial profiles on host from the validated problem
// and upload to GPU. Matches vmecpp's evalRadialProfiles (radial_profiles.cc
// lines 1149-1220) and computeMagneticFluxes (440-455):
//   maxToroidalFlux = signJ*phiedge/(2π) / torflux(1)
//   phip = maxToroidalFlux * torfluxDeriv(s)
//   torflux(x) = x*Σ aphi_i*x^i, torfluxDeriv(x) = Σ (i+1)*aphi_i*x^i
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
#include "cumes/physics/profiles.hpp"
#include "cumes/config/profile_functions.hpp"  // shared host/device evaluators
#include "cumes/config/validated_problem.hpp"
#include <cstdio>
#include <cmath>

#include "cumes/runtime/cuda_status.hpp"
#include "cumes/runtime/device_arena.cuh"

// The power-series evaluators (torflux/torfluxDeriv/evalIotaProfile/
// evalMassProfile/evalCurrProfile) live in the shared header above: the host
// validator and this upload step must divide by bit-identical normalizations.

template <typename T>
cumes::Profiles<T>::Profiles(DeviceParams<T>& p, const cumes::ValidatedProblem& vp,
                             cumes::DeviceArena* arena) {
    const cumes::ProblemSpec& sp = vp.spec();
    const int ncurr = (sp.current_model == cumes::CurrentModel::kPrescribedCurrent) ? 1 : 0;
    delta_s_ = T(1.0) / T(p.ns - 1);

    // Normalization scalars FIRST — before any device allocation. The host
    // validator (ValidatedProblem::validate) already rejects non-finite, zero,
    // and ill-scaled normalizations before CUDA initialization; the guards
    // here are the belt-and-suspenders error boundary, and they throw a typed
    // CumesError instead of exit()ing (library code never exits).
    // maxToroidalFlux = signJ * phiedge / (2π) / torflux(1)
    // (signJ = -1, so phiedge < 0 gives a positive flux, e.g. w7x).
    T maxToroidalFlux = T(DeviceParams<T>::kSignJacobian * sp.physical.phiedge) / T(2.0 * M_PI);
    T tf1 = cumes::torflux<T>(sp, T(1.0));
    if (tf1 != T(0.0)) maxToroidalFlux /= tf1;

    // ncurr=1: normalize the enclosed toroidal current profile
    // Itor = signJ * μ0*curtor / (2π * I(1)), I(s) = ∫₀ˢ ac
    T Itor = T(0.0);
    if (ncurr == 1) {
        T edgeCurrent = cumes::evalCurrProfile<T>(sp, T(1.0));
        if (edgeCurrent == T(0.0)) {
            // The normalization is a division by the edge current integral:
            // a degenerate (all-zero) ac profile would make Itor infinite and
            // poison the current constraint. Fail with a typed error (the
            // validator rejects this case earlier, before any CUDA work).
            throw cumes::CumesError(
                "profiles: ncurr=1 with a zero edge current integral "
                "(ac profile integrates to 0 at s=1)");
        }
        Itor = T(DeviceParams<T>::kSignJacobian) * DeviceParams<T>::kMu0 * T(sp.physical.curtor) /
               (T(2.0 * M_PI) * edgeCurrent);
    }

    size_t nF = p.ns * sizeof(T);
    size_t nH = (p.ns - 1) * sizeof(T);

    auto alloc = [&](T*& dst, size_t count, const char* name) {
        if (arena) dst = arena->alloc_span<T>(name, count);
        else cumes::check_cuda(cudaMalloc(&dst, count * sizeof(T)), name);
    };
    alloc(d_iota_F_,  p.ns,      "profiles/iota_F");
    alloc(d_phip_F_,  p.ns,      "profiles/phip_F");
    alloc(d_chi_F_,   p.ns,      "profiles/chi_F");
    alloc(d_sqrtS_F_, p.ns,      "profiles/sqrtS_F");
    alloc(d_iota_H_,  p.ns - 1,  "profiles/iota_H");
    alloc(d_pres_H_,  p.ns - 1,  "profiles/pres_H");
    alloc(d_phip_H_,  p.ns - 1,  "profiles/phip_H");
    alloc(d_dVds_H_,  p.ns - 1,  "profiles/dVds_H");
    alloc(d_sqrtS_H_, p.ns - 1,  "profiles/sqrtS_H");
    alloc(d_curr_H_,  p.ns - 1,  "profiles/curr_H");
    alloc(d_chip_H_,  p.ns - 1,  "profiles/chip_H");
    arena_backed_ = (arena != nullptr);

    auto* h = new T[p.ns];
    // ---- Full grid ----
    for (int j = 0; j < p.ns; ++j) {
        T s = delta_s_ * T(j);
        T tf = fmin(torflux<T>(sp, s), T(1.0));
        h[j] = evalIotaProfile<T>(sp, tf);
    }
    cumes::check_cuda(cudaMemcpy(d_iota_F_, h, nF, cudaMemcpyHostToDevice), "iota_F cpy");
    for (int j = 0; j < p.ns; ++j) {
        T s = delta_s_ * T(j);
        h[j] = maxToroidalFlux * torfluxDeriv<T>(sp, s);
    }
    cumes::check_cuda(cudaMemcpy(d_phip_F_, h, nF, cudaMemcpyHostToDevice), "phip_F cpy");
    for (int j = 0; j < p.ns; ++j) {
        T s = delta_s_ * T(j);
        T tf = fmin(torflux<T>(sp, s), T(1.0));
        h[j] = maxToroidalFlux * evalIotaProfile<T>(sp, tf) * torfluxDeriv<T>(sp, s);
    }
    cumes::check_cuda(cudaMemcpy(d_chi_F_, h, nF, cudaMemcpyHostToDevice), "chi_F cpy");
    for (int j = 0; j < p.ns; ++j) h[j] = sqrt(delta_s_ * T(j) + T(1e-12));
    cumes::check_cuda(cudaMemcpy(d_sqrtS_F_, h, nF, cudaMemcpyHostToDevice), "sqrtS_F cpy");

    // ---- Half grid ----
    for (int j = 0; j < p.ns - 1; ++j) {
        T sh = delta_s_ * (T(j) + T(0.5));
        T tf = fmin(torflux<T>(sp, sh), T(1.0));
        h[j] = evalIotaProfile<T>(sp, tf);
    }
    cumes::check_cuda(cudaMemcpy(d_iota_H_, h, nH, cudaMemcpyHostToDevice), "iota_H cpy");
    for (int j = 0; j < p.ns - 1; ++j) {
        T sh = delta_s_ * (T(j) + T(0.5));
        T tf = fmin(torflux<T>(sp, fmin(sh, T(sp.physical.spres_ped))), T(1.0));
        h[j] = evalMassProfile<T>(sp, tf);  // pres = mass (gamma = 0)
    }
    cumes::check_cuda(cudaMemcpy(d_pres_H_, h, nH, cudaMemcpyHostToDevice), "pres_H cpy");
    for (int j = 0; j < p.ns - 1; ++j) {
        T sh = delta_s_ * (T(j) + T(0.5));
        h[j] = maxToroidalFlux * torfluxDeriv<T>(sp, sh);
    }
    cumes::check_cuda(cudaMemcpy(d_phip_H_, h, nH, cudaMemcpyHostToDevice), "phip_H cpy");
    // Note: d_mass_H (dead storage, never read) was removed; d_pres_H holds
    // the mass profile (gamma=0 -> pres = mass).
    for (int j = 0; j < p.ns - 1; ++j) {
        T sh = delta_s_ * (T(j) + T(0.5));
        T tf = fmin(torflux<T>(sp, sh), T(1.0));
        h[j] = maxToroidalFlux * evalIotaProfile<T>(sp, tf) * torfluxDeriv<T>(sp, sh);
    }
    cumes::check_cuda(cudaMemcpy(d_chip_H_, h, nH, cudaMemcpyHostToDevice), "chip_H cpy");
    for (int j = 0; j < p.ns - 1; ++j) h[j] = T(1.0);  // dVds placeholder (gamma=0)
    cumes::check_cuda(cudaMemcpy(d_dVds_H_, h, nH, cudaMemcpyHostToDevice), "dVds_H cpy");
    for (int j = 0; j < p.ns - 1; ++j) {
        T sh = delta_s_ * (T(j) + T(0.5));
        h[j] = Itor * evalCurrProfile<T>(sp, fmin(torflux<T>(sp, sh), T(1.0)));
    }
    cumes::check_cuda(cudaMemcpy(d_curr_H_, h, nH, cudaMemcpyHostToDevice), "curr_H cpy");
    // sqrt(s) on half-grid for parity mixing
    for (int j = 0; j < p.ns - 1; ++j) {
        h[j] = sqrt(delta_s_ * (T(j) + T(0.5)));
    }
    cumes::check_cuda(cudaMemcpy(d_sqrtS_H_, h, nH, cudaMemcpyHostToDevice), "sqrtS_H cpy");

    delete[] h;

    // lamscale = sqrt(deltaS * Σ phipH²) (vmecpp constants_.lamscale)
    T rmsPhiP = T(0.0);
    auto* h_phip = new T[p.ns - 1];
    cumes::check_cuda(cudaMemcpy(h_phip, d_phip_H_, nH, cudaMemcpyDeviceToHost), "phip_H get");
    for (int j = 0; j < p.ns - 1; ++j) rmsPhiP += h_phip[j] * h_phip[j];
    delete[] h_phip;
    p.lamscale = sqrt(rmsPhiP * delta_s_);
    printf("  profiles: ns=%d phip=%.6e lamscale=%.6e maxToroidalFlux=%.6e\n",
           p.ns, (double)(maxToroidalFlux * torfluxDeriv<T>(sp, T(0.5))),
           (double)p.lamscale, (double)maxToroidalFlux);
}

template <typename T>
cumes::Profiles<T>::~Profiles() {
    if (!arena_backed_) {
        cudaFree(d_iota_F_);
        cudaFree(d_phip_F_); cudaFree(d_chi_F_);
        cudaFree(d_sqrtS_F_);
        cudaFree(d_iota_H_); cudaFree(d_pres_H_);
        cudaFree(d_phip_H_);
        cudaFree(d_dVds_H_); cudaFree(d_sqrtS_H_);
        cudaFree(d_curr_H_); cudaFree(d_chip_H_);
    }
}
