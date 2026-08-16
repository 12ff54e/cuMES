// constraint.cuh — spectral condensation constraint force.
// Reference: vmecpp constraint_force_kernel.h
#pragma once
#include "vmec_types.h"
#include "fourier.cuh"
#include "precon.cuh"

namespace cumes {
class DeviceArena;
template <typename T> class AxisymmetricOperator;
}

template <typename T>
struct ConstraintWorkspace {
    // Effective constraint force and bandpass-filtered version
    T* d_gConEff;  // [ns * nZnT]  R*dR/dθ + Z*dZ/dθ
    T* d_gCon;     // [ns * nZnT]  bandpass-filtered

    // xmpq-weighted real-space combination used by the constraint
    // (vmecpp rCon/zCon: sum_m m(m-1) * sqrt(s)^{parity} * coeff * basis)
    T* d_rCon;     // [ns * nZnT]
    T* d_zCon;     // [ns * nZnT]

    // Running averages for constraint relaxation
    T* d_rCon0;    // [ns * nZnT]
    T* d_zCon0;    // [ns * nZnT]

    // Constraint force multiplier profile (device; computeTconKernel writes
    // d_tcon directly — there is no host mirror to keep in sync)
    T* d_tcon;     // [ns] device

    // faccon[m] = -0.25 * signJ / m²  (mode-dependent scaling)
    T* h_faccon;   // [mnmax] host (pinned)
    T* d_faccon;   // [mnmax] device

    // Constraint-force outputs (vmecpp frcon/fzcon in AddConstraintForces):
    // frcon = ru_full * gCon (odd parity scaled by sqrt(s)). Consumed by the
    // forward DFT as the xmpq[m] * frcon term in frcc/fzsc.
    T* d_frcon_e;  // [ns * nZnT]
    T* d_frcon_o;  // [ns * nZnT]
    T* d_fzcon_e;  // [ns * nZnT]
    T* d_fzcon_o;  // [ns * nZnT]

    // Compact deAlias bandpass workspace: only slots 0/1 (analysis) and 4/5
    // (synthesis), modes m = 1..mpol-2, surfaces jF = 1..ns-1 participate in
    // the round trip — 2*(mpol-2)*(ns-1) batch elements instead of the full
    // 12*mpol*ns. Element order: ((slot*(mpol-2) + (m-1))*(ns-1) + (jF-1)),
    // then nzeta (real) / nz2 (spectra) contiguous.
    T* d_zeta_real_c;     // [2*(mpol-2)*(ns-1) * nzeta]
    typename FftTraits<T>::Complex* d_zeta_spectra_c; // [2*(mpol-2)*(ns-1) * nz2]
    cufftHandle plan_d2z_da, plan_z2d_da;

    // true when the device arrays above are subspans of a shared DeviceArena
    // (constraintFree then frees only the pinned host faccon + cuFFT plans).
    bool arena_backed = false;

    // Phase 6B: one shared cuFFT work area for the two constraint plans
    // (d2z_da/z2d_da), with auto-allocation disabled. Their transforms
    // are sequential on one stream, so a single max-sized buffer replaces
    // cuFFT's two auto-allocated per-plan areas. Owned here, freed in
    // constraintFree after the plans.
    void* d_cufft_work = nullptr;
    size_t cufft_work_bytes = 0;
};

template <typename T>
ConstraintWorkspace<T> constraintCreate(const GridParams<T>& p,
                                        cumes::DeviceArena* arena = nullptr);
template <typename T>
void constraintFree(ConstraintWorkspace<T>& cw);

// De-alias bandpass (vmecpp deAliasConstraintForce): gConEff -> gCon via the
// compact cuFFT round trip (θ-reduce → D2Z → scale → Z2D → poloidal
// synthesis). Extracted from constraintCompute so the bandpass is testable in
// isolation (the axisymmetric backend replaces exactly this step). Reads
// cw.d_gConEff/cw.d_tcon/cw.d_faccon, writes cw.d_gCon.
template <typename T>
void constraintDealiasBandpass(const GridParams<T>& p, const FourierPlan<T>& fp,
                               ConstraintWorkspace<T>& cw,
                               cudaStream_t stream = 0);

// Reset rCon0/zCon0 to the LCFS-extrapolated profile (vmecpp
// rzConIntoVolume): rCon0 = rCon_LCFS * s. Call with the current rCon/zCon
// on the first iteration and on the reinit pass after a restart.
template <typename T>
void constraintResetRzCon0(const GridParams<T>& p, ConstraintWorkspace<T>& cw,
                           const T* d_sqrtS_F, cudaStream_t stream = 0);

// Compute constraint force and add to brmn/bzmn forces in the FourierPlan.
// Must be called after computeForces and after preconCompute; tcon is
// refreshed from the current preconditioner elements only when
// precon_updated (vmecpp: constraintForceMultiplier at each preconditioner
// update, applied in the same iteration).
// Modifies fp.d_brmn_e, fp.d_brmn_o, fp.d_bzmn_e, fp.d_bzmn_o in-place.
template <typename T>
void constraintCompute(const GridParams<T>& p, const FourierPlan<T>& fp,
                       const PreconWorkspace<T>& pw, ConstraintWorkspace<T>& cw,
                       const T* d_sqrtS_F, bool precon_updated,
                       cudaStream_t stream = 0);

// Axisymmetric variant of constraintCompute (blueprint §8.5): identical steps
// 0 (tcon refresh) / 1 (gConEff) / 3 (add to brmn/bzmn) but the step-2 bandpass
// is the AxisymmetricOperator's direct-poloidal de-alias instead of the compact
// cuFFT round trip (constraintDealiasBandpass). The two are Class B ULP-
// equivalent on the ntor=0/nzeta=1 grid (pinned by test_axisym_backend).
template <typename T>
void constraintComputeAxisym(const GridParams<T>& p, const FourierPlan<T>& fp,
                             const PreconWorkspace<T>& pw,
                             ConstraintWorkspace<T>& cw, const T* d_sqrtS_F,
                             bool precon_updated,
                             cumes::AxisymmetricOperator<T>& op,
                             cudaStream_t stream = 0);
