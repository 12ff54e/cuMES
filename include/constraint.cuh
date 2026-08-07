// constraint.cuh — spectral condensation constraint force.
// Reference: vmecpp constraint_force_kernel.h
#pragma once
#include "vmec_types.h"
#include "fourier.cuh"
#include "precon.cuh"

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

    // Constraint force multiplier profile
    T* h_tcon;     // [ns] host (pinned)
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

    // Compact rCon/zCon round trip: only the value slots 0/1/4/5 of the 12
    // participate — 4*mpol*ns batch elements instead of 12*mpol*ns. Element
    // order: ((slot*mpol + m)*ns + j), then nzeta (real) / nz2 (spectra)
    // contiguous; slot = 0,1,2,3 maps to the full slots 0,1,4,5.
    T* d_zeta_real_rz;     // [4*mpol*ns * nzeta]
    typename FftTraits<T>::Complex* d_zeta_spectra_rz; // [4*mpol*ns * nz2]
    cufftHandle plan_z2d_rz;
};

template <typename T>
ConstraintWorkspace<T> constraintCreate(const GridParams<T>& p);
template <typename T>
void constraintFree(ConstraintWorkspace<T>& cw);

// Compute the xmpq-weighted real-space combination rCon/zCon from the
// spectral state (vmecpp's rCon/zCon in dft_FourierToReal_2d_symm).
// Call every iteration before the constraint force is assembled.
template <typename T>
void constraintRzConCompute(const GridParams<T>& p, const FourierPlan<T>& fp,
                            const SpectralState<T>& st, ConstraintWorkspace<T>& cw,
                            const T* d_sqrtS_F);

// Reset rCon0/zCon0 to the LCFS-extrapolated profile (vmecpp
// rzConIntoVolume): rCon0 = rCon_LCFS * s. Call with the current rCon/zCon
// on the first iteration and on the reinit pass after a restart.
template <typename T>
void constraintResetRzCon0(const GridParams<T>& p, ConstraintWorkspace<T>& cw,
                           const T* d_sqrtS_F);

// Compute constraint force and add to brmn/bzmn forces in the FourierPlan.
// Must be called after computeForces and after preconCompute; tcon is
// refreshed from the current preconditioner elements only when
// precon_updated (vmecpp: constraintForceMultiplier at each preconditioner
// update, applied in the same iteration).
// Modifies fp.d_brmn_e, fp.d_brmn_o, fp.d_bzmn_e, fp.d_bzmn_o in-place.
template <typename T>
void constraintCompute(const GridParams<T>& p, const FourierPlan<T>& fp,
                       const PreconWorkspace<T>& pw, ConstraintWorkspace<T>& cw,
                       const T* d_sqrtS_F, bool precon_updated);
