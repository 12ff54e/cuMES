// geometry.cuh — compute metric, Jacobian, and magnetic field from real-space
// geometry with even/odd parity decomposition.
#pragma once
#include "vmec_types.h"
#include "fourier.cuh"

namespace cumes { class DeviceArena; }

// Intermediate arrays computed per iteration (on GPU).
// All shapes are (ns-1, nZnT) column-major (half-grid).
template <typename T>
struct MetricWorkspace {
    // Half-grid geometry (parity-split interpolation with sqrtSH)
    T* d_r12;        // R on half-grid
    T* d_ru12;       // dR/dθ on half-grid
    T* d_zu12;       // dZ/dθ on half-grid
    T* d_rs;         // dR/ds (radial derivative)
    T* d_zs;         // dZ/ds
    T* d_tau;        // tau = tau1 + dSHalfDsInterp * tau2

    // Jacobian: gsqrt = tau * r12
    T* d_gsqrt;

    // Covariant metric (guu = g_theta_theta, etc.)
    T* d_guu;
    T* d_guv;
    T* d_gvv;

    // Contravariant B (B^θ, B^ζ)
    T* d_bsupu;
    T* d_bsupv;

    // Covariant B (B_θ, B_ζ)
    T* d_bsubu;
    T* d_bsubv;

    // Total pressure (kinetic + magnetic) on half-grid
    T* d_totalPressure;

    // true when the device arrays above are subspans of a shared DeviceArena
    // (metricFree then only resets the struct; the arena owns the memory).
    bool arena_backed = false;
};

// metricCreate allocates the 15 half-grid arrays. With `arena == nullptr` it
// uses one cudaMalloc per array (the legacy path, unchanged); with an arena it
// carves aligned named subspans from that single stage allocation.
template <typename T>
MetricWorkspace<T> metricCreate(const DeviceParams<T>& p,
                                cumes::DeviceArena* arena = nullptr);
template <typename T>
void metricFree(MetricWorkspace<T>& mw);

// Compute Jacobian, metric, B, and total pressure on the half-grid
// from parity-split real-space geometry.
// `update_iota_chi` gates the full-grid iota/chip update (updateIotaChipF):
// it must run every pass for ncurr=1 (the half-grid profiles evolve via the
// current closure), but for ncurr=0 the half-grid iotaH/chipH are fixed, so the
// full-grid values are computed once on the first pass and the per-iteration
// relaunch is skipped (Phase 6A fixed-iota update skip).
template <typename T>
void computeGeometry(const cumes::RealSpaceStorage<T>& rs, const DeviceParams<T>& p,
                     const RadialProfiles<T>& rp, MetricWorkspace<T>& mw,
                     cudaStream_t stream = 0, bool update_iota_chi = true);

// Split geometry pipeline (blueprint §6.7): base geometry (staggered
// interpolation, Jacobian, covariant metric — no 1/√g division) is computed
// before the magnetic field (the 1/√g contravariant B + covariant B + total
// pressure + ncurr closure). The full-grid iota/chip update (`update_iota_chi`)
// reads the half-grid iotaH/chipH that ncurr=1 finalization refreshes, so it
// lives with the field stage, not the base geometry. `computeGeometry` below is
// the full pipeline (base + field) kept for the tests' convenience; the solver
// drives the two stages separately through GeometryOperator (base) and
// MagneticFieldOperator (field) so the field is ordered after the
// Jacobian-status chain.
template <typename T>
void computeBaseGeometry(const cumes::RealSpaceStorage<T>& rs, const DeviceParams<T>& p,
                         const RadialProfiles<T>& rp, MetricWorkspace<T>& mw,
                         cudaStream_t stream = 0);
template <typename T>
void computeMagneticField(const cumes::RealSpaceStorage<T>& rs, const DeviceParams<T>& p,
                          const RadialProfiles<T>& rp, MetricWorkspace<T>& mw,
                          cudaStream_t stream = 0, bool update_iota_chi = true);

// Force-norm partial sums for the residual normalization (vmecpp
// computeForceNorms): writes dVdsH[jH] = signJ * sum(gsqrt * wInt) and the
// per-surface sums (guu*r12^2, bsubu^2+bsubv^2, gsqrt*|B|^2/2, gsqrt) to
// psum (4 * (ns-1)). dVdsH: (ns-1), psum: 4*(ns-1), both device arrays.
template <typename T>
void computeForceNormPartials(const DeviceParams<T>& p, const MetricWorkspace<T>& mw,
                              T* dVdsH, T* psum, cudaStream_t stream = 0);

// Jacobian validity stats (vmecpp's bad-jacobian detection): writes to
// d_stats[4] = {min signJ·√g over the half grid, max |√g|, count of non-finite
// entries, linear index (jH*nZnT + point) of the minimum}. d_stats is a
// caller-owned 4-element device scratch (allocate once; never cudaMalloc per
// call). Device-only — the caller folds the 4 values into its combined
// ControlRecord and transfers them with one async copy (Phase 6A one-fence
// control path); there is no host copy or fence here.
// NOTE: |√g| -> 0 at the innermost half-grid (axis singularity) is expected;
// the solver's threshold is relative to the run's own |√g| scale.
template <typename T>
void computeJacobianStats(const DeviceParams<T>& p, const MetricWorkspace<T>& mw,
                          T* d_stats, cudaStream_t stream = 0);
