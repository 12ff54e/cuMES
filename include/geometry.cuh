// geometry.cuh — compute metric, Jacobian, and magnetic field from real-space
// geometry with even/odd parity decomposition.
#pragma once
#include "vmec_types.h"
#include "fourier.cuh"

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
};

template <typename T>
MetricWorkspace<T> metricCreate(const GridParams<T>& p);
template <typename T>
void metricFree(MetricWorkspace<T>& mw);

// Compute Jacobian, metric, B, and total pressure on the half-grid
// from parity-split real-space geometry.
template <typename T>
void computeGeometry(const FourierPlan<T>& fp, const GridParams<T>& p,
                     const RadialProfiles<T>& rp, MetricWorkspace<T>& mw);

// Force-norm partial sums for the residual normalization (vmecpp
// computeForceNorms): writes dVdsH[jH] = signJ * sum(gsqrt * wInt) and the
// per-surface sums (guu*r12^2, bsubu^2+bsubv^2, gsqrt*|B|^2/2, gsqrt) to
// psum (4 * (ns-1)). dVdsH: (ns-1), psum: 4*(ns-1), both device arrays.
template <typename T>
void computeForceNormPartials(const GridParams<T>& p, const MetricWorkspace<T>& mw,
                              T* dVdsH, T* psum);
