// geometry.cuh — compute metric, Jacobian, and magnetic field from real-space
// geometry with even/odd parity decomposition.
#pragma once
#include "vmec_types.h"
#include "fourier.cuh"

// Intermediate arrays computed per iteration (on GPU).
// All shapes are (ns-1, nZnT) column-major (half-grid).
struct MetricWorkspace {
    // Half-grid geometry (parity-split interpolation with sqrtSH)
    double* d_r12;        // R on half-grid
    double* d_ru12;       // dR/dθ on half-grid
    double* d_zu12;       // dZ/dθ on half-grid
    double* d_rs;         // dR/ds (radial derivative)
    double* d_zs;         // dZ/ds
    double* d_tau;        // tau = tau1 + dSHalfDsInterp * tau2

    // Jacobian: gsqrt = tau * r12
    double* d_gsqrt;

    // Covariant metric (guu = g_theta_theta, etc.)
    double* d_guu;
    double* d_guv;
    double* d_gvv;

    // Contravariant B (B^θ, B^ζ)
    double* d_bsupu;
    double* d_bsupv;

    // Covariant B (B_θ, B_ζ)
    double* d_bsubu;
    double* d_bsubv;

    // Total pressure (kinetic + magnetic) on half-grid
    double* d_totalPressure;
};

MetricWorkspace metricCreate(const GridParams& p);
void metricFree(MetricWorkspace& mw);

// Compute Jacobian, metric, B, and total pressure on the half-grid
// from parity-split real-space geometry.
void computeGeometry(const FourierPlan& fp, const GridParams& p,
                     const RadialProfiles& rp, MetricWorkspace& mw);

// Force-norm partial sums for the residual normalization (vmecpp
// computeForceNorms): writes dVdsH[jH] = signJ * sum(gsqrt * wInt) and the
// per-surface sums (guu*r12^2, bsubu^2+bsubv^2, gsqrt*|B|^2/2, gsqrt) to
// psum (4 * (ns-1)). dVdsH: (ns-1), psum: 4*(ns-1), both device arrays.
void computeForceNormPartials(const GridParams& p, const MetricWorkspace& mw,
                              double* dVdsH, double* psum);
