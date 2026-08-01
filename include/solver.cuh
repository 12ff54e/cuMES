// solver.cuh — GPU-resident fixed-point iteration with Garabedian acceleration.
// All computation happens on device; host only orchestrates.
#pragma once
#include "vmec_types.h"
#include "fourier.cuh"
#include "geometry.cuh"
#include "forces.cuh"

// Run the full fixed-point solve on GPU.
// Returns: iterations used, and whether converged.
// On exit, state contains the converged (or final) spectral coefficients.
struct SolverResult {
    bool converged;
    int iterations;
    double fsqr, fsqz, fsql;   // final force residuals
    double delt;                 // final time step
};

SolverResult solverRun(SpectralState& state, const GridParams& p,
                       const RadialProfiles& rp, FourierPlan& fp,
                       MetricWorkspace& mw);
