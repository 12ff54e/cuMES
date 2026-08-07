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
template <typename T>
struct SolverResult {
    bool converged;
    int iterations;
    T fsqr, fsqz, fsql;   // final force residuals
    T delt;               // final time step
};

template <typename T>
SolverResult<T> solverRun(SpectralState<T>& state, const GridParams<T>& p,
                          const RadialProfiles<T>& rp, FourierPlan<T>& fp,
                          MetricWorkspace<T>& mw);
