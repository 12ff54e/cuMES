// solver.cuh — GPU-resident fixed-point iteration with Garabedian acceleration.
// All computation happens on device; host only orchestrates.
#pragma once
#include "vmec_types.h"
#include "fourier.cuh"
#include "geometry.cuh"
#include "forces.cuh"
#include "cumes/state/spectral_storage.hpp"

namespace cumes {
class DeviceArena;
struct SolverBench;
template <typename T> class AxisymmetricOperator;
template <typename T> class GeometryOperator;
template <typename T> class ToroidalFftOperator;
template <typename T> struct RealSpaceStorage;
}

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

// `arena`, when non-null, backs the preconditioner/constraint workspaces the
// solver allocates internally (one stage allocation instead of per-array
// cudaMalloc). nullptr keeps the legacy per-array allocation path.
//
// `axisym`, when non-null, selects the axisymmetric transform backend
// (blueprint §8.5) for ntor=0/nzeta=1: the direct-poloidal operator replaces
// the length-one cuFFT inverse/forward/de-alias round trips. nullptr keeps the
// generic ToroidalFft backend. The two are Class B ULP-equivalent
// (test_axisym_backend); selecting axisym is a trajectory re-freeze.
template <typename T>
SolverResult<T> solverRun(cumes::SpectralStorage<T>& state, const GridParams<T>& p,
                          const RadialProfiles<T>& rp,
                          cumes::ToroidalFftOperator<T>& transform,
                          cumes::RealSpaceStorage<T>& rs,
                          cumes::GeometryOperator<T>& geometry,
                          cumes::DeviceArena* arena = nullptr,
                          cudaStream_t stream = 0,
                          cumes::SolverBench* bench = nullptr,
                          cumes::AxisymmetricOperator<T>* axisym = nullptr);
