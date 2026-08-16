// solver.cuh — GPU-resident fixed-point iteration with Garabedian acceleration.
// All computation happens on device; host only orchestrates.
#pragma once
#include "vmec_types.h"
#include "cumes/state/spectral_storage.hpp"

namespace cumes {
class DeviceArena;
struct SolverBench;
template <typename T> class SpectralOperator;
template <typename T> class GeometryOperator;
template <typename T> class ToroidalFftOperator;
template <typename T> class Profiles;
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
// `transform` is the generic ToroidalFft operator — always present: the solver
// reads its mode tables (xm()/xn(), the resolution-scoped DeviceModeTable) and
// binds the cuFFT plans to the compute stream; the transform scratch/plans are
// sealed behind the operator's dump-only accessors. `profiles` carries the radial
// profiles as typed `RadialProfileViews` (the solver never reads the raw
// RadialProfiles `d_*` pointers in the hot loop). `op`, when non-null, is the
// selected transform backend the solver drives (a `SpectralOperator<T>*`); the
// solver has no `axisym_active` branch — inverse/forward/de-alias all go through
// `op`. When `op` is null it defaults to `&transform` (the generic backend). The
// two backends are Class B ULP-equivalent (test_axisym_backend); selecting the
// axisymmetric backend for ntor=0/nzeta=1 is a trajectory re-freeze.
template <typename T>
SolverResult<T> solverRun(cumes::SpectralStorage<T>& state, const DeviceParams<T>& p,
                          const cumes::Profiles<T>& profiles,
                          cumes::ToroidalFftOperator<T>& transform,
                          cumes::RealSpaceStorage<T>& rs,
                          cumes::GeometryOperator<T>& geometry,
                          cumes::DeviceArena* arena = nullptr,
                          cudaStream_t stream = 0,
                          cumes::SolverBench* bench = nullptr,
                          cumes::SpectralOperator<T>* op = nullptr);
