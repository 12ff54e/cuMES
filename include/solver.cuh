// solver.cuh — GPU-resident fixed-point iteration with Garabedian acceleration.
// All computation happens on device; host only orchestrates.
#ifndef CUMES_INCLUDE_SOLVER_CUH_
#define CUMES_INCLUDE_SOLVER_CUH_
#include "cumes/io/run_report.hpp"
#include "cumes/state/spectral_storage.hpp"
#include "vmec_types.h"

#include <functional>
#include <optional>
#include <vector>

namespace cumes {
class DeviceArena;
struct SolverBench;
template <typename T>
class SpectralOperator;
template <typename T>
class GeometryOperator;
template <typename T>
class ToroidalFftOperator;
template <typename T>
class Profiles;
template <typename T>
struct RealSpaceStorage;
template <typename T>
class FreeBoundaryOperator;
}  // namespace cumes

// Run the full fixed-point solve on GPU.
// Returns: iterations used, and whether converged.
// On exit, state contains the converged (or final) spectral coefficients.
template <typename T>
struct SolverResult {
    using val_type = T;

    bool converged;
    int iterations;
    T fsqr, fsqz, fsql;  // final force residuals
    T delt;              // final time step
    // Every pass that restored the checkpoint and re-anchored (maintenance
    // reset, Jacobian gate, nonfinite recovery, bad-jacobian / bad-progress),
    // in order, with the effective iteration at the event. Carried so the
    // multigrid driver can fill StageReport.restarts for the v1 container.
    std::vector<cumes::RestartEvent> restarts;
};

// `arena`, when engaged, backs the preconditioner/constraint workspaces the
// solver allocates internally (one stage allocation instead of per-array
// cudaMalloc). nullopt keeps the legacy per-array allocation path.
//
// `transform` is the generic ToroidalFft operator — always present: the solver
// reads its mode tables (xm()/xn(), the resolution-scoped DeviceModeTable) and
// binds the cuFFT plans to the compute stream; the transform scratch/plans are
// sealed behind the operator's dump-only accessors. `profiles` carries the
// radial profiles as typed `RadialProfileViews` (the solver never reads the raw
// profile `d_*` pointers in the hot loop). `op`, when engaged, is the
// selected transform backend the solver drives (a `SpectralOperator<T>`); the
// solver has no `axisym_active` branch — inverse/forward/de-alias all go
// through `op`. When `op` is nullopt it defaults to `&transform` (the generic
// backend). The two backends are Class B ULP-equivalent (test_axisym_backend);
// selecting the axisymmetric backend for ntor=0/nzeta=1 is a trajectory
// re-freeze. `enable_step_recovery` opts a stage into the conservative,
// one-shot recovery of a time step reduced by an early transient.
template <typename T>
SolverResult<T> solver_run(
    cumes::SpectralStorage<T>& state,
    const DeviceParams<T>& p,
    const cumes::Profiles<T>& profiles,
    cumes::ToroidalFftOperator<T>& transform,
    cumes::RealSpaceStorage<T>& rs,
    cumes::GeometryOperator<T>& geometry,
    std::optional<std::reference_wrapper<cumes::DeviceArena>> arena =
        std::nullopt,
    cudaStream_t stream = 0,
    std::optional<std::reference_wrapper<cumes::SolverBench>> bench =
        std::nullopt,
    std::optional<std::reference_wrapper<cumes::SpectralOperator<T>>> op =
        std::nullopt,
    std::optional<std::reference_wrapper<cumes::FreeBoundaryOperator<T>>>
        vacuum = std::nullopt,
    bool enable_step_recovery = false,
    bool verbose = true,
    bool use_process_environment = true);

#endif  // CUMES_INCLUDE_SOLVER_CUH_
