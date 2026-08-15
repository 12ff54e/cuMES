// stage_solver.hpp — one radial stage's resource lifecycle + solve
// (blueprint §6.11). The stage owns its ns-dependent profile/Fourier/metric
// workspaces and invokes the solver; the spectral state is owned by the
// multigrid driver (it persists across stages via prolongation).
#pragma once

#include "cumes/state/spectral_storage.hpp"
#include "fourier.cuh"
#include "geometry.cuh"
#include "input.h"
#include "profiles.cuh"
#include "solver.cuh"

namespace cumes {

// Runs a single radial stage on `p` (already carrying this stage's
// ns/max_iter/ftol). Creates the stage's workspaces, runs the fixed-point
// solver on `state`, and frees the workspaces before returning. `state` stays
// owned by the caller; profilesCreate sets p.lamscale in place for the stage.
template <typename T>
class StageSolver {
  public:
    static SolverResult<T> run(GridParams<T>& p, const InputParams& ip,
                               SpectralStorage<T>& state) {
        RadialProfiles<T> rp = profilesCreate<T>(p, ip);
        FourierPlan<T> fp = fourierCreate<T>(p);
        MetricWorkspace<T> mw = metricCreate<T>(p);
        SolverResult<T> result = solverRun<T>(state, p, rp, fp, mw);
        fourierFree(fp);
        metricFree(mw);
        profilesFree(rp);
        return result;
    }
};

}  // namespace cumes
