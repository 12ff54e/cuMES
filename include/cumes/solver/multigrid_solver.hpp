// multigrid_solver.hpp — the radial-grid stage loop (blueprint §6.11).
//
// MultigridSolver owns the spectral state across stages, prolongs a converged
// state onto each finer grid, drives StageSolver per stage, and emits a
// RunReport with full per-stage history. It validates the schedule (at least
// one stage) and preserves vmecpp's grid-sequencing semantics: a stage that
// exhausts its iteration cap without meeting ftol fails the whole run. It
// never calls exit() or writes output — the CLI maps the outcome to an exit
// code.
#pragma once

#include <cstdio>
#include <utility>

#include "cumes/io/run_report.hpp"
#include "cumes/solver/stage_solver.hpp"
#include "refine.cuh"

namespace cumes {

template <typename T>
struct MultigridOutcome {
    SpectralStorage<T> state;       // final (finest-stage) state
    SolverResult<T> result;         // final stage's solver result
    int total_iterations = 0;
    RunReport report;
    int failed_stage = -1;          // stage index that failed the run, or -1
};

template <typename T>
class MultigridSolver {
  public:
    // `p` is the base GridParams; its ns/max_iter/ftol are overwritten per
    // stage from ip (exactly the legacy stage loop). `seed` is the stage-0
    // cold-start state (already interpolated from boundary + axis).
    static MultigridOutcome<T> run(GridParams<T>& p, const InputParams& ip,
                                   SpectralStorage<T> seed) {
        MultigridOutcome<T> out;
        SpectralStorage<T> storage = std::move(seed);
        GridParams<T> p_prev;
        SolverResult<T> result{false, 0, T(1.0), T(1.0), T(1.0), T(0.9)};
        int total_iter = 0;

        for (int g = 0; g < ip.n_grids; ++g) {
            p_prev = p;                       // previous stage's params
            p.ns = ip.ns_array[g];
            p.max_iter = ip.niter_array[g];
            p.ftol = ip.ftol_array[g];
            std::printf("\n=== grid stage %d/%d: ns=%d mnmax=%d max_iter=%d "
                        "ftol=%.0e ===\n",
                        g + 1, ip.n_grids, p.ns, p.mnmax, p.max_iter,
                        (double)p.ftol);
            if (g > 0) {
                // Prolong the previous stage's converged state onto this grid.
                storage = interpolateState<T>(p, storage, p_prev);
            }
            result = StageSolver<T>::run(p, ip, storage);

            StageReport sr;
            sr.ns = p.ns;
            sr.effective_iterations = result.iterations;
            sr.converged = result.converged;
            sr.final_residual = ResidualTriple{(double)result.fsqr,
                                               (double)result.fsqz,
                                               (double)result.fsql};
            out.report.stages.push_back(sr);
            total_iter += result.iterations;

            // vmecpp semantics (vmec.cc:367-392): a stage that exhausts its
            // cap without meeting ftol fails the whole multigrid run.
            if (!result.converged && ip.n_grids > 1) {
                out.failed_stage = g;
                break;
            }
        }

        out.state = std::move(storage);
        out.result = result;
        out.total_iterations = total_iter;
        out.report.total_effective_iterations = total_iter;
        out.report.status = result.converged ? RunStatus::kConverged
                                             : RunStatus::kNotConverged;
        return out;
    }
};

}  // namespace cumes
