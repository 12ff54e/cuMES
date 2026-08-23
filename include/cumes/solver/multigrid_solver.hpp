// multigrid_solver.hpp — the radial-grid stage loop (blueprint §6.11).
//
// MultigridSolver owns the spectral state across stages, prolongs a converged
// state onto each finer grid, drives StageSolver per stage, and emits a
// RunReport with full per-stage history. It validates the schedule (at least
// one stage) and preserves vmecpp's grid-sequencing semantics: a stage that
// exhausts its iteration cap without meeting ftol fails the whole run. It
// never calls exit() or writes output — the CLI maps the outcome to an exit
// code.
#ifndef CUMES_INCLUDE_CUMES_SOLVER_MULTIGRID_SOLVER_HPP_
#define CUMES_INCLUDE_CUMES_SOLVER_MULTIGRID_SOLVER_HPP_

#include "cumes/config/validated_problem.hpp"
#include "cumes/io/run_report.hpp"
#include "cumes/numerics/prolongation.hpp"
#include "cumes/solver/stage_solver.hpp"

#include <cstdio>
#include <utility>

namespace cumes {

template <typename T>
struct MultigridOutcome {
    SpectralStorage<T> state;  // final (finest-stage) state
    SolverResult<T> result;    // final stage's solver result
    int total_iterations = 0;
    RunReport report;
    int failed_stage = -1;  // stage index that failed the run, or -1
};

template <typename T>
class MultigridSolver {
   public:
    // `p` is the base DeviceParams; its ns/max_iter/ftol are overwritten per
    // stage from the validated problem's stage schedule (exactly the legacy
    // stage loop). `seed` is the stage-0 cold-start state (already
    // interpolated from boundary + axis).
    static MultigridOutcome<T> run(DeviceParams<T>& p,
                                   const ValidatedProblem& vp,
                                   SpectralStorage<T> seed,
                                   cudaStream_t stream = 0) {
        MultigridOutcome<T> out;
        SpectralStorage<T> storage = std::move(seed);
        DeviceParams<T> p_prev;
        SolverResult<T> result{false, 0, T(1.0), T(1.0), T(1.0), T(0.9), {}};
        int total_iter = 0;
        const auto& stages = vp.spec().stages;
        const int n_grids = static_cast<int>(stages.size());

        for (int g = 0; g < n_grids; ++g) {
            p_prev = p;  // previous stage's params
            p.ns = static_cast<int>(stages[g].radial_surfaces);
            p.max_iter = static_cast<int>(stages[g].max_iterations);
            p.ftol = T(stages[g].tolerance);
            std::printf(
                "\n=== grid stage %d/%d: ns=%d mnmax=%d max_iter=%d "
                "ftol=%.0e ===\n",
                g + 1, n_grids, p.ns, p.mnmax, p.max_iter, (double)p.ftol);
            if (g > 0) {
                // Prolong the previous stage's converged state onto this grid
                // on the same compute stream (ordered before the next stage).
                storage = cumes::Prolongation<T>{}.enqueue(p, storage, p_prev,
                                                           stream);
            }
            result = StageSolver<T>::run(p, vp, storage, stream);

            StageReport sr;
            sr.ns = p.ns;
            sr.effective_iterations = result.iterations;
            sr.converged = result.converged;
            sr.final_residual = ResidualTriple{
                (double)result.fsqr, (double)result.fsqz, (double)result.fsql};
            // Restart history the controller recorded during this stage
            // (serialized into the v1 container's per-stage section).
            sr.restarts = result.restarts;
            out.report.stages.push_back(sr);
            total_iter += result.iterations;

            // vmecpp semantics (vmec.cc:367-392): a stage that exhausts its
            // cap without meeting ftol fails the whole multigrid run.
            if (!result.converged && n_grids > 1) {
                out.failed_stage = g;
                break;
            }
        }

        out.state = std::move(storage);
        out.result = result;
        out.total_iterations = total_iter;
        out.report.total_effective_iterations = total_iter;
        out.report.status =
            result.converged ? RunStatus::kConverged : RunStatus::kNotConverged;
        return out;
    }
};

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_SOLVER_MULTIGRID_SOLVER_HPP_
