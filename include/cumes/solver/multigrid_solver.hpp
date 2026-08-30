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
#include "cumes/io/equilibrium_snapshot.hpp"
#include "cumes/io/run_report.hpp"
#include "cumes/numerics/prolongation.hpp"
#include "cumes/physics/free_boundary_operator.hpp"
#include "cumes/solver/stage_solver.hpp"
#include "cumes/solver/start_policy.hpp"

#include <cstdio>
#include <memory>
#include <utility>

namespace cumes {

template <typename T>
struct MultigridOutcome {
    using val_type = T;

    SpectralStorage<T> state;  // final (finest-stage) state
    SolverResult<T> result;    // final stage's solver result
    int total_iterations = 0;
    RunReport report;
    EquilibriumSnapshot snapshot;  // final derived fields; state filled by CLI
    int failed_stage = -1;         // stage index that failed the run, or -1
};

template <typename T>
class MultigridSolver {
   public:
    using val_type = T;

    // `p` is the base DeviceParams; its ns/max_iter/ftol are overwritten per
    // stage from the validated problem's stage schedule (exactly the legacy
    // stage loop). `seed` is the stage-0 cold-start state (already
    // interpolated from boundary + axis).
    static MultigridOutcome<T> run(DeviceParams<T>& p,
                                   const ValidatedProblem& vp,
                                   SpectralStorage<T> seed,
                                   cudaStream_t stream = 0,
                                   bool hot_start = false) {
        MultigridOutcome<T> out;
        SpectralStorage<T> storage = std::move(seed);
        DeviceParams<T> p_prev;
        SolverResult<T> result{false, 0, T(1.0), T(1.0), T(1.0), T(0.9), {}};
        int total_iter = 0;
        const auto& stages = vp.spec().stages;
        const int n_grids = static_cast<int>(stages.size());
        const T configured_delt = p.delt;

        // Free-boundary operator: constructed ONCE per run (vmecpp's
        // persistent vacuum solvers — the accumulated response matrix and the
        // LU factors survive the multigrid transitions; nothing is
        // serialized to checkpoints). Only ns varies per stage; the operator
        // receives the per-stage edge pressure and ns at update time.
        std::unique_ptr<FreeBoundaryOperator<T>> vac;
        if (vp.spec().free_boundary.lfreeb) {
            typename FreeBoundaryOperator<T>::HostParams hp;
            hp.mgrid_file = vp.spec().free_boundary.mgrid_file;
            hp.coils_file = vp.spec().free_boundary.coils_file;
            hp.makegrid_parameters_file =
                vp.spec().free_boundary.makegrid_parameters_file;
            hp.embedded_makegrid_parameters =
                vp.spec().free_boundary.embedded_makegrid_parameters;
            hp.extcur = vp.spec().free_boundary.extcur;
            hp.nvacskip = vp.spec().free_boundary.nvacskip;
            hp.hot_start = hot_start;
            vac = std::make_unique<FreeBoundaryOperator<T>>(hp, p);
        }

        for (int g = 0; g < n_grids; ++g) {
            p_prev = p;  // previous stage's params
            p.ns = static_cast<int>(stages[g].radial_surfaces);
            p.max_iter = static_cast<int>(stages[g].max_iterations);
            p.ftol = T(stages[g].tolerance);
            p.delt = initial_step_for_stage(configured_delt, p.ntor, p.nzeta,
                                            vp.spec().free_boundary.lfreeb,
                                            p.ns, n_grids, g);
            std::printf(
                "\n=== grid stage %d/%d: ns=%d mnmax=%d max_iter=%d "
                "ftol=%.0e delt=%.6g ===\n",
                g + 1, n_grids, p.ns, p.mnmax, p.max_iter, (double)p.ftol,
                (double)p.delt);
            if (g > 0) {
                // Prolong the previous stage's converged state onto this grid
                // on the same compute stream (ordered before the next stage).
                storage = cumes::Prolongation<T>{}.enqueue(p, storage, p_prev,
                                                           stream);
                // vmecpp vmec.cc :536-539: the converged coarse-stage vacuum
                // state stays valid; re-mark INITIALIZED so the new stage's
                // first pass runs the vacuum block.
                if (vac) vac->on_stage_transition(p_prev.ns, p.ns);
            }
            // Only the finest stage can be published. Capturing coarse-grid
            // fields would both waste device-to-host transfers and leave a
            // snapshot whose radial extent conflicts with the next stage.
            EquilibriumSnapshot* output_snapshot =
                (g + 1 == n_grids) ? &out.snapshot : nullptr;
            result = StageSolver<T>::run(
                p, vp, storage, stream, std::nullopt,
                vac ? std::optional<
                          std::reference_wrapper<FreeBoundaryOperator<T>>>(
                          std::ref(*vac))
                    : std::nullopt,
                output_snapshot,
                // The recovery policy is qualified for fixed-boundary stages.
                // Free-boundary stages retain their vacuum-coupled reference
                // trajectory until separately qualified.
                !vac);
            if (vac) {
                const cumes::VacuumState before = vac->state();
                vac->on_stage_end();
                if (before == cumes::VacuumState::INITIALIZED) {
                    std::printf(
                        "  VACUUM PRESSURE TURNED ON AT %d ITERATIONS\n",
                        result.iterations);
                }
                std::printf(
                    "  VACUUM: rBtor=%.6e cTor=%.6e bSubUVac=%.6e "
                    "bSubVVac=%.6e delBSq=%.3e\n",
                    vac->rbtor(), vac->ctor(), vac->bsubu_vac(),
                    vac->bsubv_vac(), vac->delbsq_mean());
            }

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
            result.converged ? RunStatus::CONVERGED : RunStatus::NOT_CONVERGED;
        return out;
    }
};

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_SOLVER_MULTIGRID_SOLVER_HPP_
