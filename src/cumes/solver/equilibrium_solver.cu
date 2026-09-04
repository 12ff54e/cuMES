// equilibrium_solver.cu — public host facade over the existing CUDA
// multigrid orchestration.
#include "cumes/config/device_params.hpp"
#include "cumes/io/input_params.hpp"
#include "cumes/io/snapshot_bridge.cuh"
#include "cumes/runtime/cuda_status.hpp"
#include "cumes/runtime/stream.hpp"
#include "cumes/solver/dump_windows.hpp"
#include "cumes/solver/equilibrium_solver.hpp"
#include "cumes/solver/multigrid_solver.hpp"
#include "cumes/state/seed_state.hpp"
#include "vmec_types.h"

#include <chrono>
#include <optional>
#include <string>
#include <utility>

namespace cumes {

class EquilibriumSolver::Impl {};

EquilibriumSolver::EquilibriumSolver() : impl_(std::make_unique<Impl>()) {}
EquilibriumSolver::~EquilibriumSolver() = default;
EquilibriumSolver::EquilibriumSolver(EquilibriumSolver&&) noexcept = default;
EquilibriumSolver& EquilibriumSolver::operator=(EquilibriumSolver&&) noexcept =
    default;

SolveOutcome EquilibriumSolver::solve(const ValidatedProblem& problem,
                                      const SolveRequest& request) {
    static_cast<void>(impl_);

    const auto solve_start = std::chrono::steady_clock::now();

    DeviceParams<Real> params = init_params<Real>(problem);
    const StageRequest& first_stage = problem.spec().stages.front();
    params.ns = static_cast<int>(first_stage.radial_surfaces);
    params.max_iter = static_cast<int>(first_stage.max_iterations);
    params.ftol = Real(first_stage.tolerance);

    SpectralStorage<Real> seed;
    if (request.restart.has_value()) {
        const EquilibriumSnapshot& restart = request.restart->get();
        if (restart.ns != params.ns || restart.mnmax != params.mnmax) {
            throw CumesError(
                "restart snapshot (ns=" + std::to_string(restart.ns) +
                ", mnmax=" + std::to_string(restart.mnmax) +
                ") does not match stage-0 grid (ns=" +
                std::to_string(params.ns) +
                ", mnmax=" + std::to_string(params.mnmax) + ")");
        }
        for (const auto& family : restart.families) {
            if (family.size() != restart.family_size()) {
                throw CumesError("restart snapshot has an invalid family size");
            }
        }
        seed = restart_state<Real>(params, problem, restart, request.verbose);
    } else {
        seed = init_state<Real>(params, problem, request.verbose,
                                request.use_process_environment);
    }

    ScopedDumpEnvironment dump_environment(request.use_process_environment);
    Stream compute_stream;
    std::optional<RadialInterpolation> radial_interpolation;
    switch (request.radial_transfer) {
        case RadialTransferPolicy::AUTOMATIC:
            break;
        case RadialTransferPolicy::LINEAR:
            radial_interpolation = RadialInterpolation::LINEAR;
            break;
        case RadialTransferPolicy::CATMULL_ROM:
            radial_interpolation = RadialInterpolation::CATMULL_ROM;
            break;
        case RadialTransferPolicy::BSPLINE:
            radial_interpolation = RadialInterpolation::BSPLINE;
            break;
    }
    const auto setup_end = std::chrono::steady_clock::now();
    MultigridOutcome<Real> internal = MultigridSolver<Real>::run(
        params, problem, std::move(seed), compute_stream.get(),
        request.restart.has_value(), request.verbose,
        request.use_process_environment, radial_interpolation);
    const auto multigrid_end = std::chrono::steady_clock::now();

    SolveOutcome outcome;
    outcome.equilibrium = std::move(internal.snapshot);
    outcome.profiles = std::move(internal.profiles);
    populate_snapshot_state_from_device(internal.state, outcome.equilibrium);
    const auto transfer_end = std::chrono::steady_clock::now();
    outcome.report = std::move(internal.report);
    outcome.report.input_params = make_input_params(problem);
    outcome.converged = internal.result.converged;
    outcome.iterations = internal.result.iterations;
    outcome.fsqr = static_cast<double>(internal.result.fsqr);
    outcome.fsqz = static_cast<double>(internal.result.fsqz);
    outcome.fsql = static_cast<double>(internal.result.fsql);
    outcome.delt = static_cast<double>(internal.result.delt);
    outcome.total_iterations = internal.total_iterations;
    outcome.total_device_time_ms = internal.total_device_time_ms;
    outcome.timings.setup_wall_ms =
        std::chrono::duration<double, std::milli>(setup_end - solve_start)
            .count();
    outcome.timings.multigrid_wall_ms =
        std::chrono::duration<double, std::milli>(multigrid_end - setup_end)
            .count();
    outcome.timings.final_state_transfer_wall_ms =
        std::chrono::duration<double, std::milli>(transfer_end - multigrid_end)
            .count();
    outcome.timings.total_wall_ms =
        std::chrono::duration<double, std::milli>(transfer_end - solve_start)
            .count();
    outcome.failed_stage = internal.failed_stage;
    return outcome;
}

}  // namespace cumes
