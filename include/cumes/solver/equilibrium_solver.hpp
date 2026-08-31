// equilibrium_solver.hpp — host-facing in-process equilibrium solve facade.
//
// This is the supported embedding boundary. Consumers provide an immutable
// ValidatedProblem and receive a complete host snapshot/report; CUDA streams,
// device storage, operator workspaces, and multigrid orchestration remain
// implementation details.
#ifndef CUMES_INCLUDE_CUMES_SOLVER_EQUILIBRIUM_SOLVER_HPP_
#define CUMES_INCLUDE_CUMES_SOLVER_EQUILIBRIUM_SOLVER_HPP_

#include "cumes/config/validated_problem.hpp"
#include "cumes/io/equilibrium_profiles.hpp"
#include "cumes/io/equilibrium_snapshot.hpp"
#include "cumes/io/run_report.hpp"

#include <functional>
#include <memory>
#include <optional>

namespace cumes {

struct SolveRequest {
    // Optional hot-start state. Its shape must match the first radial stage
    // and the validated mode count. The referenced snapshot must outlive the
    // solve call.
    std::optional<std::reference_wrapper<const EquilibriumSnapshot>> restart;

    // Library calls are quiet by default. The CLI opts into the legacy
    // progress/diagnostic printout while structured logging is introduced.
    bool verbose = false;

    // Embedding calls are deterministic with respect to their arguments by
    // default. The CLI enables this to preserve its documented CUMES_*
    // environment controls.
    bool use_process_environment = false;
};

struct SolveOutcome {
    EquilibriumSnapshot equilibrium;
    EquilibriumProfiles profiles;
    RunReport report;

    bool converged = false;
    int iterations = 0;
    double fsqr = 1.0;
    double fsqz = 1.0;
    double fsql = 1.0;
    double delt = 0.9;
    int total_iterations = 0;
    double total_device_time_ms = 0.0;
    int failed_stage = -1;

    bool has_complete_equilibrium() const {
        const std::size_t family_size = equilibrium.family_size();
        if (family_size == 0) return false;
        for (const auto& family : equilibrium.families) {
            if (family.size() != family_size) return false;
        }
        return equilibrium.has_derived_fields() &&
               profiles.has_half_grid_profiles(equilibrium.ns);
    }
};

class EquilibriumSolver {
   public:
    EquilibriumSolver();
    ~EquilibriumSolver();

    EquilibriumSolver(const EquilibriumSolver&) = delete;
    EquilibriumSolver& operator=(const EquilibriumSolver&) = delete;
    EquilibriumSolver(EquilibriumSolver&&) noexcept;
    EquilibriumSolver& operator=(EquilibriumSolver&&) noexcept;

    SolveOutcome solve(const ValidatedProblem& problem,
                       const SolveRequest& request = {});

   private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_SOLVER_EQUILIBRIUM_SOLVER_HPP_
