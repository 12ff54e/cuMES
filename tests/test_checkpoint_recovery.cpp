#include "cumes/config/json_reader.hpp"
#include "cumes/solver/equilibrium_solver.hpp"
#include "cumes_test.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <utility>

namespace {

using cumes::test::check;
using Snapshot = cumes::EquilibriumSnapshot;

void check_solution(const cumes::ValidatedProblem& problem,
                    const cumes::SolveOutcome& solved) {
    check(solved.converged && solved.has_complete_equilibrium(),
          "checkpoint recovery: complete converged equilibrium");
    const auto& stages = problem.spec().stages;
    check(solved.report.stages.size() == stages.size(),
          "checkpoint recovery: full radial schedule completes");
    if (solved.report.stages.size() != stages.size()) return;
    for (std::size_t g = 0; g < solved.report.stages.size(); ++g) {
        const auto& stage = solved.report.stages[g];
        const auto& residual = stage.final_residual;
        check(stage.converged && residual.fsqr <= stages[g].tolerance &&
                  residual.fsqz <= stages[g].tolerance &&
                  residual.fsql <= stages[g].tolerance,
              "checkpoint recovery: all original residual tolerances hold");
    }
    if (!solved.has_complete_equilibrium()) return;
    const auto& state = solved.equilibrium;
    const auto finite = [](const auto& fields) {
        for (const auto& field : fields)
            if (!std::all_of(field.begin(), field.end(),
                             [](double x) { return std::isfinite(x); }))
                return false;
        return true;
    };
    check(finite(state.families) && finite(state.half_fields) &&
              finite(state.full_fields),
          "checkpoint recovery: state and published fields are finite");
    bool boundary_exact = true;
    for (int mode = 0; mode < state.mnmax; ++mode) {
        const std::size_t edge =
            static_cast<std::size_t>((mode + 1) * state.ns - 1);
        boundary_exact &= state.families[Snapshot::RMNCC][edge] ==
                              problem.boundary().rbcc[mode] &&
                          state.families[Snapshot::RMNSS][edge] ==
                              problem.boundary().rbss[mode] &&
                          state.families[Snapshot::ZMNSC][edge] ==
                              problem.boundary().zbsc[mode] &&
                          state.families[Snapshot::ZMNCS][edge] ==
                              problem.boundary().zbcs[mode];
    }
    check(boundary_exact, "checkpoint recovery: prescribed boundary is exact");
    bool physical = true;
    const auto& fields = state.half_fields;
    for (std::size_t i = 0; i < state.half_field_size(); ++i) {
        const double b_squared =
            fields[Snapshot::BSUPS][i] * fields[Snapshot::BSUBS][i] +
            fields[Snapshot::BSUPU][i] * fields[Snapshot::BSUBU][i] +
            fields[Snapshot::BSUPV][i] * fields[Snapshot::BSUBV][i];
        physical &= fields[Snapshot::SQRTG][i] < 0.0 && b_squared > 0.0 &&
                    std::isfinite(b_squared);
    }
    check(physical,
          "checkpoint recovery: oriented geometry and magnetic energy valid");
}

}  // namespace

int main() {
    try {
        const auto problem = cumes::read_and_validate(
            "tests/fixtures/qh_checkpoint_recovery.json",
            cumes::SolverOptions{});
        check(problem.has_value(), "checkpoint recovery: input validates");
        if (!problem.has_value()) return cumes::test::summary();
        cumes::EquilibriumSolver solver;
        for (const auto transfer : {cumes::RadialTransferPolicy::AUTOMATIC,
                                    cumes::RadialTransferPolicy::CATMULL_ROM}) {
            // This cold solve used to restore an invalid post-descent
            // checkpoint at iteration 26 until the first stage exhausted its
            // 10,000-pass budget. Both transfer policies must retain the full
            // schedule: a single-grid cold start uses a different initial seed.
            cumes::SolveRequest request;
            request.radial_transfer = transfer;
            const auto solved = solver.solve(problem.value(), request);
            check_solution(problem.value(), solved);
            if (!solved.converged || !solved.has_complete_equilibrium())
                continue;

            auto spec = problem.value().spec();
            spec.stages = {spec.stages.back()};
            const auto final_problem =
                cumes::validate(std::move(spec), problem.value().options());
            check(final_problem.has_value(),
                  "checkpoint recovery: replay problem validates");
            if (!final_problem.has_value()) continue;
            request.restart = std::cref(solved.equilibrium);
            const auto replay = solver.solve(final_problem.value(), request);
            check_solution(final_problem.value(), replay);
            check(replay.total_iterations == 1 &&
                      replay.report.stages.size() == 1 &&
                      replay.report.stages.front().restarts.empty(),
                  "checkpoint recovery: final state is a one-pass fixed point");
            check(solved.equilibrium.families == replay.equilibrium.families,
                  "checkpoint recovery: fixed-point coefficients preserved");
        }
    } catch (const std::exception& error) { check(false, error.what()); }
    return cumes::test::summary();
}
