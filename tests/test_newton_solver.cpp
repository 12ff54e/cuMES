#include "cumes/config/json_reader.hpp"
#include "cumes/solver/equilibrium_solver.hpp"
#include "cumes_test.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>

namespace {

using cumes::test::check;
using Snapshot = cumes::EquilibriumSnapshot;

struct Case {
    std::string_view name;
    std::array<int, 3> baseline_iterations;
    std::array<int, 3> newton_iterations;
};

bool finite_state(const Snapshot& state) {
    const auto finite = [](const auto& fields) {
        for (const auto& field : fields)
            for (const double value : field)
                if (!std::isfinite(value)) return false;
        return true;
    };
    return finite(state.families) && finite(state.half_fields) &&
           finite(state.full_fields);
}

void check_outcome(const cumes::ValidatedProblem& problem,
                   const cumes::SolveOutcome& outcome,
                   const std::array<int, 3>& expected_iterations) {
    check(outcome.converged && outcome.has_complete_equilibrium(),
          "Newton integration: complete converged equilibrium");
    check(finite_state(outcome.equilibrium),
          "Newton integration: all state and result fields are finite");
    const auto& stages = problem.spec().stages;
    check(outcome.report.stages.size() == stages.size(),
          "Newton integration: every configured stage is reported");
    if (outcome.report.stages.size() != stages.size()) return;
    for (std::size_t g = 0; g < stages.size(); ++g) {
        const auto& actual = outcome.report.stages[g];
        const auto& residual = actual.final_residual;
        check(
            static_cast<std::size_t>(actual.ns) == stages[g].radial_surfaces &&
                actual.converged && actual.effective_iterations > 0 &&
                static_cast<std::size_t>(actual.effective_iterations) <=
                    stages[g].max_iterations &&
                residual.fsqr <= stages[g].tolerance &&
                residual.fsqz <= stages[g].tolerance &&
                residual.fsql <= stages[g].tolerance,
            "Newton integration: original stage cap and tolerance hold");
#if defined(CUMES_HAVE_BSPLINE_PROLONGATION) && \
    defined(CUMES_PRECISION_POLICY_NAME)
        if (std::string_view(CUMES_PRECISION_POLICY_NAME) == "verify-double")
            check(actual.effective_iterations == expected_iterations[g],
                  "Newton integration: qualified multigrid trajectory");
#else
        static_cast<void>(expected_iterations);
#endif
    }
    const auto& state = outcome.equilibrium;
    bool boundary_exact = true;
    for (int m = 0; m < state.mnmax; ++m) {
        const std::size_t edge =
            static_cast<std::size_t>((m + 1) * state.ns - 1);
        boundary_exact &=
            state.families[Snapshot::RMNCC][edge] ==
                problem.boundary().rbcc[m] &&
            state.families[Snapshot::ZMNSC][edge] == problem.boundary().zbsc[m];
    }
    check(boundary_exact, "Newton integration: fixed boundary is exact");
    bool physical_fields = true;
    for (std::size_t i = 0; i < state.half_field_size(); ++i) {
        const auto& fields = state.half_fields;
        const double b_squared =
            fields[Snapshot::BSUPS][i] * fields[Snapshot::BSUBS][i] +
            fields[Snapshot::BSUPU][i] * fields[Snapshot::BSUBU][i] +
            fields[Snapshot::BSUPV][i] * fields[Snapshot::BSUBV][i];
        physical_fields &= fields[Snapshot::SQRTG][i] < 0.0 &&
                           std::isfinite(b_squared) && b_squared > 0.0;
    }
    check(physical_fields,
          "Newton integration: oriented Jacobian and magnetic energy valid");
}

void check_replay(cumes::EquilibriumSolver& solver,
                  const cumes::ValidatedProblem& problem,
                  const cumes::SolveOutcome& original) {
    // Shape matching for a converged checkpoint is the only reason to use
    // the final grid alone here; all cold solves retain the full schedule.
    auto spec = problem.spec();
    spec.stages = {spec.stages.back()};
    const auto final_problem =
        cumes::validate(std::move(spec), problem.options());
    check(final_problem.has_value(), "Newton replay: final stage validates");
    if (!final_problem.has_value()) return;
    cumes::SolveRequest request;
    request.restart = std::cref(original.equilibrium);
    const auto replay = solver.solve(final_problem.value(), request);
    check(replay.converged && replay.total_iterations == 1 &&
              replay.report.stages.size() == 1 &&
              replay.report.stages[0].restarts.empty(),
          "Newton replay: converges in one pass with Newton disabled");
    check(replay.fsqr <= problem.spec().stages.back().tolerance &&
              replay.fsqz <= problem.spec().stages.back().tolerance &&
              replay.fsql <= problem.spec().stages.back().tolerance,
          "Newton replay: every original residual tolerance holds");
    const auto& before = original.equilibrium;
    const auto& after = replay.equilibrium;
    bool preserved = before.ns == after.ns && before.mnmax == after.mnmax;
    for (std::size_t c = 0; c < before.families.size() && preserved; ++c) {
        preserved &= before.families[c].size() == after.families[c].size();
        for (std::size_t i = 0; i < before.families[c].size() && preserved;
             ++i) {
            const double a = before.families[c][i], b = after.families[c][i];
            if (std::bit_cast<std::uint64_t>(a) ==
                std::bit_cast<std::uint64_t>(b))
                continue;
            // Axis extrapolation may canonicalize a dependent zero's sign.
            // Every active/nonzero coefficient must retain all of its bits.
            preserved =
                i % before.ns == 0 && i / before.ns > 0 && a == 0.0 && b == 0.0;
        }
    }
    check(preserved, "Newton replay: coefficient state is preserved");
}

}  // namespace

int main() {
    try {
        constexpr std::array cases = {
            Case{"00_solovev_reference", {235, 193, 326}, {144, 113, 227}},
            // Includes the rejected fine-grid iteration-200 GMRES probe.
            Case{"10_solovev_mpol4", {240, 189, 277}, {155, 103, 227}},
            // Prescribed current requires re-evaluating iota/chi per probe.
            Case{"15_vmecpp_analytical_ncurr1",
                 {249, 211, 344},
                 {152, 103, 200}},
        };
        cumes::EquilibriumSolver solver;
        for (const auto& test_case : cases) {
            std::cout << "Newton integration: " << test_case.name << '\n';
            const auto problem = cumes::read_and_validate(
                "benchmarks/axisymmetric_newton/inputs/" +
                    std::string(test_case.name) + ".json",
                cumes::SolverOptions{});
            check(problem.has_value(), "Newton integration: input validates");
            if (!problem.has_value()) continue;
            const auto baseline = solver.solve(problem.value());
            cumes::SolveRequest request;
            request.enable_newton = true;
            const auto newton = solver.solve(problem.value(), request);
            check_outcome(problem.value(), baseline,
                          test_case.baseline_iterations);
            check_outcome(problem.value(), newton, test_case.newton_iterations);
            check(baseline.report.input_params == newton.report.input_params,
                  "Newton integration: solver flag preserves input metadata");
            check_replay(solver, problem.value(), newton);
        }
    } catch (const std::exception& error) { check(false, error.what()); }
    return cumes::test::summary();
}
