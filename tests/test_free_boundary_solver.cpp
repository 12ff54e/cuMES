#include "cumes/config/json_reader.hpp"
#include "cumes/solver/equilibrium_solver.hpp"
#include "cumes_test.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <exception>
#include <string_view>

namespace {

using cumes::test::check;

bool same_bits(double a, double b) {
    return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

template <class Fields>
bool same_fields(const Fields& a, const Fields& b) {
    for (std::size_t c = 0; c < a.size(); ++c) {
        if (a[c].size() != b[c].size()) return false;
        for (std::size_t i = 0; i < a[c].size(); ++i)
            if (!std::isfinite(a[c][i]) || !same_bits(a[c][i], b[c][i]))
                return false;
    }
    return true;
}

void check_stages(const cumes::ValidatedProblem& problem,
                  const cumes::SolveOutcome& first,
                  const cumes::SolveOutcome& second) {
    const auto& configured = problem.spec().stages;
    check(first.converged && second.converged &&
              first.has_complete_equilibrium() &&
              second.has_complete_equilibrium(),
          "repeated free-boundary solves converge completely");
    check(first.report.stages.size() == configured.size() &&
              second.report.stages.size() == configured.size(),
          "free-boundary solves retain every multigrid stage");
    if (first.report.stages.size() != configured.size() ||
        second.report.stages.size() != configured.size())
        return;
    for (std::size_t g = 0; g < configured.size(); ++g) {
        const auto& a = first.report.stages[g];
        const auto& b = second.report.stages[g];
        const auto& ra = a.final_residual;
        const auto& rb = b.final_residual;
#ifdef CUMES_PRECISION_POLICY_NAME
        if (std::string_view(CUMES_PRECISION_POLICY_NAME) == "verify-double") {
            constexpr std::array expected_iterations = {389, 636};
            check(a.effective_iterations == expected_iterations.at(g),
                  "free-boundary precise solve retains the frozen trajectory");
        }
#endif
        check(a.converged && b.converged && a.ns == b.ns &&
                  static_cast<std::size_t>(a.ns) ==
                      configured[g].radial_surfaces &&
                  a.effective_iterations == b.effective_iterations &&
                  a.effective_iterations > 0 &&
                  static_cast<std::size_t>(a.effective_iterations) <=
                      configured[g].max_iterations,
              "repeated free-boundary solves preserve stage counts and caps");
        check(same_bits(ra.fsqr, rb.fsqr) && same_bits(ra.fsqz, rb.fsqz) &&
                  same_bits(ra.fsql, rb.fsql) && ra.fsqr >= 0 && ra.fsqz >= 0 &&
                  ra.fsql >= 0 && ra.fsqr < configured[g].tolerance &&
                  ra.fsqz < configured[g].tolerance &&
                  ra.fsql < configured[g].tolerance,
              "repeated free-boundary solves preserve all converged residuals");
        bool same_restarts = a.restarts.size() == b.restarts.size();
        for (std::size_t i = 0; i < a.restarts.size() && same_restarts; ++i)
            same_restarts = a.restarts[i].iteration == b.restarts[i].iteration;
        check(same_restarts,
              "repeated free-boundary solves preserve activation and restart "
              "records");
    }
}

}  // namespace

int main() {
    try {
        // Embedded MAKEGRID keeps this integration gate independent of large
        // external mgrid files. The original two-stage solve crosses vacuum
        // activation, a soft restart, norm refreshes, and radial refinement.
        const auto problem = cumes::read_and_validate(
            "inputs/free_bdy/solovev_free_bdy_embedded.json",
            cumes::SolverOptions{});
        check(problem.has_value(), "free-boundary fixture validates");
        if (!problem.has_value()) return cumes::test::summary();
        cumes::EquilibriumSolver solver;
        // Reusing the public facade must start a fresh vacuum lifecycle.
        const auto first = solver.solve(problem.value());
        const auto second = solver.solve(problem.value());
        check_stages(problem.value(), first, second);
        if (!first.has_complete_equilibrium() ||
            !second.has_complete_equilibrium())
            return cumes::test::summary();
        const auto& a = first.equilibrium;
        const auto& b = second.equilibrium;
        check(
            a.ns == b.ns && a.mnmax == b.mnmax &&
                same_fields(a.families, b.families) &&
                same_fields(a.half_fields, b.half_fields) &&
                same_fields(a.full_fields, b.full_fields),
            "repeated free-boundary state and published fields are bit exact");
        bool physical_fields = true;
        using Snapshot = cumes::EquilibriumSnapshot;
        for (std::size_t i = 0; i < a.half_field_size(); ++i) {
            const auto& f = a.half_fields;
            const double b_squared =
                f[Snapshot::BSUPS][i] * f[Snapshot::BSUBS][i] +
                f[Snapshot::BSUPU][i] * f[Snapshot::BSUBU][i] +
                f[Snapshot::BSUPV][i] * f[Snapshot::BSUBV][i];
            physical_fields &= f[Snapshot::SQRTG][i] < 0.0 &&
                               std::isfinite(b_squared) && b_squared > 0.0;
        }
        check(physical_fields,
              "free-boundary geometry and magnetic energy are valid");
    } catch (const std::exception& error) { check(false, error.what()); }
    return cumes::test::summary();
}
