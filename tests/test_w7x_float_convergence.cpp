#include "cumes/config/json_reader.hpp"
#include "cumes/config/validated_problem.hpp"
#include "cumes/solver/equilibrium_solver.hpp"
#include "cumes_test.h"

#include <cstdio>
#include <exception>
#include <utility>

int main() {
    using cumes::test::check;
    try {
        cumes::SolverOptions options;
        options.precision = cumes::PrecisionPolicy::MIXED_FLOAT;
        auto parsed = cumes::read_problem_spec("inputs/w7x.json", options);
        check(parsed.report.ok(), "W7-X input parses");
        if (!parsed.report.ok()) return cumes::test::summary();
        constexpr double FTOL = 1e-5;
        parsed.spec.stages = {cumes::StageRequest{99, 5000, FTOL}};
        auto problem = cumes::validate(std::move(parsed.spec), options);
        check(problem.has_value(), "single-grid float W7-X validates");
        if (!problem.has_value()) return cumes::test::summary();

        cumes::SolveRequest request;
        request.odd_geometry = cumes::OddGeometryPrecision::COMPENSATED;
        cumes::EquilibriumSolver solver;
        auto cold = solver.solve(problem.value(), request);
        check(
            cold.converged && cold.fsqr <= FTOL && cold.fsqz <= FTOL &&
                cold.fsql <= FTOL,
            "single-grid float cold start meets all three residual tolerances");
        check(cold.report.stages.size() == 1 && cold.equilibrium.ns == 99,
              "cold start used only the requested ns=99 grid");
        if (!cold.converged) return cumes::test::summary();

        request.restart = std::cref(cold.equilibrium);
        auto replay = solver.solve(problem.value(), request);
        check(replay.converged && replay.iterations == 1 &&
                  replay.fsqr <= FTOL && replay.fsqz <= FTOL &&
                  replay.fsql <= FTOL,
              "single-grid float checkpoint is a converged fixed point");
        std::printf("cold iterations=%d residuals=%.9g %.9g %.9g; replay=%d\n",
                    cold.iterations, cold.fsqr, cold.fsqz, cold.fsql,
                    replay.iterations);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return cumes::test::summary();
}
