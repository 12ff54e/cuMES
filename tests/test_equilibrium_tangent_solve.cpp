#include "cumes/config/json_reader.hpp"
#include "cumes/runtime/cuda_status.hpp"
#include "cumes/solver/equilibrium_linearization.hpp"
#include "cumes/solver/equilibrium_solver.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>

int main() {
    const auto problem =
        cumes::read_and_validate("inputs/solovev.json", cumes::SolverOptions{});
    if (!problem.has_value()) {
        std::cerr << "FAIL: Solovev input did not validate\n";
        return 1;
    }
    cumes::EquilibriumSolver solver;
    const cumes::SolveOutcome equilibrium = solver.solve(problem.value());
    if (!equilibrium.converged || !equilibrium.has_complete_equilibrium()) {
        std::cerr << "FAIL: Solovev primal equilibrium did not converge\n";
        return 1;
    }

    cumes::EquilibriumLinearization linearization(problem.value(),
                                                  equilibrium.equilibrium);
    cumes::BoundaryTangent direction =
        cumes::BoundaryTangent::zero(problem.value());
    const int m1n0 = problem.value().spec().ntor + 1;
    direction.rbcc[m1n0] = 1e-3;
    direction.zbsc[m1n0] = -5e-4;
    cumes::TangentLinearOptions options;
    options.max_iterations = 300;
    options.restart = 80;
    options.relative_tolerance = 1e-4;
    options.absolute_tolerance = 1e-11;
    const cumes::SpectralTangentSolve tangent =
        linearization.solve_boundary_tangent(direction, options);
    std::cout << "Solovev tangent GMRES initial=" << tangent.initial_residual
              << " final=" << tangent.final_residual
              << " iterations=" << tangent.iterations
              << " converged=" << tangent.converged << '\n';
    if (!std::isfinite(tangent.final_residual) ||
        tangent.state_tangent.size() != linearization.state_size()) {
        std::cerr << "FAIL: Solovev tangent solve returned invalid data\n";
        return 1;
    }
    if (!tangent.converged) {
        std::cerr << "FAIL: Solovev tangent solve did not reach tolerance\n";
        return 1;
    }
    return 0;
}
