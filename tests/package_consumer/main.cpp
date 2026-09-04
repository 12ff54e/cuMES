#include <cumes/config/problem_spec.hpp>
#include <cumes/solver/equilibrium_linearization.hpp>
#include <cumes/solver/equilibrium_solver.hpp>

std::size_t tangent_linkage(const cumes::ValidatedProblem& problem,
                            const cumes::EquilibriumSnapshot& equilibrium) {
    cumes::EquilibriumLinearization linearization(problem, equilibrium);
    return linearization.state_size();
}

int main() {
    cumes::ProblemSpec problem;
    cumes::EquilibriumSolver solver;
    return problem.nfp == 1 ? 0 : 1;
}
