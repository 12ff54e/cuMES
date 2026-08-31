#include <cumes/config/problem_spec.hpp>
#include <cumes/solver/equilibrium_solver.hpp>

int main() {
    cumes::ProblemSpec problem;
    cumes::EquilibriumSolver solver;
    return problem.nfp == 1 ? 0 : 1;
}
