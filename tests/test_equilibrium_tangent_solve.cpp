#include "cumes/config/json_reader.hpp"
#include "cumes/runtime/cuda_status.hpp"
#include "cumes/solver/equilibrium_linearization.hpp"
#include "cumes/solver/equilibrium_solver.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

#ifndef CUMES_USE_FLOAT
double relative_difference(const std::vector<double>& tangent,
                           const std::vector<double>& plus,
                           const std::vector<double>& minus,
                           double epsilon) {
    double difference_sq = 0.0;
    double reference_sq = 0.0;
    for (std::size_t i = 0; i < tangent.size(); ++i) {
        const double reference = (plus[i] - minus[i]) / (2.0 * epsilon);
        const double difference = tangent[i] - reference;
        difference_sq += difference * difference;
        reference_sq += reference * reference;
    }
    return std::sqrt(difference_sq / reference_sq);
}
#endif

}  // namespace

int main() {
#ifdef CUMES_USE_FLOAT
    cumes::SolverOptions options;
    options.precision = cumes::PrecisionPolicy::MIXED_FLOAT;
    auto parsed = cumes::read_problem_spec("inputs/solovev.json", options);
    for (auto& stage : parsed.spec.stages) stage.tolerance = 1.0e-6;
    auto problem = cumes::validate(std::move(parsed.spec), options);
    if (!problem.has_value()) {
        std::cerr << "FAIL: mixed-float Solovev input did not validate\n";
        return 1;
    }
    try {
        const cumes::EquilibriumSnapshot unused_equilibrium;
        const cumes::EquilibriumLinearization unsupported(problem.value(),
                                                          unused_equilibrium);
        static_cast<void>(unsupported);
    } catch (const cumes::CumesError& error) {
        const std::string message = error.what();
        if (message.find("precise-double") != std::string::npos) return 0;
        std::cerr << "FAIL: unexpected mixed-float tangent error: " << message
                  << '\n';
        return 1;
    }
    std::cerr << "FAIL: mixed-float tangent construction did not fail\n";
    return 1;
#else
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
    const cumes::ResidualJvp physical_residual = linearization.residual_jvp(
        std::vector<double>(linearization.state_size(), 0.0));
    double physical_residual_norm = 0.0;
    for (double value : physical_residual.residual)
        physical_residual_norm += value * value;
    std::cout << "Solovev physical residual norm="
              << std::sqrt(physical_residual_norm) << '\n';
    cumes::BoundaryTangent direction =
        cumes::BoundaryTangent::zero(problem.value());
    const int m1n0 = problem.value().spec().ntor + 1;
    direction.rbcc[m1n0] = 1e-3;
    direction.zbsc[m1n0] = -5e-4;

    std::unique_ptr<cumes::EquilibriumLinearization> detached_linearization;
    {
        auto short_lived_problem = cumes::validate(problem.value().spec(), {});
        if (!short_lived_problem.has_value()) {
            std::cerr << "FAIL: short-lived tangent problem did not validate\n";
            return 1;
        }
        detached_linearization =
            std::make_unique<cumes::EquilibriumLinearization>(
                short_lived_problem.value(), equilibrium.equilibrium);
    }
    const cumes::ResidualJvp detached_boundary =
        detached_linearization->boundary_residual_jvp(direction);
    if (detached_boundary.tangent.size() !=
        detached_linearization->state_size()) {
        std::cerr << "FAIL: tangent session retained its input problem\n";
        return 1;
    }
    detached_linearization.reset();

    cumes::TangentLinearOptions options;
    options.backend = cumes::TangentLinearBackend::HOST;
    options.max_iterations = 1000;
    options.restart = 300;
    options.relative_tolerance = 2e-6;
    options.absolute_tolerance = 1e-11;
    cumes::TangentLinearOptions invalid_options = options;
    invalid_options.relative_tolerance =
        std::numeric_limits<double>::quiet_NaN();
    try {
        static_cast<void>(
            linearization.solve_boundary_tangent(direction, invalid_options));
        std::cerr << "FAIL: non-finite tangent tolerance was accepted\n";
        return 1;
    } catch (const cumes::CumesError&) {}
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
    auto device_options = options;
    device_options.backend = cumes::TangentLinearBackend::DEVICE;
    const auto device_tangent =
        linearization.solve_boundary_tangent(direction, device_options);
    double reference_squared = 0.0, difference_squared = 0.0;
    for (std::size_t i = 0; i < tangent.state_tangent.size(); ++i) {
        const double difference =
            device_tangent.state_tangent[i] - tangent.state_tangent[i];
        difference_squared += difference * difference;
        reference_squared +=
            tangent.state_tangent[i] * tangent.state_tangent[i];
    }
    const double device_error =
        std::sqrt(difference_squared / reference_squared);
    std::cout << "Solovev device GMRES final=" << device_tangent.final_residual
              << " iterations=" << device_tangent.iterations
              << " host_relative_difference=" << device_error << '\n';
    if (!device_tangent.converged || !(device_error < 1e-7)) {
        std::cerr << "FAIL: device tangent disagrees with the host reference\n";
        return 1;
    }
    const auto zero_tangent = linearization.solve_boundary_tangent(
        cumes::BoundaryTangent::zero(problem.value()), device_options);
    if (!zero_tangent.converged || zero_tangent.iterations != 0 ||
        zero_tangent.final_residual != 0.0) {
        std::cerr << "FAIL: reused device tangent workspace did not stop for "
                     "zero RHS\n";
        return 1;
    }
    for (const double value : zero_tangent.state_tangent) {
        if (value != 0.0) {
            std::cerr << "FAIL: zero boundary retained an earlier tangent\n";
            return 1;
        }
    }
    auto absolute_options = device_options;
    absolute_options.relative_tolerance = 0.0;
    absolute_options.absolute_tolerance = 2.0 * tangent.initial_residual;
    const auto absolute_tangent =
        linearization.solve_boundary_tangent(direction, absolute_options);
    if (!absolute_tangent.converged || absolute_tangent.iterations != 0) {
        std::cerr
            << "FAIL: device tangent ignored absolute stopping tolerance\n";
        return 1;
    }
    auto capped_options = device_options;
    capped_options.max_iterations = 1;
    capped_options.restart = 8;
    const auto capped_tangent =
        linearization.solve_boundary_tangent(direction, capped_options);
    if (capped_tangent.converged || capped_tangent.iterations != 1 ||
        !std::isfinite(capped_tangent.final_residual)) {
        std::cerr << "FAIL: device tangent ignored its iteration budget\n";
        return 1;
    }
    const auto repeated_tangent =
        linearization.solve_boundary_tangent(direction, device_options);
    if (!repeated_tangent.converged ||
        repeated_tangent.state_tangent != device_tangent.state_tangent) {
        std::cerr << "FAIL: changed restart size corrupted reused tangent "
                     "workspace\n";
        return 1;
    }
    auto unpreconditioned_options = device_options;
    unpreconditioned_options.use_equilibrium_preconditioner = false;
    unpreconditioned_options.restart = 2;
    unpreconditioned_options.max_iterations = 3;
    const auto unpreconditioned_device = linearization.solve_boundary_tangent(
        direction, unpreconditioned_options);
    unpreconditioned_options.backend = cumes::TangentLinearBackend::HOST;
    const auto unpreconditioned_host = linearization.solve_boundary_tangent(
        direction, unpreconditioned_options);
    if (unpreconditioned_device.converged ||
        unpreconditioned_device.iterations != 3 ||
        !std::isfinite(unpreconditioned_device.final_residual) ||
        std::abs(unpreconditioned_device.final_residual -
                 unpreconditioned_host.final_residual) >
            1e-10 * unpreconditioned_host.final_residual) {
        std::cerr << "FAIL: unpreconditioned device restart disagrees with "
                     "the host residual\n";
        return 1;
    }
    const auto device_fields = linearization.materialize_tangent(
        device_tangent.state_tangent, equilibrium.equilibrium,
        equilibrium.profiles);
    const cumes::EquilibriumTangent fields = linearization.materialize_tangent(
        tangent.state_tangent, equilibrium.equilibrium, equilibrium.profiles);
    if (!fields.matches(equilibrium.equilibrium, equilibrium.profiles)) {
        std::cerr << "FAIL: materialized equilibrium tangent shape mismatch\n";
        return 1;
    }
    for (const auto& field : fields.equilibrium.half_fields) {
        for (double value : field) {
            if (!std::isfinite(value)) {
                std::cerr << "FAIL: non-finite materialized field tangent\n";
                return 1;
            }
        }
    }
    for (double value : fields.profiles.rotational_transform) {
        if (!std::isfinite(value)) {
            std::cerr << "FAIL: non-finite iota tangent\n";
            return 1;
        }
    }
    const std::size_t family_size = equilibrium.equilibrium.family_size();
    const std::size_t lcfs = static_cast<std::size_t>(
        (problem.value().spec().ntor + 1) * equilibrium.equilibrium.ns +
        equilibrium.equilibrium.ns - 1);
    if (device_fields.equilibrium.families[cumes::EquilibriumSnapshot::RMNCC]
                                          [lcfs] != direction.rbcc[m1n0] ||
        device_fields.equilibrium.families[cumes::EquilibriumSnapshot::ZMNSC]
                                          [lcfs] != direction.zbsc[m1n0] ||
        fields.equilibrium.families[cumes::EquilibriumSnapshot::RMNCC][lcfs] !=
            direction.rbcc[m1n0] ||
        fields.equilibrium.families[cumes::EquilibriumSnapshot::ZMNSC][lcfs] !=
            direction.zbsc[m1n0] ||
        family_size == 0) {
        std::cerr << "FAIL: materialized tangent lost its LCFS direction\n";
        return 1;
    }

    constexpr double epsilon = 1e-1;
    auto perturbed_problem = [&](double sign) {
        cumes::ProblemSpec spec = problem.value().spec();
        spec.stages = {spec.stages.back()};
        bool found_r = false;
        bool found_z = false;
        for (auto& harmonic : spec.rbc) {
            if (harmonic.m == 1 && harmonic.n == 0) {
                harmonic.value += sign * epsilon * direction.rbcc[m1n0];
                found_r = true;
            }
        }
        for (auto& harmonic : spec.zbs) {
            if (harmonic.m == 1 && harmonic.n == 0) {
                harmonic.value += sign * epsilon * direction.zbsc[m1n0];
                found_z = true;
            }
        }
        if (!found_r || !found_z) {
            std::cerr << "FAIL: Solovev m=1,n=0 boundary harmonic missing\n";
            std::exit(1);
        }
        auto validated = cumes::validate(std::move(spec), {});
        if (!validated.has_value()) {
            std::cerr << "FAIL: perturbed Solovev problem did not validate\n";
            std::exit(1);
        }
        return std::move(validated.value());
    };
    const cumes::ValidatedProblem plus_problem = perturbed_problem(1.0);
    const cumes::ValidatedProblem minus_problem = perturbed_problem(-1.0);
    cumes::SolveRequest hot_restart;
    hot_restart.restart = std::cref(equilibrium.equilibrium);
    const cumes::SolveOutcome plus = solver.solve(plus_problem, hot_restart);
    const cumes::SolveOutcome minus = solver.solve(minus_problem, hot_restart);
    if (!plus.converged || !minus.converged) {
        std::cerr << "FAIL: perturbed Solovev oracle did not converge\n";
        return 1;
    }
    std::cout << "hot-restart oracle iterations: plus=" << plus.total_iterations
              << " minus=" << minus.total_iterations << '\n';
    std::vector<double> nonlinear_fd_state(linearization.state_size());
    for (std::size_t component = 0; component < 6; ++component) {
        for (std::size_t index = 0; index < family_size; ++index) {
            nonlinear_fd_state[component * family_size + index] =
                (plus.equilibrium.families[component][index] -
                 minus.equilibrium.families[component][index]) /
                (2.0 * epsilon);
        }
    }
    const cumes::ResidualJvp nonlinear_fd_residual =
        linearization.residual_jvp(nonlinear_fd_state);
    const cumes::ResidualJvp implicit_residual =
        linearization.residual_jvp(tangent.state_tangent);
    double nonlinear_fd_residual_norm = 0.0;
    double implicit_residual_norm = 0.0;
    for (std::size_t index = 0; index < nonlinear_fd_residual.tangent.size();
         ++index) {
        nonlinear_fd_residual_norm += nonlinear_fd_residual.tangent[index] *
                                      nonlinear_fd_residual.tangent[index];
        implicit_residual_norm +=
            implicit_residual.tangent[index] * implicit_residual.tangent[index];
    }
    std::cout << "combined residual derivative: nonlinear-FD="
              << std::sqrt(nonlinear_fd_residual_norm)
              << " implicit=" << std::sqrt(implicit_residual_norm) << '\n';
    double spectral_error_sq = 0.0;
    for (std::size_t component = 0; component < 6; ++component) {
        const double error =
            relative_difference(fields.equilibrium.families[component],
                                plus.equilibrium.families[component],
                                minus.equilibrium.families[component], epsilon);
        std::cout << "  family " << component << " relative error=" << error
                  << '\n';
        if (std::isfinite(error)) spectral_error_sq += error * error;
    }
    const double spectral_error = std::sqrt(spectral_error_sq);
    double field_error_sq = 0.0;
    for (const auto field :
         {cumes::EquilibriumSnapshot::SQRTG, cumes::EquilibriumSnapshot::BSUPU,
          cumes::EquilibriumSnapshot::BSUPV, cumes::EquilibriumSnapshot::BSUBS,
          cumes::EquilibriumSnapshot::BSUBU,
          cumes::EquilibriumSnapshot::BSUBV}) {
        const double error =
            relative_difference(fields.equilibrium.half_fields[field],
                                plus.equilibrium.half_fields[field],
                                minus.equilibrium.half_fields[field], epsilon);
        std::cout << "  half field " << static_cast<int>(field)
                  << " relative error=" << error << '\n';
        if (std::isfinite(error)) field_error_sq += error * error;
    }
    const double field_error = std::sqrt(field_error_sq);
    std::cout << "Solovev nonlinear-FD tangent spectral error="
              << spectral_error << " field error=" << field_error << '\n';
    if (!(nonlinear_fd_residual_norm < 1e-5) ||
        !(implicit_residual_norm < 1e-5)) {
        std::cerr << "FAIL: tangent directions do not satisfy the linearized "
                     "equilibrium equations\n";
        return 1;
    }
    // This comparison also catches a disconnected condensation-constraint
    // path: the dual inverse must populate the ConstraintOperator-owned
    // rCon/zCon views, exactly as the nonlinear equilibrium pass does.
    if (!(spectral_error < 1e-2) || !(field_error < 2e-2)) {
        std::cerr << "FAIL: tangent solve selected a response inconsistent "
                     "with the converged nonlinear restart\n";
        return 1;
    }
    return 0;
#endif
}
