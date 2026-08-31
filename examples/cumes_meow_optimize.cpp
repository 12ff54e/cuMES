// Example integration application: optimize one boundary harmonic against
// major/minor-radius targets while keeping cuMES and meow independent.
#include "plasma_size_target.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include <cumes/config/json_reader.hpp>
#include <cumes/config/validated_problem.hpp>
#include <cumes/solver/equilibrium_solver.hpp>
#include <meow/trf.hpp>

namespace {

enum class BoundaryFamily { RBC, ZBS };

struct BoundaryVariable {
    BoundaryFamily family;
    int m;
    int n;
    double reference;
    double scale;
};

class BoundaryParameterization {
   public:
    explicit BoundaryParameterization(BoundaryVariable variable)
        : variable_(variable) {}

    cumes::ProblemSpec apply(const cumes::ProblemSpec& baseline,
                             const meow::Vector& x) const {
        if (x.size() != 1 || !std::isfinite(x[0])) {
            throw std::invalid_argument(
                "boundary parameterization expects one finite variable");
        }
        cumes::ProblemSpec problem = baseline;
        std::vector<cumes::BoundaryHarmonic>& family =
            variable_.family == BoundaryFamily::RBC ? problem.rbc : problem.zbs;
        auto harmonic =
            std::find_if(family.begin(), family.end(), [&](const auto& value) {
                return value.m == variable_.m && value.n == variable_.n;
            });
        const double value = variable_.reference + variable_.scale * x[0];
        if (harmonic == family.end()) {
            family.push_back(
                cumes::BoundaryHarmonic{variable_.m, variable_.n, value});
        } else {
            harmonic->value = value;
        }
        return problem;
    }

   private:
    BoundaryVariable variable_;
};

class EquilibriumResidual {
   public:
    EquilibriumResidual(cumes::ProblemSpec baseline,
                        BoundaryParameterization boundary,
                        double target_major_radius,
                        double target_minor_radius)
        : baseline_(std::move(baseline)),
          boundary_(std::move(boundary)),
          target_major_radius_(target_major_radius),
          target_minor_radius_(target_minor_radius) {
#ifdef CUMES_USE_FLOAT
        validation_options_.precision = cumes::PrecisionPolicy::MIXED_FLOAT;
#endif
    }

    meow::Vector operator()(const meow::Vector& x) {
        if (cached_x_.has_value() && *cached_x_ == x) return cached_residual_;

        cumes::ProblemSpec problem = boundary_.apply(baseline_, x);
        cumes::ValidationResult validated =
            cumes::validate(std::move(problem), validation_options_);
        if (!validated.has_value()) {
            const auto errors = validated.error().errors();
            throw std::runtime_error(
                errors.empty()
                    ? "cuMES boundary validation failed"
                    : "cuMES boundary validation failed: " + errors.front());
        }

        cumes::SolveOutcome solved = solver_.solve(validated.value());
        if (!solved.converged || !solved.has_complete_equilibrium()) {
            throw std::runtime_error(
                "cuMES did not produce a converged complete equilibrium");
        }

        const cumes_meow_example::PlasmaSize size =
            cumes_meow_example::calculate_plasma_size(
                solved.equilibrium, solved.report.input_params);
        meow::Vector residual(2);
        residual[0] = size.major_radius - target_major_radius_;
        residual[1] = size.minor_radius - target_minor_radius_;
        cached_x_ = x;
        cached_residual_ = residual;
        return residual;
    }

   private:
    cumes::ProblemSpec baseline_;
    BoundaryParameterization boundary_;
    double target_major_radius_;
    double target_minor_radius_;
    cumes::SolverOptions validation_options_;
    cumes::EquilibriumSolver solver_;
    std::optional<meow::Vector> cached_x_;
    meow::Vector cached_residual_;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: cumes_meow_optimize INPUT.json TARGET_RMAJOR "
                     "TARGET_AMINOR\n";
        return 2;
    }

    try {
        cumes::SolverOptions validation_options;
#ifdef CUMES_USE_FLOAT
        validation_options.precision = cumes::PrecisionPolicy::MIXED_FLOAT;
#endif
        cumes::ParsedProblem parsed =
            cumes::read_problem_spec(argv[1], validation_options);
        if (!parsed.report.ok()) {
            throw std::runtime_error("input JSON mapping failed");
        }

        auto harmonic =
            std::find_if(parsed.spec.rbc.begin(), parsed.spec.rbc.end(),
                         [](const auto& h) { return h.m == 1 && h.n == 0; });
        if (harmonic == parsed.spec.rbc.end()) {
            throw std::runtime_error("input has no rbc(m=1,n=0) harmonic");
        }

        const double target_major_radius = std::stod(argv[2]);
        const double target_minor_radius = std::stod(argv[3]);
        if (!std::isfinite(target_major_radius) ||
            !std::isfinite(target_minor_radius) ||
            !(target_major_radius > 0.0) || !(target_minor_radius > 0.0)) {
            throw std::invalid_argument(
                "radius targets must be finite and positive");
        }
        EquilibriumResidual evaluator(
            parsed.spec,
            BoundaryParameterization(BoundaryVariable{BoundaryFamily::RBC, 1, 0,
                                                      harmonic->value, 0.01}),
            target_major_radius, target_minor_radius);

        meow::Vector initial = meow::Vector::Zero(1);
        meow::Bounds bounds{meow::Vector::Constant(1, -10.0),
                            meow::Vector::Constant(1, 10.0)};
        meow::TrfOptions options;
        options.x_scale = meow::Vector::Ones(1);
        options.verbose = 1;

        const meow::TrfResult result = meow::trf_least_squares(
            [&evaluator](const meow::Vector& x) { return evaluator(x); },
            initial, bounds, options);
        std::cout << "success=" << result.success << " x=" << result.x[0]
                  << " cost=" << result.cost << '\n';
        return result.success ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "cumes-meow: " << error.what() << '\n';
        return 1;
    }
}
