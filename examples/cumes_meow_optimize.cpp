// Example integration application: optimize one boundary harmonic against an
// equilibrium-derived target while keeping cuMES and meow independent.
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
#include <cumes/io/equilibrium_snapshot.hpp>
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

double outer_half_surface_mean(const cumes::EquilibriumSnapshot& equilibrium,
                               cumes::EquilibriumSnapshot::HalfField field) {
    const std::size_t points = equilibrium.points_per_surface();
    const std::size_t offset =
        static_cast<std::size_t>(equilibrium.ns - 2) * points;
    const auto& values = equilibrium.half_fields[field];
    double sum = 0.0;
    for (std::size_t point = 0; point < points; ++point) {
        sum += values[offset + point];
    }
    return sum / static_cast<double>(points);
}

class EquilibriumResidual {
   public:
    EquilibriumResidual(cumes::ProblemSpec baseline,
                        BoundaryParameterization boundary,
                        double target)
        : baseline_(std::move(baseline)),
          boundary_(std::move(boundary)),
          target_(target) {
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

        const double achieved = outer_half_surface_mean(
            solved.equilibrium, cumes::EquilibriumSnapshot::BSUPV);
        meow::Vector residual(1);
        residual[0] = achieved - target_;
        cached_x_ = x;
        cached_residual_ = residual;
        return residual;
    }

   private:
    cumes::ProblemSpec baseline_;
    BoundaryParameterization boundary_;
    double target_;
    cumes::SolverOptions validation_options_;
    cumes::EquilibriumSolver solver_;
    std::optional<meow::Vector> cached_x_;
    meow::Vector cached_residual_;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr
            << "usage: cumes_meow_optimize INPUT.json TARGET_MEAN_BSUPV\n";
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

        const double target = std::stod(argv[2]);
        EquilibriumResidual evaluator(
            parsed.spec,
            BoundaryParameterization(BoundaryVariable{BoundaryFamily::RBC, 1, 0,
                                                      harmonic->value, 0.01}),
            target);

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
