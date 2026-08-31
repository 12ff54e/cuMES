// Example integration application: optimize one boundary harmonic with a
// QS/QH/QA residual while keeping cuMES, target policy, and meow independent.
#include "quasisymmetry_target.hpp"

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
enum class TargetFamily { QH, QA };

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

class QuasisymmetryResidual {
   public:
    QuasisymmetryResidual(cumes::ProblemSpec baseline,
                          BoundaryParameterization boundary,
                          TargetFamily target_family,
                          cumes_meow_example::QsFluxGradient flux_gradient,
                          int helicity_m,
                          int helicity_n,
                          double target_aspect_ratio,
                          std::optional<double> target_iota_integral)
        : baseline_(std::move(baseline)),
          boundary_(std::move(boundary)),
          target_family_(target_family),
          flux_gradient_(flux_gradient),
          helicity_m_(helicity_m),
          helicity_n_(helicity_n),
          target_aspect_ratio_(target_aspect_ratio),
          target_iota_integral_(target_iota_integral) {
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

        cumes_meow_example::QuasisymmetryTargetSpec spec;
        spec.helicity_m = helicity_m_;
        spec.helicity_n = helicity_n_;
        spec.half_surface_indices =
            cumes_meow_example::all_half_grid_surfaces(solved.equilibrium.ns);
        spec.flux_gradient = flux_gradient_;

        cumes_meow_example::CompositeQuasisymmetryTarget target;
        if (target_family_ == TargetFamily::QH) {
            target = cumes_meow_example::calculate_qh_target(
                solved.equilibrium, solved.profiles, solved.report.input_params,
                spec, target_aspect_ratio_);
        } else {
            target = cumes_meow_example::calculate_qa_target(
                solved.equilibrium, solved.profiles, solved.report.input_params,
                spec, target_aspect_ratio_, *target_iota_integral_);
        }

        meow::Vector residual(
            static_cast<meow::Vector::Index>(target.residuals.size()));
        for (std::size_t index = 0; index < target.residuals.size(); ++index) {
            residual[static_cast<meow::Vector::Index>(index)] =
                target.residuals[index];
        }
        cached_x_ = x;
        cached_residual_ = residual;
        return residual;
    }

   private:
    cumes::ProblemSpec baseline_;
    BoundaryParameterization boundary_;
    TargetFamily target_family_;
    cumes_meow_example::QsFluxGradient flux_gradient_;
    int helicity_m_;
    int helicity_n_;
    double target_aspect_ratio_;
    std::optional<double> target_iota_integral_;
    cumes::SolverOptions validation_options_;
    cumes::EquilibriumSolver solver_;
    std::optional<meow::Vector> cached_x_;
    meow::Vector cached_residual_;
};

void print_usage() {
    std::cerr
        << "usage: cumes_meow_qs_optimize INPUT.json qh|qa "
           "poloidal|toroidal M N TARGET_ASPECT [TARGET_IOTA_INTEGRAL]\n"
        << "N is the physical toroidal helicity; multiply a per-period mode "
           "by nfp. QA requires N=0 and TARGET_IOTA_INTEGRAL; QH requires "
           "N!=0.\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 7 && argc != 8) {
        print_usage();
        return 2;
    }

    try {
        const std::string target_name = argv[2];
        const TargetFamily target_family =
            target_name == "qh" ? TargetFamily::QH
            : target_name == "qa"
                ? TargetFamily::QA
                : throw std::invalid_argument("target family must be qh or qa");
        const std::string flux_name = argv[3];
        const auto flux_gradient =
            flux_name == "poloidal"
                ? cumes_meow_example::QsFluxGradient::NORMALIZED_POLOIDAL
            : flux_name == "toroidal"
                ? cumes_meow_example::QsFluxGradient::NORMALIZED_TOROIDAL
                : throw std::invalid_argument(
                      "flux gradient must be poloidal or toroidal");
        const int helicity_m = std::stoi(argv[4]);
        const int helicity_n = std::stoi(argv[5]);
        const double target_aspect_ratio = std::stod(argv[6]);
        std::optional<double> target_iota_integral;
        if (target_family == TargetFamily::QA) {
            if (argc != 8) {
                throw std::invalid_argument("QA requires TARGET_IOTA_INTEGRAL");
            }
            target_iota_integral = std::stod(argv[7]);
        } else if (argc != 7) {
            throw std::invalid_argument(
                "QH does not take TARGET_IOTA_INTEGRAL");
        }

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

        QuasisymmetryResidual evaluator(
            parsed.spec,
            BoundaryParameterization(BoundaryVariable{BoundaryFamily::RBC, 1, 0,
                                                      harmonic->value, 0.01}),
            target_family, flux_gradient, helicity_m, helicity_n,
            target_aspect_ratio, target_iota_integral);

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
        std::cerr << "cumes-meow-qs: " << error.what() << '\n';
        return 1;
    }
}
