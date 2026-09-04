#include "cumes/config/json_reader.hpp"
#include "cumes/config/validated_problem.hpp"
#include "cumes/solver/equilibrium_solver.hpp"
#include "cumes_test.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <future>
#include <stdexcept>
#include <utility>

#include <unistd.h>

namespace {

class StdoutCapture {
   public:
    StdoutCapture() {
        stream_ = std::tmpfile();
        if (stream_ == nullptr)
            throw std::runtime_error("could not create stdout capture");
        saved_fd_ = dup(fileno(stdout));
        if (saved_fd_ < 0) {
            std::fclose(stream_);
            stream_ = nullptr;
            throw std::runtime_error("could not capture solver stdout");
        }
        std::fflush(stdout);
        if (dup2(fileno(stream_), fileno(stdout)) < 0) {
            close(saved_fd_);
            saved_fd_ = -1;
            std::fclose(stream_);
            stream_ = nullptr;
            throw std::runtime_error("could not redirect solver stdout");
        }
    }

    ~StdoutCapture() {
        if (saved_fd_ >= 0) {
            std::fflush(stdout);
            dup2(saved_fd_, fileno(stdout));
            close(saved_fd_);
        }
        if (stream_ != nullptr) std::fclose(stream_);
    }

    long finish() {
        std::fflush(stdout);
        if (std::fseek(stream_, 0, SEEK_END) != 0) {
            throw std::runtime_error("could not inspect captured stdout");
        }
        const long size = std::ftell(stream_);
        if (dup2(saved_fd_, fileno(stdout)) < 0) {
            throw std::runtime_error("could not restore solver stdout");
        }
        close(saved_fd_);
        saved_fd_ = -1;
        return size;
    }

   private:
    std::FILE* stream_ = nullptr;
    int saved_fd_ = -1;
};

}  // namespace

int main() {
    try {
        using cumes::test::check;
        cumes::SolverOptions options;
#ifdef CUMES_USE_FLOAT
        options.precision = cumes::PrecisionPolicy::MIXED_FLOAT;
#endif
        cumes::ParsedProblem parsed =
            cumes::read_problem_spec("inputs/solovev.json", options);
        check(parsed.report.ok(), "solver API: JSON mapping succeeds");

        // Exercise the optimizer-facing path: edit one boundary coefficient
        // in memory, then rebuild the immutable validated problem.
        check(!parsed.spec.rbc.empty(), "solver API: boundary is nonempty");
        if (parsed.spec.rbc.empty()) return cumes::test::summary();
        parsed.spec.rbc.front().value += 1.0e-4;
        parsed.spec.stages = {cumes::StageRequest{5, 1,
#ifdef CUMES_USE_FLOAT
                                                  1.0e-6
#else
                                                  1.0e-16
#endif
        }};

        cumes::ValidationResult validated =
            cumes::validate(std::move(parsed.spec), options);
        check(validated.has_value(), "solver API: edited problem validates");
        if (!validated.has_value()) return cumes::test::summary();

        cumes::EquilibriumSolver solver;
        // The embedding API is argument-deterministic by default even when
        // the surrounding process carries legacy CLI environment controls.
        setenv("CUMES_MAX_ITER", "0", 1);
        cumes::SolveOutcome outcome;
        long captured_bytes = 0;
        {
            StdoutCapture capture;
            outcome = solver.solve(validated.value());
            captured_bytes = capture.finish();
        }

        check(outcome.failed_stage == -1,
              "solver API: single grid has no failed multigrid stage");
        check(captured_bytes == 0,
              "solver API: embedding calls are quiet by default");
        check(outcome.iterations > 0,
              "solver API: process environment is ignored by default");
        check(outcome.equilibrium.ns == 5,
              "solver API: snapshot has requested radial size");
        check(outcome.equilibrium.mnmax ==
                  static_cast<int>(validated.value().shape().modes()),
              "solver API: snapshot has validated mode count");
        check(outcome.has_complete_equilibrium(),
              "solver API: state and derived fields are complete");
        check(outcome.profiles.has_half_grid_profiles(outcome.equilibrium.ns),
              "solver API: equilibrium flux profiles are complete");
        check(!outcome.report.input_params.rbc_value.empty() &&
                  outcome.report.input_params.rbc_value.front() ==
                      validated.value().spec().rbc.front().value,
              "solver API: report embeds normalized input");
        check(outcome.report.stages.size() == 1,
              "solver API: report contains the stage");
        check(std::isfinite(outcome.fsqr) && std::isfinite(outcome.fsqz) &&
                  std::isfinite(outcome.fsql),
              "solver API: residuals are finite");
        const double timed_phase_sum =
            outcome.timings.setup_wall_ms + outcome.timings.multigrid_wall_ms +
            outcome.timings.final_state_transfer_wall_ms;
        check(outcome.timings.setup_wall_ms >= 0.0 &&
                  outcome.timings.multigrid_wall_ms > 0.0 &&
                  outcome.timings.final_state_transfer_wall_ms >= 0.0 &&
                  outcome.timings.total_wall_ms > 0.0,
              "solver API: structured wall timings are nonnegative");
        check(std::abs(timed_phase_sum - outcome.timings.total_wall_ms) <=
                  1.0e-9 * outcome.timings.total_wall_ms,
              "solver API: structured wall timings cover the timed call");

        bool profile_identity = true;
#ifdef CUMES_USE_FLOAT
        constexpr double profile_tolerance = 1.0e-6;
#else
        constexpr double profile_tolerance = 1.0e-13;
#endif
        for (std::size_t j = 0;
             j < outcome.profiles.rotational_transform.size(); ++j) {
            const double expected =
                outcome.profiles.rotational_transform[j] *
                outcome.profiles.toroidal_flux_derivative[j];
            profile_identity =
                profile_identity &&
                std::abs(outcome.profiles.poloidal_flux_derivative[j] -
                         expected) < profile_tolerance;
        }
        check(profile_identity,
              "solver API: fixed-iota flux derivatives are consistent");

        bool covariant_flux_functions_match = true;
        const std::size_t points = outcome.equilibrium.points_per_surface();
        const auto& bsubu =
            outcome.equilibrium.half_fields[cumes::EquilibriumSnapshot::BSUBU];
        const auto& bsubv =
            outcome.equilibrium.half_fields[cumes::EquilibriumSnapshot::BSUBV];
        for (int surface = 0; surface < outcome.equilibrium.ns - 1; ++surface) {
            const std::size_t offset =
                static_cast<std::size_t>(surface) * points;
            double poloidal_sum = 0.0;
            double toroidal_sum = 0.0;
            for (std::size_t point = 0; point < points; ++point) {
                poloidal_sum += bsubu[offset + point];
                toroidal_sum += bsubv[offset + point];
            }
            const std::size_t radial = static_cast<std::size_t>(surface);
            covariant_flux_functions_match =
                covariant_flux_functions_match &&
                std::abs(outcome.profiles.poloidal_covariant_field[radial] -
                         poloidal_sum / static_cast<double>(points)) <
                    profile_tolerance &&
                std::abs(outcome.profiles.toroidal_covariant_field[radial] -
                         toroidal_sum / static_cast<double>(points)) <
                    profile_tolerance;
        }
        check(covariant_flux_functions_match,
              "solver API: I and G match covariant field averages");

        cumes::SolveRequest restart_request;
        restart_request.restart = std::cref(outcome.equilibrium);
        restart_request.radial_transfer =
            cumes::RadialTransferPolicy::CATMULL_ROM;
        cumes::SolveOutcome restarted =
            solver.solve(validated.value(), restart_request);
        check(restarted.has_complete_equilibrium(),
              "solver API: in-memory restart returns a complete equilibrium");
        check(restarted.report.input_params == outcome.report.input_params,
              "solver API: cold and hot runs retain the same input metadata");

        cumes::SolveOutcome repeated = solver.solve(validated.value());
        check(repeated.fsqr == outcome.fsqr && repeated.fsqz == outcome.fsqz &&
                  repeated.fsql == outcome.fsql &&
                  repeated.equilibrium.families == outcome.equilibrium.families,
              "solver API: repeated cold evaluation is deterministic");

        cumes::ParsedProblem concurrent_parsed =
            cumes::read_problem_spec("inputs/w7x.json", options);
        check(concurrent_parsed.report.ok(),
              "solver API: concurrent 3-D input mapping succeeds");
        cumes::ValidationResult concurrent_validated =
            cumes::validate(std::move(concurrent_parsed.spec), options);
        check(concurrent_validated.has_value(),
              "solver API: concurrent 3-D problem validates");
        if (!concurrent_validated.has_value()) {
            return cumes::test::summary();
        }
        const auto concurrent_solve = [&concurrent_validated]() {
            cumes::EquilibriumSolver independent_solver;
            cumes::SolveRequest request;
            request.radial_transfer = cumes::RadialTransferPolicy::CATMULL_ROM;
            return independent_solver.solve(concurrent_validated.value(),
                                            request);
        };
        auto first_future = std::async(std::launch::async, concurrent_solve);
        auto second_future = std::async(std::launch::async, concurrent_solve);
        const cumes::SolveOutcome first_concurrent = first_future.get();
        const cumes::SolveOutcome second_concurrent = second_future.get();
        check(first_concurrent.converged && second_concurrent.converged,
              "solver API: concurrent 3-D solves converge");
        check(first_concurrent.equilibrium.families ==
                  second_concurrent.equilibrium.families,
              "solver API: concurrent 3-D solves are deterministic");

        return cumes::test::summary();
    } catch (const std::exception& error) {
        cumes::test::check(false, error.what());
        return cumes::test::summary();
    }
}
