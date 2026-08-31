#include "cumes/config/json_reader.hpp"
#include "cumes/config/validated_problem.hpp"
#include "cumes/solver/equilibrium_solver.hpp"
#include "cumes_test.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
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
        check(!outcome.report.input_params.rbc_value.empty() &&
                  outcome.report.input_params.rbc_value.front() ==
                      validated.value().spec().rbc.front().value,
              "solver API: report embeds normalized input");
        check(outcome.report.stages.size() == 1,
              "solver API: report contains the stage");
        check(std::isfinite(outcome.fsqr) && std::isfinite(outcome.fsqz) &&
                  std::isfinite(outcome.fsql),
              "solver API: residuals are finite");

        cumes::SolveRequest restart_request;
        restart_request.restart = std::cref(outcome.equilibrium);
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

        return cumes::test::summary();
    } catch (const std::exception& error) {
        cumes::test::check(false, error.what());
        return cumes::test::summary();
    }
}
