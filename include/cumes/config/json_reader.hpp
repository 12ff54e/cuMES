// json_reader.hpp — vmecpp-style JSON -> ProblemSpec (blueprint §6.1).
//
// The host-only JSON adapter. It maps a flat vmecpp indata document onto the
// dynamic ProblemSpec, reproducing the legacy parser's defaults and folding.
// Unlike the legacy parser (which threw on the first error), it records type
// errors, integer narrowing, unsupported features, and unknown keys into a
// ValidationReport so one pass reports every finding. JSON syntax/file errors
// still throw (there is no document to validate).
#pragma once

#include "cumes/config/problem_spec.hpp"
#include "cumes/config/validated_problem.hpp"

#include <string>

namespace cumes {

struct ParsedProblem {
    ProblemSpec spec;
    ValidationReport report;
};

// Parse + map a JSON document. Throws std::runtime_error on a missing file or
// a JSON syntax error; records semantic/type findings in `report`.
ParsedProblem read_problem_spec(const std::string& path,
                                const SolverOptions& options);

// parse + validate in one call (the CLI entry point). Throws std::runtime_error
// (path-prefixed) on JSON syntax/file errors; returns the immutable model, or
// the combined report (mapping + validation) on any error.
ValidationResult read_and_validate(const std::string& path,
                                   const SolverOptions& options);

}  // namespace cumes
