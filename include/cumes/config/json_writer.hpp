// json_writer.hpp — ProblemSpec -> strict, read-back-compatible input JSON.
#ifndef CUMES_INCLUDE_CUMES_CONFIG_JSON_WRITER_HPP_
#define CUMES_INCLUDE_CUMES_CONFIG_JSON_WRITER_HPP_

#include "cumes/config/problem_spec.hpp"

#include <string>

namespace cumes {

// Serialize the flat input schema accepted by parse_problem_spec(). Unlike
// ValidatedProblem::normalize_to_json(), this representation intentionally
// omits derived/folded records and can be consumed as a subsequent solve.
std::string problem_spec_to_json(const ProblemSpec& problem);

// Write problem_spec_to_json(problem) to path, throwing std::runtime_error on
// an open or write failure.
void write_problem_spec(const std::string& path, const ProblemSpec& problem);

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_CONFIG_JSON_WRITER_HPP_
