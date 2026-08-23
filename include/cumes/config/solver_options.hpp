// solver_options.hpp — validation/behavior options (blueprint §6.1).
#ifndef CUMES_INCLUDE_CUMES_CONFIG_SOLVER_OPTIONS_HPP_
#define CUMES_INCLUDE_CUMES_CONFIG_SOLVER_OPTIONS_HPP_

#include "cumes/config/precision_policy.hpp"

namespace cumes {

struct SolverOptions {
    PrecisionPolicy precision = PrecisionPolicy::VERIFY_DOUBLE;
    // Strict schema (the schema-v1 default, completion plan step 2.1):
    // unknown JSON keys are a hard validation error. The named --compatibility
    // CLI policy clears this to restore the vmecpp-style warn-and-ignore
    // behavior.
    bool strict_schema = true;
};

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_CONFIG_SOLVER_OPTIONS_HPP_
