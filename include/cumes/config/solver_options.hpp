// solver_options.hpp — validation/behavior options (blueprint §6.1).
#pragma once

#include "cumes/config/precision_policy.hpp"

namespace cumes {

struct SolverOptions {
    PrecisionPolicy precision = PrecisionPolicy::kVerifyDouble;
    // Strict schema (the schema-v1 default, completion plan step 2.1):
    // unknown JSON keys are a hard validation error. The named --compatibility
    // CLI policy clears this to restore the vmecpp-style warn-and-ignore
    // behavior.
    bool strict_schema = true;
};

}  // namespace cumes
