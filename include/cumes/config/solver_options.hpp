// solver_options.hpp — validation/behavior options (blueprint §6.1).
#pragma once

#include "cumes/config/precision_policy.hpp"

namespace cumes {

struct SolverOptions {
    PrecisionPolicy precision = PrecisionPolicy::kVerifyDouble;
    // Strict mode rejects unknown JSON keys (schema v1 default). Compatibility
    // mode warns and preserves the vmecpp-style ignored-key behavior.
    bool strict_schema = false;
};

}  // namespace cumes
