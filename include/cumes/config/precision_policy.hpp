// precision_policy.hpp — explicit precision policies (blueprint §8.12).
//
// Replaces the single CUMES_USE_FLOAT preprocessor alias with named policies
// and a tolerance floor derived from the scalar type, so an impossible
// tolerance (float ftol < 1e-6) is a validation error rather than a run that
// silently stalls at the float rounding floor.
#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

namespace cumes {

enum class PrecisionPolicy : std::uint8_t {
    kVerifyDouble = 0,  // double state/geometry/FFT/reduction, precise math
    kFastDouble = 1,    // double + selected fast intrinsics (opt-in)
    kMixedFloat =
        2,  // float state/geometry/FFT, double reductions (experimental)
    kDebugDouble = 3,  // double + precise + device checks
};

// The lowest tolerance a policy can meaningfully meet. Double reaches the
// ~1e-16 residual floor (the shipped configs request 1e-16 and converge);
// float stalls at ~1e-7, and the legacy float gate used 1e-6 as a safety
// margin.
inline double tolerance_floor(PrecisionPolicy policy) {
    switch (policy) {
        case PrecisionPolicy::kVerifyDouble:
        case PrecisionPolicy::kFastDouble:
        case PrecisionPolicy::kDebugDouble:
            return 1e-16;
        case PrecisionPolicy::kMixedFloat:
            return 1e-6;
    }
    return 1e-16;
}

// Is `ftol` achievable under `policy`? False means validation must reject it
// (or the caller opts into "iterate to stagnation" explicitly).
inline bool tolerance_achievable(double ftol, PrecisionPolicy policy) {
    if (!(ftol > 0.0)) return false;
    return ftol >= tolerance_floor(policy);
}

}  // namespace cumes
