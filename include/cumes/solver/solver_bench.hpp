// solver_bench.hpp — opt-in fixed-iteration benchmark observer (blueprint
// §8.1).
//
// The solver records one host wall-clock timestamp per *evaluated* pass at the
// single control fence (the one per-pass host sync already on the path). This
// is pure host-side observability: no arithmetic, scheduling, or stream-order
// change, so enabling it is Class A (the trajectory is bitwise-unchanged). The
// fixed-iteration harness discards the first `warmup` samples and reports
// median/p95 over the remainder.
#pragma once

#include <cstddef>
#include <vector>

namespace cumes {

struct SolverBench {
    // When false (default) the solver records nothing and the hot loop is
    // byte-identical to a production run.
    bool enabled = false;
    // Per-pass host wall-clock microseconds, one entry per evaluated pass
    // (control fence to control fence). Passes that exit early (maintenance
    // reset / bad-Jacobian / nonfinite restore) do not reach the fence and are
    // not recorded; in a steady-state fixed window there are no such passes.
    std::vector<double> pass_wall_us;
};

}  // namespace cumes
