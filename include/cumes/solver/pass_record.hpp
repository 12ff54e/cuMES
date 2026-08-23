// pass_record.hpp — the structured per-pass scalar telemetry record.
//
// This is the typed form of the legacy 15-column `per_iter_residuals_cumes.bin`
// trajectory record. The field order IS the frozen on-disk contract (the dump
// writer serializes the fields column-major, byte-identically to the legacy
// array-of-doubles). Observers consume PassRecord values and can never affect
// the controller's decisions — they see only scalars this pass already
// produced.
#ifndef CUMES_INCLUDE_CUMES_SOLVER_PASS_RECORD_HPP_
#define CUMES_INCLUDE_CUMES_SOLVER_PASS_RECORD_HPP_

#include <type_traits>

namespace cumes {

// One evaluated pass's control scalars. All 15 fields are double (the legacy
// on-disk layout stores iter2/iter1/reason as doubles), contiguous and
// standard-layout so the column-major serializer can walk them by offset.
struct PassRecord {
    double invariant_fsqr = 0.0;       // fsqr_i (unpreconditioned)
    double invariant_fsqz = 0.0;       // fsqz_i
    double invariant_fsql = 0.0;       // fsql_i
    double preconditioned_fsqr = 0.0;  // fsqr
    double preconditioned_fsqz = 0.0;  // fsqz
    double preconditioned_fsql = 0.0;  // fsql
    double delta_t = 0.0;              // time step of the pass
    double otav = 0.0;                 // 10-sample mean of 1/tau
    double dtau = 0.0;                 // delt * otav / 2
    double b1 = 0.0;                   // 1 - dtau
    double fac = 0.0;                  // 1 / (1 + dtau)
    double iter2 = 0.0;                // effective iteration
    double iter1 = 0.0;                // restart anchor
    double reason = 0.0;               // RestartReason as int (0/1/2)
    double axis_r = 0.0;               // axis R at zeta=0, pre-descent

    static constexpr int COLUMN_COUNT = 15;
};

// The serializer relies on the fields being exactly 15 contiguous doubles with
// no padding; enforce that once at compile time.
static_assert(std::is_standard_layout_v<PassRecord>);
static_assert(sizeof(PassRecord) == PassRecord::COLUMN_COUNT * sizeof(double));

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_SOLVER_PASS_RECORD_HPP_
