// control_record.hpp — the scalar records exchanged between the device
// residual path and the pure host IterationController (blueprint §6.9, §6.10).
//
// These are plain trivially-copyable structs: the solver fills them from the
// per-fence device reductions, the controller reads them, and an observer may
// subscribe to them without ever seeing a device pointer.
#ifndef CUMES_INCLUDE_CUMES_SOLVER_CONTROL_RECORD_HPP_
#define CUMES_INCLUDE_CUMES_SOLVER_CONTROL_RECORD_HPP_

#include "cumes/solver/control_policy.hpp"

#include <cstdint>
#include <type_traits>

namespace cumes {

// vmecpp's restart reason taxonomy (Evolve / VMEC_8_52). The integer values are
// part of the frozen telemetry contract: the legacy per-iteration record
// writes `(int)reason` (0 = none, 1 = bad-Jacobian, 2 = bad-progress).
enum class RestartReason : std::uint8_t {
    NONE = 0,
    BAD_JACOBIAN = 1,
    BAD_PROGRESS = 2,
};

// Oriented Jacobian statistics (computeJacobianStats): min(signJ·√g),
// max|√g|, the nonfinite entry count, and the surface index of the minimum.
template <typename T>
struct JacobianStatus {
    using val_type = T;

    T min_oriented = T(0);     // min(signJ·√g); negative => sign flip
    T max_abs = T(0);          // max |√g|
    T nonfinite_count = T(0);  // number of non-finite entries
    int min_index = 0;         // flat index (jH*nZnT + point) of the minimum
};

// Result of classifying the invariant (unpreconditioned, normalized) residual
// triple. The controller owns the delt/restart-anchor bookkeeping for the
// nonfinite path; the solver only restores device state and continues.
struct InvariantVerdict {
    bool nonfinite = false;
    bool converged = false;
};

// Damping coefficients plus the intermediate telemetry scalars the legacy
// per-iteration record reproduces byte-for-byte.
template <typename T>
struct Damping {
    using val_type = T;

    T b1 = T(0);    // 1 - dtau
    T fac = T(0);   // 1 / (1 + dtau)
    T otav = T(0);  // 10-sample mean of 1/tau (telemetry)
    T dtau = T(0);  // delt * otav / 2 (telemetry)
};

// The controller's restart/refresh decision for one evaluated pass.
template <typename T>
struct RestartDecision {
    using val_type = T;

    RestartReason reason = RestartReason::NONE;
    bool do_refresh = false;  // refresh the checkpoint AFTER descent
    Damping<T> damping;
};

// ---------------------------------------------------------------------------
// Combined per-pass control record (blueprint §6.9; completion plan step 1.3)
// ---------------------------------------------------------------------------
// Device-visible validity/status flags. The finalize/predicate kernels write
// them; the guarded field/constraint/preconditioner/force/reduction kernels
// read them. Plain uint32s, no atomics: every writer and reader is ordered on
// the single compute stream, and the whole record is memset at pass start so
// no stale bit can leak across passes.
struct ControlStatus {
    std::uint32_t jacobian_valid = 0;       // finalize kernel: geometry usable
    std::uint32_t invariant_nonfinite = 0;  // terminal predicate
    std::uint32_t invariant_converged = 0;  // terminal predicate
    std::uint32_t preconditioned_evaluated = 0;  // set by the guarded reduction
    std::uint32_t force_norms_evaluated = 0;     // set by the guarded reduction
    std::uint32_t reserved = 0;                  // explicit tail for growth
};

// The trivially-copyable record the per-pass DAG reduces into and the host
// controller consumes after the single control fence. The numeric field order
// is the frozen 16-slot telemetry contract: Jacobian stats [0..3], invariant
// raw sums [4..6], preconditioned raw sums [7..9], force-norm partials
// [10..15]. One cudaMemcpyAsync of sizeof(ControlRecord) delivers the whole
// struct; the status bits travel with the numbers they describe.
struct ControlRecord {
    double jacobian_min_oriented = 0.0;         // min(signJ·√g)
    double jacobian_max_abs = 0.0;              // max |√g|
    double jacobian_nonfinite_count = 0.0;      // non-finite entry count
    double jacobian_min_index = 0.0;            // flat argmin index (as double)
    double invariant_raw[3] = {0.0, 0.0, 0.0};  // ΣF²/(mnmax·ns) per group
    double preconditioned_raw[3] = {0.0, 0.0, 0.0};  // after the preconditioner
    double force_norms[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    // {sRZ, sL, sMag, eTherm, vol, rzNorm} (before the deltaS scaling)
    // Device-finalized force-norm factors (completion-plan follow-up §2.3):
    // force_norm_finalize_kernel fills them from force_norms on refresh passes
    // (valid iff status.force_norms_evaluated). The device terminal predicate
    // consumes them for the convergence classification, and the host reads the
    // SAME fields at the fence instead of recomputing — the two decisions then
    // share bit-identical inputs, so a converged refresh pass no-ops the
    // in-place preconditioner exactly like any other terminal pass.
    double final_f_norm_rz = 0.0;
    double final_f_norm_l = 0.0;
    double final_f_norm1 = 0.0;
    ControlStatus status;
};

static_assert(std::is_trivially_copyable<ControlRecord>::value,
              "ControlRecord must be trivially copyable for the single D2H "
              "control transfer");
static_assert(std::is_trivially_copyable<ControlStatus>::value,
              "ControlStatus must be trivially copyable (device-visible)");

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_SOLVER_CONTROL_RECORD_HPP_
