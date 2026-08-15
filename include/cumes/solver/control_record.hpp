// control_record.hpp — the scalar records exchanged between the device
// residual path and the pure host IterationController (blueprint §6.9, §6.10).
//
// These are plain trivially-copyable structs: the solver fills them from the
// per-fence device reductions, the controller reads them, and an observer may
// subscribe to them without ever seeing a device pointer.
#pragma once

#include <cstdint>

namespace cumes {

// vmecpp's restart reason taxonomy (Evolve / VMEC_8_52). The integer values are
// part of the frozen telemetry contract: the legacy per-iteration record
// writes `(int)reason` (0 = none, 1 = bad-Jacobian, 2 = bad-progress).
enum class RestartReason : std::uint8_t {
    kNone = 0,
    kBadJacobian = 1,
    kBadProgress = 2,
};

// Oriented Jacobian statistics (computeJacobianStats): min(signJ·√g),
// max|√g|, the nonfinite entry count, and the surface index of the minimum.
template <typename T>
struct JacobianStatus {
    T min_oriented = T(0);    // min(signJ·√g); negative => sign flip
    T max_abs = T(0);         // max |√g|
    T nonfinite_count = T(0); // number of non-finite entries
    int min_index = 0;        // flat index (jH*nZnT + point) of the minimum
};

// Result of classifying the invariant (unpreconditioned, normalized) residual
// triple. The controller owns the delt/restart-anchor bookkeeping for the
// nonfinite path; the solver only restores device state and continues.
template <typename T>
struct InvariantVerdict {
    bool nonfinite = false;
    bool converged = false;
};

// Damping coefficients plus the intermediate telemetry scalars the legacy
// per-iteration record reproduces byte-for-byte.
template <typename T>
struct Damping {
    T b1 = T(0);    // 1 - dtau
    T fac = T(0);   // 1 / (1 + dtau)
    T otav = T(0);  // 10-sample mean of 1/tau (telemetry)
    T dtau = T(0);  // delt * otav / 2 (telemetry)
};

// The controller's restart/refresh decision for one evaluated pass.
template <typename T>
struct RestartDecision {
    RestartReason reason = RestartReason::kNone;
    bool do_refresh = false;  // refresh the checkpoint AFTER descent
    Damping<T> damping;
};

// The full per-pass scalar record. Phase 4 feeds the controller across two
// fences (the Jacobian gate, then the invariant/preconditioned residuals); the
// single-fence `advance(ControlRecord)` form is the Phase 6A target.
template <typename T>
struct ControlRecord {
    T invariant[3];       // fsqr_i, fsqz_i, fsql_i (normalized)
    T preconditioned[3];  // fsqr, fsqz, fsql (normalized)
    JacobianStatus<T> jacobian;
};

}  // namespace cumes
