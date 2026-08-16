// device_params.hpp — the compact per-stage device parameter pack (blueprint
// §6.1). This is the "fourth stage" of the config pipeline: ProblemSpec ->
// ValidationReport -> ValidatedProblem -> DeviceParams<T>. It carries exactly
// the trivially-copyable constants and scalar knobs the GPU kernels index on —
// no pointers, no owning storage, no host-only vectors.
//
// It replaces the legacy `GridParams<T>` (vmec_types.h) field-for-field: the
// extents (ns/mnmax/ntheta/nzeta/nfp/nZnT/mpol/ntor) plus the run knobs
// (ncurr/delt/ftol/max_iter/tcon0/lamscale) and the two physical constants
// (kSignJacobian/kMu0). The type stays in the global namespace, matching the
// legacy placement, so the 65 kernel/operator headers that named GridParams
// keep resolving to the same symbol after the rename.
#pragma once

// Self-contained: M_PI (used by kMu0 below) is a POSIX/GNU extension, not ISO
// C++. Host-only .cpp TUs compiled by g++ (rather than nvcc, which pre-defines
// it) need it visible here before any other header includes <cmath>. The
// fallback is glibc's exact double value, so kMu0 is bit-identical on every
// toolchain.
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// The computation scalar type, templated on T throughout; the app-level
// compile-time switch between double and float is `Real` (vmec_types.h).
template <typename T>
struct DeviceParams {
    int ns; int mnmax; int ntheta; int nzeta;
    int nfp; int nZnT; int mpol; int ntor;
    // Runtime input knobs (host-side; the validated problem fills them).
    int ncurr;              // 0: prescribed iota, 1: prescribed current
    T delt = T(0.9);        // initial time step
    T ftol = T(1e-16);      // convergence tolerance (invariant residuals)
    int max_iter = 1000;
    T tcon0 = T(1.0);       // constraint-force multiplier (vmecpp indata
                            // tcon0; scales the tcon profile in constraint.cu)
    T lamscale = T(0.0);    // sqrt(deltaS * sum phipH^2), vmecpp constants_,
                            // set by profilesCreate
    static constexpr int kSignJacobian = -1;
    static constexpr T kMu0 = 4.0 * M_PI * 1.0e-7;  // exact, = vmecpp MU_0
};
