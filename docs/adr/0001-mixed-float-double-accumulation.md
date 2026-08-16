# ADR-0001 — double accumulation for norm reductions (mixed-float)

Status: accepted (Phase 9, blueprint §8.8/§8.12)

## Context

The float build (`CUMES_USE_FLOAT=ON`) stalls at a ~1e-7 residual floor. The
control-feeding reductions — `computeResidualsKernel` (invariant + preconditioned
residuals), `rzNormKernel` (decomposed R/Z state norm), and
`forceNormReduceKernel` (force-norm partials) — accumulated their sums in `T`,
so a float build summed up to `mnmax*ns` (~1.5e4 for W7-X) terms with per-add
float rounding. The summation error is independent of the float-state floor and
is the cheapest precision win available.

## Decision

Introduce `cumes::NormAccum<T>` (`include/cumes/numerics/accumulation.hpp`):

- `NormAccum<float>::type = double` — mixed-float accumulates norms in double;
- `NormAccum<double>::type = double` — the verified double build is unchanged.

The three kernels accumulate in `NormAccum<T>::type` (their shared-memory trees
are widened to match) and still store the scalar into the `T` control record.
The *terms* keep their `T` arithmetic; only the summation width changes. This is
the minimal reading of §8.8's "accumulation type is a policy, defaulting to
double for norms" — a full double `ControlRecord` (double invariants threaded
through the host controller) is a separate, larger follow-up.

## Consequences

- **Class A** for the double build: `NormAccum<double>::type == double`, so every
  widened expression is an identity and the Solovev trajectory is bit-identical
  (251 → 199 → 456, FSQR 9.583e-17).
- **Class B** for the float build: the summation width changes. On a 1M-term
  dynamic-range case (`test_accumulation.cu`) float accumulation errs by 4.6e1
  while double accumulation errs by 2.3e0 against an extended-precision
  reference — a ~20× reduction in summation error.
- It does **not** unlock lower float `ftol`: the float-state rounding floor
  (forces/state are float-precise) still dominates. Double accumulation makes
  the *reported* residuals more faithful, not the physics more precise.

## Alternatives considered

- **Full double `ControlRecord`** (`double invariant[3]` etc. threaded through the
  host controller): strictly better, but it changes the host/controller scalar
  types and the `DeviceBuffer<T>`/`PinnedBuffer<T>` control path. Deferred as a
  follow-up; the accumulation-only change captures most of the benefit cheaply.
- **Always-float accumulation**: the prior default; rejected because the
  summation error is avoidable and indistinguishable from a real residual
  signal in a float build.

## Status update (2026-08-16): full double `ControlRecord` landed

The deferred follow-up is implemented: the per-pass control record — the 16
scalars {Jacobian stats [0..3], invariant residuals [4..6], preconditioned
residuals [7..9], force-norm factors [10..15]} — is now **double in both
builds**, so the double accumulations reach the host controller unrounded:

- Device stores widen: `computeResidualsKernel`, `forceNormReduceKernel`,
  `rzNormKernel` (already accumulating in `NormAccum<T>::type` = double for
  mixed-float) now store `double` instead of rounding to `T`;
  `jacobianStatsKernel` stores its (T-accumulated) stats into the double
  record unchanged. `EquilibriumOperator::d_control_` is a
  `DeviceBuffer<double>`; the pinned D2H mirror is `PinnedBuffer<double>`.
- Host derivation widens: `plainPerEl`, the force-norm normalization factors
  (`finalizeForceNorms`), the invariant/preconditioned triple arithmetic, and
  the Jacobian gate all run in double; the controller is
  `IterationController<double>` in both builds (its `delta_t`/`fsqz_prev`
  accessors and all decisions are double).
- The descent physics stays `T`: the existing double `DescentAction` casts to
  `T` at the kernel boundary exactly as before. `SolverResult<T>` stays `T`
  (output-facing; float builds now report the float-rounded double residual).

Verification:

- **Class A for the double build** — every widened expression is an identity
  when `T = double`: Solovev `251→199→456` FSQR 9.583e-17 and W7-X
  `1877→1617→2011` FSQR 9.778e-13 **bit-identical** (compare_runs.py,
  0.000e+00 max rel diff on all six families, identical restart sequences,
  converged-iter delta 0), 35/35 CTest.
- **Class B for the float build** — 23/23 CTest; an end-to-end float smoke
  (relaxed `ftol_array` to 1e-6) converges at FSQR 8.933e-07, and its
  controller decisions now derive from the double accumulations instead of
  the float-rounded record.

The durable limitation from the original ADR is unchanged: this does not
unlock a lower float `ftol` — the float-state rounding floor (~1e-7) still
dominates the physics. The win is a faithful control signal: the convergence
test, damping log-ratio, and restart thresholds see the true double-precision
residuals rather than their float-rounded surrogates.
