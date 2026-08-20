# ADR-0004 — axisymmetric transform backend wired into the Solovev production path

Status: accepted and active (blueprint §8.5)

## Context

The overhaul built `cumes::AxisymmetricOperator` — a direct-poloidal
synthesis/projection backend for `ntor=0, nzeta=1` that elides the generic
backend's length-one cuFFT Z2D/D2Z round trips — and `test_axisym_backend`
proved it Class B ULP-equivalent to the generic `inverseDFTFused`/`forwardDFT`/
`constraintDealiasBandpass` on identical frozen inputs. It was left *built but
unwired*: `solverRun` always called the generic backend, so the Solovev shape
paid the full submission cost of ~24 launches/pass dominated by length-one
cuFFT transforms. It was initially left unwired, then promoted to the
production Solovev path and used for the real-pass CUDA-graph measurement in
ADR-0003.

## Decision

Wire the operator into the production hot loop for the axisymmetric shape:

- `StageSolver::run` constructs one `AxisymmetricOperator<T>` per stage when
  `ntor==0 && nzeta==1` (the operator's poloidal tables are ns-independent, but
  its kernels launch on `p.ns`, so one is built per stage; re-uploading the
  ~430-element tables is negligible) and passes it to `solverRun`.
- `solverRun` selects the backend per pass: `enqueue_inverse` + `enqueue_rzcon`
  replace `inverseDFTFused`; `enqueue_forward` replaces `forwardDFT`; and a new
  `constraintComputeAxisym` replaces `constraintCompute`'s step-2
  `constraintDealiasBandpass` with `enqueue_dealias` (steps 0/1/3 are shared,
  extracted as `constraintComputeHead`/`constraintComputeTail`).
- `CUMES_FORCE_GENERIC=1` restores the generic backend for A/B runs; the
  generic path is byte-for-byte unchanged, so the frozen baseline stays
  reproducible.
- `cumes_benchmark_fixed_iteration` mirrors the production decision and now
  reports the *actual* `transform_backend` (the previous `axisymmetric-available`
  label was a stale placeholder that did not reflect what ran).

## Measured result (TITAN Xp, sm_61, Solovev ns=55, 200 passes / 20 warmup)

| backend     | median | p95    |
| ----------- | ------ | ------ |
| toroidal-fft (generic) | 214.59 µs/iter | 245.28 µs/iter |
| axisymmetric           | 153.23 µs/iter | 180.89 µs/iter |

A **28.6%** median wall-time reduction on the submission-bound axisymmetric
shape, above the §10.7 `max(5%, noise floor)` acceptance floor.

## Consequences

- **Class B** for the Solovev trajectory: the axisymmetric run converges
  `251 → 199 → 456` with `FSQR 9.583e-17` — the *same* iteration counts and
  restart sequence as the frozen generic baseline — and the final state agrees
  at ULP level (max |Δ| ≈ 1e-13 across the six families). No controller decision
  changes, so the re-freeze is a documented trajectory member, not a divergence.
- **Class A (bit-identical) for W7-X**: `ntor=12/nzeta=36` never selects the
  axisymmetric backend, so the W7-X trajectory is byte-for-byte the frozen
  `1877 → 1617 → 2011 (5505)`, `FSQR 9.778e-13`.
- The generic backend remains selectable via `CUMES_FORCE_GENERIC=1` and is the
  differential oracle in `test_axisym_backend`.
- The ADR-0003 real-pass re-measurement is complete. It confirmed a modest
  axisymmetric benefit, but production graph integration remains deferred on
  complexity and portability grounds.

## Alternatives considered

- **Construct the operator once per run and update `ns` per stage.** Rejected
  as over-engineered for a pedagogical codebase: the tables are ~1.7 KB of
  doubles; the per-stage reconstruction is negligible setup cost and keeps the
  operator's `p_` snapshot consistent with the stage it launches for.
- **Make the enqueue methods `const` and pass a const pointer.** Rejected for
  this change: the operator is the active backend; non-const is semantically
  honest and avoids a larger public-API edit that `test_axisym_backend` would
  also have to track.
- **Leave it unwired.** Rejected: the measured 28.6% is a free, low-risk win on
  the primary axisymmetric shape, and it unblocks the graph question.
