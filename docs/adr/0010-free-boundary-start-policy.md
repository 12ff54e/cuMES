# ADR-0010: Tune free-boundary stage starts

- Status: Accepted
- Date: 2026-08-31

## Context

Free-boundary convergence is defined by the configured finite force residuals,
not by reproducing another solver's trajectory. The reference cold start uses
the regular `s^(m/2)` interpolation and zero lambda. Measurements on both a
coarse CTH-like 3-D continuation and the `ns=51` W7-X free-boundary case show
that a geometry closer to the interior equilibrium reduces full vacuum-coupled
passes, but that fine 3-D grids cannot tolerate the coarse predictor's full
correction without additional startup restarts.

Axisymmetric free-boundary geometry supports the same analytical
straight-field-line lambda predictor as the fixed-boundary path. Because the
boundary subsequently moves, its usefulness must be established independently.

## Decision

Keep the exact axis and LCFS and apply the existing regular envelope

```
w(m, s) = s^(m/2) * (1 + c * (1 - s))
```

to 3-D free-boundary cold starts. Use `c=0.12` when the initial grid has at
most 25 radial surfaces and the conservative `c=0.03` on finer cold starts.
The resolution split reflects the observed startup stability boundary rather
than the machine or input name. `CUMES_SEED_ENVELOPE=0` remains the reference
opt-out.

For axisymmetric free-boundary cold starts, retain the reference R/Z envelope
and use the full geometry-derived lambda predictor (`scale=1.0`).
`CUMES_AXISYM_LAMBDA_SEED=0` restores zero lambda. Checkpoint restarts are
unchanged.

Double-precision 3-D free-boundary stages with at most 25 radial surfaces use
`17/14` times the configured initial descent step. For CTH-like `delt=0.7`,
this is 0.85. Finer, axisymmetric, and mixed-float free-boundary stages retain
the configured step. `CUMES_DELT0` remains an absolute diagnostic override.

Activate the vacuum edge force when `FSQR + FSQZ < 3e-2`, rather than
over-solving the temporary no-vacuum force model to `1e-3`. The activation
still performs the existing velocity reset, damping re-anchor, full vacuum
update, and residual/Jacobian validation. `CUMES_VACUUM_ACTIVATION_THRESHOLD`
is a diagnostic override; `1e-3` restores the reference handover gate.

## Evidence

All values below are precise-double results and meet every component of the
configured force tolerance:

| case | reference passes | seeded passes | final FSQR |
| ---- | ---------------: | ------------: | ---------: |
| CTH-like `ns=15` | 489 | 384 | `9.932e-11` |
| CTH-like `15 -> 25` | `242 -> 350` (592) | `240 -> 323` (563) | `9.942e-11` |
| W7-X `ns=51` | 1831 | 1797 | `9.705e-13` |
| Solovev `16 -> 32` | `417 -> 630` (1047) | `389 -> 636` (1025) | `9.822e-15` |

Combining the coarse 3-D seed with the `17/14` step reduces CTH-like
single-grid further to 347 passes (29.0% below reference, FSQR `9.359e-11`)
and multigrid to `208 -> 238` = 446 passes (24.7% below reference, FSQR
`9.745e-11`). The step policy does not select W7-X `ns=51` or axisymmetric
Solovev, so their qualified seeded trajectories remain unchanged.

The earlier vacuum handover further reduces CTH-like single-grid to 309
passes (FSQR `9.635e-11`) and multigrid to `198 -> 233` = 431 passes (FSQR
`9.885e-11`). W7-X falls from the seeded 1797 passes to 1733 (FSQR
`9.996e-13`); Solovev remains exactly on its 1025-pass seeded trajectory. A
threshold sweep established a broad W7-X minimum from `3e-2`
through `4e-2`; `3e-2` is the conservative edge. CTH-like also selects
`3e-2` (309 single-grid and 431 multigrid) rather than `2e-2` (314) or `4e-2`
(340).

For W7-X, applying the coarse `0.12` correction caused extra startup restarts
and regressed to 1953 passes; the fine-grid `0.03` correction is therefore a
deliberate stability limit. A full fixed-boundary predictor was also rejected:
1449 predictor passes plus 1723 free-boundary passes exceeds the direct solve.

## Consequences

- Free-boundary cold starts perform only extra host arithmetic already present
  in state construction; the per-pass GPU and vacuum DAGs are unchanged.
- Default free-boundary trajectories change (Class C), but convergence remains
  governed by the residual and oriented-Jacobian gates.
- The policy is resolution-based and applies equally to CTH-like and W7-X-like
  inputs.
- Vacuum pressure turns on earlier, but the same edge-force residual,
  factorization, and geometry-validity gates qualify every subsequent pass.
