# ADR-0010: Seed free-boundary stages from the resolved geometry

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

## Evidence

All values below are precise-double results and meet every component of the
configured force tolerance:

| case | reference passes | seeded passes | final FSQR |
| ---- | ---------------: | ------------: | ---------: |
| CTH-like `ns=15` | 489 | 384 | `9.932e-11` |
| CTH-like `15 -> 25` | `242 -> 350` (592) | `240 -> 323` (563) | `9.942e-11` |
| W7-X `ns=51` | 1831 | 1797 | `9.705e-13` |
| Solovev `16 -> 32` | `417 -> 630` (1047) | `389 -> 636` (1025) | `9.822e-15` |

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
