# ADR-0008: Retain more interior shaping in fixed-boundary 3-D cold starts

- Status: Accepted
- Date: 2026-08-30

## Context

The reference cold start interpolates `m=0` between the supplied magnetic axis
and LCFS and multiplies every `m>0` boundary harmonic by `s^(m/2)`. This is
regular at the magnetic axis and exact at the boundary, but it suppresses the
interior non-axisymmetric shaping more strongly than the converged W7-X state.
The solver must spend early iterations rebuilding that shaping before entering
its asymptotic convergence regime.

An initial-state improvement must preserve axis regularity and the fixed LCFS.
It must not affect restart states, where the stored equilibrium is already the
best available guess.

## Decision

For fixed-boundary 3-D cold starts, use

```
w(m, s) = s^(m/2) * (1 + 0.12 * (1 - s))
```

for all `m>0` R/Z boundary families. The factor is one at the LCFS and finite
at the axis, so the exact boundary and the `s^(m/2)` regularity order are
unchanged. The `m=0` axis-to-boundary interpolation and zero-lambda seed remain
unchanged.

Axisymmetric and free-boundary cold starts keep the reference envelope.  The
later axisymmetric policy in ADR-0009 supersedes this for coarse fixed-boundary
starts.
`CUMES_SEED_ENVELOPE` overrides the coefficient for experiments; zero is the
diagnostic reference opt-out.

## Evidence

With the qualified one-shot step recovery already enabled:

| workload | reference seed | shaped seed | reduction |
| -------- | -------------: | ----------: | --------: |
| W7-X single-grid | 2711 | 2627 | 3.10% |
| W7-X multigrid | 1741 → 1568 → 1635 (4944) | 1315 → 1559 → 1633 (4507) | 8.84% |
| Solovev multigrid | 251 → 199 → 456 | 251 → 199 → 456 | 0% |

The shaped multigrid result ends at FSQR `9.967e-13`, FSQZ `1.563e-13`, FSQL
`4.956e-14`. Restarting its checkpoint on the final grid converges at iteration
1 with the identical residual triple.

Native RTX 4090 A/B measurements on gervais reduced median single-grid wall
time from 1.9238 s to 1.8823 s and multigrid wall time from 3.4419 s to
3.2983 s. GPU P-state outliers are reported in `docs/performance.md`; the
deterministic pass reductions are the primary timing evidence.

## Alternatives considered

- Uniform corrections from `-0.20` through `0.18` were swept. Negative values
  regressed, values near `0.20` inverted the initial geometry, and the response
  became trajectory-sensitive above the retained conservative value.
- An `m`-scaled correction was less effective for both W7-X schedules.
- Loose coarse-grid predictor stages generated interior lambda and missing
  modes, but their own passes cost more than they saved. The best direct
  predictor without seed shaping totaled 2800 passes versus 2711.
- Combining an 11-surface, 11–13-pass predictor with shaping reached 2577
  total passes, only 50 below the direct shaped seed, and adds stage/reporting
  complexity. It is not retained.

## Consequences

- W7-X spends fewer full GPU passes reconstructing interior shaping.
- No device memory, kernel, synchronization, checkpoint, or output format is
  added or changed.
- The default 3-D fixed-boundary cold-start trajectory is a Class C numerical
  change. Axisymmetric, free-boundary, restart, and explicit opt-out paths
  retain their prior seed.
