# ADR-0009: Tune the fixed-boundary axisymmetric start policy

- Status: Accepted
- Date: 2026-08-30

## Context

ADR-0008 improved the fixed-boundary 3-D cold start but deliberately left the
axisymmetric trajectory unchanged.  The same residual-driven measurement on
Solovev shows that its coarse-grid equilibrium has slightly less interior
shaping than the reference `s^(m/2)` interpolation.  Starting from the
reference envelope therefore spends several full passes removing shaping
before approaching the coarse-grid fixed point.

Any axisymmetric seed change must retain the exact magnetic axis and LCFS,
the `s^(m/2)` regularity order, the reference fine-grid cold start, and the
free-boundary trajectory.

## Decision

For a fixed-boundary axisymmetric cold start whose first grid has at most 11
surfaces, use

```
w(m, s) = s^(m/2) * (1 - 0.07 * (1 - s))
```

for the `m>0` R/Z boundary families.  Finer axisymmetric cold starts retain
the reference envelope.  `CUMES_SEED_ENVELOPE=0` remains the explicit
reference opt-out and can select another coefficient for diagnostics.

## Evidence

On the three-grid Solovev fixture, the seed changes the trajectory from
`251 -> 199 -> 456` (906 passes) to `241 -> 199 -> 455` (895 passes), a 1.21%
reduction.  Every stage still satisfies its configured `1e-16` force-residual
tolerance; final FSQR is `9.999e-17`.  The final state differs from the
reference trajectory by at most `2.874e-10` relatively and is therefore the
same equilibrium to well within the project verification threshold.

A family- and mode-specific fit to the converged coarse state was also tested.
It required 244 coarse passes and was rejected as both slower and less
general than the single correction.

## Consequences

- Coarse axisymmetric continuation starts require fewer GPU passes.
- Single-grid fine starts, free-boundary starts, checkpoints, and 3-D starts
  retain their existing policies.
- This is a Class C trajectory change judged by configured force convergence,
  not by reproduction of the previous iteration count.
