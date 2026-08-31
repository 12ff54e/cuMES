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

Fixed-boundary axisymmetric stages also use a larger initial descent step than
the general 3-D controller:

| stage context | multiplier on configured `delt` |
| ------------- | ------------------------------: |
| single grid | `7/6` |
| first grid of a continuation | `17/15` |
| prolonged grid | `6/5` |

For Solovev's configured `delt=0.9`, these are 1.05, 1.02, and 1.08. The
first continuation grid is deliberately more conservative because it starts
from the analytic seed; later grids start from a converged prolongation.
Three-dimensional and free-boundary runs retain the configured step.
`CUMES_DELT0` remains an absolute diagnostic override of this policy.
Mixed-float axisymmetric runs also retain the configured step: applying the
double-tuned boost makes the finest stage oscillate at `2e-6` to `6e-6`, while
`delt=0.9` converges through a `1e-6` three-grid experiment in 285 passes.

Finally, initialize axisymmetric lambda from the cold R/Z geometry. Defining

```
q(theta) = sqrt(g) / R^2
         = (R_s Z_theta - R_theta Z_s) / R,
```

the straight-field-line relation gives

```
lambda_theta = 1 - q / <q>_theta.
```

Direct host quadrature projects this derivative into the resolved sine-family
lambda coefficients. The coupled R/Z state will still move, so the cold seed
uses 65% of the geometric predictor. A scale sweep from zero through two has
a broad optimum from 0.64 through 0.655. Invalid initial geometry rejects the
predictor atomically and falls back to zero lambda. The diagnostic override
`CUMES_AXISYM_LAMBDA_SEED=0` restores the zero-lambda reference.

## Evidence

In isolation, the seed changes the three-grid Solovev trajectory from
`251 -> 199 -> 456` (906 passes) to `241 -> 199 -> 455` (895 passes), a 1.21%
reduction.  Every stage still satisfies its configured `1e-16` force-residual
tolerance; final FSQR is `9.999e-17`.  The final state differs from the
reference trajectory by at most `2.874e-10` relatively and is therefore the
same equilibrium to well within the project verification threshold.

Combined with the staged initial-step policy, multigrid converges in
`238 -> 193 -> 387` (818 passes), 9.71% below the 906-pass reference. Final
FSQR is `9.781e-17`; checkpoint replay on the final grid converges at iteration
1 with a bit-identical state and residual triple. The generic transform
backend gives the same 818 decisions (FSQR `9.789e-17`).

Adding the geometric lambda predictor reduces the coarse stage by three more
passes, producing `235 -> 193 -> 387` (815 total), 10.04% below the reference;
final FSQR is `9.792e-17`. The generic backend gives the same 815 decisions
(FSQR `9.788e-17`), and final-grid checkpoint replay remains bit-identical at
iteration 1. The new and 818-pass final states differ by at most `9.400e-12`.

For a cold single `ns=55` grid, the reference seed plus the `7/6` step reduces
533 passes to 416 (21.95%), final FSQR `9.691e-17`. Its final state differs
from the old trajectory by at most `5.626e-8` absolutely, while satisfying the
same force tolerance; the multigrid fixed-point replay is the stronger
equilibrium check.

With the geometric predictor, the cold single `ns=55` solve falls further to
354 passes, 33.58% below the original 533, with FSQR `9.835e-17`.

A family- and mode-specific fit to the converged coarse state was also tested.
It required 244 coarse passes and was rejected as both slower and less
general than the single correction.

A native precise-double sm_89 build on gervais reproduced the exact final pass
counts. GPU P-state and process-startup outliers are large at this subsecond
scale, so the deterministic 10.04%/33.58% pass reductions are the primary
evidence.

## Consequences

- Coarse axisymmetric continuation starts require fewer GPU passes.
- Axisymmetric cold starts pay a small host-only quadrature cost once and
  begin with a nonzero straight-field-line lambda predictor.
- Single-grid fine starts retain the reference R/Z seed but use the tuned
  step and lambda predictor.
- Free-boundary starts, 3-D starts, and checkpoint contents retain their
  existing policies.
- Mixed-float starts use the same geometric seeds but retain their configured
  time step.
- This is a Class C trajectory change judged by configured force convergence,
  not by reproduction of the previous iteration count.
