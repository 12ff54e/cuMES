# ADR-0007: Recover a reduced step after a stable fixed-boundary window

- Status: Accepted
- Date: 2026-08-30

## Context

The W7-X `ns=99` single-grid solve encounters five invalid-Jacobian transients
near startup. The reference VMEC_8_52 controller correctly restores the last
checkpoint and multiplies `delt` by 0.9 each time, but then retains the reduced
step for the remaining thousands of stable passes. The final reduction is
therefore useful for crossing the transient yet unnecessarily conservative in
the asymptotic regime.

The reference input converges in 2953 effective iterations at FSQR
`9.924e-13`. A controller change is Class C. cuMES's finite discrete force
residuals and geometry validity gates define convergence; independent solver
comparisons diagnose differences but are not the convergence oracle and do
not select a unique member of the weak lambda-gauge family.

## Decision

Opt every fixed-boundary stage into one conservative recovery. After 250
accepted passes since the latest restart, when `delt` remains
below its initial value, set

```
delt = min(delt0, 1.1 * delt)
```

The 250-pass window is ten radial-preconditioner periods. The attempt is made
at most once even if a later restart reduces the step again. Existing
residual-growth and oriented-Jacobian gates validate the new step and provide
the normal checkpoint rollback path.

Free-boundary runs do not enable the policy because the vacuum-coupled
trajectory requires separate qualification. `CUMES_DISABLE_STEP_RECOVERY=1`
is a diagnostic A/B opt-out for every fixed-boundary stage. The pure-host
controller test covers the stable window, disabled behavior, 10% cap, and
one-attempt lifetime.

## Evidence

For `inputs/w7x-single-grid.json` in precise double:

| policy | effective iterations | FSQR | FSQZ | FSQL |
| ------ | -------------------: | ---: | ---: | ---: |
| reference controller | 2953 | 9.924e-13 | 2.196e-13 | 2.242e-13 |
| one-shot recovery | 2711 | 9.988e-13 | 2.229e-13 | 2.255e-13 |

The iteration reduction is 242 passes (8.20%). An independent vmecpp 0.7.0
solve of the same input was converted with `compare_wout`; maximum absolute
differences, skipping the state-file-only axis row, were:

| family | max absolute difference |
| ------ | ----------------------: |
| rmncc | 2.583e-6 |
| zmnsc | 1.809e-6 |
| lmnsc | 1.676e-5 |
| rmnss | 1.499e-6 |
| zmncs | 1.587e-6 |
| lmncs | 1.547e-5 |

The worst family remains below `1e-4`. This independent comparison is a
diagnostic cross-check rather than the convergence criterion.

For `inputs/w7x.json`, recovery changes the stages from
`1877 -> 1617 -> 2011` (5505 total) to `1741 -> 1568 -> 1635` (4944 total), a
561-pass or 10.19% reduction. The final residual triple is FSQR `9.989e-13`,
FSQZ `1.590e-13`, FSQL `5.116e-14`. Restarting the resulting checkpoint on the
`ns=99` grid converges at iteration 1 with exactly the same residuals, proving
the result is a cuMES fixed point. Solovev remains byte-for-byte on its
`251 -> 199 -> 456` trajectory.

The recovered multigrid trajectory differs from vmecpp by up to `9.666e-5` in
R, `7.135e-5` in Z, and `5.796e-4` in the weak lambda families. These values
are reported for diagnosis; they do not override the satisfied cuMES residual,
Jacobian, and restart fixed-point criteria.

A native RTX 4090 build on gervais reproduced `2953 -> 2711`. The thermally
stable median subset of the alternating, preheated measurements reduced total
wall time from 2.0583 s to 1.9217 s (6.63%); unlocked P-state outliers are
reported in `docs/performance.md` rather than omitted.

The native gervais multigrid A/B reproduced both trajectories. Six alternating
runs reduced median wall time from 3.3109 s to 3.0021 s (9.33%).

## Alternatives considered

- Raising the initial step does not avoid the transient and was slower or
  unstable across the tested `0.6` to `2.0` range.
- Changing the preconditioner cadence gave a best result of 2853 iterations
  at interval 10 but regressed other intervals and changes more of the
  trajectory.
- A nonzero damping floor (`0.0025` through `0.08`) did not converge within
  5000 iterations.
- Repeated or periodic recovery caused recurring invalid Jacobians and failed
  to converge. Secant/tangent extrapolation also regressed convergence.
- More aggressive controller predicate changes reached fewer iterations but
  changed the equilibrium by up to `2.34e-3` without the present policy's
  conservative trigger and fixed-point evidence, and were rejected.

## Consequences

- The target single-grid and multigrid solves perform 8.20% and 10.19% fewer
  full GPU passes, respectively, with no additional device memory or
  synchronization.
- Default fixed-boundary output is converged but not trajectory identical to
  the diagnostic reference controller; it is explicitly a Class C result.
- Free-boundary runs and the diagnostic opt-out retain the prior controller
  behavior.
