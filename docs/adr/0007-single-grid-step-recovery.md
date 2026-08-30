# ADR-0007: Recover a reduced step after a stable single-grid window

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
`9.924e-13`. A controller change is Class C: reducing the residual is not
sufficient evidence because a different trajectory can converge to a
different member of the weak lambda-gauge family.

## Decision

Opt fixed-boundary single-grid solves into one conservative recovery per
stage. After 250 accepted passes since the latest restart, when `delt` remains
below its initial value, set

```
delt = min(delt0, 1.1 * delt)
```

The 250-pass window is ten radial-preconditioner periods. The attempt is made
at most once even if a later restart reduces the step again. Existing
residual-growth and oriented-Jacobian gates validate the new step and provide
the normal checkpoint rollback path.

Multigrid and free-boundary runs do not enable the policy, preserving their
qualified trajectories. `CUMES_DISABLE_STEP_RECOVERY=1` is a diagnostic A/B
opt-out for single-grid runs. The pure-host controller test covers the stable
window, disabled behavior, 10% cap, and one-attempt lifetime.

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

The worst family remains below `1e-4` and is within the established W7-X
independent-solver envelope. Solovev remains byte-for-byte on its
`251 -> 199 -> 456` trajectory and W7-X multigrid remains byte-for-byte on
`1877 -> 1617 -> 2011`.

A native RTX 4090 build on gervais reproduced `2953 -> 2711`. The thermally
stable median subset of the alternating, preheated measurements reduced total
wall time from 2.0583 s to 1.9217 s (6.63%); unlocked P-state outliers are
reported in `docs/performance.md` rather than omitted.

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
  failed the independent equilibrium comparison by up to `2.34e-3` and were
  rejected.

## Consequences

- The target solve performs 8.20% fewer full GPU passes and uses no additional
  device memory or synchronization.
- Default single-grid output is numerically equivalent, not trajectory
  identical, to the reference controller; it is explicitly a Class C result.
- Multigrid, free-boundary, and the diagnostic opt-out retain the prior
  controller behavior.
