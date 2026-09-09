# Convergence-speed experiments, September 2026

Baseline: `9316169`, following the Class-A transform optimizations. These
experiments permit different convergence trajectories but retain cuMES's
residual and geometry gates. Timings below measure the sum of stage CUDA-event
solve intervals, including host submission gaps; they exclude startup and
output. They are exploratory measurements, not the paired confidence-interval
qualification used for the transform changes.

## Grid scheduling

On gervais, RTX 4090 GPU 2, CUDA 12.9, native `sm_89`, five repeated runs of
each schedule gave the following median times. Explicitly generated temporary
JSON inputs selected the schedules; the shipped inputs were unchanged.

| Case | Schedule and coarse tolerances | Effective iterations | Device ms |
| --- | --- | --- | ---: |
| W7-X | 33/66/99, 1e-12/1e-12 | 1315/1419/1372 | 1695.220 |
| W7-X | 33/66/99, 1e-6/1e-8 | 195/294/2516 | 1485.617 |
| W7-X | 33/66/99, 1e-4/1e-4 | 119/57/2665 | 1458.719 |
| W7-X | 99 only | 2465 | 1285.620 |
| Solovev | 5/11/55, 1e-16/1e-16 | 235/193/326 | 67.635 |
| Solovev | 5/11/55, 1e-6/1e-8 | 76/60/326 | 44.702 |
| Solovev | 5/11/55, 1e-4/1e-4 | 45/2/364 | 40.728 |
| Solovev | 55 only | 354 | 33.642 |

All cases retain the original final tolerance: 1e-12 for W7-X and 1e-16 for
Solovev. All sixteen tested schedules met every final residual tolerance and
replayed their checkpoints at iteration 1. Coefficient values were unchanged
by replay; signed-zero bits and the recomputed residuals can differ.

Single-grid solves are already supported: select the last element of
`ns_array`, `niter_array`, and `ftol_array`. The W7-X single-grid policy also
uses its previously qualified seed envelope 0.129, versus 0.12 for multigrid.
Forcing envelope 0.12 increases its single-grid count from 2465 to 2627;
skipping coarse solves still helps. The existing defaults need not be the
fastest schedule for every input. Caller-specified intermediate tolerances
must not be silently weakened.

Changing the schedule changes W7-X's final state beyond roundoff. For coarse
1e-6/1e-8, maximum absolute differences from the baseline include Rcc 1.65e-4,
Rss 4.59e-4, Zcs 6.48e-4, and lambda_cs 2.84e-3. Native-coordinate relative
RMS differences include sqrt(g) 6.88e-4, B^zeta 3.66e-5, and B^theta 5.86e-3.
Single-grid lambda_cs differs by up to 3.30e-3. These are coordinate-sensitive
comparisons, not physical error bounds. Solovev is less sensitive: the coarse
1e-6/1e-8 schedule differs by about 1e-9 in R/Z and 3.2e-9 in lambda.

## Safeguarded residual extrapolation — rejected

A GPU prototype retained four matching spectral states and residuals, formed
three residual differences, and solved a regularized weighted least-squares
problem for a Pulay extrapolation. It preserved fixed R/Z boundaries and
dependent axis entries, bounded extrapolation coefficients, and evaluated
each trial through the full solver. Invalid geometry or insufficient reduction
of the invariant residual sum restored both the ordinary descent state and
its momentum before any controller decision. Accepted trials reset momentum
and its damping history. All scratch was allocated at stage construction;
trial status used the existing control fence.

The design is related to residual-subspace acceleration described by
[Walker and Ni (2011)](https://epubs.siam.org/doi/10.1137/10078356X).
The acceptance check is a local safeguard, not a convergence theorem.

On the TITAN Xp with CUDA 12.1, precise double arithmetic, experiments varied
raw versus preconditioned residuals, sample spacing from 2 to 50 iterations,
proposal intervals from 8 to 200, required merit reduction from 0.1% to 5%,
and whether history could cross preconditioner refreshes. Representative counts:

| Variant | Solovev effective iterations | W7-X effective iterations |
| --- | ---: | ---: |
| Baseline | 754 | 4106 |
| Raw, spacing 2, interval 8, 5% reduction | 1483 | 4276 |
| Raw, spacing 2, interval 16, 5% reduction | 741 | 4268 |
| Preconditioned, spacing 4, interval 16, 5% reduction | 731 | 4059 |
| Raw, spacing 25, interval 100, across refreshes | 724 | 4042 |

Rejected trials add full residual evaluations beyond these effective counts.
For example, W7-X's first raw variant accepted only 34 proposals and rejected
314, requiring 4608 actual evaluations. Even the small count reductions in
other variants did not establish a reliable solve-time improvement. The raw
spacing-4/interval-16 variant with the weaker acceptance threshold exhausted
the second W7-X stage's configured iteration budget. The prototype is therefore
excluded from the production solver, including its experimental environment
variables and controller changes.

## Reproduction artifacts

The local experiment directory is
`../tmp/cumes-aggressive-20260908/`: logs, outputs, checkpoints,
`pulay-results.json`, the archived prototype, and the grid-schedule report.
The schedule source, generated inputs, scripts, and full-precision state
comparisons are also archived at `gervais:/tmp/cumes-stage-policy.6aSYtc`.
These scratch paths describe this run; the tables above preserve the principal
findings independently of their lifetime.
