# Two-level FAS correction experiment

The September 8, 2026 experiment performed genuine coarse corrections during
fine solves, while retaining every configured grid, tolerance and iteration
cap. Both tested settings converged correctly but increased measured solver
time. The production solver is unchanged. The standalone diagnostic remains in
`benchmarks/coarse_correction.cu` and `benchmarks/coarse_correction.cuh`.

## Equation and transfers

Let `F_h(x)` be the production spectral force, `P_h^-1` its cached
preconditioner, and `D_h` the production map from decomposed, normalized,
m=1-mixed force coordinates to physical coefficient increments. Define

```
G_h(x) = D_h P_h^-1 F_h(x).
x_H0   = R_x x_h
g_H    = R_G G_h(x_h)
tau_H  = G_H(x_H0) - g_H
G_H(y) = tau_H
```

The coarse operator uses the same production geometry, field, constraint,
transform and preconditioner operators on `ns_H = (ns_h + 1)/2` surfaces,
with unchanged angular resolution. It owns separate profiles and caches.
The first coarse evaluation initializes its reference and preconditioner;
both remain fixed during the cycle. The current-closure update still runs.
The internal coarse `ftol=0` exposes the shifted equation's force; it cannot
declare convergence for an outer stage.

Both the state and the physical preconditioned fine force are restricted.
Radial interpolation is linear in the regularized odd-m coefficient
`x/sqrt(s)`, using the production axis extrapolation convention; even-m
coefficients are interpolated directly. The same operation prolongs the
coarse error. Fixed R/Z boundary entries, dependent axis entries and null
parities are masked out of the returned correction. Lambda boundary unknowns
remain active. The fixed mixed m=1 gauge requires equal physical Rss/Zcs
increments. The coarse error is projected onto that common-increment space
before prolongation, because subtracting two physical states can otherwise
round the two accumulated errors differently.

The coarse smoothing recurrence is

```
r = (G_H(y) - G_H(x_H0)) + g_H
v = 0.8 * (0.75*v + delta*r)
y = y + delta*v
```

with zero initial velocity. This evaluates `G_H(y)-tau_H` in an order that
preserves an exact zero-forcing fixed point. It solves a shifted FAS equation;
it does not ask the restricted state to solve the coarse equilibrium.
Each cycle uses one initial coarse evaluation plus 8 or 16 smoothing/evaluation
pairs. All buffers, profiles and plans are allocated before the solve loop.
A failed coarse validity/nonfinite gate discards the whole direction.

## Fine trial and rollback

The private full-solver integration proposes corrections before the host
controller consumes the current pass. It requires a valid, nonterminal state,
the stable fixed m=1 gauge (`FSQZ < 1e-6`), more than 50 iterations since the
restart anchor, and a pass that neither refreshes the preconditioner nor resets
the constraint reference. The smallest Solovev grid is solved normally; the
correction is eligible on grids with `ns > 5`.

The hook saves the fine state, preconditioned force and control record, then
tries physical corrections at scales 1, 1/2 and 1/4. Each trial runs the full
fine production operator using the existing fine caches and norm factors.
Acceptance requires a valid Jacobian, finite values and at least a 0.5%
reduction in the sum of the three actual invariant residuals. The usual
per-component configured stopping tests remain authoritative.

Rejected trials restore the saved state, force and record before any controller
call; velocities and controller history are untouched. Accepted trials zero
velocity and reset only the damping history and running residual minimum.
They do not change the step, counters, restart anchor or refresh schedule.
The ordinary controller and descent then process the accepted state. The
accepted axis telemetry is refreshed as well. Fine trial current-closure
arrays are recomputed independently on the next evaluation; they are not a
history-dependent recurrence.

## Validation and anchor measurements

The standalone harness passed these checks on saved Solovev `ns=55, iter=250`
and W7-X `ns=99, iter=500` states on Pascal:

- A zero restricted fine force produces exactly zero correction while every
  coarse evaluation remains valid, despite the nonzero coarse discretization
  defect.
- Returned corrections preserve fixed boundary/inactive entries and the m=1
  common-increment condition exactly.
- Re-evaluating the saved fine base after all coarse work and trials reproduces
  all three baseline invariant residuals exactly.
- The header compiles with the existing precise CUDA flags for both scalar
  types. All nine instantiated float kernels contain zero FP64 arithmetic
  instructions in the SASS audit. Full equilibrium experiments used double.

At `delta=0.5` and 8 smooths, W7-X accepted scale 1/4, reducing full fine merit
by about 0.70%; Solovev rejected all scales. Coarse-cycle plus three fine-trial
CUDA event time was 15.82 ms and 1.96 ms respectively. The earlier 16-smooth,
`delta=0.9` screen produced no accepted Solovev correction and triggered the
W7-X coarse failure guard. The conservative full solves below use `delta=0.5`.

## Full original-schedule solves

Runs used the RTX 4090 on gervais GPU2, precise Release `sm89`, CUDA 12.9.41,
GCC 12.4 and CPU8. The numerical baseline is the existing multigrid solver;
only the private solver translation unit was rebuilt and inserted into a
copied library. Its disabled mode supplies the instrumented baseline.

- Solovev: grids `5, 11, 55`; all tolerances `1e-16`; caps `1000, 2000, 2000`.
- W7-X: grids `33, 66, 99`; all tolerances `1e-12`; caps `3000, 4000, 5000`.

Every one of the eight runs completed every configured stage with all three
actual residuals below that stage's original tolerance. Structured records
come directly from `SolverResult`, including the configured cap and tolerance.

| Input | Variant | Effective iterations by stage | Solver wall ms | Extra coarse evaluations | Fine trial evaluations | Accepted cycles |
|---|---|---|---:|---:|---:|---:|
| Solovev | Disabled | 235 / 193 / 326 | 68.315 | 0 | 0 | 0 |
| Solovev | Forced rejection, 8 / 100 | 235 / 193 / 326 | 74.582 | 36 | 12 | 0 |
| Solovev | 8 smooths / period 100 | 235 / 185 / 339 | 74.665 | 36 | 8 | 2 |
| Solovev | 16 smooths / period 200 | 235 / 187 / 351 | 73.722 | 51 | 7 | 2 |
| W7-X | Disabled | 1315 / 1419 / 1372 | 1695.322 | 0 | 0 | 0 |
| W7-X | Forced rejection, 8 / 100 | 1315 / 1419 / 1372 | 1838.460 | 342 | 114 | 0 |
| W7-X | 8 smooths / period 100 | 1299 / 1420 / 1318 | 1796.907 | 333 | 103 | 6 |
| W7-X | 16 smooths / period 200 | 1315 / 1410 / 1298 | 1755.348 | 306 | 50 | 3 |

The period gate begins at effective iteration 100. No coarse validity failures
occurred in these full runs. Fine evaluations, including restart passes but
excluding trials, were 755/760/774 for Solovev disabled/8/16 and
4124/4055/4041 for W7-X. Effective iterations alone therefore understate work.

Forced rejection preserves final coefficient hashes, stage iteration counts,
fine-pass counts and all three residuals exactly for both cases. Final enabled
Solovev FSQR values were `9.944132084780487e-17` and
`9.949395298993118e-17`; W7-X values were `9.963946385961908e-13` and
`9.963773082362199e-13`. Each corresponding FSQZ/FSQL also passed its original
tolerance.

The table measures `steady_clock` solver wall time, including every coarse
evaluation, transfer, fine trial and fence. It starts after construction of
the coarse corrector. Separate whole-process times include that setup and
output: Solovev disabled/8/16 were 0.376/0.885/0.372 s; W7-X were
2.225/3.236/3.379 s. These are bounded single-run screens, not paired speed
estimates; process times include startup variability. Both enabled settings
regress the measured iteration work, so no further performance qualification
or production integration was pursued.

## Reproduction and artifacts

```sh
cmake --build build --target cumes_benchmark_coarse_correction
build/cumes_benchmark_coarse_correction --input inputs/w7x.json \
  --restart ../tmp/cumes-block-20260908/states/w7x-ns99-iter500.ckpt \
  --steps 8 --delta 0.5 --out /tmp/w7x-coarse.json
```

The complete local record is `../tmp/cumes-two-level-20260908/`: anchor JSON
and logs, float SASS audit, `prepare_live.py`, `build_live.py`, `run_full.py`
and the private solver source. `ada/` contains the exact tested source,
compile/link commands, hashes, all eight native/checkpoint outputs, logs and
structured stage reports. The combined stdout/stderr stream can place a
structured marker inside a progress line; `ada/reparse_ada.py` finds the marker
anywhere and decodes its JSON object. The original parser failure and corrected
reports are both retained for provenance.

The exact tested full-solver hook is retained as
`benchmarks/coarse_live.patch`. It modifies only the solver implementation
and the controller's experimental momentum-reset method. Apply it in a
disposable worktree; the patch is not part of the production build:

```sh
git worktree add --detach ../cumes-coarse-experiment HEAD
cd ../cumes-coarse-experiment
git apply benchmarks/coarse_live.patch
cmake --preset verify
cmake --build build -j
ctest --test-dir build --output-on-failure

# Instrumented baseline, with the original input unchanged.
CUMES_COARSE_INPUT=inputs/w7x.json CUMES_COARSE_PERIOD=0 \
  build/cumes inputs/w7x.json --output /tmp/coarse-disabled.bin \
  --checkpoint /tmp/coarse-disabled.ckpt

# Guarded FAS; repeat for Solovev and the 16/200 setting.
CUMES_COARSE_INPUT=inputs/w7x.json CUMES_COARSE_STEPS=8 \
  CUMES_COARSE_DELTA=0.5 CUMES_COARSE_PERIOD=100 CUMES_COARSE_START=100 \
  build/cumes inputs/w7x.json --output /tmp/coarse-enabled.bin \
  --checkpoint /tmp/coarse-enabled.ckpt
```

For the rollback check, add `CUMES_COARSE_FORCE_REJECT=1` to the enabled
command and compare stage records and checkpoint coefficient bytes with the
disabled run. The private hook also accepts `CUMES_COARSE_NS` to restrict
correction eligibility to one outer grid and `CUMES_COARSE_GRID` to select
the internal coarse size; neither override was used for the table.
