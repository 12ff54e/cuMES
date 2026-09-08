# Fully coupled Newton–Krylov experiments

The September 8, 2026 experiments tested Newton corrections across all active
R, Z and lambda coefficients during complete production multigrid solves.
Every configured grid, tolerance and iteration cap was retained. Newton reduced
the Solovev iteration count substantially and gave a modest measured speedup;
W7-X did not show a reliable speedup across Pascal and Ada. The implementation
and full-solver patch remain experiments, with production defaults unchanged.
The separate [FAS experiment](coarse-correction-experiments.md) completes the
other remaining proposal.

The subsequent [19-case axisymmetric qualification](axisymmetric-newton-qualification.md)
tests the same fixed forward32 policy on both GPUs, including shape,
resolution and finite-pressure variations. It confirms larger gains for the
prescribed-current fixture and regressions for some circular/low-resolution
Ada workloads; all cases converge through every configured stage. Its complete
timing and solution checks refine the scope of the axisymmetric opportunity.

## Operator and coordinates

For a fixed production preconditioner epoch, write

```
G(x) = pack(P^-1 F(x))
A q  = -dG(x + D q)/dq
A delta = G(x)
x_trial = x + alpha D delta
```

`D` is the exact production descent map: Fourier normalization, mixed m=1
Rss/Zcs coordinates, fixed-boundary masking, dependent axis entries and null
parities. Lambda remains active at the boundary. Tests compare this map
exactly against `DescentOperator` for both scalar types and gauge settings.

The matrix-free product uses the same `EquilibriumOperator` as the live solve.
Its preconditioner, constraint multiplier, constraint reference and m=1 gauge
are frozen; geometry, magnetic field, all three force components and the
`ncurr=1` current closure are recomputed for every perturbation. A separately
constructed, freshly preconditioned derivative would differentiate a different
operator and was deliberately avoided.

The physical perturbation has maximum coefficient magnitude `epsilon`:
`h = epsilon / max(abs(D q))`. Central differences use
`(G(x-h Dq)-G(x+h Dq))/(2h)`. The cheaper forward variant uses the immutable
saved right-hand side, `(G(x)-G(x+h Dq))/h`. A zero direction returns zero.
Invalid/nonfinite geometry or a probe that does not evaluate preconditioning
causes Krylov breakdown and rejection. The live stage's positive tolerance is
unchanged; the standalone diagnostic uses `ftol=0` to expose the full map.

Restarted GPU GMRES uses two-pass classical Gram–Schmidt,
parallel basis projections and vector updates, GPU Givens rotations and a GPU
triangular solve. All work vectors are allocated before iteration. There is
no host Arnoldi loop or transfer of Krylov vectors. The submitted map count is
`steps + ceil(steps/basis)`, including a true-residual evaluation per restart.
Inactive directions still submit maps, so early inner convergence does not
erase their cost. Central products cost two equilibrium evaluations each;
forward products cost one. The inner relative target is `1e-3`; reaching a
fixed iteration budget is reported separately from reaching this target.

## Full-solve integration

`benchmarks/newton_live.patch` contains the exact tested private hook. It
proposes corrections before the host controller classifies the current pass,
only for valid nonterminal states with a stable m=1 gauge, more than 20 passes
since the restart anchor, and no preconditioner/reference refresh that pass.
The normal graph and all configured multigrid stages remain present.

Full fine-grid trials use scales 1, 1/2, 1/4 and 1/8. Acceptance requires valid,
finite geometry and at least a 5% decrease in the sum of the three actual
invariant residuals. Each component must still pass the original stage
tolerance before convergence is declared. Accepted corrections zero momentum
and reset the old damping history and running residual minimum; they preserve
the step, counters, restart epoch and refresh cadence. The ordinary controller
then captures its usual post-descent checkpoint. Axis telemetry is refreshed.

A rejected correction restores the state and re-evaluates it with the same
cached operator before any controller decision. Its invariant residual triple
must match exactly; velocity and controller history remain untouched. The
disabled hook reproduces the original baseline. Full forced-reject Solovev
and W7-X runs reproduced baseline checkpoints byte for byte. Native structured
stage residuals, iteration counts and restart sequences also match the
preserved reference executable exactly. Earlier exploratory runs without the damping-history reset are archived in `runs/` and are not
the basis of the final performance comparison.

Structured `EXPERIMENT_STAGE` records come from `SolverResult`; they contain
actual stage iterations, convergence, all three residuals, extra equilibrium
evaluations, correction time and total scoped time. `EXPERIMENT_NEWTON` records
acceptance, scale, Krylov steps/breakdown and residual information. No iteration
count or convergence claim is inferred from the human progress table.

## Numerical validation

- GPU GMRES passes double/float manufactured nonsymmetric and restarted solves,
  a skew rotation, near breakdown, singular/nonfinite maps, zero RHS reuse,
  budget exhaustion, and tiny/huge RHS tests. Memcheck, initcheck, synccheck and
  racecheck report zero errors/hazards on RTX 4090.
- All 13 captured production states pass the central-difference halving check
  at `epsilon=1e-8` and restore all three residuals exactly. Ada relative
  halving errors range from `2.52e-8` to `3.05e-7`. W7-X `ns=99, iter=500`
  passes memcheck/initcheck/synccheck. Nineteen invalid CLI cases reject.
- W7-X's default screening perturbation `1e-6` was too large: the random
  direction's halving discrepancy was `7.69e-4`; `1e-8` reduced it to
  `1.42e-7` on Pascal. The full W7-X experiments therefore use `1e-8`.
- Forward Solovev at `epsilon=1e-6` passes all six anchor halving/restoration
  checks and the final-grid memcheck/initcheck checks. Random-direction errors
  are `9.46e-6`–`1.70e-5`. Forward versus central products on the actual
  correction agree to `3.02e-7`–`6.99e-6` at five anchors, but differ by
  `1.24e-3` at the late `ns=55, iter=250` anchor. This weak-direction accuracy
  limit is retained in the record; the actual nonlinear residual acceptance
  test remains essential. Forward differences did not provide a sufficiently
  accurate W7-X correction in the tested perturbation range.

Standalone local merit reductions are not interpreted as speedups. For
example, central32 reduces Solovev's fine-grid iteration-100 merit by about
99.33%, while the complete solve still pays for all 66 matrix-product
equilibrium evaluations and its accepted trial.

## Complete configured solves

The unchanged inputs specify Solovev grids `5/11/55`, tolerances all `1e-16`
and caps `1000/2000/2000`; W7-X grids `33/66/99`, tolerances all `1e-12` and
caps `3000/4000/5000`. Every reported complete solve passes every stage.

| Input | Correction policy | Effective iterations by stage | Extra equilibrium evaluations |
|---|---|---|---:|
| Solovev | Baseline | 235 / 193 / 326 | 0 |
| Solovev | Central16 every 100 | 158 / 140 / 273 | 140 |
| Solovev | Central32 every 100 | 144 / 113 / 226 | 268 |
| Solovev | Forward32 every 100 | 144 / 113 / 227 | 136 |
| Solovev | Forward32 once at 100 per stage | 144 / 113 / 270 | 102 |
| Solovev | Forward64 every 100 | 144 / 110 / 200 | 264 |
| W7-X | Baseline | 1315 / 1419 / 1372 | 0 |
| W7-X | Central32 once at fine-grid 100 | 1315 / 1419 / 1259 | 67 |
| W7-X | Central64 once at fine-grid 100 | 1315 / 1419 / 1227 | 131 |
| W7-X | Central128 once at fine-grid 100 | 1315 / 1419 / 1193 | 259 |
| W7-X | Central16 every 200 | 1379 / 1463 / 1307 | 723 |

Larger W7-X subspaces saved progressively fewer additional outer iterations
per extra map. The periodic policy also worsened the coarse/middle trajectory.
The selected repeated candidates are forward32 every 100 for Solovev and
central32 once at fine-grid iteration 100 for W7-X. Their trajectories repeat
exactly within each architecture, including final coefficient hashes.

Five alternating-order pairs follow two warm-up runs of each variant. The
reported confidence intervals use 20,000 paired percentile-bootstrap samples.
Both builds use precise Release arithmetic: TITAN Xp/CUDA12.1/GCC12 and
RTX4090/CUDA12.9.41/GCC12.4.

| GPU / input | Scoped median ms, baseline → Newton | Scoped p95 ms, baseline → Newton | Paired reduction, 95% CI |
|---|---:|---:|---:|
| TITAN Xp / solovev | 98.754 → 88.829 | 99.240 → 89.091 | 10.050% [9.939%, 10.471%] |
| TITAN Xp / w7x | 4644.216 → 4603.336 | 4657.641 → 4617.810 | 0.891% [0.782%, 1.062%] |
| RTX 4090 / solovev | 71.951 → 69.149 | 72.245 → 69.711 | 4.159% [2.284%, 5.009%] |
| RTX 4090 / w7x | 1691.465 → 1696.694 | 1694.191 → 1707.150 | -0.746% [-0.832%, 0.168%] |

The MAD/median noise estimates (baseline/Newton) are 0.159%/0.056% for
Pascal Solovev and 0.218%/0.228% for Pascal W7-X; Ada gives
0.441%/0.611% and 0.187%/0.416%, respectively.

Clocks were unlocked. Pascal active samples at 200 ms intervals recorded
SM 1759–1885 MHz, memory 5508–5702 MHz, temperature 34–71 °C and
power 68–197 W; exact samples are retained. Ada endpoint samples recorded
SM 2520–2760 MHz, memory 10501 MHz, 31–39 °C and 53–146 W, which do
not bound all in-process excursions.

Whole-process baseline/Newton medians are 0.486/0.471 s (Pascal Solovev),
5.382/5.381 s (Pascal W7-X), 0.372/0.372 s (Ada Solovev) and
2.781/3.034 s (Ada W7-X).

These measurements sum CUDA-event intervals around each solver call, starting
before equilibrium/corrector construction and ending after stream drain.
They include correction setup, all extra maps, trials, fences, and host gaps
inside the interval. Outer stage construction, inter-stage transfer, output
and process startup are outside it. Separate process wall times are archived;
their allocation/startup outliers prevent an end-to-end speedup claim.
Five pairs are a small sample. No production adoption claim is made.

The Ada lower confidence bound falls below the required 5% improvement in
[the existing acceptance policy](performance.md#4-acceptance-policy-verificationmd-7).
W7-X also lacks a reliable benefit across the two architectures. Accordingly,
these experiments do not change solver defaults or establish a new Class C
production trajectory. The residual reductions are useful evidence for future
work, but counting only outer passes would materially overstate their value.

## Final-state diagnostics

Both selected candidate checkpoints converge at iteration 1 in a final-grid
replay with Newton disabled, using the original final tolerance. These replay
inputs exist only to match the checkpoint shape; they are not performance
workloads. The complete timed solves above retain all original stages.

Against the baseline, Solovev's maximum R/Z coefficient changes excluding the
dependent axis row are `5.91e-9`/`2.34e-9`, and lambda changes by `2.27e-8`.
Native half-grid `B^theta` and `B^zeta` relative L2 differences are `5.97e-9`
and `4.03e-10`. W7-X changes by up to `2.97e-5` in R, `2.27e-5` in Z and
`1.90e-4` in lambda; its two contravariant field differences are `3.05e-4`
and `6.38e-6`. These are differences in the native coordinates, not a
gauge-invariant field comparison.

The archived independent VMEC++ 0.7.0 solves provide a diagnostic cross-check.
Signed-n modes are folded and wout lambda is converted back with
`phipF/lamscale`. Solovev's maximum R/Z/lambda differences from VMEC++ are
`2.55e-8`, `1.44e-8`, `5.13e-8`; baseline values were `2.14e-8`, `1.24e-8`,
`3.48e-8`. W7-X retains the known gauge-family discrepancy: the largest
lambda-family difference is about `5.74e-3` for both trajectories. VMEC++ is
not used as the convergence oracle. The exact arrays are retained in the
native outputs/checkpoints; hashes and all six-family differences are in
`scientific-comparison.json`.

## Reproduction and artifacts

The committed standalone probe accepts an existing captured checkpoint:

```sh
cmake --build build --target cumes_benchmark_newton_correction
build/cumes_benchmark_newton_correction --input inputs/w7x.json \
  --restart ../tmp/cumes-block-20260908/states/w7x-ns99-iter500.ckpt \
  --iterations 32 --basis 32 --epsilon 1e-8 --difference central
```

To reproduce complete solves, apply the experimental patch in a disposable
worktree and build the precise double preset. The patch and its environment
controls are excluded from the normal solver build.

```sh
git worktree add --detach ../cumes-newton-experiment HEAD
cd ../cumes-newton-experiment
git submodule update --init --recursive
git apply benchmarks/newton_live.patch
cmake --preset verify
cmake --build build -j

# Instrumented baseline.
build/cumes inputs/solovev.json --output /tmp/newton-baseline.bin

# Every configured Solovev stage, with guarded forward corrections.
CUMES_NEWTON_PERIOD=100 CUMES_NEWTON_START=100 \
CUMES_NEWTON_STEPS=32 CUMES_NEWTON_BASIS=32 \
CUMES_NEWTON_FORWARD=1 CUMES_NEWTON_EPSILON=1e-6 \
  build/cumes inputs/solovev.json --output /tmp/newton-solovev.bin \
  --checkpoint /tmp/newton-solovev.ckpt

# Every configured W7-X stage; one correction on the final stage.
CUMES_NEWTON_PERIOD=100 CUMES_NEWTON_START=100 CUMES_NEWTON_END=100 \
CUMES_NEWTON_NS=99 CUMES_NEWTON_STEPS=32 CUMES_NEWTON_BASIS=32 \
CUMES_NEWTON_EPSILON=1e-8 \
  build/cumes inputs/w7x.json --output /tmp/newton-w7x.bin \
  --checkpoint /tmp/newton-w7x.ckpt
```

Use `CUMES_NEWTON_RATIO=0` for forced rejection. These are private experimental
controls, not additions to the public CLI/environment contract.

The full artifact tree is `../tmp/cumes-newton-20260908/`: exact tested sources,
the private binary and compile commands, anchor probes, screening runs,
`paired-pascal-warm/`, `ada/paired/`, replay logs, scientific comparisons and
input/environment records. `../tmp/cumes-block-20260908/ada-gmres/` and
`ada-newton/` hold the manufactured tests, all anchor JSON, sanitizer logs,
tested source copies and SHA256 manifests. The FAS results and independent
private hook are documented separately.
