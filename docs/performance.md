# cuMES performance

This is the current performance contract: what has been *measured* (never
claimed), on which hardware, and the acceptance policy that governs future
optimization. The dated tuning-session notes have been archived in
`overhaul-history.md`; this document supersedes them for decision-making.

> **Measurement provenance.** The numbers
> in §2 were re-measured with the overhaul's final build configuration:
> precise double math (`CUMES_PRECISION_POLICY=verify-double`, no
> `--use_fast_math` — its removal proved **trajectory byte-identical**, so
> the frozen baselines stand under precise math), the single-construction
> stage arena (one allocation + one module construction per stage, no
> measuring pass), and the fence-delivered axis/boundary telemetry (the
> production path has exactly one deliberate control fence per pass and no
> per-print synchronization or allocation). Setup and iteration time are
> reported separately. The frozen Pascal baseline is kept below, followed by
> the 2026 RTX 4090 optimization measurement; these are different code states
> and are not used as a cross-GPU comparison.

## 1. Measurement harness (§8.1)

`cumes_benchmark_fixed_iteration` runs one radial stage at a config's final
shape (Solovev `ns=55`, W7-X `ns=99`) with `ftol=0` (never converges), discards
a warm-up, and emits JSON: GPU identity, shape, arena/cuFFT bytes, setup vs
solve time, median/p95 wall µs/iter, and an FNV final-state hash. The seed
construction is shared with the CLI via `cumes/state/seed_state.hpp`.

A `cumes::SolverBench` observer samples the single control fence; it is pure
host observability (`bench == nullptr` in production leaves the hot loop
byte-identical). A 2-pass smoke gate (`cumes_benchmark_smoke`) is registered in
CTest.

## 2. Steady-state wall time (TITAN Xp, sm_61)

Measured by the fixed-iteration harness with 300 timed passes after 50 warmup
passes, three thermally stable repeats each under `verify-double`:

| shape | median µs/iter | p95 µs/iter | repeats |
| ----- | -------------: | ----------: | ------: |
| Solovev `ns=55` (axisymmetric) | 167.0 / 167.1 / 166.5 | 175.7 / 177.2 / 175.2 | 3 |
| Solovev `ns=55` (generic backend, `CUMES_FORCE_GENERIC=1`) | 179.1 | 194.9 | 1 |
| W7-X `ns=99` (3D, generic toroidal-FFT) | 1757.8 / 1757.7 (first repeat 1874.4: cold-clock ramp, discarded) | 1772.4 / 1772.5 | 3 |

Setup (arena + module construction + profile/table upload, one pass per
stage): ~3.5 ms Solovev / ~3.9 ms W7-X, reported separately from the
iteration time. The axisymmetric backend retains a measured ~7% advantage
over the generic backend on the same shape (167 vs 179 µs/iter).

CUDA Graph re-measurement on the REAL pass DAG (ADR-0003 follow-up,
`cumes_benchmark_graph_realpass`, Solovev `ns=55`): enqueue submission
63.9 µs/pass vs graph launch 9.8 µs/pass (54.2 µs saved), production-pattern
wall 124.0 vs 111.3 µs/pass (12.8 µs ≈ 10% saved), GPU time 125.3 vs
112.9 µs. The real-pass bound is therefore confirmed beneficial on the
submission-bound Solovev workload; the later production decision is recorded
in §3.3.

### 2.1 RTX 4090 optimization measurement (CUDA 12.9, 2026-08-26)

The profiling build used precise double math, NetCDF/HDF5/vacuum disabled, GPU
2, 50 warm-up passes and 300 timed fixed iterations. After a 1000-pass preheat,
six graph-on/graph-off pairs with alternating execution order were used for the
production graph comparison. Medians of the six per-run medians were:

| shape | direct stream | production graph | graph wall reduction |
| ----- | ------------: | ---------------: | -------------------: |
| Solovev `ns=55` | 117.14 µs/iter | 82.33 µs/iter | 29.7% |
| W7-X `ns=99` | 562.10 µs/iter | 534.53 µs/iter | 4.9% |

Relative to the pre-optimization fixed-iteration baselines from the same RTX
session, the final W7-X latency fell from 660.81 to 534.53 µs/iter (19.1%, or
1.236x throughput), and Solovev fell from 115.59 to 82.33 µs/iter (28.8%, or
1.404x throughput). The final-state hashes are identical with graphs enabled
and disabled: W7-X `ce46a7cbe693601a`, Solovev `3c09fd3260b003a8`.

The retained kernel changes are a parallel two-stage Jacobian-statistics
reduction (W7-X kernel time 104.7 to 7.2 µs) and a full-theta inverse-transform
mapping. Register-capping and split forward-reduction prototypes were removed
because they did not clear the 5% acceptance threshold.

Historical hotspot profiles (different sessions; not cross-GPU ratios — see the
blueprint §2) put the W7-X iteration work in the transform accumulation/reduction
(~0.4 ms each on TITAN Xp) and the forward reduction (~0.11 ms on RTX 4090),
with force, geometry, tridiagonal, and residual work smaller — i.e. structural
transform work dominates over generic occupancy tuning, and host synchronization
matters more as kernels shorten.

### 2.2 W7-X fixed-boundary convergence recovery (2026-08-30)

ADR-0007 adds a qualified one-shot recovery of a time step reduced by the
early W7-X transient. It reduces deterministic effective iterations from 2953
to 2711 (8.20%) without changing the per-pass CUDA DAG or its memory use.

A native `sm_89` precise-double build on gervais (RTX 4090, CUDA 12.9,
NetCDF/HDF5 disabled) reproduced both iteration counts and residuals. Six
preheated, alternating A/B measurements pinned to NUMA-local CPU 10 gave
2.0583 s baseline versus 1.9217 s recovery median, a 6.63% end-to-end wall
reduction. Median absolute deviations were 6.1 ms and 7.6 ms. Unlocked GPU
P-state transitions produced isolated slow samples (ranges 2.0510–2.5015 s
and 1.9102–2.5244 s; interpolated p95 2.4893 s and 2.4846 s), so this set does
not by itself establish the performance-policy confidence bound. The exact
8.20% pass reduction, independent-solver state comparison, and frozen
non-target trajectories are the primary acceptance evidence; the diagnostic
opt-out retains direct A/B measurement.

The same policy applied once per multigrid stage reduces W7-X from
`1877 → 1617 → 2011` (5505 total) to `1741 → 1568 → 1635` (4944 total), a
10.19% pass reduction. The final residual triple is FSQR `9.989e-13`, FSQZ
`1.590e-13`, FSQL `5.116e-14`; restarting its checkpoint on the final grid
converges at iteration 1 with the identical triple. Six alternating native
gervais measurements gave 3.3109 s baseline versus 3.0021 s recovery median,
a 9.33% end-to-end reduction (MAD 0.1090 s / 0.2123 s; unlocked-clock ranges
3.1856–3.6662 s / 2.7096–3.5031 s). Solovev remains on its exact established
trajectory because no reduced step survives to its recovery window.

### 2.3 Shaped 3-D cold start (2026-08-30)

ADR-0008 augments the regular `s^(m/2)` boundary-harmonic seed by the factor
`1 + 0.12(1-s)` for fixed-boundary 3-D cold starts. It changes no LCFS value,
preserves the required near-axis order, adds no GPU work, and can be disabled
with `CUMES_SEED_ENVELOPE=0`.

On W7-X single-grid it reduces effective iterations from 2711 to 2627 (3.10%).
On W7-X multigrid it changes `1741 → 1568 → 1635` (4944 total) to
`1315 → 1559 → 1633` (4507 total), an 8.84% reduction. Combined with the
reference controller baseline, total multigrid passes fall by 18.13%. The
final residual triple is FSQR `9.967e-13`, FSQZ `1.563e-13`, FSQL
`4.956e-14`, and checkpoint replay converges at iteration 1 with the identical
triple. ADR-0008 alone leaves Solovev byte-identical because axisymmetric
seeds are excluded; the later axisymmetric policy is measured separately
below.

Six alternating native gervais runs measured 1.9238 s versus 1.8823 s median
for single-grid (2.16% wall reduction; MAD 6.3/14.8 ms) and 3.4419 s versus
3.2983 s for multigrid (4.17%; MAD 0.331/0.388 s). Unlocked-clock outliers make
these wall measurements noisier than the deterministic pass counts.

### 2.4 Axisymmetric start policy (2026-08-30)

ADR-0009 uses a `-0.07` envelope correction only for coarse fixed-boundary
axisymmetric cold starts and raises the stage-initial descent step according
to whether the state is cold, single-grid, or prolonged. It changes no GPU
kernel, allocation, or per-pass DAG.

Solovev multigrid falls from `251 -> 199 -> 456` (906 passes) to
`235 -> 193 -> 387` (815), a 10.04% reduction, with final FSQR `9.792e-17`.
A final-grid checkpoint replay converges at iteration 1 with a bit-identical
state. A cold single `ns=55` grid falls from 533 to 354 passes (33.58%), with
FSQR `9.835e-17`. W7-X remains exactly on its accepted
`1315 -> 1559 -> 1633` trajectory.

Before adding the lambda predictor, the local TITAN Xp gave 0.445 s versus
0.395 s median multigrid wall time (11.2%), and a native sm_89 gervais build
gave a 3.7% median reduction for the staged-step policy. The final sm_89 build
reproduces the 815/354 pass counts. At this subsecond scale, CUDA process
startup and unlocked P-state outliers dominate, so pass counts are the primary
timing evidence.

### 2.5 Free-boundary cold predictors (2026-08-31)

ADR-0010 extends the host-only cold predictor to free-boundary starts without
changing the vacuum-coupled iteration DAG. A resolution-limited 3-D envelope
reduces CTH-like single-grid from 489 to 384 passes and `15 -> 25` multigrid
from 592 to 563 total passes. The conservative fine-grid predictor reduces
W7-X `ns=51` from 1831 to 1797 passes. The full axisymmetric lambda predictor
reduces Solovev free-boundary from 1047 to 1025 total passes. Final FSQR values
are `9.932e-11`, `9.942e-11`, `9.705e-13`, and `9.822e-15`, respectively.
Adding the qualified `17/14` step on 3-D free-boundary grids through `ns=25`
reduces CTH-like single-grid further to 347 passes and multigrid to
`208 -> 238` (446 total), with final FSQR `9.359e-11` and `9.745e-11`.
Activating the vacuum force at residual sum `3e-2` reduces those trajectories
again to 309 and `198 -> 233` (431 total), with FSQR `9.635e-11` and
`9.885e-11`. W7-X falls from the seeded 1797 to 1733 passes (FSQR
`9.996e-13`), while Solovev remains at 1025.

Seven alternating local TITAN Xp A/B pairs reduced CTH-like multigrid median
wall time from 1.93 s to 1.50 s (22.3%; ranges 1.91--2.01 s and
1.48--1.50 s). Five alternating W7-X pairs reduced the median from 8.37 s to
7.93 s (5.3%; ranges 8.32--8.39 s and 7.85--7.97 s). Pass counts were
identical in every repetition.

## 3. Phase 9 experiments and their outcomes

The exit gate for §8.10–§8.12 was *"measure, then adopt or remove."* Three
experiments were run on the TITAN Xp; one was adopted.

### 3.1 Mixed-float double accumulation — adopted (ADR-0001)

`cumes::NormAccum<T>` (`float → double`, `double → double`) widened the
accumulator of the three control-feeding reductions. **Class A** for the double
build (identity); **Class B** for float, where `test_accumulation.cu` shows a
~20× lower summation error (4.6e1 → 2.3e0 on a 1M-term dynamic-range case). It
does not unlock a lower float `ftol` — the float-state rounding floor dominates.

### 3.2 R/Z vs λ force-kernel split — not adopted (ADR-0002)

The split reduced registers 108 → 82/54 but was **1.20–1.45× slower**: the
force kernel is input-traffic-bound, so the two-kernel split doubled the
geometry/field loads. The prototype has been removed; the conclusion
(§8.10's remaining radial-tile / force+projection fusion ideas trade global
traffic for registers/shared memory and are unlikely to pay) is the durable
result.

### 3.3 CUDA Graph capture — adopted for fixed boundary (ADR-0003)

Empty-kernel microbenchmark: enqueue 1.94 µs/kernel, graph launch 6.43 µs/pass
for 24 kernels → **~40 µs/pass upper-bound saving**. The real-pass
measurement (see §2) confirmed a ~12.8 µs/pass (≈10%) wall saving on Solovev,
with bitwise replay fidelity (`test_cuda_graph`). Modern re-measurement then
showed a 29.7% wall reduction on Solovev and a 4.9% reduction on W7-X. The graph
change therefore clears the adoption threshold on one primary shape while
remaining a non-regressing improvement on the other. Fixed-boundary schedules
are captured lazily and cached by shape. Free-boundary and verification-dump
paths remain direct; `CUMES_DISABLE_CUDA_GRAPHS=1` is the diagnostic opt-out.

## 4. Acceptance policy (verification.md §7)

A performance-motivated change is accepted only when, on one named target
workload, the lower bound of the 95% confidence interval shows an improvement
`> max(5%, measured noise floor)`, while the upper bound on the other primary
workload's regression is `<= 2%` — unless a separately approved correctness or
memory benefit justifies it. Requirements:

- repeated thermally stable warm runs; report median, p95, CI, clocks, and the
  noise floor (never a single timing);
- setup and output reported separately from effective-iteration time;
- compared on the legacy Pascal target *and* one modern architecture;
- peak arena/cuFFT/graph memory growth beyond an agreed baseline is rejected
  unless the measured benefit justifies it;
- the old backend is retained until the new one passes both numerical and
  performance gates.

**Modern-GPU validation status:** the 2026 RTX 4090 tuning run validates the
retained optimizations on the modern target and includes bitwise final-state
gates. A same-code-state Pascal re-run is still outstanding, so the complete
two-architecture acceptance matrix remains open. That run must use the same
commit, precision policy, toolkit provenance, warm-up, and measurement method
as the TITAN Xp baseline; record GPU model, compute capability, driver/toolkit,
clocks, power/thermal state, arena/cuFFT/graph memory, setup/output time,
median, p95, measured noise floor, and a 95% confidence interval; re-run the
complete numerical trajectory/state gate afterwards.

Equivalence class precedes any timing claim: Class A requires bitwise equality;
Class B requires per-operator ULP bounds and identical controller decisions;
Class C requires residual/validity qualification, independent comparisons with
differences reported, and a written ADR.

## 5. Decision records and historical inputs

- `overhaul-history.md` archives the TITAN Xp optimization pass and RTX
  4090 profiling session. Their measurements are historical comparison inputs,
  not acceptance goldens.
- `docs/adr/0001..0003` — the decisions behind §3.
