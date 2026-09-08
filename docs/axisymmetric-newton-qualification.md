# Axisymmetric Newton–Krylov qualification

The September 8, 2026 follow-up confirms that Newton–Krylov improves several
axisymmetric workloads beyond the original Solovev input. The strongest case,
the adapted VMEC++ prescribed-current Solovev fixture, saves 19.21% of scoped
solver time on TITAN Xp and 14.51% on RTX 4090. However, the same policy slows
some circular and low-resolution cases on RTX 4090. Axisymmetry alone is not a
sufficient condition for enabling this policy by default.

All 19 inputs converge with and without Newton on both GPUs, using every
configured stage and the original tolerances. This is an experimental Class C
trajectory change; production defaults remain unchanged. The results below
include the regressions and their uncertainty.

## Fixed experiment

The [input matrix](../benchmarks/axisymmetric_newton/README.md) fixes geometry,
pressure profiles, spectral/radial resolution and quadrature variations. Its
original 16 cases were committed before their first runs (`8e2ae73`). Cases
00–13 are correlated variations of the cuMES Solovev input; cases 14–15 are
adapted VMEC++ v0.7.0 fixtures, with adaptations recorded field by field.
These are not 16 independent devices. A separately predeclared three-case
supplement (`e809507`) adds 1,000 Pa and 5,000 Pa profiles after inspection
identified the base input's central pressure as only 0.125 Pa. The initial
16-case manifest and all its results remain unchanged.

All cases are fixed-boundary, stellarator symmetric, axisymmetric, precise
double. They use grids `5/11/55`, except the explicit radial-resolution case
`5/11/99`; caps are `1000/2000/2000` and all three stage tolerances are `1e-16`.
Each convergence check requires all three residual components below tolerance.
No grid, tolerance or cap is changed to obtain a timing result.

The [existing private full-solver hook](newton-correction-experiments.md) uses
the same policy for every input and both GPUs:

```text
forward differences, epsilon 1e-6
32 Krylov steps, basis 32
period 100, start 100, on every configured stage
acceptance merit ratio 0.95
```

Its validity, epoch, gauge and refresh guards, nonlinear trial acceptance,
rollback and controller-history handling are unchanged. Baseline and candidate
use the same preserved executable; the baseline disables the hook. The runner
clears inherited `CUMES_*` variables and checks each input's SHA-256. No new
numerical implementation or per-case tuning enters this qualification.

## Timing protocol

TITAN Xp uses CUDA 12.1/GCC 12 and native sm61; RTX 4090 on gervais uses CUDA
12.9.41/GCC 12.4 and native sm89. Both use precise Release arithmetic and CPU
8 affinity. The full binary hashes, build commands, exact tested source,
input hashes and endpoint GPU telemetry are archived. A resident Chromium
process on TITAN Xp was left untouched. Telemetry is sampled before and after
each solve; it does not bound clock or utilization changes inside the solve.
The Ada build's scratch CUDA-header compatibility adaptation is recorded in
its archive; no mathematical body was changed.

The metric is the sum of CUDA-event intervals from `solver_run` entry before
EquilibriumOperator/Newton construction through final solver synchronization.
It includes that setup, all Krylov/trial/rollback evaluations and host gaps
inside the intervals. It excludes outer stage setup, interstage transfer,
process startup and output. These percentages describe scoped solver time;
full process wall medians are recorded separately and need not improve by the
same percentage.

Every case receives a screen pair, two warmups per variant and five measured
pairs in alternating baseline/Newton order. No timing outlier is removed.
For each pair, gain is `100 * (1 - Newton_ms / baseline_ms)`; the table reports
its median. The 95% percentile-bootstrap interval uses 20,000 resamples of
whole pairs, seed 20260908. Per-variant medians therefore need not reproduce
the median paired gain exactly. These are exploratory per-case intervals,
without adjustment for multiple comparisons; five pairs give limited
precision. p95, MAD, process-wall medians, raw paired reductions, exact
trajectory checks and final state hashes are also in the committed
[machine-readable results](../benchmarks/axisymmetric_newton/results-20260908.json).

## Initial five-pair results: every case

Positive gains mean faster Newton solves; negative gains mean slower.
Times are milliseconds. Full case definitions are in the input matrix.

| Case | TITAN Xp ms, baseline → Newton | Gain % [95% CI] | RTX 4090 ms, baseline → Newton | Gain % [95% CI] |
|---|---:|---:|---:|---:|
| 00_solovev_reference | 98.469 → 88.547 | +10.39 [9.72, 10.66] | 71.586 → 68.719 | +3.96 [3.18, 5.81] |
| 01_solovev_circular | 96.894 → 94.020 | +2.91 [2.75, 15.09] | 70.923 → 72.824 | -1.90 [-8.15, 5.15] |
| 02_solovev_elongation_low | 97.106 → 86.577 | +10.78 [10.24, 11.38] | 70.734 → 67.191 | +5.06 [4.79, 15.59] |
| 03_solovev_elongation_high | 110.022 → 104.544 | +4.80 [3.93, 6.17] | 80.402 → 81.458 | -1.52 [-1.96, 2.09] |
| 04_solovev_triangularity_stronger | 115.086 → 103.523 | +10.06 [8.72, 10.40] | 83.951 → 79.741 | +4.65 [-6.60, 53.38] |
| 05_solovev_triangularity_reversed | 91.873 → 84.092 | +8.65 [7.94, 10.20] | 67.432 → 65.734 | +2.48 [-21.62, 2.89] |
| 06_solovev_pressure_half | 98.860 → 88.884 | +10.09 [9.92, 10.14] | 71.883 → 68.911 | +4.37 [3.55, 7.37] |
| 07_solovev_pressure_double | 99.104 → 88.964 | +10.30 [10.11, 10.33] | 71.570 → 68.993 | +3.48 [-4.14, 4.38] |
| 08_solovev_pressure_quadratic | 98.767 → 88.796 | +10.00 [9.78, 10.43] | 72.062 → 68.993 | +4.19 [-1.58, 5.97] |
| 09_solovev_pressure_flatcore | 98.511 → 88.724 | +10.22 [9.44, 11.20] | 71.937 → 68.907 | +4.29 [-31.55, 5.66] |
| 10_solovev_mpol4 | 86.215 → 82.707 | +4.14 [3.90, 4.25] | 64.208 → 66.116 | -2.89 [-35.32, -2.71] |
| 11_solovev_mpol10 | 122.012 → 108.587 | +10.76 [10.09, 11.81] | 84.682 → 79.846 | +5.70 [4.87, 6.53] |
| 12_solovev_radial99 | 138.414 → 122.300 | +11.64 [10.76, 12.54] | 101.700 → 93.061 | +4.67 [3.33, 14.64] |
| 13_solovev_ntheta36 | 107.596 → 96.387 | +10.36 [10.20, 10.83] | 77.917 → 73.582 | +5.58 [4.52, 6.04] |
| 14_vmecpp_circular | 183.999 → 178.799 | +2.83 [2.16, 2.97] | 120.562 → 128.148 | -6.29 [-26.54, -2.87] |
| 15_vmecpp_analytical_ncurr1 | 142.061 → 114.822 | +19.21 [19.11, 19.63] | 97.442 → 83.310 | +14.51 [11.08, 14.98] |
| 16_solovev_pressure1000_linear | 94.567 → 85.003 | +10.12 [2.80, 10.44] | 68.830 → 66.450 | +3.63 [-129.43, 3.92] |
| 17_solovev_pressure5000_linear | 116.403 → 110.341 | +5.06 [4.89, 5.29] | 84.777 → 84.412 | +0.52 [-0.46, 49.05] |
| 18_solovev_pressure1000_quadratic | 98.630 → 86.680 | +12.10 [11.89, 12.85] | 72.010 → 67.402 | +5.71 [1.19, 6.95] |

Only case 15 has an initial RTX 4090 lower confidence bound above 5%.
A modest positive estimate for the original Solovev input is reproducible,
but it does not by itself qualify an axisymmetric default.

## Fresh measurements where uncertainty was large

Before follow-up runs, selection was fixed to every case whose initial
interval exceeded 10 percentage points in width, regardless of gain sign.
This selects one TITAN Xp case and ten RTX 4090 cases. Each receives 15 fresh
alternating-order pairs and two warmups per variant, using the same binary,
policy and affinity. All samples are retained. The table below is additional
evidence; it does not overwrite the initial five-pair results or pool away
their outliers. No further timing selection or parameter tuning follows it.

| GPU | Case | Baseline → Newton ms | Gain % [95% CI] |
|---|---|---:|---:|
| TITAN Xp | 01_solovev_circular | 97.147 → 94.093 | +3.04 [2.79, 3.35] |
| RTX 4090 | 01_solovev_circular | 70.820 → 73.009 | -3.11 [-9.16, -2.65] |
| RTX 4090 | 02_solovev_elongation_low | 70.776 → 67.487 | +4.85 [3.22, 5.00] |
| RTX 4090 | 04_solovev_triangularity_stronger | 84.037 → 79.839 | +5.40 [5.16, 9.36] |
| RTX 4090 | 05_solovev_triangularity_reversed | 67.607 → 65.737 | +3.41 [1.71, 9.70] |
| RTX 4090 | 09_solovev_pressure_flatcore | 71.656 → 69.258 | +3.35 [1.51, 4.26] |
| RTX 4090 | 10_solovev_mpol4 | 63.924 → 66.067 | -3.53 [-10.71, -3.07] |
| RTX 4090 | 12_solovev_radial99 | 99.791 → 95.113 | +7.50 [3.85, 8.51] |
| RTX 4090 | 14_vmecpp_circular | 120.780 → 125.119 | -3.77 [-4.69, -3.48] |
| RTX 4090 | 16_solovev_pressure1000_linear | 69.042 → 66.646 | +3.38 [2.94, 4.04] |
| RTX 4090 | 17_solovev_pressure5000_linear | 84.034 → 84.656 | -0.71 [-3.21, -0.48] |

The follow-up confirms RTX 4090 regressions for both circular cases, `mpol=4`
and the 5,000 Pa case. Stronger triangular shaping benefits both GPUs, with a
5.16% lower bound on its Ada follow-up gain. Some other positive estimates
remain below the adoption threshold or have wide intervals. The input family
and implementation cost both matter; merely saving outer iterations is
insufficient evidence of a speedup.

## Work counts and convergence

The following stage counts and extra evaluation counts are identical on both
architectures and repeat within each case/variant, including the fresh timing
follow-ups. Newton reduces outer iterations in every case, but adds 136–272
equilibrium evaluations and Krylov vector/reduction work. That cost is included
in all timings above.

| Case | Baseline stage iterations | Newton stage iterations | Extra equilibrium evaluations |
|---|---|---|---:|
| 00_solovev_reference | 235 / 193 / 326 | 144 / 113 / 227 | 136 |
| 01_solovev_circular | 241 / 180 / 315 | 154 / 115 / 257 | 137 |
| 02_solovev_elongation_low | 242 / 188 / 314 | 154 / 118 / 200 | 136 |
| 03_solovev_elongation_high | 246 / 202 / 407 | 161 / 116 / 300 | 170 |
| 04_solovev_triangularity_stronger | 244 / 208 / 438 | 151 / 125 / 331 | 136 |
| 05_solovev_triangularity_reversed | 240 / 181 / 291 | 152 / 101 / 200 | 136 |
| 06_solovev_pressure_half | 235 / 193 / 327 | 144 / 113 / 227 | 136 |
| 07_solovev_pressure_double | 235 / 193 / 326 | 144 / 113 / 227 | 136 |
| 08_solovev_pressure_quadratic | 235 / 193 / 327 | 144 / 113 / 227 | 136 |
| 09_solovev_pressure_flatcore | 235 / 193 / 326 | 144 / 113 / 227 | 136 |
| 10_solovev_mpol4 | 240 / 189 / 277 | 155 / 103 / 227 | 136 |
| 11_solovev_mpol10 | 232 / 205 / 354 | 150 / 122 / 245 | 136 |
| 12_solovev_radial99 | 235 / 193 / 597 | 144 / 113 / 400 | 170 |
| 13_solovev_ntheta36 | 235 / 193 / 326 | 144 / 113 / 227 | 136 |
| 14_vmecpp_circular | 310 / 368 / 649 | 200 / 253 / 476 | 272 |
| 15_vmecpp_analytical_ncurr1 | 249 / 211 / 344 | 152 / 103 / 200 | 136 |
| 16_solovev_pressure1000_linear | 244 / 192 / 293 | 160 / 101 / 200 | 136 |
| 17_solovev_pressure5000_linear | 249 / 229 / 421 | 173 / 143 / 300 | 170 |
| 18_solovev_pressure1000_quadratic | 248 / 196 / 312 | 160 / 110 / 200 | 136 |

Across screens, warmups and measured runs, all 982 full configured solves
converge: 338 on TITAN Xp and 644 on RTX 4090. Native structured RunReports
agree with the stage instrumentation. Repeated trajectories and final state
hashes are exact within each GPU and variant. No baseline failure or candidate
failure is omitted.

The `mpol=4` candidate rejects a fine-grid iteration-200 trial with GMRES
breakdown code 2 and an infinite linear relative residual. Rollback succeeds
and every full solve still converges. The private diagnostic printer emits
that nonfinite value as invalid JSON; the runner records the parse error and
preserves the raw marker rather than treating the inner solve as successful.
Stage records retain the actual extra evaluation counts, and all native
reports and stage residuals remain valid. This reinforces the need for the
existing nonlinear acceptance and rejection path.

## Solution checks

All 982 native outputs and their checkpoints pass the input-provenance,
configured-stage, finite-field, boundary/parity, dependent-axis and native-state
consistency checks. All sampled half-grid Jacobians have the required
orientation and nonzero magnitude; B² is positive. The validator's targeted
negative cases also reject altered provenance, boundary coefficients, replay
tolerances and impermissible state changes.

For each of 19 cases, both variants on both GPUs were restarted once from a
converged checkpoint with Newton disabled: all 76 replays converge at
iteration 1 using the original final tolerance. Their coefficient values are
numerically exact. Twelve states are also bit exact; the other 64 only
canonicalize 68 signed-zero entries at dependent `m>0` axis positions. No
active coefficient, boundary coefficient or nonzero coefficient changes any
bit. The gate permits only that explicit zero-sign exception. Fresh replay
normalization can slightly change the residual triple (eight are bit exact),
so each replay independently checks all three original tolerances. Final-grid
replay inputs exist only to match checkpoint shape; they are not speed
workloads.

Maximum candidate/baseline differences across the matrix are:

| Diagnostic | Maximum | Case / GPU |
|---|---:|---|
| R/Z surface displacement | 6.7212e-7 m | Low elongation / RTX 4090 |
| Lambda coefficient relative L2 | 3.4835e-6 | Low elongation / RTX 4090 |
| Native B² relative L2 | 7.0656e-8 | Low elongation / RTX 4090 |
| Native B² absolute difference | 4.2434e-7 T² | VMEC++ circular / RTX 4090 |

R/Z surfaces use plain physical Fourier synthesis at equal native poloidal
angles. Coefficient and surface comparisons exclude only the dependent
full-grid axis row; axis R00 is checked separately. B² is computed from native
contravariant/covariant fields on every half-grid point, including the
innermost half surface. These are diagnostics at equal coordinate labels,
without gauge alignment or an imposed blanket equivalence tolerance.

Independent VMEC++ v0.7.0 CPU solves use the exact adapted inputs for cases
00, 08, 14, 15, 16, 17 and 18. All seven converge with `ier_flag=0` and all
three residuals below `1e-16`. Signed-n modes are folded and exported lambda
is converted with `phipF / sqrt(sum(phipH²)/(ns-1))`; the conversion is checked
against stored internal lambda. The largest Newton/VMEC++ R/Z displacement is
6.7807e-8 m (1,000 Pa quadratic, RTX 4090), and the largest lambda relative L2
difference is 4.6316e-7 (VMEC++ circular, TITAN Xp). VMEC++ B² equivalence was
not assessed. These reference comparisons support the solution diagnostics;
cuMES's own discrete residuals and validity gates define convergence.

## Sanitizer diagnostic

The full prescribed-current candidate passes CUDA 12.1 memcheck with graphs
enabled, CUDA 12.1 initcheck with graphs disabled, and CUDA 12.9 initcheck on
RTX 4090 with graphs enabled. All full runs retain `152 / 103 / 200` iterations.

CUDA 12.1 initcheck with graphs enabled reports uninitialized PCR pivot-scale
reads; its initial full candidate run reached the 180-second timeout. A short
baseline run with Newton disabled reproduces the same report before any
correction. Disabling graphs gives zero errors and the identical residual
triple; CUDA 12.9's corresponding graph-enabled baseline check also reports
zero errors. These intentionally capped two-pass diagnostics are excluded
from the full-convergence and timing totals.

A scratch probe links the unchanged production preconditioner and checks all
13 mode scales against the exact CPU maximum of the six assembled matrix
bands. Fresh graph/direct execution and a refresh after replacing scales with
NaN sentinels produce finite, exact expected scales; matrix bands and residual
slabs also match between graph and direct paths. CUDA 12.1 initcheck still
reports the same reads during the passing probe. This supports a tool/version
tracking issue, but post-pass readback does not alone prove ordering at every
read. The old-tool graph diagnostic remains recorded as unresolved; no
blanket clean initcheck claim is made for that configuration, and no
initialization change was added merely to suppress it.

## Reproduction and retained evidence

The [benchmark directory](../benchmarks/axisymmetric_newton/README.md) contains
pinned inputs, generators, hash manifests, paired runner, scientific validator,
checkpoint replay driver and the complete timing summaries. Build the private
hook in a disposable worktree using the instructions in the original
[Newton report](newton-correction-experiments.md#reproduction-and-artifacts),
including recursive submodule initialization, then run the matrix commands
in the benchmark README. The hook is excluded from the normal solver build.

The full artifact tree is `../tmp/cumes-axisymmetric-newton-20260908/`:

- `pascal/` and `ada/`: all screen, warmup, measured and replay logs; native
  outputs/checkpoints; input/environment/command records; follow-up selection
  plans; sample and summary JSON. Original and fresh timing results remain in
  separate phase directories.
- `tested-source/`, `compile-command.json`, `binary-provenance.json` and the
  corresponding Ada source/build archive: exact private numerical sources and
  preserved executables. The committed benchmark tools add no production
  numerical behavior.
- `ref/`: seven independently converged VMEC++ 0.7.0 references with exact
  input, source/binary provenance and lambda normalization checks.
- `validation/`: checks of all native outputs/checkpoints, physical differences
  and actual checkpoint replays. `pascal/sanitizer/` and `ada-sanitizer/`
  record additional diagnostic runs separately from timing results.

This study establishes a useful axisymmetric opportunity and its limits.
It does not establish a universal policy, free-boundary/float qualification,
or a new production trajectory. A selective policy would need its own frozen
selection rule and validation rather than selecting only favorable rows from
this matrix.
