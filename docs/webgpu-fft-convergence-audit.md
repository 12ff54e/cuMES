# W7-X FFT convergence causality audit (2026-09-06)

## Conclusion

For this WebGPU solver, a tiny change in **one iteration's forward projection**
can change the number of iterations needed to reach `1e-12` by about 10%.
This is demonstrated by an intervention, not inferred solely from separate
full FFT/direct runs. The measured transform discrepancies are small numerical
errors; the audit finds no sign, normalization or indexing mismatch in the
tested W7-X projections. It does not prove the absence of every possible bug,
nor establish a universal sensitivity property of VMEC or FFTs.

The faster direct trajectory is not an accuracy oracle. More accurate zeta
coefficients in the direct projection also change its iteration count, and
the independent reference generally favors FFT over legacy direct for the
toroidal stage. End-to-end solve times remain valid measurements of the
current routes, but are not isolated transform-speed comparisons.

## Controlled interventions

Served build: cuMES `bb590a0`, standalone FFT `c985220`; Chrome/D3D12,
RTX 3060 Ti. All runs use the unchanged single-grid W7-X input, paired state,
`trace=1&gpu_norms=1&gpu_control=jacobian&timing=0`, and tolerance `1e-12`.
Each experiment has its own visible browser tab; GPU solves run sequentially.
Timings are not the acceptance criterion for these diagnostics.

`webgpu_route_experiment.mjs` changes only the existing `fft` URL option after
a specified controller record. No state, force, damping, residual, grid or
controller setting is overwritten. The next evaluation consumes that option
for both unconstrained and constrained forward projections. Native operator
code is unchanged.

| Experiment | Effective iterations |
| --- | ---: |
| Direct throughout | 2812 |
| Direct, sham switches after 100 and 101 (still direct) | 2812 |
| Direct except FFT at iteration 2 | 2812 |
| **Direct except FFT at iteration 101** | **3100** |
| Same iteration-101 intervention repeated | 3100 |
| Same intervention with GPU snapshots enabled | 3100 |
| FFT throughout | 3091 |
| Direct with canonical zeta table throughout | 3149 |
| Direct through 2500, then FFT | 2792 |
| FFT through 2500, then direct | 3287 |

Iteration 101 is after the final restart anchor at 93. The iteration-2
intervention is discarded: only that controller record differs, and the
subsequent trace rejoins direct exactly. In contrast, the iteration-101 pulse
matches all **105 recorded controller entries through effective iteration
100**, then diverges at attempt 110 / iteration 101. Its 3105 controller
records reproduce exactly in both the repeat and the capture-enabled run.
The sham's 2817 records match the direct baseline exactly.

For the pulse versus direct:

- State high/low fingerprints still match when iteration 101 is evaluated.
- The preconditioned search-direction fingerprint first differs there.
- The state high/low fingerprints first differ at iteration 102.
- The damping values used by descent (`b1` and `fac`, after conversion to
  f32) first differ at iteration 127, not at the intervention itself.
- Time step, restart reason, restart anchor and preconditioner-refresh
  schedule match over the shared trace. Checkpoint selection is not included
  in this assertion; there are no later restores that consume checkpoints.

This demonstrates propagation through a repeated state update. It does not
identify a single unstable eigenmode or quantify the separate contributions
of the f32 preconditioner and adaptive damping. Calling it simply an immediate
controller-branch discrepancy would be incorrect.

## Exact intervention inputs and measured perturbation

`webgpu_transform_capture.js` copies buffers immediately after the poloidal
pass, in the same encoder, before a subsequent projection can overwrite them.
It only adds diagnostic COPY_SRC access and copies; no shadow result is fed
to the solver. Selected radial surfaces are **1, 50 and 97** of ns=99.
All 20 fields, all 30 theta points and all 36 zeta points are captured on those
surfaces. Full residual vectors (all six families, all 156 modes, all 99
surfaces) are captured, as are the selected toroidal intermediates and the
actual basis/parameter buffers.

At iteration 101, pulse and sham have **bit-identical captured high/low input
fields, basis buffers and parameters**, in both projection phases. Across
each entire 92664-coefficient residual vector:

| Projection phase | FFT/direct relative L2 difference | Maximum absolute difference | Different high words |
| --- | ---: | ---: | ---: |
| Unconstrained force | 7.4586e-13 | 9.9481e-13 | 66 / 92664 |
| Constrained force | 3.4987e-13 | 5.1248e-13 | 70 / 92664 |

The invariant FSQR at that evaluation is `0.008485486555416673` direct and
`0.008485486555416710` with the FFT pulse. The unchanged damping and changed
search direction show that the initial state perturbation is not caused by
different damping at that pass. The radial preconditioner consumes scalar-f32
forces and returns the scalar-f32 direction consumed by paired descent; paired
state/invariant reductions do not make those directions identical.

## Independent higher-precision reference

`webgpu_transform_oracle.cpp` is an offline CPU implementation using Boost
`cpp_bin_float_quad`: **113 significand bits**, including its trigonometry.
It does not call the GPU arithmetic or FFT generator. It evaluates toroidal
DFTs followed by the six-family poloidal projection. Analytic constant and
harmonic tests exercise normalization, cosine/sine signs, derivatives and
field-period scaling (maximum error 1.54e-15 from paired input rounding).
A native extended-precision recomputation provides a second precision check.

It reports three quantities separately:

1. Toroidal intermediates versus high-precision mathematical zeta roots.
2. Complete residuals using those roots but retaining the **actual shared
   theta basis, normalization and sqrt(2) coefficients**. This isolates the
   changed stage and arithmetic from pre-existing common basis errors.
3. Complete residuals using high-precision mathematical coefficients for
   both angles, normalization and sqrt(2).

An additional legacy-direct comparison uses its own captured zeta roots to
separate coefficient error from summation/arithmetic error. These are
projection-error metrics, not normalized squared solver residuals; they must
not be compared numerically to `ftol` as though the units were the same.

Selected results, relative L2 error against the 113-bit reference:

| Iteration / phase | Quantity | Optimized FFT | Legacy direct |
| --- | --- | ---: | ---: |
| 1 / force | Toroidal intermediate | 2.919e-15 | 1.093e-14 |
| 101 / force | Toroidal intermediate | 2.293e-15 | 1.593e-14 |
| 101 / force | Residual, shared theta coefficients | 6.231e-13 | 1.414e-12 |
| 101 / constrained | Residual, shared theta coefficients | 4.943e-13 | 8.377e-13 |
| 2800 / force | Toroidal intermediate | 2.437e-15 | 1.566e-14 |
| 2800 / constrained | Residual, shared theta coefficients | 6.513e-9 | 1.238e-8 |

The late force vector is cancellation-dominated: small absolute errors become
larger relative errors as the net force decreases. The reference precision
check is substantially below GPU errors (e.g. extended/quad agreement is
2.22e-13 relative L2 for the late constrained projection versus GPU errors
of order 1e-8). FFT is not uniformly best under every metric: with fully
mathematical theta coefficients, direct is slightly closer on the iteration-101
constrained sample (1.480e-12 versus FFT's 1.545e-12). There is no basis here
for attributing the FFT trajectory's extra iterations to systematically worse
transform accuracy.

The early/late shadow capture run retained all 3096 FFT controller records'
state, low-state, preconditioned-direction, time-step, restart, anchor,
refresh and checkpoint fields. It converged in the same 3091 iterations.
The selected-surface oracle is not exhaustive coverage of every radial point
or every possible problem; full-buffer pulse differences and existing
transform conformance supply complementary checks.

## Reproduction and evidence

All evidence below is under the workspace's `../tmp`. The scripts create and
operate only on their own Chrome tabs, through the user-provided port 9333.
No `EM_JS` or production numerical changes were introduced.

```bash
node scripts/webgpu_route_experiment.mjs \
  'http://localhost:6969/magnetic-equilibrium-solver/tmp/cumes-build-webgpu-ds/webgpu/cumes_webgpu.html?solve=w7x&trace=1&gpu_norms=1&gpu_control=jacobian&fft=0&timing=0&audit_iterations=101' \
  ../tmp/w7x-causality-pulse-capture \
  '[{"after":100,"fft":1},{"after":101,"fft":0}]' \
  scripts/webgpu_transform_capture.js
# Sham: use fft=0 in both schedule entries and a distinct output prefix.
# Early/late audit: use fft=1&compare_fft=1, omit audit_iterations, schedule [].

export TMPDIR=/lustre/qzhong/magnetic-equilibrium-solver/tmp
g++-12 -std=c++20 -O2 -ffp-contract=off -Wall -Wextra -Werror \
  scripts/webgpu_transform_oracle.cpp -o ../tmp/webgpu-transform-oracle
../tmp/webgpu-transform-oracle --self-test
../tmp/webgpu-transform-oracle \
  ../tmp/w7x-causality-audit-capture-0.bin \
  ../tmp/w7x-causality-audit-capture-1.bin \
  ../tmp/w7x-causality-audit-capture-2.bin \
  ../tmp/w7x-causality-audit-capture-3.bin
```

Capture groups 0–3, 4–7, 8–11 and 12–15 are respectively early force, early
constrained, late force and late constrained. Within each group: optimized
FFT, generic FFT, legacy direct and canonical direct. The JSON capture
manifest describes every binary section. For a two-route pulse/sham check,
the oracle accepts full primary captures in the comparison slots; duplicated
slots do **not** represent additional measured variants.

Evidence prefixes/files:

- `w7x-causality-{canonical,one-fft,one-fft-101,one-fft-101-repeat,sham,late-fft,late-direct}`:
  full logs, controller traces and route-switch records.
- `w7x-causality-{pulse-capture,sham-capture}`: full captured intervention runs,
  section manifests and binary snapshots.
- `w7x-causality-audit`: full early/late shadow run and 16 snapshots.
- `w7x-causality-pulse-difference.json`, `w7x-causality-oracle.json`,
  `w7x-causality-pulse-oracle.json`: same-input differences and oracle metrics.
  The early/late oracle was first evaluated from the identical snapshots
  exported while that run was still completing (`w7x-causality-early`).

## Remaining question

The causal claim is established for this case. It is **not yet established**
how much a paired preconditioner or revised damping would reduce the
sensitivity, whether the weak lambda/gauge directions dominate it, or which
trajectory is preferable beyond satisfying cuMES's own convergence criterion.
Those require separate numerical experiments; this audit does not silently
change the solver or choose a trajectory based on iteration count alone.
