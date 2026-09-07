# W7-X: single-grid float convergence

The previous compensated mode stalled on a cold ns=99 float solve at
`ftol=1e-5`, despite converging through multigrid and replaying its checkpoint.
The retained correction adds compensated accumulation for only the four
`m=1` toroidal R/Z position channels and preserves their low words through
poloidal reconstruction. All production float kernels remain free of FP64.

## Isolation

NVIDIA TITAN Xp, CUDA 12.1, precise float build. Start from `inputs/w7x.json`
with `ns_array=[99]`, `niter_array=[5000]`, `ftol_array=[1e-5]`. Use the
existing radius reference, seed envelope 0.129, and normal controller.

| Reconstruction/control | Result |
| --- | --- |
| Native geometry | Not converged in 5,000 passes; best maximum residual 2.8622e-5 |
| Previous poloidal compensation | Not converged in 20,000 passes; best maximum residual 1.6972e-5 |
| Accurate split odd scale + poloidal compensation | Not converged in 5,000 passes |
| Seed envelope 0 or 0.12, initial step 0.5, or damping floor 0.05 (separate poloidal-only runs) | None converged in 5,000 passes |
| Full float-float direct odd reconstruction | Converged in 1,214 effective iterations |
| All odd toroidal channels compensated, fused poloidal reconstruction | Converged in 1,338 effective iterations |
| Only m=1 toroidal channels, compensated products and sums | Converged in 1,378 effective iterations |
| Only m=1 toroidal channels, float products and compensated sums | Converged in 1,354 effective iterations; retained |

On the same stalled checkpoint, the independent host long-double Fourier
reference gives the following radial-difference RMS errors. These measure
reconstruction error at fixed stored coefficients, independently of the
solver's normalization and controller.

| Reconstruction | R radial RMS error | Z radial RMS error |
| --- | ---: | ---: |
| Native | 5.4386e-6 | 6.3197e-6 |
| Poloidal compensation | 4.1920e-6 | 4.9788e-6 |
| Poloidal compensation + split scale | 3.4473e-6 | 4.0418e-6 |
| Retained m=1 sums + poloidal compensation | 1.6552e-6 | 1.7167e-6 |
| Full float-float reconstruction | 1.4479e-6 | 1.5835e-6 |

The retained correction reduces radial RMS error by 60.5% in R and 65.5%
in Z relative to the preceding poloidal-only scope.

The remaining problem is precision in the odd-position reconstruction before
its final poloidal sum. The dominant m=1 channels contain the large odd
position contributions. Toroidal rounding and subsequent odd normalization
produce surface-to-surface errors that radial differentiation amplifies.
Poloidal compensation cannot recover information already lost in the
single-precision toroidal result. Higher odd toroidal channels need no
additional compensation for the tested case.

The stalled state is not itself a converged solution hidden by inaccurate
residual reporting: evaluating that exact checkpoint with the full correction
still gives FSQR about 3.26e-5. Correcting reconstruction throughout the
iteration allows the state to converge. Tolerances, normalization definitions,
seed shaping, time steps and controller rules are unchanged.

## Implementation and usage

`CUMES_GEOMETRY_PRECISION=compensated` selects the new correction in float.
Embedding callers use `OddGeometryPrecision::COMPENSATED`; the diagnostic
`POLOIDAL` / `--odd-geometry poloidal` retains the preceding implementation.
Native geometry remains the default. Double compensated solves keep their
previous poloidal double-double arithmetic.

One additional kernel forms four m=1 toroidal sums with float products and
float-float accumulators. The existing fused R/Z kernels consume both words
for only the odd positions, retaining compensation through poloidal products,
sums and the final split odd scale. Derivatives, even modes, lambda and
constraints still consume their original cuFFT outputs. The additional m=1
branch changes compiler unrolling/FMA contraction: the four odd derivative
outputs differ from native by at most 1.2e-7 in the unit fixture, with no
change to their expressions or precision. Tests allow a few float ULPs for
those four fields and require exact preservation of the other sampled fields.
No field is converted
to double and no allocation is added inside the iteration.

Scratch contains `4 * ns * nzeta` float pairs (114,048 bytes at ns=99,
nzeta=36), plus immutable toroidal basis and radial scale tables prepared at
startup. The full odd reconstruction diagnostic uses six times as much
channel scratch for W7-X and a separate poloidal kernel; neither is required
by the retained correction.

```sh
CUMES_GEOMETRY_PRECISION=compensated ./build-float/cumes w7x-ns99-1e-5.json \
  --checkpoint w7x-single.ckpt --output w7x-single.bin
```

Use the same input and setting with `--restart w7x-single.ckpt` for replay.

Local experiment commands, checkpoints, source variants and logs are in
`/lustre/qzhong/cumes-diagnostics/w7x-single-grid-fix/`; the preceding stall
records are in `/lustre/qzhong/cumes-diagnostics/w7x-single-grid-float/`.


## Final qualification and cost

| Solve | Effective iterations | Final FSQR / FSQZ / FSQL |
| --- | --- | --- |
| Single-grid cold start, ns=99 | 1,354 (1,363 evaluated passes including restarts) | 9.784521353e-6 / 4.695623375e-6 / 2.392124498e-9 |
| Single-grid checkpoint replay | 1 | 9.784496797e-6 / 4.647216883e-6 / 2.392124721e-9 |
| Multigrid cold start, 33/66/99 | 149 → 277 → 311 | 9.883718121e-6 / 5.675343800e-6 / 2.955646394e-9 |
| Multigrid checkpoint replay, ns=99 | 1 | 9.883718121e-6 / 5.614867405e-6 / 2.955646394e-9 |

Final CLI commands and telemetry are in `qualified/` under the artifact root.
The native double W7-X and Solovev checkpoints and final-stage telemetry are
byte-identical to the previous qualified results. `compare_runs` reports zero
state/residual differences and identical restart sequences. Double compensated
reconstruction is also checked bit-for-bit against its existing poloidal path.

The full suites pass 103/103 verify and 66/66 float tests. After retaining the
separate single-word higher-mode expressions, the single-grid/replay test,
float/double inverse tests, FP64 audit, memcheck and initcheck pass again.
The audit covers every configured architecture (61/75/80/86/89): 355 cuMES
and 175 vacuum function sections contain no FP64 instructions. Graph
replay agrees with ordinary launches. The cold-start regression uses the
normal input, one ns=99 stage, a 5,000-pass cap and the unchanged 1e-5 tolerance;
it does not freeze an exact iteration count across GPUs.

Three alternating trials on the same imported tight W7-X checkpoint, with
300 warmup and 500 measured passes per trial, give these median per-pass costs:

| Reconstruction | Median time |
| --- | ---: |
| Previous poloidal-only compensation | 546.79 µs |
| Retained m=1 toroidal + poloidal compensation | 562.85 µs |
| Full float-float odd reconstruction | 660.94 µs |

The retained correction costs 2.9% more per pass than the preceding poloidal
scope and 14.8% less than the full diagnostic. The old poloidal scope does not
converge this single-grid cold start, so per-pass timing alone is not a
solution-time comparison. The first poloidal timing was 601.44 µs; the other
two were 546.68 and 546.79 µs. All samples are retained in `timing/`.

Reproduce the float timing with the explicitly float benchmark target:

```sh
./build-float/cumes_benchmark_fixed_iteration_float --config w7x \
  --restart tight-w7x.ckpt --odd-geometry compensated \
  --warmup 300 --passes 500 --out timing.json
```
