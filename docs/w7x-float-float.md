# W7-X: selective float-float reconstruction

This experiment follows the [inverse-transform isolation](w7x-float-inverse-diagnostic.md).
The smallest successful scope in the original multigrid experiment used
float-float only for the poloidal
products, accumulation and final odd-scale multiplication of `r_o` and `z_o`.
It retains the float cuFFT, float basis tables and float scale value, along with
float state and downstream geometry. It is fused into the existing R/Z kernels:
no additional launches, scratch arrays or hot-loop allocations are needed for
the diagnostic `poloidal` scope. The current `compensated` setting additionally
corrects m=1 toroidal sums and the odd scale to support
[single-grid cold starts](w7x-single-grid-float.md). Native remains the default.

The measurements below describe the original reconstruction experiment.
The subsequent [float-only device policy](adr/0015-float-only-device-arithmetic.md)
also replaces double norm accumulation/control records and reference arithmetic.
Its W7-X multigrid qualification was 149 → 278 → 314 iterations at all-stage `1e-5`.

## Usage and scope

Set **every** W7-X `ftol_array` entry to `1e-5`, then run:

```sh
CUMES_GEOMETRY_PRECISION=compensated \
  ./build-float/cumes w7x-1e-5.json --checkpoint w7x.ckpt --output w7x.bin
```

`CUMES_GEOMETRY_PRECISION` accepts `native` (default) or `compensated`.
Compensation improves the odd R/Z position reconstruction with selective
paired-scalar arithmetic: float-float in float builds, double-double in double
builds. It does not change the precision of every geometry operation. This
document reports the float experiment; see the separate
[double measurements](w7x-double-compensation.md).

Embedding callers select the same correction with
`SolveRequest::odd_geometry = OddGeometryPrecision::COMPENSATED`. The `POLOIDAL` scope retains the original
multigrid experiment.
Environment parsing is enabled only when `use_process_environment` is true.
The solver applies it only to fixed-boundary 3-D runs. Reference storage
is enabled by default for float solves; `CUMES_RADIUS_REFERENCE=0` or
`SolveRequest::use_radius_reference=false` restores absolute coefficients.
The measurements below were taken with reference storage enabled.
Checkpoint replay must use the same options.

For diagnostic experiments, the library API and the benchmark's
`--odd-geometry` option retain these detailed float scopes. Double supports
`native`, `poloidal` and `compensated` (the last two are identical in double).

| Diagnostic option | Extra precision in the two odd position fields |
| --- | --- |
| `native` | None; existing transform |
| `float-order` | None; diagnostic control moving scaling after the sum |
| `sum` | Float products, float-float accumulation and final multiplication |
| `poloidal` | Float-float products, accumulation and final multiplication |
| `poloidal-scale` | `poloidal`, plus a scale stored as a split float pair |
| `compensated` | Float products and float-float sums for m=1 toroidal positions, retained through `poloidal-scale` reconstruction |
| `float-float` | Additional direct toroidal/poloidal reconstruction using float-float throughout |

The full `float-float` reconstruction calculates only odd R/Z positions, using
four toroidal channels and a separable two-kernel reconstruction. Other outputs
still require the existing cuFFT. The historical `double` diagnostic used the
same algorithm and tables; it has been removed to keep float kernels free of
FP64. The basis and scale constants are prepared in double on the host at setup; float-float stores high/low float
parts, and its new device arithmetic uses float instructions. The
`poloidal-scale` option similarly prepares only the radial scale table at setup.

`Compensated<T>` carries an unevaluated `hi + lo` sum; `FloatFloat` names its
float specialization. Explicit rounded CUDA
addition/subtraction and FMA product residuals protect its compensation from
compiler contraction. Both words are retained until the final float output.
The minimal successful path evaluates

```
q = sum over odd m [FF(c0) * FF(t0) + FF(c1) * FF(t1)]
r_o or z_o = float(q * FF(facO))
```

It does not first round each product pair or each scaled modal contribution to
float. Even modes, angular derivatives, lambda and constraint arithmetic retain
their existing expressions. The original reconstruction experiment left norm
reductions and controller gates unchanged; ADR-0015 documents their subsequent
precision change. Tolerances are unchanged. This is an opt-in numerical
experiment, not a frozen trajectory refactor or a guarantee for other
configurations/tolerances.

## Precision and convergence

TITAN Xp, CUDA 12.1, precise mixed-float build; W7-X `mpol=12`, `ntor=12`,
`ntheta=30`, `nzeta=36`, radial grids 33/66/99, all stage tolerances `1e-5`.
All comparisons below enable the radius reference.

The initial scope-isolation experiment, using supplementary reconstruction
kernels on an identical imported tight double checkpoint, gives:

| Reconstruction | FSQR |
| --- | ---: |
| Native float | 1.100810e-5 |
| Float scaling moved after sum | 1.059034e-5 |
| Float-float sum, rounded products | 8.792237e-6 |
| Float-float poloidal products and sum | 8.585289e-6 |
| Same, with split scale | 6.437060e-6 |
| Full odd reconstruction, float-float | 2.016604e-6 |
| Full odd reconstruction, double | 2.016604e-6 |

The one-pass improvement alone is insufficient evidence of cold-start
convergence: rounded products with compensated addition still stall in the
cold-start experiment. The minimal successful version retains product errors
as well. Moving the scale alone also fails.

The fused poloidal and split-scale variants evaluate that same checkpoint at
FSQR 8.585131e-6 and 6.436789e-6, respectively. Their small arithmetic
differences do not change the scope conclusion.

The retained fused implementation converges from a cold start:

| Reconstruction | Effective iterations by grid | Final FSQR | Final FSQZ |
| --- | --- | ---: | ---: |
| Poloidal float-float | 149 → 277 → 322 | 9.837748e-6 | 4.785764e-6 |
| Poloidal + split scale | 149 → 277 → 313 | 8.838591e-6 | 4.165528e-6 |
| Full odd float-float | 148 → 208 → 315 | 4.988024e-6 | 2.491077e-6 |
| Full odd double control | 148 → 204 → 310 | 4.864334e-6 | 2.627977e-6 |

All four checkpoints converge on their first replay pass. Graph-enabled runs
with dumps disabled reproduce these iteration counts. Earlier supplementary
kernel versions had slightly different trajectories; their logs are preserved
under `mg-*`, while the table reports the retained `fused-mg-*` implementation.

An independent long-double host Fourier sum, using the exact imported float
coefficients at ns=99, measures the following RMS errors. Radial error is the
error in adjacent-surface differences divided by `ds=1/98`.

| Reconstruction | `r_o` RMS | `z_o` RMS | R radial RMS | Z radial RMS |
| --- | ---: | ---: | ---: | ---: |
| Native | 6.86e-8 | 6.96e-8 | 5.42e-6 | 6.06e-6 |
| Poloidal float-float | 6.37e-8 | 6.39e-8 | 4.11e-6 | 4.67e-6 |
| Poloidal + split scale | 6.15e-8 | 6.07e-8 | 3.40e-6 | 3.73e-6 |
| Full odd float-float/double | 1.04e-8 | 1.13e-8 | 1.44e-6 | 1.56e-6 |

The minimal correction particularly reduces the surface-to-surface error that
radial differentiation amplifies. Full reconstruction reaches essentially the
float output rounding floor; it is unnecessary for this `1e-5` target.

## Timing

The values in this section precede caching of the fixed radius reference.
The [subsequent cache measurement](adr/0014-float-radius-reference.md#cache-the-fixed-reference-2026-09-07)
reduces native-reference passes from 541.90 to 525.55 µs and poloidal
float-float passes from 559.29 to 543.02 µs, preserving the numerical results.


The final-grid fixed-iteration harness starts from the same tight checkpoint,
uses `ftol=0` to prevent early termination, discards 300 warmup passes and times
500 passes. Three repetitions use different ordering; the table gives the
median of the three per-run medians. Dumps are off and no other GPU process is
run concurrently. These are measurements on this TITAN Xp, not portable speed
claims. The native first repetition was slower during initial warmup; retaining
all three and taking their median gives the values below.

| Reconstruction | Median solver pass | Relative to native |
| --- | ---: | ---: |
| Native | 543.64 µs | 1.000 |
| Poloidal float-float | 560.93 µs | 1.032 |
| Poloidal + split scale | 560.65 µs | 1.031 |
| Full odd float-float | 673.41 µs | 1.239 |
| Full odd double | 612.96 µs | 1.128 |

A separate inverse-only CUDA Graph benchmark measured approximately 117 µs
native, 137 µs for either fused poloidal version, 248 µs for full odd
float-float, and 192 µs for the full odd double control. This isolates the
transform cost; the whole-pass benchmark includes the real controller's
maintenance/restart work and host submission/fence time. These are per-pass
costs, not time-to-convergence speedups against the nonconverging baseline.

For this configuration, `poloidal` is the smallest successful scope tested.
`poloidal-scale` has indistinguishable per-pass cost in this measurement, less
radial error and slightly more convergence margin. Full float-float is slower
than the same full double reconstruction in this implementation, reinforcing
the value of limiting its scope.

Reproduce timing after a float build with benchmarks enabled:

```sh
./build-float/cumes_benchmark_fixed_iteration_float --config w7x \
  --passes 500 --warmup 300 --restart tight-w7x.ckpt \
  --radius-reference 1 --odd-geometry poloidal --out timing.json
./build-float/tests/test_odd_geometry --benchmark tight-w7x.ckpt
```

## Verification and artifacts

`test_odd_geometry` compares against independent host direct synthesis, checks
that poloidal compensation reduces radial-difference error, and verifies
nondefault-stream graph capture/replay and unchanged odd angular derivatives.
The test is registered for ordinary runs and Compute Sanitizer variants.
The float suite passes 64/64 tests. The verify suite passes 102/102, including
19 memcheck and 19 initcheck cases. The default double W7-X cold-start run
retains 1315 → 1419 → 1372 iterations at 1e-12, with a byte-identical
checkpoint and final-stage per-pass telemetry compared with the pre-change
executable. `git diff --check` passes.

Commands, checkpoints, per-pass telemetry, precision comparisons, timing JSON
and logs are preserved under
`/lustre/qzhong/cumes-diagnostics/w7x-float-investigation/float-float/`.
`run_bench.py` reproduces the timing order. The parent directory contains the
input variants, tight checkpoint and `run_case.py` convergence runner.
