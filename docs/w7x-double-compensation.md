# W7-X: compensated double reconstruction

`CUMES_GEOMETRY_PRECISION=compensated` originally applied the same poloidal
compensation algorithm to either state scalar. `Compensated<float>` uses two
floats; `Compensated<double>` uses two doubles. Products, their sums and the
final odd-position scaling retain the low word until the result is rounded
back to the state scalar. Double inputs and outputs are never narrowed to
float.

The correction still touches only `r_o` and `z_o`. FFTs, basis/scale tables,
angular derivatives, even modes, lambda, constraints and downstream physics
retain their existing arithmetic. The solver enables the setting only for
fixed-boundary 3-D problems. `native` remains the default in both builds.

The later [float-only device policy](adr/0015-float-only-device-arithmetic.md)
changes float reduction/control precision and its trajectory. The float
byte-identity checks below apply to this double-compensation experiment.

The subsequent [single-grid float correction](w7x-single-grid-float.md) adds
m=1 toroidal compensation only for float; double arithmetic below is unchanged.

## Accuracy

NVIDIA TITAN Xp, CUDA 12.1, precise verify-double build, W7-X `mpol=12`,
`ntor=12`, `ntheta=30`, `nzeta=36`, ns=99. Both inverse transforms evaluate the
same imported double checkpoint. The independent reference sums the direct
Fourier product basis and odd normalization in host long double (64 mantissa
bits on this host), retaining long double through the error calculation.
Radial errors are errors in adjacent-surface differences divided by `ds=1/98`.

| RMS error | Native double | Compensated double | Reduction |
| --- | ---: | ---: | ---: |
| `r_o` | 1.40458e-16 | 1.32008e-16 | 6.0% |
| `z_o` | 1.39057e-16 | 1.29290e-16 | 7.0% |
| radial difference of `r_o` | 9.85304e-15 | 7.24642e-15 | 26.5% |
| radial difference of `z_o` | 1.10513e-14 | 8.41413e-15 | 23.9% |

This is a modest reduction in reconstruction error, not an extra digit of
accuracy throughout the solver. FFT and table errors remain in these fields.
The exact cancellation tests separately verify preservation of low sum/product
bits for both scalar types. Both also exercise nondefault-stream CUDA Graph
capture/replay and unchanged angular derivatives.

## Solver behavior

Cold starts with all three stage tolerances set to `1e-12`:

| Mode | Effective iterations, ns=33/66/99 | Final FSQR |
| --- | --- | ---: |
| native | 1315 → 1419 → 1372 | 9.9972988876e-13 |
| compensated | 1315 → 1419 → 1372 | 9.9972621773e-13 |

Starting both modes from the same native checkpoint and requesting `1e-16`
on ns=99 gives **3433 iterations in both modes**. Final FSQR is
9.9742903364e-17 (native) versus 9.9749005662e-17 (compensated). The small
residual changes do not reduce iteration counts at either tested tolerance.

The production input floor remains `1e-16`. To inspect smaller residuals,
the existing fixed-iteration benchmark disables convergence with its internal
`ftol=0`, starting both modes from the same native `1e-16` checkpoint and
running 10,000 passes. This does not change production validation or gates.

| Maximum of the three residuals | Native | Compensated |
| --- | ---: | ---: |
| minimum over the run | 7.26249e-20 | 7.55776e-20 |
| median over the last 1000 passes | 1.06476e-19 | 1.02725e-19 |
| range over the last 1000 passes | 7.26249e-20 – 1.29636e-19 | 7.55776e-20 – 1.30183e-19 |

These similar residual tails do not establish a meaningful solver advantage.
The tail medians are still declining, so this finite run does not establish
the ultimate residual floor. The native mode attains a slightly lower best
residual; the compensated mode has a slightly lower tail median.

## Timing

No other GPU workload or build ran during timing. The full solver benchmark
starts each run from the same native `1e-12` checkpoint, discards 300 warmup
passes and measures 500 passes. Three repetitions alternate mode ordering.
Reported values are medians of the three per-run medians.

| Per evaluated solver pass | Native double | Compensated double |
| --- | ---: | ---: |
| median time | 1574.61 µs | 1867.31 µs |

The full-pass overhead is **18.6%** on this TITAN Xp. The isolated inverse graph
measures 493.26 → 782.82 µs (one run, 3000 warmup replays followed by 15 samples
of 100 replays). Double-double uses additional FP64 instructions; the result
is hardware-dependent and does not predict the overhead on other GPUs.

Given the measured cost and unchanged convergence counts, native double
remains the default. Compensated double is available as an opt-in experiment.
The float reconstruction still uses precisely the previous float-float
arithmetic sequence.

## Regression checks

- Verify and mixed-float builds pass all 102 and 64 CTest cases, respectively.
  Verify includes 19 memcheck and 19 initcheck cases; the reconstruction test
  exercises both scalar types.
- The native double cold-start checkpoint is byte-identical to the prior
  qualified result. Float compensated W7-X also retains its byte-identical
  checkpoint and 149 → 277 → 322 iteration counts.
- Default precision, radius-reference policy, residual definitions, input
  tolerance floors and controller gates are unchanged.

## Reproduction

```sh
# Both precisions run cancellation and reconstruction correctness checks.
./build/tests/test_odd_geometry

# Accuracy and inverse timing on an ns=99 physical double checkpoint.
./build/tests/test_odd_geometry --benchmark-double w7x-native.ckpt

# Every stage in this input has tolerance 1e-12.
CUMES_GEOMETRY_PRECISION=compensated ./build/cumes w7x-double.json \
  --checkpoint w7x-compensated.ckpt --output w7x-compensated.bin

# Full-pass timing; use --odd-geometry native for the control.
./build/cumes_benchmark_fixed_iteration --config w7x \
  --restart w7x-native.ckpt --warmup 300 --passes 500 \
  --odd-geometry poloidal --out timing.json
```

Raw local experiment artifacts: `/tmp/cumes-double-compensation/` (commands,
input variants, checkpoints, residual telemetry, reconstruction output and
individual timing records).
