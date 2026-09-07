# ADR-0015: Float arithmetic throughout float GPU kernels

- Status: Accepted; supersedes the float device policy in ADR-0001
- Date: 2026-09-07

## Context

The float solver still used FP64 in control-feeding reductions, the device
control record and its finalization/predicate kernels, radius-reference
reconstruction, and an optional full-double odd-geometry diagnostic. The
vacuum dependency also had implicit promotions and CUDA float trigonometric
calls whose compiled large-angle fallback used FP64. The requested policy is
float arithmetic throughout float kernel functions, while retaining selective
compensation and the existing double solver.

## Decision

- `NormAccum<float>` is `FloatFloat`, a pair of floats with compensated sums.
  The existing per-element norm expressions still evaluate in float. Shared
  reduction trees retain both words; final results are rounded to float.
  `NormAccum<double>` remains double.
- `DeviceControlRecord<T>` uses T for all floating fields. The Jacobian argmin
  is an integer so large grid indices remain exact. Force-norm finalization,
  normalization, Jacobian validity and terminal classification all use T.
- The normalized residual triple is stored in the device record. After the
  single transfer/fence, the host widens the record to double and consumes
  that triple directly, avoiding a second normalization with different
  rounding. Adaptive damping/restart bookkeeping remains in
  `IterationController<double>`. Jacobian classification uses the device
  scalar on both sides of the fence.
- The immutable radius reference retains double host metadata for physical
  checkpoint export and subtraction before float import. Device views and
  the scalar mean use T. The cached angular reconstruction uses a cosine
  table prepared on the host and uploaded in T, with float-float accumulation.
  This still runs once per stage before graph capture.
- Remove the full-double reconstruction diagnostic from float dispatch and
  from the diagnostic parser. The public `native|compensated` setting stays
  unchanged. Compensated double solves still use the retained templated
  double-double implementation.
- In vacuum-field, use T for kernel metadata and arithmetic constants. Float
  axis/period rotations read immutable T sine/cosine tables prepared during
  host startup. This also removes the FP64 fallback embedded in CUDA's float
  trigonometric implementations. Double kernels retain their existing path.

Host setup, controller arithmetic, vacuum's host dense solve, and on-disk
state remain double. cuFFT continues to use its single-precision API for
float transforms. No hot-loop allocations or new per-pass transfers are
introduced. The `mixed-float` provenance name is retained: it now describes
float device arithmetic with a double host controller, rather than FP64
GPU reductions.

## Verification

`float_kernel_types` inspects SASS for FP64 arithmetic/conversion instructions
and PTX for `.f64` types in `cumes_cuda_float` and `vfield_cuda_float`. It checks
all embedded architectures, not just the installed GPU. Registration requires
`cuobjdump`; real-only builds may have no PTX. The configured 61/75/80/86/89
build contains 340 cuMES and 175 vacuum function sections and passes. Auditing
the double archive as a negative control fails as expected.

The safety-predicate tests instantiate float and double and compare device
and host decisions, including exactly-at-tolerance and adjacent-ULP cases.
The accumulation test compares float-float sums against a host long-double
reference. Existing reference-storage, geometry, free-boundary and sanitizer
checks cover the changed operators.

## Consequences

Float normalization rounds differently from the previous double path, so its
trajectory is intentionally changed; no tolerance or residual definition is
relaxed. W7-X with reference storage and compensated reconstruction still
converges at all-stage `1e-5`. The input floor remains `1e-6`, with convergence
case-dependent. Removing FP64 does not imply every reduction is faster:
float-float uses multiple FP32 operations to retain small summands.

## Qualification on TITAN Xp, CUDA 12.1

Both precise builds complete. CTest passes **103/103 verify** (including 19
memcheck and 19 initcheck cases) and **65/65 float**.

| Case | Effective stage iterations | Final FSQR / FSQZ / FSQL |
| --- | --- | --- |
| Float W7-X, all-stage `1e-5`, reference + compensated | 149 → 278 → 314 | 9.288121873e-6 / 4.655083558e-6 / 2.253401465e-9 |
| Same float checkpoint, ns=99 replay | 1 | 9.288121873e-6 / 4.561993137e-6 / 2.253401465e-9 |
| Native double W7-X, all-stage `1e-12` | 1315 → 1419 → 1372 | 9.997298888e-13 / 2.818063973e-13 / 1.299116996e-13 |
| Native double Solovev, all-stage `1e-16` | 235 → 193 → 326 | 9.972963054e-17 / 5.409546518e-18 / 9.512710225e-21 |

The native double checkpoints and final-stage telemetry are byte-identical
to the preceding qualified runs; `compare_runs` reports zero state/residual
differences and identical restart sequences. Float's previous 149 → 277 → 322 trajectory
changes as expected from the new reduction/finalization precision. The replay
converges immediately; its normalization refresh changes FSQZ, so no identical
float residual-triple claim is made.

Raw commands, inputs, checkpoints, logs, telemetry and summaries are in
`/lustre/qzhong/cumes-diagnostics/float-only-device/`. The reference inputs
use `inputs/w7x.json` and `inputs/solovev.json` with every stage tolerance set
to the corresponding value in the table. Reproduce the float solve with:

```sh
CUMES_GEOMETRY_PRECISION=compensated ./build-float/cumes w7x-1e-5.json \
  --checkpoint w7x.ckpt --output w7x.bin
```

For replay, set `ns_array=[99]`, `niter_array=[5000]`, `ftol_array=[1e-5]`
and pass `--restart w7x.ckpt` with the same geometry setting.
