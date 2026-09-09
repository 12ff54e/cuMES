# ADR-0019: WebGPU vacuum backend with shared host coupling

Date: 2026-09-09

Status: Implemented, opt-in. HOST remains the browser default.

## Decision

Compile the dependency's `VFIELD_BACKEND=WEBGPU` and select its paired-f32
GPU solver with `vacuum=webgpu`. The CUDA portion of vacuum-field is portable:
its six WGSL modules implement geometry, external fields, integrals,
Fourier/Laplace assembly and reconstruction. MAKEGRID and immutable coefficient
setup remain Wasm double, as does the dense LU step already used by native CUDA.
The dependency owns the numerical implementation; cuMES owns scheduling and
LCFS coupling. Its [precision ADR](../../deps/vacuum-field/docs/adr/0001-webgpu-paired-vacuum.md)
records the maintained native/WGSL correspondence, accurate twofold arithmetic,
recurrence conditioning, tangent poles and complete-solver reference gates.

`FreeBoundaryOperator::enable_webgpu` reuses its already loaded MgridProvider.
It is called before the first factorization. The integration retains the HOST
operator storage, allowing a small change to the existing setup path; it does
not regenerate the coil grid. GPU buffers, parameters and scratch persist
across updates. The existing host controller calls an Asyncify compatibility
method that yields once while the asynchronous GPU driver maps matrix/RHS,
runs double LU, uploads the potential and maps the final vacuum outputs.
There is no polling or map per integral kernel. Diagnostic captures add a
separate map only in the dependency tests.

The existing activation/current extrapolation, `nvacskip`, first-full-update,
soft restart, constraint decay, LCFS pressure and preconditioner behavior remain
in the shared coupling. GPU pressure replaces the HOST pressure array only
when that backend is selected. Multigrid keeps the same vacuum instance.
`vacuum=host` retains the original HOST/Wasm double update. Coil selection,
uploads and in-memory MAKEGRID generation are unchanged.

The public dependency API also provides resident output views for GPU consumers.
The current cuMES compatibility path still reconstructs LCFS coupling inputs
and applies the small edge correction in Wasm. Moving that remaining coupling
or LU onto GPU is a separate optimization, not a claim of this implementation.

## Numerical qualification

This backend changes arithmetic precision and can change discrete controller
decisions. Treat consumer qualification as Class C; paired-f32 is not native
binary64. Default HOST behavior must remain exact. The implementation does not
weaken configured residual thresholds to accommodate the new backend.

The WebGPU build and 14 browser CTests passed. Real GPU checks used Chrome
152 on the forwarded NVIDIA GeForce RTX 3060 Ti:

- The complete integrated WGSL conformance suite, including m=1 geometry and
  Newton numerical fixtures.
- The dependency's 1,770 arithmetic comparisons, 30 integral fixtures, full and
  partial trusted CTH-like/axisymmetric/asymmetric solves, per-stage arrays at
  native bounds, concurrent-update rejection and failure recovery.
- All 19 dependency tests separately on HOST, Wasm and CUDA, and the native
  cuMES vacuum-bridge test on a TITAN Xp.

Matched HOST and GPU runs use the bundled free-boundary presets, paired plasma
precision and the page's `1e-12` threshold. All configured stages and all three
residuals converge after the geometry/finite gates.

| Preset | HOST effective iterations | GPU effective iterations | GPU final residual triple |
| --- | ---: | ---: | --- |
| Solovev, two grids | 1,055 | 1,055 | `(9.936e-13, 4.744e-14, 2.695e-14)` |
| W7-X | 1,845 | 1,842 | `(9.687e-13, 2.914e-13, 6.219e-13)` |
| cth_like | 625 | 626 | `(9.635e-13, 2.759e-13, 2.684e-13)` |

The paired runs retain the same 3/6/1 restart counts. Exported fields are finite,
the Jacobian has one orientation, radial B is exactly zero, and B² is nonnegative.
GPU versus HOST maximum LCFS R/Z coefficient differences are respectively
`8.92e-9`, `5.56e-6`, and `7.02e-8` m; relative L2 B² differences are
`1.02e-8`, `2.16e-6`, and `1.15e-7`. These are recorded diagnostics at converged
states, not universal error bounds. The dependency's trusted Fortran-VMEC
vacuum data supplies an independent field reference; these browser captures
do not establish independent full-equilibrium agreement for arbitrary inputs.

The HOST runs retain exactly the original merged controller records and
scientific state/field hashes: 1,056/1,847/625 records. Scalar Solovev with the
GPU vacuum also retains all 75 original records and both scientific hashes
at the browser's scalar `1e-5` threshold. This does not qualify scalar W7-X.

A browser fixed-point checkpoint replay is not covered by the current input
path. Other adapters, Firefox, arbitrary coil grids and higher-resolution
or strongly conditioned cases need their own qualification. The artificial
near-unstable singular recurrence remains explicitly limited by its propagated
precision error; it does not relax any consumer tolerance.

## Performance and retained defaults

The CUDA parallel axisymmetric source-term decomposition is implemented, and
singular workgroups 8/16/32/64 were compared in rotating order over twelve warmed
samples of 128 dispatches. Every output word matched; timing quartiles overlap,
so the dependency keeps 64. See its ADR for the measured workloads.

Before the arithmetic-preserving optimization below, single end-to-end paired
captures, including setup/tracing/output, took
HOST/GPU approximately 13.9/61.0 s for Solovev, 76.7/212.9 s for W7-X, and
12.5/40.9 s for cth_like. These are not warmed repeated performance claims and
should not be generalized. They do not support making GPU vacuum the default
on this adapter. Accurate twofold operations and readback/compatibility costs
remain targets for further profiling. HOST remains the default; the GPU
backend is available for continued numerical and performance work.

## Arithmetic-preserving integral acceleration

The dependency now computes singular RHS point contributions in parallel,
then performs the original recurrence-order, row, four-lane and remainder
sums. The 3-D regularized RHS similarly separates expensive image terms from
its ordered source/image sum. GPU-computed immutable Fourier quotients persist
across updates, and singular preparation reuses identical rounded expressions.
Both words, the rounding barriers, full/partial scheduling and host LU remain
unchanged. Optional scratch respects actual device buffer and dispatch limits;
the original fused paths handle cases that exceed those limits.

This is a Class A optimization relative to the initial GPU vacuum port.
Chrome 152 / RTX 3060 Ti passed all 30 integral fixtures with identical words
against the original implementation and against the retained fused paths,
plus 27 fused HOST-reference fixtures. Trusted CTH-like full/partial, torus,
asymmetric, concurrency/recovery and capacity-fallback runtime checks pass.
The dependency's 19 Wasm CTests and cuMES's 14 browser CTests pass.

Seven warmed alternating GPU timestamp samples of eight complete integral
sequences measured the 35-mode / 104-point singular full update at
8.243 → 1.571 ms (5.25×), its partial update at 7.154 → 1.215 ms (5.89×), and
the 104-source / 192-target 3-D regularized sequence at 6.101 → 0.908 ms
(6.72×). These include new preparation passes and preserve pass boundaries;
they exclude compilation, CPU LU, uploads and final readback. The dependency's
[precision ADR](../../deps/vacuum-field/docs/adr/0001-webgpu-paired-vacuum.md)
records sample quartiles, smaller axisymmetric results and unchanged costs.

The integrated optimized build also preserves every numerical controller record
(ignoring wall-clock timestamps) and the complete scientific output digest against the original GPU build for paired
Solovev (1,056 records, two grids), W7-X (1,844), cth_like (626), and scalar
Solovev (75). All configured residuals and validity gates pass, with identical
restart decisions and final spectral/field data. No convergence threshold was
changed. This establishes Class A equivalence on these tested cases; it does
not expand the initial backend's precision or adapter qualification.

Two matched complete W7-X captures took 175.2–179.3 s before and 90.8–100.6 s
after, including startup, tracing and output. These are repeated complete-run
observations, with visible variability; they are separate from the warmed
kernel timestamp measurements. Single optimized Solovev captures took 32.7 s
paired and 2.5 s scalar and are correctness evidence, not repeated speed
claims. HOST remains the default: these measurements compare two GPU vacuum
revisions and do not establish superiority over a matched HOST run.

For cth_like, discarding the first warmup solve of each revision and retaining
three further serial solves gave a median complete time of 45.67 → 20.33 s
(2.25×). Original samples ranged 45.33–46.52 s; optimized samples ranged
19.82–29.94 s, so the variability must accompany the median. Time through the
first controller record was 0.45–0.68 s, and time after the final record was
0.17–0.18 s. The median interval between first and final controller records
fell from 44.94 to 19.71 s. These intervals separate startup and output tails
without presenting controller wall time as isolated GPU execution.

All full-run comparisons used the same bundled preset, paired plasma,
`vacuum=webgpu&trace=1&timing=0`, Chrome 152 and RTX 3060 Ti, with no concurrent
GPU tests. The baseline dependency was `3dbc6aa`; the optimized dependency is
`09f682d`. Repeated traces and scientific digests remain exact. Raw captures,
including all samples and the per-run timing split, live outside the repository
in `../tmp/vacuum-webgpu-speed/`. Other adapters need their own qualification.
