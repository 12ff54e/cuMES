# Applying the cuMES 1.5 optimizations to WebGPU

The `main` merge at `49f5e41` brought `6756fd6` into the browser branch.
The merge itself preserved the browser's controller traces and scientific
outputs. The subsequent ports below have separate numerical qualification;
CUDA timing results do not establish browser speedups.

## Optimization priorities

Include host/device exchange and synchronization in every investigation of
browser solver performance. For free boundary, audit the plasma/vacuum
interface as well as the vacuum kernels:

- Identify the host consumer and dependency for each readback, upload, and
  completion wait. Account for bytes and round trips separately for full and
  partial vacuum updates; partial updates still require vacuum work.
- Keep spectral state, physics fields, and reusable intermediates resident
  wherever their consumers permit it. Read only required axis/LCFS slices and
  compact controller values, and upload only host-modified results.
- Batch independent readbacks and copies at existing dependency boundaries.
  Reuse staging buffers and scratch. Evaluate GPU boundary coupling against
  the existing formulas to remove transfers of fields used only on the GPU.
- Preserve vacuum activation, restart state, `nvacskip`, pressure coupling,
  preconditioner terms, and multigrid persistence. Moving a reduction changes
  the numerical qualification unless its original arithmetic is preserved;
  retain finite, geometry, Jacobian, and residual gates.
- Measure warmed complete solves with matching configurations and retain
  controller and scientific comparisons. Report transfer/wait reductions
  separately from measured wall-time improvement; a faster vacuum kernel
  does not establish a faster free-boundary solve.

These priorities guide further work and do not describe completed ports.
See [the performance contract](performance.md#11-hostdevice-exchange-and-synchronization)
for measurement and acceptance requirements.

## Port results

| Merged update | WebGPU implementation and qualification |
| --- | --- |
| Skip unused inverse constraint accumulations (`66a557a`) | Already present: the inverse computes only the required R/Z constraint accumulators. No duplicate implementation was added. |
| Cache weighted forward basis (`21b6043`) | Implemented a setup-time GPU cache for scalar axisymmetric forward projection, retaining each rounded product and endpoint weight. Conformance, fixed Solovev and free Solovev controller traces and scientific hashes are exact. Twenty alternating warmed runs had overlapping timing ranges; no speedup is claimed. |
| Compensate m=1 toroidal odd-position sums (`9c59702`) | Implemented `geometry=compensated-m1` for scalar 3-D. Paired m=1 intermediates and radial scaling survive through poloidal reconstruction. New signed-family, nonzero-n, sub-ULP and minimal-grid fixtures pass. Default behavior remains exact. |
| Keep higher odd-mode inputs single-word (`0d8482a`) | The m=1 option retains the existing scalar toroidal products for higher odd modes and the established compensated poloidal reconstruction. The always-paired path retains its own precision contract. |
| Parallel axisymmetric vacuum source evaluation (`0bc92e2`) | Ported into vacuum-field's WGSL: independent source/image terms followed by the original ordered sum. Scratch is persistent. The fused HOST operator remains the independent reference. |
| Spread small singular RHS systems over blocks (`f8bbfa2`) | Singular RHS systems are independent WebGPU invocations with the original four-lane sum association. Warmed 8/16/32/64 trials preserve every output word and have overlapping timing ranges; 64 is retained. |
| Combine free-boundary copies with fences (`cc2d91d`) | The new vacuum backend batches matrix/RHS into one map and final outputs into a second map around Wasm LU. Integral intermediates stay resident. Plasma already uses its own batched readbacks; CUDA streams/copies are not part of this backend. |
| Opt-in Newton–GMRES (`bcdd3da`, `ca33025`) | Implemented resident f32 Krylov algebra, paired trials, frozen physics probes, native eligibility/backtracking policy, and exact rollback. `newton=1` is restricted to paired fixed-boundary axisymmetric solves. Solovev and prescribed-current cases pass all configured residuals and independent scientific checks. |

## Vacuum backend

The GPU portion of vacuum-field now has a WebGPU implementation inside
`deps/vacuum-field`. It provides a separate `vfield::webgpu::Solver` API with
owned WebGPU buffers and asynchronous updates. It does not emulate CUDA
pointers in Wasm memory. The corresponding six WGSL modules execute:

1. Surface synthesis, derivatives, metrics and curvatures.
2. Coil-grid interpolation, axis-current field, normal and covariant fields.
3. Singular and regularized integrals, Fourier transforms, matrix/RHS assembly.
4. Potential derivatives, vacuum magnetic field, pressure and surface integrals.

Coil parsing, MAKEGRID, base coefficient setup and dense LU use shared
Wasm-double C++; paired Fourier quotients are cached once on the GPU. LU
remains between the assembly and reconstruction GPU
batches; partial updates reuse its factorization. Inputs are copied before
asynchronous submission, intermediate arrays remain resident, and output
views are available directly to GPU consumers. The existing cuMES host
coupling currently consumes the final readback arrays. An Asyncify wrapper
yields the worker while the same asynchronous solver completes, without polling.

Select `vacuum=webgpu` on a free-boundary page. `vacuum=host` remains the
reference/default. Selection precedes the first vacuum update and survives
multigrid transitions with the existing activation, restart, `nvacskip`,
current-consistency and LCFS-pressure policies. Only coil geometry and small
configuration assets are served; response grids are generated in memory.

Paired-f32 vacuum arithmetic has independent sqrt/log/recurrence checks and
retains the binary32 exponent range. Miller normalization and tangent-pole
handling avoid double-only seed/sentinel magnitudes. The qualification and
conditioning limits live in the dependency's
[precision ADR](../deps/vacuum-field/docs/adr/0001-webgpu-paired-vacuum.md)
and [browser test instructions](../deps/vacuum-field/tests/webgpu/README.md).
A passed stress recurrence estimate does not relax any physical residual or
complete-solver comparison bound. Paired Solovev/W7-X/cth_like and scalar
Solovev pass the consumer gates; [ADR-0019](adr/0019-webgpu-vacuum-backend.md)
records trajectory differences, scientific diagnostics and timing limits.
The subsequent arithmetic-preserving acceleration splits singular and 3-D
regularized RHS terms from their original ordered sums, with capacity-limited
fused fallbacks. Warmed integral-sequence measurements improve about 5–7× on
the recorded 3-D fixtures; the optimized GPU path retains its prior component
words and consumer controller records.

### Free-boundary residency and compact force readbacks

Free-boundary production solves now retain spectral state and descent velocity
on the GPU across normal iterations. The next inverse consumes the pending
device state, while its host snapshot joins the next prefix's readback batch.
The host commits that snapshot before vacuum coupling. Vacuum promotion still
precedes the next iteration's update schedule; checkpoint restores and stage
initialization clear pending device state. Normal continuing iterations use two
plasma completion maps instead of three. With GPU vacuum's two unchanged maps,
the corresponding total is four instead of five.

The force prefix reads only the four LCFS rows needed by the existing Wasm
pressure correction. A GPU finite scan checks every word of all 16 force
fields, including interior fields that remain resident. Corrected LCFS rows
are uploaded before the suffix consumes the complete device force arrays.
For paired W7-X at `ns=51, ntheta=20, nzeta=36, mpol=7, ntor=6`, force readback
payload falls from 4,700,160 bytes to 41,400 bytes: 23,040 bytes of LCFS values
and 18,360 bytes of finite flags. These are logical copy/map payloads, not
measured bus traffic or a wall-time result.

The batched host continuation also avoids copying full arrays into inputs for
operators that have already executed. Geometry is moved into vacuum coupling
and restored to its canonical owner before normalization and output; magnetic
low words move directly to their consumer. This removes nine vector copies,
or 18,259,200 bytes of host copying per paired pass at the W7-X shape above.

This is a Class A ownership/scheduling change: physics arithmetic and original
host reduction order are preserved. The WebGPU build and all 14 CTests pass.
Real Chrome conformance passes, including fixed/free LCFS device descent,
compact/full force word equality, and rejection of nonfinite interior force
words. Complete solves preserve every controller record (excluding time) and
scientific payload digest against the original baseline: 1,847 records for
paired W7-X with HOST vacuum, 625 for paired cth_like with HOST vacuum, and 75
for multigrid scalar Solovev with WebGPU vacuum. Native CUDA and Firefox were
not rerun for this browser-only ownership change.

The 2026-09-09 comparison against `4c06de4` used Chrome 152.0.7977.77 on an
NVIDIA GeForce RTX 3060 Ti, the bundled W7-X free-boundary preset, paired
plasma precision, HOST vacuum, `trace=1`, and `timing=0`. Runs were serial,
with one warmup and two measured complete solves per revision. All measured
runs retained the exact baseline trajectory and scientific digest.

| Interval | Baseline median (range), s | Updated median (range), s |
| --- | ---: | ---: |
| Full page run | 74.769 (74.473–75.066) | 52.522 (52.506–52.538) |
| First-to-last controller record | 69.276 (69.152–69.401) | 47.459 (47.448–47.471) |

This is 29.75% less full-run time and 31.49% less time across the controller
records on this measured setup. The first controller record arrived at worker
ages 4.789/5.265 s before and 4.693/4.644 s after; these ages are not pure setup
times. Window completion and worker trace clocks have different origins, so no
output-only interval is inferred. Two measured samples on one adapter are a
bounded browser comparison, not the full cross-architecture performance
qualification in `performance.md`. Logs, traces, digests, and the comparison
are retained outside the repository in `../tmp/free-boundary-transfers/`.

The extension below reduces the remaining geometry and magnetic-field
readbacks. Original host norm reductions and Wasm vacuum/pressure coupling
remain. `field_readbacks=full` retains full field and velocity snapshots;
`resident=0`, `fences=1`, or `compare_fft=1` select the separate-dispatch
free-boundary reference path.

### Boundary snapshots and cached vacuum reconstruction

Both vacuum choices now receive compact plasma snapshots. The inverse returns
12 angular rows containing axis R/Z and the LCFS geometry used by the existing
coupling. On non-refresh passes, base geometry returns gsqrt/guv for the
unchanged ordered host Jacobian gate; magnetic fields return their outer two
half-grid rows and radial profiles. GPU finite scans include omitted fields
and both precision words. Refresh passes retain full base/magnetic arrays for
normalization. Final output downloads the accepted full fields once, without
another physics pass or controller transition. Full-readback fallbacks remain
for unsupported compact shapes or finite-scan dispatch sizes.

For paired W7-X (`ns=51`, angular grid `20×36`), the affected non-refresh
readbacks, including finite flags and magnetic profiles, change as follows:

| Snapshot | Before, bytes | After, bytes |
| --- | ---: | ---: |
| Inverse geometry and constraint reconstructions | 5,875,200 | 92,072 |
| Base geometry | 2,880,000 | 587,252 |
| Magnetic fields and profiles | 1,440,800 | 40,992 |
| Total for these snapshots | 10,196,000 | 720,316 |

These are logical readback payloads, not measured bus traffic. The existing
prefix map carries all three, so no extra completion wait is introduced.

Vacuum-field now caches potential Fourier factors separately in HOST-native T
arithmetic and paired-f32 WGSL. Both retain the original factor expressions
and ascending mode accumulation; full/partial factorization scheduling is
unchanged. The WebGPU cache has an uncached capacity fallback. The browser
requests pressure and surface integrals only from GPU vacuum; all other
outputs stay resident and retain their finite checks. This removes eight
output copy commands and six recurring uploads from ordinary cuMES GPU
updates. The public dependency API retains full results by default.

This is a Class A change. The WebGPU build and all 14 parent CTests pass,
as do all 19 native HOST and all 19 Wasm dependency tests. Cached HOST
reconstruction matches the original kernel bitwise in float and double.
Real Chrome conformance checks compact/full scalar and paired words, retained
device layouts, and finite gates. The dependency's Chrome API gate also checks
full/compact/full result transitions and nonzero-to-omitted input reuse.
Complete paired W7-X solves preserve their own backend's controller records
(1,847 HOST; 1,844 WebGPU), excluding timestamps, and scientific output digest.
Paired cth_like and multigrid scalar Solovev likewise preserve exact traces and
digests with both vacuum backends. CUDA float/double libraries compile; the
CUDA execution path is unchanged and was not benchmarked. Firefox was not run.

The 2026-09-09 comparison against `b3a77c9` used Chrome 152.0.7977.77 on
Windows with an NVIDIA GeForce RTX 3060 Ti, the bundled free-boundary W7-X
preset above, paired plasma precision, and `trace=1`. Both revisions used the
page's default detailed timing and GPU timestamp instrumentation; these are
instrumented default-page timings. Runs were serial, with one warmup and two
measured complete solves per revision and vacuum backend. Measured runs had
no CPU profiler attached.

| Vacuum backend / interval | Baseline median (range), s | Updated median (range), s |
| --- | ---: | ---: |
| HOST/Wasm full page run | 56.311 (56.244–56.378) | 41.857 (41.725–41.989) |
| HOST/Wasm controller span | 51.035 (51.014–51.057) | 36.746 (36.570–36.922) |
| WebGPU full page run | 78.711 (77.964–79.458) | 65.598 (65.166–66.031) |
| WebGPU controller span | 73.149 (72.411–73.886) | 60.431 (60.165–60.696) |

This is 25.67% less full-run time with HOST/Wasm vacuum and 16.66% less with
WebGPU vacuum on the measured setup. HOST/Wasm remains the faster default.
Median worker age at the first controller record changed from 4.859 to
4.723 s for HOST and 5.158 to 4.749 s for WebGPU; these include setup and
are not pure setup intervals. Controller spans use the first and last worker
records. Output time is included in the page run but not isolated by these
clocks. The two-repeat comparison on one adapter is not full performance
qualification or an isolated measurement of each optimization. Captures and
comparison data remain in `../tmp/free-boundary-second-pass/`.

### Resident vacuum boundary-force correction

With WebGPU vacuum, the LCFS pressure correction now runs on the GPU. The
operator consumes existing geometry, plasma-pressure and vacuum-pressure
buffers, then updates the first four force planes at the LCFS in place.
Interior force words remain unchanged. The prefix retains its full-force
finite scan but omits the four host LCFS rows. Pressure-error contributions
and corrected-word validity join the existing suffix readback.

Vacuum-field's resident completion API returns after the matrix/RHS readback,
Wasm-double LU solve and reconstruction submission. Pressure stays in the
vacuum arena. The two surface integrals and raw validity flags (24 bytes)
join the plasma suffix map. The parent validates them before controller
decisions, checkpoint changes or output. Pre-activation updates also finish
this validation even though they do not yet apply an edge force. The known
activation/edge gates are prepared before dispatch; errors abort the pending
evaluation. A cancelled or failed full update clears factorization reuse.
The dependency retains its ordinary full/compact APIs and guards pending
resident results with a generation token and explicit finish/cancel methods.

Normal active WebGPU-vacuum iterations now have three sequential completion
maps: plasma prefix, vacuum matrix/RHS, and plasma suffix. The previous final
vacuum map is removed. For paired W7-X at the shape above, correction removes
23,040 bytes of LCFS readback, 23,040 bytes of corrected-force uploads and
3,168 bytes of vacuum-pressure readback. It adds 8,640 bytes of ordered
pressure-error/validity data to the suffix and a 64-byte uniform upload: a net
40,544-byte reduction in logical exchange per applying pass. The 24-byte
vacuum summary is retained. Synchronization is the main target of this change;
the existing constraint/reference and host norm readbacks remain.

`vacuum_force=host` retains the original correction with GPU vacuum. HOST/Wasm
vacuum keeps its original correction by default, since it has no separate
vacuum map to eliminate. `vacuum=host&vacuum_force=webgpu` explicitly selects
the new correction with a reduced-pressure upload. That option saves 34,208
bytes per applying W7-X pass and retains two plasma maps. Full-readback and
nonresident reference paths keep the host correction.

The dependency completion change preserves arithmetic (Class A). Moving the
correction from Wasm double to paired-f32 WGSL is Class B; scalar plasma also
uses paired correction arithmetic before storing its high word. The new
operator tests compare against independent double expressions, with force
error bounded by `4e-12 * (1 + abs(prior) + abs(increment))` for paired output
and `1.3e-7` times that scale for scalar output. Scaling by the operands retains
a useful bound through cancellation. Pressure-mean error is bounded by
`4e-12 * (1 + abs(reference))`. Real Chrome conformance passes scalar/paired,
axisymmetric/3-D, mirrored reduced-grid indexing, unchanged interior/padding,
malformed ranges and nonfinite/overflow cases. The measured combined scaled
force maximum is `5.196e-8` (dominated by scalar rounding); pressure-mean error
is `2.395e-16`. The paired variants independently pass their tighter bound.

For paired W7-X, the first 100 controller records, including vacuum
activation, remain exact against the frozen `ee3b26b` runs. Longer trajectories
are not bitwise equivalent: rounding eventually changes checkpoint choices
and the stopping iteration. Every configured residual still reaches `1e-12`
after the existing validity gates. The six spectral families, axis/LCFS rows
and all 13 derived-field arrays were compared and remain finite.

| Vacuum choice with GPU correction | Baseline / updated controller records | First nonidentical record (zero-based) | Maximum final R/Z coefficient difference, m | Maximum lambda coefficient difference |
| --- | ---: | ---: | ---: | ---: |
| WebGPU | 1,844 / 1,859 | 172 | 5.255e-6 | 2.178e-5 |
| HOST, explicit opt-in | 1,847 / 1,838 | 146 | 3.489e-6 | 1.385e-5 |

First checkpoint-decision differences occur at records 537 and 710,
respectively. Maximum derived-field error normalized by each reference
array's maximum magnitude is `8.50e-4` for WebGPU and `5.36e-4` for the HOST
opt-in, in the contravariant radial current. These are measured full-solve
differences, not per-operator tolerance changes or general qualification for
other inputs. The unchanged HOST default preserves all 1,847 records and
the exact scientific digest.

Paired cth_like with WebGPU vacuum converges in 625 controller records versus
626, retains its first 100 records exactly, and has maximum R/Z coefficient
difference `7.48e-8 m` and lambda difference `8.06e-7`. Its maximum normalized
derived-field difference is `3.83e-4`. Evaluating the final boundaries on the
same `128×128` angular grid with the existing output reader gives maximum
LCFS displacement `32.91 µm` for W7-X and `0.424 µm` for cth_like; corresponding
axis displacements are `0.890 µm` and `0.00247 µm`. Scalar multigrid Solovev
preserves all 75 controller records and its scientific digest exactly.
The `vacuum_force=host` fallback preserves all 626 cth_like WebGPU-vacuum
records and its original scientific digest exactly.

The WebGPU build and all 14 CTests pass. The dependency's real Chrome API
gate preserves exact full/partial outputs through ordinary/resident
transitions and checks stale, cancelled and invalid pending results. Parent
out-of-grid and reversed-current cases retain their original error
classification, identical accepted controller prefix and absence of output.
Both native CUDA free-boundary translation units compile; their numerical
execution path is unchanged. Native GPU solves and Firefox were not rerun.

The 2026-09-09 timing comparison used Chrome 152.0.7977.77 on Windows with an
NVIDIA GeForce RTX 3060 Ti, the bundled paired W7-X free-boundary preset,
`vacuum=webgpu&trace=1&timing=0`, and the frozen `ee3b26b` application as baseline.
Each revision had one warmup and two serial measured runs. The updated repeats
preserve the new 1,859-record trajectory and scientific digest exactly.

| Interval | Baseline median (range), s | Updated median (range), s |
| --- | ---: | ---: |
| Full page run | 63.696 (63.613–63.778) | 58.560 (58.419–58.701) |
| First-to-last controller record | 58.819 (58.756–58.881) | 53.541 (53.433–53.650) |

This is 8.06% less full-run time and 8.97% less controller-span time despite
15 additional controller records. It measures the complete port, including
its numerical trajectory change, rather than isolating fence latency.
Median worker age at the first controller record is 4.513 s before and
4.628 s after; it includes setup and is not a pure setup interval. Output is
included in page time but cannot be isolated by the differing window/worker
clocks. This two-repeat, one-adapter result is a bounded browser comparison,
not full performance qualification. Raw captures, numerical differences,
error cases and provenance remain in `../tmp/boundary-force-resident/`.

### Resident vacuum LU

`vacuum=webgpu&vacuum_lu=webgpu` keeps the dense vacuum solve on WebGPU.
The paired-f32 implementation uses Gaussian elimination with partial pivoting,
strict first-maximum pivot selection, complete row swaps and the existing
forward/back-substitution order. Full updates factorize; partial updates reuse
the resident factors and pivots. Separate factor storage preserves the original
assembled matrix for diagnostics. A single 64-thread workgroup coordinates
the solve; matrices stay in storage buffers, with workgroup scratch bounded
at 256 unknowns. Larger systems must select Wasm LU (`vacuum_lu=host`).

Assembly, LU and field reconstruction submit without a matrix/RHS map or
potential upload. The resident callback also avoids an unnecessary Asyncify
wait. Singular-pivot and nonfinite-system status share the existing 24-byte
vacuum summary and are checked before accepting any controller result. Failed
or cancelled full updates invalidate the factors; partial updates cannot reuse
an unaccepted full factorization.

Normal active passes with resident boundary-force correction now require two
sequential maps, plasma prefix and suffix, versus three with Wasm LU. For
W7-X's 117 unknowns, each full vacuum update removes 110,448 bytes of
matrix/RHS readback and 936 bytes of potential upload, totaling 111,384 bytes.
A partial update removes 936 bytes in each direction, totaling 1,872 bytes.
These are logical payloads, not measured bus traffic; the unchanged plasma
readbacks and compact vacuum summary remain. Ordinary full-output and
`vacuum_force=host` paths retain their final vacuum-output map.

This is a Class B precision port of the same pivoted solver. Real Chrome
executes 11 direct matrix fixtures, including dense nonsymmetric orders 117
and 256, first-maximum and low-word pivot ties, late row swaps, singular pivot
indices, nonfinite inputs, overflow, a valid large diagonal system and repeated
RHS solves. The independent componentwise backward-error bound is `2e-12`;
the measured maximum is `1.053e-13`. The constructed-solution absolute bound
is `2e-10`. Original matrix words and reused factor words remain exact.
Solving the same GPU-assembled vacuum matrix/RHS independently with Wasm-double
LU gives maximum scaled potential difference `2.536e-16`, below `2e-12`.
Existing trusted, axisymmetric/asymmetric, full/partial, compact/resident and
failure/recovery vacuum gates pass with both LU choices and unchanged bounds.

LU uses local power-of-two scaling for extreme operands because GPU arithmetic
can flush small divisors or produce finite values on overflow. The restoration
checks paired values against the supported exponent range before rescaling.
This retains ordinary-scale operation order; general subnormal arithmetic
remains unqualified. See vacuum-field's
[LU decision](../deps/vacuum-field/docs/adr/0002-resident-webgpu-lu.md).

Paired W7-X preserves the first 100 controller records exactly against the
frozen `d2c721a` build with Wasm LU. The first numerical difference occurs at
record 254 and the first checkpoint-decision difference at 543. GPU LU
converges in 1,832 records versus 1,859; all three configured residuals reach
`1e-12` after the existing validity gates. All six spectral families, axis and
LCFS rows, and 13 derived-field arrays were compared and remain finite.
Maximum final R/Z coefficient difference is `6.60e-6 m`, lambda difference
`2.74e-5`, and derived-field difference normalized by its reference array's
maximum magnitude `1.04e-3` (contravariant radial current). These are full-solve
differences from later controller rounding, not changes to operator bounds.

Paired cth_like reaches `1e-12` in 626 records versus 625; paired Solovev
reaches it in 1,056 records with either LU. Their first 100 records remain
exact; first checkpoint differences occur at 350 and 1,042, respectively.
Maximum R/Z coefficient differences are `5.44e-8 m` and `2.99e-8 m`, lambda
differences `6.33e-7` and `7.26e-8`, and normalized derived-field differences
`3.74e-4` and `8.00e-6`. All compared arrays remain finite. Scalar cth_like
and Solovev preserve all 108 and 75 records and their scientific digests
exactly. Existing output reconstruction on a `128×128` angular grid gives
maximum LCFS displacements of `41.274 µm` for W7-X and `0.312 µm` for cth_like;
the plotted-axis displacements are `1.175 µm` and `2.138 nm`, using the
reader's existing `converged_axis` definition.

The 2026-09-09 timing comparison uses Chrome 152.0.7977.77 on Windows,
NVIDIA GeForce RTX 3060 Ti, and the bundled paired W7-X free-boundary preset.
Both variants use `vacuum=webgpu&trace=1&timing=0`; the candidate adds
`vacuum_lu=webgpu`. Each has one warmup and two serial measured solves. The
updated repeats reproduce all 1,832 controller records and their scientific
digest exactly.

| Interval | Wasm LU median (range), s | WebGPU LU median (range), s |
| --- | ---: | ---: |
| Full page run | 61.860 (61.628–62.092) | 59.460 (59.446–59.474) |
| First-to-last controller record | 56.479 (56.274–56.684) | 54.272 (54.218–54.326) |

Warmed page time is 3.88% lower and controller-span time 3.91% lower. This
includes the changed stopping iteration and does not isolate LU kernel time
or fence latency. Median worker age at the first controller record is
4.971 s with Wasm LU and 4.795 s with GPU LU. The first GPU-LU warmup instead
took 80.789 s with 25.671 s until the first controller record, consistent
with a substantial initial shader-compilation cost. These worker ages include
other setup; output time is included in page time but not isolated. GPU LU
remains opt-in, with Wasm LU and the overall HOST vacuum default retained.
This two-repeat comparison on one adapter is not full performance
qualification. Raw captures, comparisons and runtime provenance remain in
`../tmp/vacuum-lu-webgpu/`.

The parent WebGPU build and all 14 CTests pass, along with all 19 dependency
Wasm reference tests and the real Chrome gates above. Native numerical
execution paths are unchanged; native GPU solves and Firefox were not rerun.
The explicit `vacuum_lu=host` fallback preserves all 625 paired cth_like
controller records and its scientific digest exactly.

## Geometry and Newton options

`geometry=compensated-m1` changes scalar W7-X trajectories and is a Class C
experiment. The single-grid case converges in 895 effective iterations;
three grids converge in 1,163. Untouched inverse fields are bitwise preserved.
Finite fields, oriented Jacobians, fixed LCFS, and an independent VMEC++ 0.7.0
comparison are recorded in [ADR-0017](adr/0017-webgpu-m1-geometry-compensation.md).
The option remains disabled by default; it is not a qualification below the
browser scalar `1e-5` tolerance or a general speed result.

`newton=1&precision=double` is an opt-in Class C change. It reuses the physics
DAG with a frozen preconditioner, constraint references, normalization and
gauge. Probes enqueue without individual host maps. Eager GMRES work has one
control map, and actual trial states pass geometry and all three residual
checks before a correction is accepted. Rejection reevaluates the original
base; accepted steps clear velocity and reset controller momentum.

The matched paired Solovev case takes 333 iterations versus 507 without
Newton; the prescribed-current case takes 397 versus 596. Deliberate inner
breakdown reproduces the exact baseline controller trace and scientific
hashes. A step sweep measures finite differences of the actual browser
f32-preconditioned residual; paired state alone does not make that oracle
binary64. [ADR-0018](adr/0018-webgpu-newton-experiment.md) records the fixtures,
independent VMEC++ diagnostics, rollback, and remaining qualification limits.
Iteration reductions do not by themselves establish a wall-time speedup.

The 3-D forward projector weights the combined force expression. Moving the
weight into its basis would change the arithmetic association, unlike the
qualified scalar axisymmetric cache. That further change is outside the
merged cache optimization and has not been introduced.
