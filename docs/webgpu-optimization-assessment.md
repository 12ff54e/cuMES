# Applying the cuMES 1.5 optimizations to WebGPU

The `main` merge at `49f5e41` brought `6756fd6` into the browser branch.
The merge itself preserved the browser's controller traces and scientific
outputs. The subsequent ports below have separate numerical qualification;
CUDA timing results do not establish browser speedups.

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
