# Applying the cuMES 1.5 optimizations to WebGPU

Assessment of `main` at `6756fd6` against the WebGPU implementation at
`c7eceda`. The merge retains the browser's arithmetic and controller behavior.
Native CUDA performance measurements do not qualify browser speedups.

## Applicability

| Update | Browser applicability | Decision |
| --- | --- | --- |
| Skip unused inverse constraint accumulations (`66a557a`) | Already present structurally: the WebGPU inverse computes only the two needed R/Z constraint accumulators. | No additional port. |
| Cache weighted forward basis (`21b6043`) | Scalar axisymmetric projection still multiplies four basis values by the integration weight per theta, mode, and surface. A separate immutable weighted table could remove that repeated work. | Candidate for a measured follow-up; retain current arithmetic in this merge. |
| Compensate m=1 toroidal odd-position sums (`9c59702`) | The scalar browser path rounds toroidal sums to scalar intermediates before compensated poloidal reconstruction. The paired path already retains high/low intermediates. | Scalar precision improvement is feasible, but requires new cancellation tests and trajectory qualification. |
| Keep higher odd-mode products single-word (`0d8482a`) | Scalar WebGPU already uses single-word toroidal intermediates and compensated poloidal products. Its precision split differs from the new CUDA implementation. | Do not weaken the paired path or copy the CUDA shortcut independently of its precision contract. |
| Parallel axisymmetric vacuum source evaluation (`0bc92e2`) | CUDA benefits from distributing source terms over blocks. The HOST/Wasm kernel dispatcher is serial; the split would add scratch traffic without GPU parallelism. | Merge the CUDA implementation and retain the fused HOST evaluation. |
| Spread small singular RHS systems over blocks (`f8bbfa2`) | CUDA block size affects occupancy; HOST dispatch ignores it. | Merge using the existing backend launch abstraction. |
| Combine free-boundary copies with fences (`cc2d91d`) | These are CUDA stream/copy changes. The browser already reads the edge residual after its force operation and uses its own batched WebGPU readbacks. | No direct browser port. |
| Opt-in Newton–GMRES (`bcdd3da`, `ca33025`) | Requires WebGPU linear algebra, frozen residual probes, trial acceptance, and precision qualification. Native support is restricted to fixed-boundary axisymmetric double. | Keep the browser option unavailable until a separate Class C port is qualified. |

The vacuum dependency must contain both the HOST backend from `4f724ed` and
the CUDA optimizations from `4d19939`. Replacing its gitlink with the native
revision alone removes the browser backend. The merged HOST path retains the
existing source-summation order; only CUDA allocates the split source scratch.

## Fourier follow-ups

The lowest-scope candidate is the scalar axisymmetric projector in
[`axisymmetric_forward.wgsl`](../src/webgpu/shaders/axisymmetric_forward.wgsl).
[`cached_separable_gpu_basis`](../src/webgpu/toroidal.cpp) already caches its
unweighted trigonometric and derivative tables. A weighted cache must preserve
the four separately rounded products, endpoint half weights, and existing
normalization. Keep it separate from the unweighted tables shared with inverse
transforms and dealiasing. Check exact operator outputs and controller traces
on the same adapter before calling this Class A. Measure warmed repeated runs
before claiming that removing four multiplies offsets any added setup/storage.

The 3-D projector in
[`toroidal_forward.wgsl.in`](../src/webgpu/shaders/templates/toroidal_forward.wgsl.in)
weights the combined residual expression. Moving that weight into its basis
changes arithmetic order, so it is not the same mechanical cache optimization.

The scalar inverse in
[`toroidal_inverse.wgsl.in`](../src/webgpu/shaders/templates/toroidal_inverse.wgsl.in)
uses Kahan toroidal summation but stores a scalar result; its correction is
not a paired low word. Applying the native m=1 precision improvement requires
explicit paired toroidal sums carried through poloidal products, together with
a paired radial normalization (the scalar shader currently computes it in
f32). Existing odd-position cancellation tests
in [`float_geometry_tests.cpp`](../src/webgpu/float_geometry_tests.cpp) populate
n=0 odd coefficients; add nonzero-n cancellation coverage before changing this
path. Use the browser's scalar W7-X baselines and tolerances, not native CUDA
iteration counts. If controller decisions change, apply the Class C gates. The already paired inverse has a different precision
contract and does not need this selective compensation retrofit.

## Newton prerequisites

The native
[`NewtonCorrection`](../include/cumes/numerics/newton_correction.hpp) and
[`DeviceGmres`](../include/cumes/numerics/device_gmres.hpp) own CUDA buffers and
launch CUDA kernels. Their mathematics and fixtures are reusable; those
implementations cannot link into the Wasm target. A first browser experiment
should be opt-in, paired-f32, fixed-boundary, and axisymmetric.

Reuse the existing WebGPU physics DAG and cached preconditioner through
[`IterationCase`](../include/cumes/webgpu/iteration.hpp), with these additions:

- WGSL active-coordinate packing, finite-difference JVPs, and stable restarted
  GMRES. Retain axis/LCFS exclusions, parity zeros, and the m=1 coordinate map.
- A probe enqueue path that keeps residuals on device. The current
  [`IterationDispatch`](../src/webgpu/iteration.cpp) always maps a readback;
  invoking it for each Krylov probe would introduce a host fence per probe.
  Own frozen RHS/base/trial snapshots because result handles can alias scratch.
- Freeze preconditioner, constraint references and multiplier, force
  normalization, and gauge during probes; recompute geometry, fields, forces,
  and current closure.
- Study finite-difference step size and cancellation. Paired spectral state
  does not make the complete residual oracle paired: the current
  [`preconditioner application`](../src/webgpu/preconditioner_apply.cpp) still
  consumes and produces scalar f32. Native epsilon and inner tolerance cannot
  be assumed valid for this path.
- Preserve native finite/Jacobian gates, stable-epoch guards, merit decrease,
  and accepted-correction velocity reset. Rejected trials must restore and
  reevaluate the base with the frozen caches, requiring all three residuals to
  match their pre-probe values. Coefficient restoration alone leaves probe
  intermediates active. The merged shared controller's
  `reset_correction_momentum()` is reusable.

Follow the native [Newton ADR](adr/0016-opt-in-newton-corrections.md) and the
[Class C gates](verification.md#6-equivalence-gates): every configured residual,
valid geometry, physical invariants, rollback, fixed-point replay where
practical, representative cases including prescribed current, an independent
comparison, and a browser ADR. Real adapter qualification and warmed timings
must precede a browser performance claim or a default-policy change.
