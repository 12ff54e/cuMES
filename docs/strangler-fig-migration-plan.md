# Strangler-fig migration plan — legacy kernels → `cumes` operators

Status date: 2026-08-16. This is the plan for the final leg of the
`docs/cuda-overhaul-blueprint.md` (§11 Phase 10 exit gate): *"source dependency
graph matches section 5, … and no legacy code is required for normal
execution."* It is the handover's "next step 3" and the one remaining large item
after the Phase 10 tail (axisymmetric wiring, schema v1).

**Progress (2026-08-16):** all five owning operators are landed and wired into
the solver — `Profiles`, `ToroidalFftOperator`, `GeometryOperator`,
`Preconditioner`, `ConstraintOperator` (commits `87cec03`, `75dcc65`) — each
owns its workspace (RadialProfiles / FourierPlan / MetricWorkspace /
PreconWorkspace / ConstraintWorkspace) and `solverRun`/`StageSolver` construct
and drive the operators instead of the raw structs. The §4 step-1 `FourierPlan`
split landed (`a692200`: transform scratch vs stage-owned `RealSpaceStorage`).
Then the `SpectralOperator` unification landed (`d03640f`): the interface grew
rCon/zCon + `enqueue_dealias`, `ToroidalFftOperator` became a
`SpectralOperator<T>` peer, the compact de-alias scratch moved into the
FourierPlan (transform-owned), and `solverRun` drives a single
`SpectralOperator<T>*` with no `axisym_active` branch. The four stateless
operators landed (`3296ae1`): `ForceOperator`/`ResidualOperator`/
`DescentOperator`/`Prolongation` as thin wrappers over `computeForces`/
`computeResidualsKernel`/`descentStepKernel`/`interpolateState`. All verified
Class A bit-identical (Solovev `251→199→456` FSQR 9.583e-17, W7-X
`1877→1617→2011` FSQR 9.778e-13, 35/35 CTest, float build clean).

Legacy-struct deletion then began with three ownership slices (all Class A):
`53ed57a` extracted the folded-mode table out of `FourierPlan` into a stage-owned
`cumes::DeviceModeTable` (`FourierBasis` deleted from `vmec_types.h`; the
transform free functions + `preconCompute` now take `const int* xm/xn`);
`67ff2fd` moved the cuFFT stream binding into `ToroidalFftOperator::bind_stream`;
`b8fa3a0` moved `m1PreconScale` into `Preconditioner::enqueue_m1_scale`.

Remaining: step 5 (base-geometry vs B/pressure split — Class B), step 12
(`EquilibriumOperator` composition), and step 13 (legacy-struct deletion),
gated on `solverRun` naming no legacy struct. Steps 5 and 12 are the blocking
predecessors of step 13. After the three slices `solverRun`'s remaining legacy
naming is:

- `FourierPlan` — only the dump-gated `inverseDFT`/`fourierCombineParity`
  observability path (the hot loop is sealed behind the operator +
  `bind_stream`); `FourierBasis`/`fp.basis` and `fp.plan_*` are already gone.
- `PreconWorkspace` / `MetricWorkspace` — named as arguments (`constraint.enqueue`
  takes `pw`; `precon.enqueue_compute`/`enqueueForceNorms` take `mw`); their
  `.d_*` field reads are now dump-only.
- `ConstraintWorkspace` / `RadialProfiles` — still read in the hot loop (the
  `RealFieldView`s over `cw.d_rCon/d_zCon/d_frcon/d_fzcon`, and `rp.d_sqrtS_F`/
  `rp.delta_s`/`rp.d_dVds_H`/`rp.d_pres_H`).
- `GridParams<T>` everywhere (needs the `DeviceParams<T>` replacement) and
  `InputParams` in `StageSolver`/`main` (needs `ValidatedProblem`).

No `DeviceParams<T>` replacement exists yet.

The term "strangler fig" is from blueprint §1: wrap the current implementation
behind tested operator interfaces and let the operators *replace* the legacy
structs one at a time, rather than rewriting the solver in one pass. Every step
below is Class A (bit-identical) unless explicitly marked Class B, so the frozen
Solovev `251→199→456` / W7-X `1877→1617→2011` trajectories are the regression
oracle at each step.

## 1. Definition of done (blueprint §14, §11 Phase 10 exit gate)

- `solverRun` drives only `cumes` operator classes; it holds no `FourierPlan`,
  `MetricWorkspace`, `PreconWorkspace`, `ConstraintWorkspace`, or `RadialProfiles`.
- The generic transform is a `SpectralOperator<T>` backend (`ToroidalFftOperator`)
  on equal footing with `AxisymmetricOperator`; `solverRun` picks one pointer, no
  `if (axisym_active)` branches.
- Transforms own only transform tables/plans/scratch — never geometry, force, or
  diagnostic arrays (blueprint §5.1).
- `include/*.cuh` legacy structs and `InputParams` are deleted; `src/*_impl.cuh`
  kernel bodies survive only as the operator implementations (renamed/moved into
  `src/cumes/` or left in place, not as free-function entries the solver calls).
- Full precise trajectories are bit-identical to the frozen baseline (or a
  documented Class B re-freeze), all CTest + Compute Sanitizer entries pass, and
  the optional-backend matrix (none/netcdf/hdf5) still compiles.

## 2. Current state (2026-08-16)

### 2.1 Legacy layer — the *production* implementation

| Module | Header | Impl | Owned struct(s) | Solver entry points |
| ------ | ------ | ---- | --------------- | ------------------- |
| transforms | `include/fourier.cuh` | `src/fourier_impl.cuh` | `FourierPlan<T>` (transform scratch + 18 geometry + 16 force + 9 combined arrays + cuFFT plans) | `inverseDFTFused`, `forwardDFT`, `fourierCombineParity` |
| geometry | `include/geometry.cuh` | `src/geometry_impl.cuh` | `MetricWorkspace<T>` | `computeGeometry`, `computeJacobianStats`, `computeForceNormPartials` |
| forces | `include/forces.cuh` | `src/forces_impl.cuh` | — (writes `FourierPlan` force arrays) | `computeForces` |
| profiles | `include/profiles.cuh` | `src/profiles_impl.cuh` | `RadialProfiles<T>` | `profilesCreate/Free` |
| preconditioner | `include/precon.cuh` | `src/precon_impl.cuh` | `PreconWorkspace<T>` | `preconCreate/Free/Compute/Apply` |
| constraint | `include/constraint.cuh` | `src/constraint_impl.cuh` | `ConstraintWorkspace<T>` | `constraintCreate/Free/Compute/ComputeAxisym/ResetRzCon0/DealiasBandpass` |
| solver | `include/solver.cuh` | `src/solver_impl.cuh` | — (owns the loop) | `solverRun` |
| refine | `include/refine.cuh` | `src/refine_impl.cuh` | — | `interpolateState` |
| axisymmetric | — | `src/axisymmetric_impl.cuh` | `cumes::AxisymmetricOperator<T>` (already a `cumes` class) | `enqueue_*` |

Each module is split into `src/<mod>_double.cu` / `src/<mod>_float.cu` explicit
instantiation TUs (one scalar type each) under `cumes_cuda_double` /
`cumes_cuda_float`.

### 2.2 `cumes` layer — boundaries declared, mostly unimplemented

| Boundary | Header | Implemented? |
| -------- | ------ | ------------ |
| `SpectralOperator<T>` (interface) | `transforms/spectral_operator.hpp` | interface only |
| `AxisymmetricOperator<T>` | `transforms/axisymmetric_operator.hpp` | **yes** (Phase 7.1, wired Phase 10 tail) |
| `ToroidalFftOperator<T>` | *(conceptual — no class yet)* | **no** |
| `GeometryOperator<T>` | `physics/geometry_operator.hpp` | no (`enqueue` declared only) |
| `MagneticFieldOperator<T>` | `physics/magnetic_field_operator.hpp` | no |
| `ForceOperator<T>` | `physics/force_operator.hpp` | no |
| `ConstraintOperator<T>` | `physics/constraint_operator.hpp` | no |
| profiles host build | `physics/profiles.hpp` | no (`RadialProfilesResult` struct only) |
| `ResidualOperator<T>` | `numerics/residual_operator.hpp` | no |
| `Preconditioner<T>` | `numerics/preconditioner.hpp` | no |
| `TridiagonalBackend<T>` (+ `PcrBackend`/`ThomasBackend`) | `numerics/tridiagonal_backend.hpp` | **yes** (Phase 8, in `src/precon_impl.cuh`) |
| `DescentOperator<T>` | `numerics/descent_operator.hpp` | no |
| `Prolongation<T>` | `numerics/prolongation.hpp` | no |
| `IterationController<T>` | `solver/iteration_controller.hpp` | **yes** (Phase 4, pure host) |
| host config / I/O / runtime | `config/*`, `io/*`, `runtime/*` | **yes** (Phases 2–6) |

So the migration is: **implement the unimplemented `enqueue` methods as thin
wrappers over the legacy kernels, split the mixed-ownership structs, then rewire
`solverRun` to drive the operators and delete the legacy structs.**

## 3. Principles

1. **One operator per step; bit-identical before the next step.** Each step is
   independently committable and re-verified against the frozen trajectory.
2. **Wrap first, decompose second.** An operator's first implementation calls the
   *exact* legacy kernel with the *exact* launch config; any kernel split that
   changes arithmetic order is a separate Class B step with a differential test.
3. **Ownership moves with the operator.** The `FourierPlan` split (step 1) is
   prerequisite: transforms must stop owning physics/diagnostic arrays before a
   transform operator is meaningful.
4. **The solver's residual/velocity/state slabs stay as-is.** `SpectralStorage`,
   the component-major slabs, and the typed domain views are already correct and
   do not change.
5. **No new dependency edges.** Constraint reaches the transform only through
   `SpectralOperator`; output never sees a device pointer; the controller stays
   pure host.

## 4. Step plan (dependency order)

### Step 1 — Split `FourierPlan` into transform-only vs storage (Class A)

`FourierPlan<T>` currently mixes: (a) transform state (`plan_z2d/d2z`,
`d_zeta_spectra/real`, the four poloidal tables, `d_fwd_w`, `d_cufft_work`);
(b) real-space geometry (18 parity + 9 combined arrays); (c) force arrays (16).

- Introduce `RealSpaceStorage<T>` (or reuse a `StageWorkspace`-owned
  `GeometryParityViews` + `ForceParityViews` + combined arrays) holding (b) and
  (c), backed by the existing `DeviceArena` spans (no layout change).
- Shrink `FourierPlan<T>` to (a) only. The `inverseDFT/forwardDFT` signatures
  change to take the geometry/force views instead of reaching into `fp.d_r_e` etc.
- `MetricWorkspace`, `PreconWorkspace`, `ConstraintWorkspace`, `RadialProfiles`
  are left untouched this step.

*Verify:* `test_fourier`, `test_operator_views`, full Solovev/W7-X bit-identical.

### Step 2 — `ToroidalFftOperator` (Class A)

Create `include/cumes/transforms/toroidal_fft_operator.hpp` + an impl TU: a
`SpectralOperator<T>` backend that owns the (now transform-only) `FourierPlan<T>`
and wraps `inverseDFTFused`/`forwardDFT` + the generic de-alias. Same `enqueue_*`
signature family as `AxisymmetricOperator` (inverse/forward + rzCon/de-alias).

- `enqueue_inverse` → `inverseDFTFused(..., rCon, zCon, stream)`.
- `enqueue_forward` → `forwardDFT(...)`.
- `enqueue_dealias` → `constraintDealiasBandpass(...)`.

*Verify:* `test_axisym_backend`-style differential test — generic operator vs the
legacy free functions, bit-identical (same kernels, so should be exact).

### Step 3 — Unify the solver transform selection (Class A)

`solverRun` takes a `SpectralOperator<T>*` (either backend) instead of
`AxisymmetricOperator<T>*` + the `axisym_active` branches. The three call sites
collapse to `op->enqueue_inverse/rzcon`, `op->enqueue_forward`, and the
constraint path via `op->enqueue_dealias` (or a small backend-neutral
`constraintCompute` that takes a `SpectralOperator&`).

*Verify:* Solovev (axisym) and W7-X (toroidal) both bit-identical to the
committed `9d18e4e` state; `CUMES_FORCE_GENERIC=1` still forces the toroidal
backend on Solovev.

### Step 4 — Host profile build (Class A)

Implement `profiles.hpp`: a `profilesCreateTyped(p, ip, arena)` that fills
`RadialProfileViews` + `delta_s` + `lamscale` by calling the existing
`profilesCreate` internals, returning `RadialProfilesResult<T>`. `StageSolver`
switches to it; the raw `RadialProfiles<T>` struct is still the internal storage
for one step, then folded in.

*Verify:* profile arrays bit-identical (`test_fourier`-adjacent profile check or
the existing trajectory).

### Step 5 — Split base geometry from B/pressure (Class A, then B)

`computeGeometry` currently produces base geometry (`tau`, `√g`, covariant
metric), then the field (B^u/B^v/B_u/B_v, total pressure) and current closure,
all in one entry point.

- Implement `GeometryOperator::enqueue` = the base-geometry + metric kernels
  (no `1/√g`), writing `BaseGeometryHalfViews`.
- Implement `MagneticFieldOperator::enqueue` = the `1/√g` field + total-pressure
  + `ncurr=0/1` closure kernels, reading base geometry + profiles.
- Split `computeJacobianStats` into the `reset→reduce→finalize` device chain
  described in blueprint §6.7 (this is the Class B part if the reduction tree or
  status finalization changes summation order; otherwise Class A).

*Verify:* pointwise geometry/B equality on manufactured cases
(`test_geometry_iso`, `test_geometry_ncurr`); full trajectory bit-identical if
Class A, ULP + identical controller decisions if Class B.

### Step 6 — `ForceOperator` (Class A)

Implement `ForceOperator::enqueue` wrapping `computeForces` verbatim. Add the
scalar CPU reference from blueprint §4.7/§10.3 as `test_force_reference`
(already exists; extend it to be the gate). No kernel change.

### Step 7 — `ConstraintOperator` (Class A)

Implement `ConstraintOperator::enqueue` wrapping the `constraintComputeHead` +
dealias (generic or axisym via `SpectralOperator`) + `constraintComputeTail`
split already in place. Move `ConstraintState` (versioned reference) into the
operator and thread the reset cadence from `IterationController`.
`ConstraintWorkspace` storage becomes the operator's private arena-backed spans.

### Step 8 — `ResidualOperator` (Class A)

Implement `enqueue_invariant`/`enqueue_preconditioned` wrapping
`computeResidualsKernel` + the host `plainPerEl`/`fNorm*` scaling into a device
`ControlRecord` (blueprint §6.9). The force-norm `enqueueForceNorms` +
`finalizeForceNorms` pair moves here or into a `ForceNormOperator`.

### Step 9 — `Preconditioner` (Class A)

Implement `enqueue_compute` (→ `preconCompute`) and `enqueue_apply` (→
`preconApply`, which already calls the Phase 8 `PcrBackend`/`ThomasBackend`).
`PreconWorkspace` becomes the operator's private storage.

### Step 10 — `DescentOperator` + checkpoint (Class A)

Implement `DescentOperator::enqueue` wrapping `descentStepKernel` + the
`backupState`/`restoreState` single-copy checkpoint, driven by `DescentAction`
(already declared). Folds the ordered apply (descent → post-descent checkpoint →
post-descent restore+zero) into one operator call.

### Step 11 — `Prolongation` (Class A)

Implement `Prolongation::enqueue` wrapping `interpolateState`; `MultigridSolver`
calls it instead of the free function.

### Step 12 — `EquilibriumOperator` + stage/multigrid rewire (Class A)

Compose the operators into `EquilibriumOperator::enqueue` (the per-iteration
DAG from blueprint §7) and rewrite `solverRun` to be a thin loop over the pure
`IterationController` + `EquilibriumOperator`. `StageSolver::run`/`MultigridSolver::run`
construct operators instead of the five legacy workspaces. This is the payoff
step: `solverRun` no longer names any legacy struct.

*Verify:* full precise trajectories bit-identical; `cumes_benchmark_fixed_iteration`
and the smoke gate still run.

### Step 13 — Delete legacy internals (Class A, deletion only)

Delete `include/*.cuh` (`fourier`, `geometry`, `forces`, `profiles`, `precon`,
`constraint`, `refine`, `solver`, `input`, `input_json`, `output`, `vmec_types`
where superseded), `src/*_impl.cuh` free functions no longer referenced, and
`InputParams`/`GridParams` legacy bridges (keeping the `cumes` `GridShape`/
`ValidatedProblem`/`DeviceParams`). Rename the surviving `src/cumes/*` to drop
the "legacy wrapper" comments. Update `CMakeLists.txt` targets and the
`docs/architecture.md` §1 "two layers" description to a single layer.

*Verify:* clean-clone build, full CTest + sanitizer matrix, both configs
bit-identical, `docs` scaffold (blueprint §5) matches the source tree.

## 5. Verification per step

- **Class A gate:** `./build/cuMES inputs/{solovev,w7x}.json` bit-identical to
  the frozen state files (`compare_converged_state.py` / `test_io_golden`), plus
  the full CTest + Compute Sanitizer matrix.
- **Class B gate (steps 5/6 if decomposed):** per-operator ULP bounds on frozen
  inputs, identical finite/status classification, identical controller decisions
  (iteration counts and restart sequence) — via `test_controller` replay and the
  fixed-iteration benchmark's `state_hash`.
- Every operator gets a `test_<operator>.cu` differential test against the legacy
  free function it wraps, in the `test_axisym_backend` style.

## 6. Risks and open questions

- **`FourierPlan` split (step 1) is the linchpin.** It touches every transform
  consumer and the dump machinery; do it first and re-verify before anything else.
- **`computeGeometry` kernel boundaries (step 5).** Whether the `1/√g` division
  is already a separate kernel or fused into base geometry is to be confirmed in
  `src/geometry_impl.cuh`; a fused division forces a Class B split with a
  differential proof (blueprint §6.7 explicitly forbids assuming a grid-wide
  barrier).
- **The `dynSharedBase()` workaround.** `constraint_impl.cuh` retains the
  non-templated dynamic-shared-memory indirection because switching to
  `extern __shared__ T[]` perturbs `--use_fast_math` FMA (~1e-10). When the
  explicit instantiation split is fully in place this indirection *can* be
  removed, but only as a Class B re-freeze — decide at step 13 whether to fold it
  in or leave it.
- **`vmec_types.h` / `GridParams`.** `DeviceParams<T>` (blueprint §6.1) is the
  eventual replacement for `GridParams<T>`; if the migration stalls, the legacy
  `GridParams` can remain as the DeviceParams backing store without violating the
  exit gate, as long as `InputParams` and the raw owning structs are gone.
- **The dump machinery (`DUMP_CUMES_VERIFY`)** reaches into `fp.d_*` arrays; the
  step-1 split must keep the dump path working (it is observability-only and must
  not change the trajectory) or gate it behind the same views.

## 7. Ordering rationale

The dependency graph (blueprint §5.1) forces this order: transforms (steps 1–3)
must be owned before constraint (step 7) can reach them through `SpectralOperator`;
base geometry (step 5) must be split before B/pressure (step 5b) and force (step
6); residual (step 8) and preconditioner (step 9) consume geometry/field; descent
(step 10) consumes residual + controller; prolongation (step 11) is independent;
the equilibrium operator (step 12) composes everything; deletion (step 13) is
last and gated on every consumer. Steps 4 and 11 can be done in parallel with
steps 5–10.
