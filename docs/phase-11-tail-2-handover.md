# cuMES Phase 11 tail #2 handover — steps 1–4 landed, step 13 remaining

Status date: 2026-08-16. Branch: `overhaul`. This handover continues
`docs/phase-11-tail-handover.md` (whose §5 listed five next steps). It records a
critical correctness fix plus the first four of those five steps — all landed
and **Class A bit-identical** — and scopes the one remaining item (migration
step 13, the legacy-struct deletion).

## 1. Critical fix: prolongation stale-read (commit `8b25458`)

The default (axisymmetric) Solovev path read an **all-zero ns=55 stage** while
the generic backend was fine (stages ns=5/ns=11 converged; ns=55 failed with
`√g=0` everywhere and `Rax=0`). Root cause: the coarse state is written on the
nonblocking compute stream, and `cudaStreamSynchronize(stream)` at the prior
stage's exit does **not** make those writes visible to `interpolateState`'s
kernel — only a full `cudaDeviceSynchronize()` orders them (the stage's
synchronous default-stream memsets/memcpys interact with the nonblocking
stream). Fix: `cudaDeviceSynchronize()` before the prolongation kernel (one
fence per stage, never in the hot loop) plus `cudaStreamSynchronize(stream)`
after, so the coarse buffer is not freed (via move-assignment of the returned
`SpectralStorage`) while the kernel still reads it.

Verified: Solovev axisym **and** generic `251→199→456` FSQR 9.583e-17; W7-X
`1877→1617→2011` FSQR 9.778e-13.

## 2. Steps landed (handover §5, items 1–4)

| Handover §5 | Migration step | Commit | What |
| ----------- | -------------- | ------ | ---- |
| 1 | — (seal) | `ed6e091` | `FourierPlan` sealed from `solverRun` via `ToroidalFftOperator::enqueue_inverse_dump`/`combine_parity` (dump-only). `solverRun` no longer holds `transform.fourier_plan()`. |
| 2 | — (views) | `bbbf4cd` | `Profiles::profile_views()/delta_s()` + `ConstraintOperator::rcon_view()/zcon_view()/constraint_force_views()`; `solverRun` reads typed `RadialProfileViews`/views, not raw `rp.d_*`/`cw.d_*`. |
| 3 | step 5 (base geom split) | `0a2e55c` | `geometryKernel` split into `baseGeometryKernel` (interpolation/Jacobian/metric, no 1/√g) + `magneticFieldKernel` (1/√g B + covariant B + total pressure); `computeGeometry` = full-pipeline wrapper; solver drives `GeometryOperator` (base) + stateless `MagneticFieldOperator` (field), ordered base → Jacobian stats → field. |
| 4 | step 12 (EquilibriumOperator) | `a9e198a` | Per-iteration device DAG (axis extrapolation → … → preconditioned residual, incl. interleaved dump machinery) extracted into `cumes::EquilibriumOperator::enqueue` (`include/cumes/solver/equilibrium_operator.hpp`). `solverRun` is now a thin loop: `next_schedule()` → `EvaluationSchedule` → `enqueue` → one fence → Jacobian/invariant/restart decisions → descent → post-descent capture/restore. |

Each commit is Class A bit-identical: Solovev axisym + generic `251→199→456`
FSQR 9.583e-17; W7-X `1877→1617→2011` FSQR 9.778e-13; 35/35 CTest; `CUMES_DUMP=1`
dump path and the float build clean.

## 3. What `solverRun` still names (post step 12)

After `a9e198a`, `solverRun`'s only legacy-struct naming is the dump
observability (NOT the hot loop): `SpectralState<T> st = storage.legacy_view()`
and `const RadialProfiles<T>& rp = profiles.workspace()` — both feed
`axisRAtZeta0`/`printIterRow`/the step-0 dumps. Plus `const GridParams<T>& p`
(everywhere) and `InputParams` in `StageSolver`/`main`. The hot-loop operators
are already behind `EquilibriumOperator`/typed views.

## 4. Remaining: migration step 13 — delete legacy structs (handover §5, item 5)

This is the whole legacy-deletion endgame and is **not** a single-session change.
It is deliberately left as the next unit of work, in dependency order:

1. **Create `DeviceParams<T>`** (`include/cumes/config/…`, blueprint §6.1). It
   must be the compact trivially-copyable stage+scalar pack that replaces
   `GridParams<T>` field-for-field: `ns, mnmax, ntheta, nzeta, nfp, nZnT, mpol,
   ntor, ncurr, delt, ftol, max_iter, tcon0, lamscale` plus the constants
   `kSignJacobian`/`kMu0`. `GridParams` is referenced in **65 files**
   (all `src/*_impl.cuh`, the `include/cumes/{physics,numerics,solver,state,
   transforms}` operator headers, `include/*.cuh`, `main.cu`, `output*.cpp`,
   and ~15 tests). The mechanical move is: `typedef`/`using` swap first, then
   shrink `GridParams` into `DeviceParams` and delete `GridParams`.

2. **`InputParams` → `ValidatedProblem`.** `main.cu` currently does
   `vr.value().to_input_params()` and threads `InputParams` through
   `init_params`/`init_state`/`restart_state` (`seed_state.hpp`),
   `Profiles`/`profilesCreate`, `StageSolver`/`MultigridSolver`, and
   `outputSave`/`outputPrint`. Delete `InputParams` (and `to_input_params()`)
   once these consumers take `ValidatedProblem` directly; `GridShape`/
   `ValidatedProblem` already carry the extents/folding.

3. **Delete the raw owning structs + free functions** once the operators own
   everything with typed views: `FourierPlan` (fourier.cuh), `MetricWorkspace`
   (geometry.cuh), `RadialProfiles` (vmec_types.h), `PreconWorkspace`
   (precon.cuh), `ConstraintWorkspace` (constraint.cuh), `SpectralState`
   (vmec_types.h), and their `*Create/*Free` + `compute*/inverseDFT/forwardDFT/
   preconCompute/preconApply/constraintCompute/interpolateState` entry points.
   The `src/*_impl.cuh` kernel bodies survive as operator implementations.

4. **Decide `dynSharedBase()` removal** (Class B re-freeze, per the
   `--use_fast_math` FMA note in `constraint_impl.cuh`/`geometry_impl.cuh`).
   This can be folded in or left as a documented follow-up; it is the only
   non-deletion arithmetic question in step 13.

5. **Update the tests** that still construct legacy structs directly
   (`test_fourier`, `test_forces`, `test_geometry_iso`, `test_geometry_ncurr`,
   `test_constraint_tcon`, `test_force_reference`, `test_force_verify`,
   `test_axisym_backend`, `test_regression_kernels`) to drive the operators.

6. **Update `CMakeLists.txt`** (drop the legacy `.cu` TUs once folded into
   `src/cumes/`) and `docs/architecture.md` §1 (two layers → one layer).

Verification bar per step: `./build/cuMES inputs/{solovev,w7x}.json` bit-identical
(`251→199→456` / `1877→1617→2011`), 35/35 CTest + the optional-backend matrix
(none/netcdf/hdf5), float build clean.

## 5. Commits (this handover)

```
a9e198a Phase 11 step 12: EquilibriumOperator — solverRun becomes a thin loop
0a2e55c Phase 11 step 5: split base geometry from the magnetic field
bbbf4cd Phase 11: typed view accessors for Profiles/ConstraintOperator
ed6e091 Phase 11: seal FourierPlan from the solver (dump-only transform accessors)
8b25458 fix: prolongation stale-read of the coarse state on the axisymmetric path
```
