# cuMES Phase 11 tail #3 handover — step 13 items 1–2 landed, 3–6 remaining

Status date: 2026-08-16. Branch: `overhaul`. This handover continues
`docs/phase-11-tail-2-handover.md`, whose §4 scoped migration step 13 (the
legacy-struct deletion endgame) as six dependency-ordered items. Items 1 and 2
have now landed, both **Class A bit-identical**; items 3–6 remain and are
re-scoped below with the exact current state.

## 1. Items landed (this handover)

| Item | Migration step | Commit | What |
| ---- | -------------- | ------ | ---- |
| 1 | step 13.1 | `a7c0030` | `GridParams<T>` renamed to `DeviceParams<T>` and moved to `include/cumes/config/device_params.hpp` (blueprint §6.1 four-stage config pipeline). Pure rename, global namespace retained so all 65 kernel/operator headers resolve unchanged. |
| 2 | step 13.2 | `22e71d0` | `InputParams` deleted. `init_params`/`init_state`/`restart_state` (`seed_state.hpp`), `Profiles`/`profilesCreate`, `StageSolver`/`MultigridSolver`, and `outputSave`/NetCDF/HDF5 now consume `ValidatedProblem` directly. The legacy JSON parser (`input.h`/`input_json.h`/`src/input_json.cpp`) and `to_input_params()` are gone; the NetCDF/HDF5 v0 writers keep their padded layout via a new `cumes::io::LegacyInputProvenance`. Tests/benchmark drive `read_and_validate`/`validate` through a `cumes_test_support` `loadValidated`/`validateSpec` helper. |

Both are Class A: Solovev `251→199→456` FSQR 9.583e-17, W7-X `1877→1617→2011`
FSQR 9.778e-13, 35/35 CTest, float + none/netcdf/hdf5 backend builds clean.

## 2. Item 4 (decided): `dynSharedBase()` is a deferred Class B follow-up

The non-templated dynamic-shared-memory base accessor (`dynSharedBase()` in
`fourier_impl.cuh`/`geometry_impl.cuh`/`constraint_impl.cuh`) is **retained**.
Switching to a direct `extern __shared__ T[]` changes `--use_fast_math` FMA
fusion in the consumers (opaque function return vs. known shared-array aliasing)
and perturbs the trajectory at ~1e-10 — a Class B change, not the Class A
bitwise equivalence the library split must preserve. Decision: leave it in
place and re-freeze only in a dedicated Class B pass (already documented in the
`.cuh` comment and `docs/architecture.md` §2). No code change in step 13.

## 3. Remaining: migration step 13 items 3, 5, 6 (the legacy-struct endgame)

This is the same "not a single-session change" endgame `tail-2` §4 described.
Item 1 and 2 were the mechanical preconditions; the rest is the coupled
deletion of the owning structs **and** the tests that construct them. In
dependency order:

### 3.1 Delete the six legacy owning structs + their free functions

The `cumes` operators **already own** the structs (strangler-fig): `Profiles`
owns `RadialProfiles`, `ToroidalFftOperator` owns `FourierPlan`,
`GeometryOperator` owns `MetricWorkspace`, `Preconditioner` owns
`PreconWorkspace`, `ConstraintOperator` owns `ConstraintWorkspace`, and
`SpectralStorage::legacy_view()` materializes `SpectralState`. Deleting them
means the operators own the raw `DeviceBuffer<T>`s directly and the
`src/*_impl.cuh` kernel bodies become the operators' method implementations
(the build/library split is already done — `cumes_cuda_double`/`_float` are the
nine `*_double.cu`/`*_float.cu` TUs).

Delete, leaf-first:

- `SpectralState` (vmec_types.h) — consumers are `outputSaveBinary`/`outputPrint`
  (`output.cpp`), `outputSaveNetcdf`/`outputSaveHdf5`, `seed_state.hpp`'s
  `init_state`/`restart_state` (they use `storage.legacy_view()` only for the
  six upload `cudaMemcpy`s), and the `solver_impl.cuh` dump machinery (`st.d_*`).
  Replace with `SpectralStorage` slab accessors; `legacy_view()` goes away.
- `RadialProfiles` (vmec_types.h) — `Profiles` should own `DeviceBuffer<T>`
  fields and expose `RadialProfileViews` directly (the view struct already
  exists in `real_fields.cuh`).
- `MetricWorkspace` (geometry.cuh) — `GeometryOperator` owns the buffers; the
  base-geometry/field split (step 5, `0a2e55c`) already put
  `computeBaseGeometry`/`computeMagneticField` behind the two operators.
- `PreconWorkspace` (precon.cuh) — `Preconditioner` owns the buffers.
- `ConstraintWorkspace` (constraint.cuh) — `ConstraintOperator` owns the buffers.
- `FourierPlan` (fourier.cuh) — `ToroidalFftOperator` owns the cuFFT plans +
  scratch + poloidal tables directly; `inverseDFT`/`inverseDFTFused`/
  `forwardDFT`/`fourierCombineParity`/`constraintDealiasBandpass` become its
  methods (the `RealSpaceStorage` split is already done).

The `*Create`/`*Free` + `compute*`/`inverseDFT`/`forwardDFT`/`preconCompute`/
`preconApply`/`constraintCompute`/`interpolateState` free-function entry points
are deleted alongside their struct. `interpolateState` (`refine.cuh`) already
has a `Prolongation` operator wrapping it.

### 3.2 Update the tests that construct legacy structs directly

`test_fourier`, `test_forces`, `test_geometry_iso`, `test_geometry_ncurr`,
`test_constraint_tcon`, `test_force_reference`, `test_force_verify`,
`test_axisym_backend`, `test_regression_kernels` all call `fourierCreate`/
`metricCreate`/`preconCreate`/`constraintCreate`/`realSpaceCreate`/
`modeTableCreate` + the free functions. They must drive the operators
(`ToroidalFftOperator`, `GeometryOperator`, `Preconditioner`,
`ConstraintOperator`, `Prolongation`, …) instead. This is coupled to 3.1 — a
struct cannot be deleted until its tests stop constructing it.

### 3.3 Update `CMakeLists.txt` + `docs/architecture.md`

- Drop the legacy `.cu` TUs from `CUMES_CUDA_MODULES` once each `*_impl.cuh` is
  folded into its operator's `src/cumes/` TU (the `cumes_cuda_double`/`_float`
  targets remain; their source lists shrink).
- `docs/architecture.md` §1 "Two layers" → "one layer", and §5's "still
  pending" list (which still names `FourierPlan`, `MetricWorkspace`,
  `InputParams`, …) shrinks to empty. `InputParams` is already gone as of
  `22e71d0`; the remaining entries are the owning structs named in 3.1.

Verification bar per step (unchanged): `./build/cuMES inputs/{solovev,w7x}.json`
bit-identical (`251→199→456` / `1877→1617→2011`), 35/35 CTest + the
optional-backend matrix (none/netcdf/hdf5), float build clean.

## 4. Commits (this handover)

```
0ef9e82 Phase 11 step 13.3 (part 4): delete ConstraintWorkspace + constraint free functions
ba88d05 Phase 11 step 13.3 (part 3): delete PreconWorkspace + preconCreate/preconFree/preconApply
c9da234 Phase 11 step 13.3 (part 2): delete MetricWorkspace + geometry/forces/precon free functions
b483428 Phase 11 step 13.3 (part 1): delete RadialProfiles + profilesCreate/profilesFree
22e71d0 Phase 11 step 13.2: InputParams -> ValidatedProblem (delete the legacy parser)
a7c0030 Phase 11 step 13.1: GridParams<T> -> DeviceParams<T> (per-stage param pack)
```

Step 13.3 is four-sixths done: `RadialProfiles`, `MetricWorkspace`,
`PreconWorkspace`, and `ConstraintWorkspace` are deleted (each operator owns its
buffers directly and exposes typed views/accessors; the matching
`*Create`/`*Free` + `compute*`/`inverseDFT`/`forwardDFT`/`precon*`/
`constraint*` free functions are gone). Remaining: `FourierPlan` (the cuFFT
plans + transform scratch, owned by `ToroidalFftOperator`), `SpectralState`
(`SpectralStorage::legacy_view()`), and the `interpolateState` free function
(`Prolongation`), followed by the remaining kernel tests and the CMake/docs
close-out. All follow the same established pattern.
