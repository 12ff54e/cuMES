# cuMES Phase 11 handover — strangler-fig migration (owning operators + FourierPlan split)

Status date: 2026-08-16. Branch: `overhaul` (… + Phase 10 tail `9d18e4e` + this
Phase 11 work). This document records what the strangler-fig migration
(`docs/strangler-fig-migration-plan.md`) delivered in its first landing, how it
was verified, and what was deliberately left for the tail.

## 1. Scope

Phase 11 begins the blueprint §11 Phase 10 exit gate: *"source dependency graph
matches section 5, … and no legacy code is required for normal execution."* It
is the migration's first committed, verified landing — **wrap the legacy
workspaces in owning operator classes and split the `FourierPlan`** so the
transform operator owns only transform scratch. It is deliberately **not** the
full exit gate (see §4).

The concrete work, in four commits:

| Commit | What |
| ------ | ---- |
| `87cec03` | four owning operators (`Preconditioner`, `ConstraintOperator`, `GeometryOperator`, `Profiles`) + solver rewire |
| `75dcc65` | `ToroidalFftOperator` owns the `FourierPlan` (all 5 workspaces owned) |
| `25bcb97` | docs: record the progress in the migration plan |
| `a692200` | split `FourierPlan` — transform scratch vs stage-owned `RealSpaceStorage` |

## 2. What changed

### 2.1 Five owning operators (`87cec03`, `75dcc65`)

Each legacy workspace moved out of `solverRun`/`StageSolver` into a `cumes`
operator that owns it and wraps the existing free function/kernel unchanged:

| Operator | Owns | Wraps |
| -------- | ---- | ----- |
| `Profiles` | `RadialProfiles` | `profilesCreate` |
| `ToroidalFftOperator` | `FourierPlan` | `inverseDFTFused` / `forwardDFT` |
| `GeometryOperator` | `MetricWorkspace` | `computeGeometry` + `computeJacobianStats` + `computeForceNormPartials` |
| `Preconditioner` | `PreconWorkspace` | `preconCompute` / `preconApply` (which already routes the solve through the Phase 8 `PcrBackend`/`ThomasBackend`) |
| `ConstraintOperator` | `ConstraintWorkspace` | `constraintCompute` (generic or axisym de-alias) |

Implemented in each module's `*_impl.cuh` (the `PcrBackend`/`ThomasBackend`
pattern), explicitly instantiated per scalar type. `StageSolver::run` constructs
them (RAII); `solverRun` calls their `enqueue` methods in the hot loop.

### 2.2 `FourierPlan` split (`a692200`)

`FourierPlan<T>` no longer owns the 43 real-space arrays (18 parity-split
geometry + 9 combined + 16 force). Those move to a new stage-owned
`cumes::RealSpaceStorage<T>` (`include/cumes/state/real_space_storage.hpp`,
`realSpaceCreate`/`realSpaceFree` in `fourier.cuh`/`fourier_impl.cuh`), so the
transform operator owns only transform scratch (plans, ζ buffers, poloidal
tables, work area, mode tables). The transform/geometry/force/constraint/
preconditioner functions now take `RealSpaceStorage&` (geometry/force) alongside
`FourierPlan&` (tables). This satisfies the blueprint §5.1 dependency rule —
*transforms no longer own force/geometry fields*.

## 3. Verification

Every commit is **pure Class A** (bit-identical): the operators wrap the exact
legacy kernels/launch configs, and the `RealSpaceStorage` pointers are bit-for-bit
the `FourierPlan`'s old members. Re-verified at each commit:

| Config | Effective iterations | Final FSQR | Verdict |
| ------ | -------------------- | ---------- | ------- |
| Solovev (generic, `CUMES_FORCE_GENERIC=1`) | 251 → 199 → 456 | 9.583e-17 | bit-identical |
| Solovev (axisymmetric, default) | 251 → 199 → 456 | 9.583e-17 (ULP-equiv) | fast path intact |
| W7-X | 1877 → 1617 → 2011 (5505) | 9.778e-13 | bit-identical |

Test matrix: **35/35 CTest** (23 unit + 12 compute-sanitizer memcheck + 1 smoke),
float build clean. The `CUMES_FORCE_GENERIC=1` knob (added in the Phase 10 tail)
remains the A/B switch for the transform backend.

## 4. Deferred / remaining (documented, not hidden)

The blueprint's full Phase 10 exit gate — *"no legacy code is required for normal
execution"* — is **not** met; the legacy `include/*.cuh` structs and `InputParams`
are still the production kernels, and the operators still *name* the legacy
structs they read (via `fourier_plan()`/`workspace()` aliases) in their
transitional form. Remaining, in dependency order:

1. **`SpectralOperator` unification.** `ToroidalFftOperator` is not yet a
   `SpectralOperator<T>` peer of `AxisymmetricOperator`; `solverRun` still has
   the `axisym_active` branch. Unifying requires resolving the fused-vs-split
   rCon/zCon impedance (the generic backend fuses rCon/zCon into the inverse via
   `inverseDFTFused`; the axisymmetric backend splits them into `enqueue_inverse`
   + `enqueue_rzcon`). The §6.6 interface currently declares only
   `enqueue_inverse`/`enqueue_forward`; it must grow rCon/zCon (and the
   de-alias) to cover both backends.
2. **Stateless operators.** `ForceOperator`, `ResidualOperator`,
   `DescentOperator`, `Prolongation` — thin wrappers over `computeForces`,
   `computeResidualsKernel`, `descentStepKernel` + checkpoint, and
   `interpolateState`. They are declaration-only in `include/cumes/*`.
3. **Legacy-struct deletion.** `include/*.cuh` (`fourier`, `geometry`, `forces`,
   `profiles`, `precon`, `constraint`, `refine`, `solver`, `input`, `input_json`,
   `output`, `vmec_types` where superseded) + `InputParams`, gated on `solverRun`
   naming no legacy struct. Also the `dynSharedBase()` non-templated shared-memory
   indirection (removable only as a Class B re-freeze, per the `--use_fast_math`
   FMA note in `constraint_impl.cuh`/`geometry_impl.cuh`).

## 5. Next steps

1. Land `SpectralOperator` unification (collapse the `axisym_active` branch).
2. Land the four stateless operators.
3. Delete the legacy structs, then re-check the §5 dependency graph and the
   blueprint §14 definition of done from a clean clone.
