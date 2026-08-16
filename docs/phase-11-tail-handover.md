# cuMES Phase 11 tail handover — operator unification + legacy-deletion begin

Status date: 2026-08-16. Branch: `overhaul`. This document records the work
that continues `docs/phase-11-handover.md` (the strangler-fig migration's first
landing): the `SpectralOperator` unification and the four stateless operators
(the handover's next-steps 1–2), and the first three ownership slices of the
legacy-struct deletion (next-step 3). Every commit is **Class A bit-identical**
to the frozen trajectory.

## 1. Scope

This is the continuation of Phase 11's strangler-fig migration. The prior
handover ended with five owning operators and a split `FourierPlan`, but left
three items open. This handover delivers:

1. **`SpectralOperator` unification** — `solverRun` drives a single
   `SpectralOperator<T>*` with no `axisym_active` branch.
2. **The four stateless operators** — `ForceOperator`, `ResidualOperator`,
   `DescentOperator`, `Prolongation`.
3. **The first legacy-deletion slices** — mode-table extraction, cuFFT stream
   binding, and `m1PreconScale` all move out of `solverRun`'s naming.

## 2. What changed

### 2.1 `SpectralOperator` unification (`d03640f`)

The interface grew to the full transform surface the solver needs, and
`ToroidalFftOperator` became a `SpectralOperator<T>` peer of
`AxisymmetricOperator`:

| Interface method | Toroidal (generic) | Axisymmetric |
| ---------------- | ------------------ | ------------ |
| `enqueue_inverse(coeff, geometry, rCon, zCon, stream)` | fused `inverseDFTFused` (rCon/zCon in the accumulate) | direct poloidal synthesis + rzCon kernel |
| `enqueue_forward(force, constraint_force, residual, stream)` | `forwardDFT` (constraint force as 4 raw ptrs) | direct reduced-θ projection |
| `enqueue_dealias(gConEff, tcon, faccon, gCon, stream)` | compact cuFFT round trip | direct poloidal bandpass |

The fused-vs-split rCon/zCon impedance was resolved by folding rCon/zCon into a
unified `enqueue_inverse` (the axisymmetric backend runs its rzCon kernel right
after its synthesis, in the same stream order the solver previously used). The
compact de-alias scratch + plans moved out of `ConstraintWorkspace` into the
`FourierPlan` (transform-owned), so the constraint reaches the bandpass through
`op->enqueue_dealias` instead of naming `FourierPlan`. The dead
`constraintComputeAxisym` free function was removed.

`solverRun`, `StageSolver`, and `benchmarks/fixed_iteration.cu` now build one
`SpectralOperator<T>*` (`nullptr` → generic) and call the three methods
unconditionally; the `if (axisym_active)` branches are gone.

### 2.2 Stateless operators (`3296ae1`)

`ForceOperator`, `ResidualOperator`, `DescentOperator`, `Prolongation` were
declaration-only; they are now thin wrappers wired into the drivers:

- `ForceOperator::enqueue(rs, p, rp, mw, stream)` → `computeForces`.
- `ResidualOperator::enqueue(residual, ns, mnmax, sq_out, stream)` →
  `computeResidualsKernel` (one launch, 3-group output; the invariant vs
  preconditioned distinction stays host-side).
- `DescentOperator::enqueue(state, velocity, residual, xm, xn, ns, mnmax,
  action, stream)` → `descentStepKernel` under a `DescentAction`; the
  single-copy checkpoint capture/restore stays with the solver's state slab
  (blueprint §6.10 keeps the checkpoint a distinct operator).
- `Prolongation::enqueue(p_new, state_old, p_old, stream)` → `interpolateState`.

### 2.3 Legacy-deletion ownership slices (`53ed57a`, `67ff2fd`, `b8fa3a0`)

Three Class-A moves began peeling the legacy structs out of `solverRun`:

| Commit | Move | Legacy naming removed from `solverRun` |
| ------ | ---- | -------------------------------------- |
| `53ed57a` | folded-mode table → `cumes::DeviceModeTable` (new `cumes/state/mode_table.cuh`); `FourierBasis` deleted | `fp.basis.d_xm/d_xn` |
| `67ff2fd` | cuFFT stream binding → `ToroidalFftOperator::bind_stream` | `fp.plan_z2d/d2z/d2z_da/z2d_da` |
| `b8fa3a0` | `m1PreconScaleKernel` → `Preconditioner::enqueue_m1_scale` | `pw.d_ard/d_brd/azd/bzd` |

The transform free functions (`inverseDFT`/`inverseDFTFused`/`forwardDFT`) and
`preconCompute` now take `const int* xm, const int* xn` instead of reaching into
`fp.basis`; `ToroidalFftOperator` holds a non-owning mode-table pointer and
exposes `xm()`/`xn()`.

## 3. Verification

Re-verified Class A bit-identical at every commit:

| Config | Effective iterations | Final FSQR | Verdict |
| ------ | -------------------- | ---------- | ------- |
| Solovev (axisym, default) | 251 → 199 → 456 | 9.583e-17 | bit-identical |
| Solovev (generic, `CUMES_FORCE_GENERIC=1`) | 251 → 199 → 456 | 9.583e-17 | bit-identical |
| W7-X | 1877 → 1617 → 2011 (5505) | 9.778e-13 | bit-identical |

Test matrix: **35/35 CTest** (23 unit + 12 compute-sanitizer memcheck + 1
smoke), float build clean.

## 4. Remaining legacy naming in `solverRun` (grepped, precise)

After `b8fa3a0`, the legacy structs `solverRun` still names are:

- **`FourierPlan`** — only the dump-gated `inverseDFT`/`fourierCombineParity`
  observability path (the hot loop is sealed behind the operator + `bind_stream`).
- **`PreconWorkspace` / `MetricWorkspace`** — named as arguments
  (`constraint.enqueue` takes `pw`; `precon.enqueue_compute`/`enqueueForceNorms`
  take `mw`); their `.d_*` field reads are now dump-only.
- **`ConstraintWorkspace` / `RadialProfiles`** — still read in the hot loop: the
  `RealFieldView`s over `cw.d_rCon/d_zCon/d_frcon/d_fzcon`, and
  `rp.d_sqrtS_F`/`rp.delta_s`/`rp.d_dVds_H`/`rp.d_pres_H`.
- **`GridParams<T>`** — everywhere (needs the `DeviceParams<T>` replacement).
- **`InputParams`** — in `StageSolver`/`main` (needs `ValidatedProblem`).

`dynSharedBase()` (the non-templated shared-memory indirection) is still
retained in `fourier_impl.cuh`/`constraint_impl.cuh`/`geometry_impl.cuh`; it is
removable only as a Class B re-freeze (the `--use_fast_math` FMA note), to be
decided at step 13.

## 5. Next steps (dependency order)

1. **Seal `FourierPlan` from the solver.** Give `ToroidalFftOperator` a
   dump-only `combine_parity()`/`inverse_dump()` accessor (or move the dump
   path behind a diagnostic operator) so `solverRun` drops the
   `transform.fourier_plan()` alias entirely.
2. **View accessors for `Profiles` / `ConstraintOperator`.** Add typed
   `RadialProfileViews`/`ConstraintForceViews`/rCon-zCon accessors so
   `solverRun` stops building raw `RealFieldView`s over `cw.d_*` and reading
   `rp.d_*` directly.
3. **Migration step 5** — split base geometry from B/pressure (Class B; the
   `1/√g` division is fused into `geometryKernel`, so a split is a re-freeze
   with a differential proof).
4. **Migration step 12** — `EquilibriumOperator` composition: `solverRun`
   becomes a thin loop over `IterationController` + `EquilibriumOperator`.
5. **Migration step 13** — delete `include/*.cuh` legacy structs + `InputParams`
   after `solverRun` names none of them; replace `GridParams` with `DeviceParams`
   and `InputParams` with `ValidatedProblem` (the `to_input_params()` bridge
   already exists); decide `dynSharedBase()` removal.

## 6. Commits (this handover)

```
f9a1b0d docs: record the three legacy-deletion ownership slices
b8fa3a0 Phase 11: move m1PreconScale into the Preconditioner operator
67ff2fd Phase 11: move cuFFT stream binding into ToroidalFftOperator
53ed57a Phase 11: extract the mode table out of FourierPlan (DeviceModeTable)
0bb4a91 docs: record strangler-fig progress (SpectralOperator unification + stateless operators)
3296ae1 Phase 11: land the four stateless operators (force/residual/descent/prolongation)
d03640f Phase 11: SpectralOperator unification — collapse the axisym_active branch
```
