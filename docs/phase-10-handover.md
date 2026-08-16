# cuMES Phase 10 handover — retire compatibility internals

Status date: 2026-08-16. Branch: `overhaul` (Phase 0 `bd26857` + Phase 1
`12bcc44` + Phase 2 `168170a` + Phase 3 `c21564c` + Phase 4 `1b0d099` + Phase 5
`2b9aaf8` + Phase 6 `759f933` + Phase 7 `a59fff1` + Phase 8 `1069939` + Phase 9
`bda03f8` + this Phase 10 work). This document records what Phase 10 of
`docs/cuda-overhaul-blueprint.md` delivered, how it was verified, and what was
deliberately left for the tail of the migration.

## 1. Scope

Phase 10 is **retire compatibility internals** (blueprint §11, last phase). Its
exit gate is: *"source dependency graph matches section 5, all release gates
pass from a clean clone, and no legacy code is required for normal execution."*
The concrete work this phase delivered is the retirement of the two
measured-but-not-adopted Phase 9 prototypes, the correction of a stale
documented iteration count, and the publication of the tested contracts as
standalone docs:

| Item | Status |
| ---- | ------ |
| Retire the §8.10 force-split prototype | **Done** (`c9caeec`) |
| Retire the Phase-7 rzCon reference path | **Done** (`c2b467e`) |
| Correct the stale W7-X count in CLAUDE.md/README.md | **Done** (`c52a84f`) |
| Publish architecture/data-layout/performance docs | **Done** (`7b60910`) |
| Delete fixed-capacity `InputParams` / raw owning workspaces / old kernels | **Blocked** (§4) |
| Freeze `configs/schema-v1.json` artifact | **Deferred** (§4) |

## 2. What changed

### 2.1 Force-split prototype retired (`c9caeec`)

Removed `computeForcesSplit`, `rzForcesKernel`, `lambdaForcesKernel`, the
`computeForcesSplit` declaration and double/float explicit instantiations,
`test_force_split.cu`, and all CMake references. `computeForces` (the monolith)
remains the sole production force path, pinned by `test_force_reference.cu`
(scalar CPU reference) and the frozen trajectories. ADR-0002 now records the
retirement; its durable conclusion — the force kernel is input-traffic-bound,
so register-reduction strategies do not pay — is preserved.

### 2.2 rzCon reference path retired (`c2b467e`)

Removed `constraintRzConCompute`, `rzConPackKernel`, `rzConAccumulateKernel`,
the `ConstraintWorkspace` rzCon round-trip (`d_zeta_real_rz`,
`d_zeta_spectra_rz`, `plan_z2d_rz`) and its cuFFT work-area accounting, the
double/float instantiations, `test_rzcon_fusion.cu`, and all CMake references.

Since Phase 7.2 the fused `inverseDFTFused` has been the production rCon/zCon
producer; the reference path was retained only as the oracle for the
differential tests. `test_constraint_tcon` and `test_axisym_backend` now
produce rCon/zCon via `inverseDFTFused` (Class B ULP-equivalent), so the
reference is no longer needed. This also drops ~2.8 MB of W7-X-double constraint
scratch (`plan_z2d_rz` + the 4-slot compact batch) and one cuFFT plan per stage.

### 2.3 Stale W7-X count corrected (`c52a84f`)

CLAUDE.md and README.md recorded the W7-X run as `1878 → 1617 → 2011 (5506)`.
The frozen baseline (Phase 6B onward) is `1877 → 1617 → 2011 (5505)` — the
stage-1 count moved with the Phase-6B device-only force-norm reduction and the
docs were never updated. Corrected in both files.

### 2.4 Docs published (`7b60910`)

- `docs/architecture.md` — the two-layer (legacy kernels + `cumes` scaffold)
  reality, the build/library split, the production per-iteration pipeline, the
  dependency rules, and the remaining strangler-fig tail.
- `docs/data-layout.md` — the spectral/real/half-grid layouts, parity and
  mode numbering, the reduced-theta quadrature, and the legacy binary v0
  contract.
- `docs/performance.md` — the measured TITAN Xp numbers, the Phase 9
  adopted/not-adopted outcomes (ADR-0001..0003), and the §10.7 acceptance
  policy.
- `docs/adr/0002-force-split.md` — status-update section recording the
  retirement.

## 3. Verification

Both retirements are **pure dead-code removals** — the production `solverRun`
path never called `computeForcesSplit` or `constraintRzConCompute`, so no
numerical expression, kernel, launch, or scheduling decision changed. The
trajectory was re-run against the frozen baseline:

| Config | Effective iterations | Final FSQR | Restarts | Verdict |
| ------ | -------------------- | ---------- | -------- | ------- |
| Solovev | 251 → 199 → 456 | 9.583e-17 | 0 | bit-identical |
| W7-X | 1877 → 1617 → 2011 (5505) | 9.778e-13 | 15 (10 BADJ + 5 BADP) | bit-identical |

Test matrix:

| Preset | Result |
| ------ | ------ |
| `verify` (double, sanitizers ON) | **35/35** (23 unit + 13 compute-sanitizer memcheck + 1 smoke) — two fewer tests than Phase 9 because `test_force_split` and `test_rzcon_fusion` were removed |
| `float` | compiles cleanly (the float CLI and the float test legs) |

The W7-X restart sequence was re-confirmed as 5 BAD PROGRESS + 10 BAD JACOBIAN
(the 3 additional "invalid √g" log lines are initial invalid-geometry passes,
not restart events), matching the Phase 6/7/8 handovers.

## 4. Deferred / blocked (documented, not hidden)

The blueprint's full Phase 10 exit gate — *"no legacy code is required for
normal execution"* — is **not** met this phase, and cannot be met until the
strangler-fig migration replaces the legacy kernels outright:

- **`include/*.cuh` kernel structs (`FourierPlan`, `MetricWorkspace`,
  `InputParams`, …) are still the production implementation.** Deleting them
  is explicitly gated on "only after their consumers are gone" (blueprint §11);
  the `cumes` operator classes do not yet own the production hot loop, so this
  is a larger multi-phase migration, not a Phase 10 cleanup.
- **`AxisymmetricOperator` (Phase 7.1) is not wired into the Solovev production
  path.** It is built and gated (`test_axisym_backend`) but swapping it in is a
  Class B trajectory change (re-freeze) and remains the highest-leverage
  submission-bound-shape follow-up. It is also the prerequisite for re-measuring
  graph benefit (ADR-0003).
- **`configs/schema-v1.json` is not emitted.** The schema-v1 and checkpoint
  compatibility contracts are specified (blueprint §6.13) and exercised by the
  host I/O tests, but the standalone schema artifact is future work.
- **`docs/mathematics.md` / `docs/verification.md`** are not written separately;
  the normative mathematics is blueprint §4 and the verification tiers are
  blueprint §10, now cross-referenced from `docs/architecture.md`.

## 5. Next steps

1. Wire `AxisymmetricOperator` into the Solovev production path behind the
   retained generic backend (Class B re-freeze), then re-measure graph
   submission savings (ADR-0003).
2. Land the §8.3 fused descent/checkpoint behind the benchmark harness.
3. Continue the strangler-fig migration toward the §5 target: replace the
   legacy kernel structs with the `cumes` operator classes, then delete the
   now-unused `include/*.cuh` legacy headers and `InputParams`.
4. Emit `configs/schema-v1.json` and freeze the checkpoint compatibility policy
   as a versioned artifact.
