# cuMES Phase 7 handover — transform specialization

Status date: 2026-08-16. Branch: `overhaul` (Phase 0 `bd26857` + Phase 1
`12bcc44` + Phase 2 `168170a` + Phase 3 `c21564c` + Phase 4 `1b0d099` + Phase 5
`2b9aaf8` + Phase 6 `759f933` + this Phase 7 work). This document records what
Phase 7 of `docs/cuda-overhaul-blueprint.md` delivered, how it was verified, and
what was deliberately deferred and why.

## 1. Scope

Phase 7 is **transform specialization** (§8.4–§8.7): specialize the spectral
transforms for the axisymmetric case and fold the constraint's duplicate
transform work into the main inverse accumulator. The blueprint lists five
deliverables:

| Blueprint § | Deliverable | Status |
| ----------- | ----------- | ------ |
| §8.5 | Axisymmetric transform + constraint/bandpass backend | **Done** (commit `26b66d0`) |
| §8.4 | Weighted R/Z constraint accumulation fused into inverse | **Done** (commit `05a59d1`) |
| §8.6 | Bounded theta/zeta/mode tiles | **Deferred** (§5) |
| §8.7 | Pack/recover transpose experiments | **Deferred** (§5) |
| §8.7 | Generalized de-alias coverage | **Already done** (Phase 0 containment, §6) |

The two delivered items are the structurally-driven transform specializations
with concrete differential/bitwise gates. The two deferred items are
performance experiments whose acceptance criterion (§8.1) is a measured speedup
on a benchmark harness that is still deferred from Phase 6.

## 2. What changed

### 2.1 Axisymmetric transform backend (`26b66d0`)

For `ntor = 0, nzeta = 1` the toroidal direction is a single point and every
folded mode has n = 0, so the product basis collapses to
`R = Σ rmncc·cos(mθ)`, `Z = Σ zmnsc·sin(mθ)`, `λ = Σ lmnsc·sin(mθ)` with zero
toroidal derivatives and no sin(nζ) families. The generic backend produces
exactly this after its length-one Z2D/D2Z, so `AxisymmetricOperator` performs
the same poloidal synthesis/projection directly and never creates or executes a
length-one cuFFT plan.

- **`include/cumes/transforms/axisymmetric_operator.hpp`** + `src/
  axisymmetric_impl.cuh` + `_double.cu`/`_float.cu`: a concrete
  `SpectralOperator` backend owning its per-mode trigonometric tables. It
  implements `enqueue_inverse` (18 parity-split geometry arrays) and
  `enqueue_forward` (6 spectral families), plus the axisymmetric constraint
  helpers `enqueue_rzcon` (xmpq-weighted rCon/zCon) and `enqueue_dealias`
  (the bandpass as a direct poloidal sum).
- **Interface refinement**: `SpectralOperator::enqueue_forward` gains a
  `ConstraintForceViews` parameter — the forward DFT folds `xmpq·frcon/fzcon`
  into the R/Z projections (blueprint §4.8), so the abstract contract now
  names the real input. `ConstraintForceViews` was added to
  `real_fields.cuh`.
- **`constraintDealiasBandpass` extracted** from `constraintCompute` so the
  bandpass is testable in isolation (the axisymmetric backend replaces exactly
  that step). This is a pure code move — the Solovev dump tree is byte-identical
  (Class A).

### 2.2 Weighted R/Z fusion (`05a59d1`)

The main inverse poloidal accumulator already stages every per-m R/Z ζ-signal,
so it now accumulates the `xmpq = m(m-1)`-weighted rCon/zCon sums at the same
time — removing the constraint's separate pack + zeta inverse + accumulation
(`constraintRzConCompute`'s rzCon path) from the hot loop.

- **`inverseAccumulateKernel` gains a `FuseRzCon` non-type parameter** (guarded
  by `if constexpr`): the `FuseRzCon=false` geometry path is bit-identical to
  the pre-change kernel (verified by byte-comparison), while the `=true` path
  additionally writes rCon (from the R-slot launch) and zCon (from the
  Z-slot launch) as full real-space fields.
- **`inverseDFTFused`** is the fused entry point; the solver calls it once per
  pass and drops the separate `constraintRzConCompute` launch. The reference
  rzCon path (function + `plan_z2d_rz` + `d_zeta_real_rz`/`d_zeta_spectra_rz`)
  is retained for the differential test and the two constraint tests that still
  read it.

## 3. Key design decisions

1. **Axisymmetric backend as a separate, selectable operator — not wired into
   the production hot loop.** The blueprint §8.5 gate is "runs both backends
   and compares every transform product"; the acceptance is a differential test
   against the retained generic backend, not an immediate swap-in. Wiring
   Solovev through `AxisymmetricOperator` is a Class B trajectory change
   (re-freeze) left as the follow-up (§5).
2. **`FuseRzCon` as a compile-time `if constexpr`, not a runtime branch.** The
   concern is real: adding live registers/statements to the hottest kernel can
   change `--use_fast_math` FMA contraction of the existing expressions and
   perturb the geometry at ULP level (the same hazard the `dynSharedBase()`
   indirection guards against). The template parameter makes the non-fused
   instantiation identical to the old kernel, so the Class A bitwise gate holds
   for the plain `inverseDFT` path while the fused path is a clearly-scoped
   Class B change.
3. **Retained reference paths, not eager deletion.** Both `constraintRzConCompute`
   and the axisymmetric constraint helpers keep their cuFFT/reference
   counterparts available for differential tests. The ~2.8 MB rzCon scratch is
   still allocated (the reference path owns it); retiring it belongs with the
   broader reference-retirement in Phase 10, not a Phase 7 cleanup that would
   remove the very oracle the differential tests compare against.

## 4. Verification

### Differential tests (the per-intermediate gate, §8.5/§8.4)

| Test | What it compares | Result |
| ---- | ---------------- | ------ |
| `test_axisym_backend` | `AxisymmetricOperator` vs the generic cuFFT backend on frozen axisymmetric inputs: inverse 18 arrays, forward 6 families, rCon/zCon, gCon | PASS — double ≤ 5e-16, float ≤ 3e-8 |
| `test_rzcon_fusion` | fused rCon/zCon vs `constraintRzConCompute`; fused geometry bitwise vs `inverseDFT` | PASS — double 1.4e-14, float 7.6e-6; geometry bitwise |

### Class A bitwise gate

The `FuseRzCon=false` geometry path and the `constraintDealiasBandpass`
extraction are pure moves: the full Solovev dump tree (235 files) is
byte-identical to the Phase-6 baseline after both commits.

### Class B trajectory gate (`compare_runs.py` vs pre-change baselines)

| Config | Restart sequence | Convergence | Final residual | Final state (interior) |
| ------ | ---------------- | ----------- | -------------- | ---------------------- |
| Solovev | identical (0 events) | 456 = 456 | identical (rel 0) | ≤ 9e-14 |
| W7-X | identical (15 events) | 2011 = 2011 | identical (rel 0) | ≤ 4e-9 |

The W7-X restart sequence (5 BADP + 10 BADJ at identical iterations) is the
critical controller gate: the fusion changes the constraint force at the ULP
level but does not change a single branch decision. The near-axis λ state
spreads the most (4e-9), consistent with the known near-degenerate λ-gauge
amplification noted in the Phase 5/6 handovers.

### Test matrix

| Preset | Result |
| ------ | ------ |
| `verify` (double, both backends) | **30/30** (20 unit + 10 compute-sanitizer memcheck) |
| `float` | compiles; both new tests run the float leg |

## 5. Deferred (documented, not hidden)

- **Bounded mode tiles (§8.6).** The kernels are already ζ-tiled (`computeKTile`
  from the Phase 0 containment) and the de-alias analyze loops over 32-point
  θ-groups, so the *bounded-launch* part is done; the remaining gap is **mode
  tiles** (the per-thread m-loop is still serial `O(mpol)`). This is a
  shared-memory tiling rewrite whose acceptance (§8.1) is a *measured* speedup.
  It should follow the benchmark harness below, not precede it.
- **Pack/recover transpose experiments (§8.7).** Shared-memory tiled transpose
  vs `cufftPlanMany` embedding/stride vs the current pack/recover — a
  benchmark comparison, not a code change to land without evidence.
- **`cumes_benchmark_fixed_iteration` (§8.1, deferred since Phase 6).** The
  fixed-iteration harness (median/p95 wall µs, host-blocking counts, arena/cuFFT
  bytes, residual/state hashes, per-kernel regs/spills/occupancy) is the
  prerequisite for the two deferred items above; it should be the first task of
  the next phase.
- **Production wiring of `AxisymmetricOperator`.** Solovev still runs the
  generic cuFFT backend; swapping in the axisymmetric operator is a Class B
  trajectory change (re-freeze) once the operator has a short-trajectory gate
  beyond the component differential test.
- **rzCon plan/scratch retirement.** The `plan_z2d_rz` + ~2.8 MB compact
  scratch are still allocated (the reference `constraintRzConCompute` owns
  them); retiring them belongs with Phase 10 reference-retirement after the
  differential tests become permanent regressions.

## 6. Generalized de-alias coverage (already delivered)

The Phase 0 containment already generalized the de-alias analysis to loop over
32-point θ-groups (so `ntheta > 32` is fully summed, not silently truncated);
`test_regression_kernels` and its sanitizer variant cover the awkward angular
shapes. No further Phase 7 work is required for this item.

## 7. Next steps

1. Land `cumes_benchmark_fixed_iteration` (§8.1) — the measurement harness that
   gates every remaining performance claim.
2. With the harness in place, attempt the §8.6 mode tiles and the §8.7
   pack/recover transpose, selecting by measured shape as the blueprint
   requires.
3. Wire `AxisymmetricOperator` into the Solovev production path behind the
   retained generic backend, with a short-trajectory + full-regression
   re-freeze (Class B).
4. Retire the reference rzCon path (plan/scratch) once the differential tests
   are permanent regressions (Phase 10 scope).
