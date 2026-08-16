# cuMES Phase 8 handover — scalable preconditioner and reductions

Status date: 2026-08-16. Branch: `overhaul` (Phase 0 `bd26857` + Phase 1
`12bcc44` + Phase 2 `168170a` + Phase 3 `c21564c` + Phase 4 `1b0d099` + Phase 5
`2b9aaf8` + Phase 6 `759f933` + Phase 7 `a59fff1` + this Phase 8 work). This
document records what Phase 8 of `docs/cuda-overhaul-blueprint.md` delivered,
how it was verified, and what was deliberately deferred and why.

## 1. Scope

Phase 8 is **scalable preconditioner and reductions** (§8.8, §8.9, §4.9): make
the radial tridiagonal solve backend-neutral and scale-aware, and replace the
shared-memory reduction trees with warp shuffles. The blueprint lists four
deliverables:

| Blueprint § | Deliverable | Status |
| ----------- | ----------- | ------ |
| §8.9 | Backend-neutral batched tridiagonal API + scalable implementation | **Done** (`f58acd6`) |
| §4.9 | Scale-aware pivot/breakdown status | **Done** (`f58acd6`) |
| §8.8 | Warp/CUB reductions with deterministic verification mode | **Done** (`fa37b7e`, Class B) |
| §8.8 | Optional refresh-stream overlap after measurement | **Deferred** (§5) |

The first three are the self-contained numerical contracts of the phase. The
fourth is gated on the benchmark harness (`cumes_benchmark_fixed_iteration`,
deferred since Phase 6), so it stays deferred.

## 2. What changed

### 2.1 Backend-neutral tridiagonal solve + scale-aware pivot (`f58acd6`)

- **`include/cumes/numerics/tridiagonal_backend.hpp`** (fleshed out from the
  Phase-5 stub): a concrete `StridedBatchTridiagonalView` carrying the
  `lower/diagonal/upper` coefficients, a `rhs` strided batch (`rhs_count` RHS
  planes sharing one elimination, `rhs_stride` between planes), the per-mode
  `first_surface` (jMin) and `scale`, and a shared `last_surface` (the excluded
  LCFS row). `TridiagonalStatus` and `PivotPolicy` name the pivot contract.
- **Two concrete backends** in `src/precon_impl.cuh`:
  - `PcrBackend` — the production 128-thread grid-stride PCR, extracted
    **bit-for-bit** from the legacy `tridiagSolveKernel` (the `rhs_count` loop
    is a compile-time template parameter so the `#pragma unroll` and the
    `-use_fast_math` FMA contraction of the staged arithmetic are preserved).
  - `ThomasBackend` — a serial Thomas reference (one thread per system), the
    small-batch fallback named in §8.9.
- **`preconApply` routes through `PcrBackend`**: the R (comps 0,3) and Z
  (comps 1,4) systems are two `enqueue_solve` calls with `rhs_count=2`, and the
  boundary zeroing + lambda-diagonal finishing move to a separate
  `preconBoundaryKernel`. The legacy `tridiagSolveKernel` is gone; the legacy
  `preconCompute`/`preconApply` signatures are unchanged so the solver and the
  regression tests keep their call sites.

### 2.2 Scale-aware pivot/breakdown (`f58acd6`)

The legacy **absolute** `1e-30` pivot clamp (which silently flipped negative
pivots to `+1e-30` before the Phase-0 copysign fix, and still silently clamped
near-singular diagonals) is replaced by the blueprint §4.9 scale-aware floor:

```
floor = kappa * eps_T * scale[mode],   scale = max |lower|,|diagonal|,|upper|
```

- `preconScaleKernel` computes `scale[mode]` once per preconditioner refresh
  (the matrix changes only on refresh) into a new `PreconWorkspace::d_preconScale`.
- The solve kernels read it and **guard with `copysign` AND count the breakdown**
  into a device `d_preconStatus` accumulator (`atomicAdd`), instead of silently
  clamping. A genuinely sub-scale diagonal is now *reported*, not absorbed.
- For the frozen trajectories the diagonals are O(1)..O(m²), far above the
  floor, so the guard never fires and the solve is byte-identical (Class A).

### 2.3 Warp-shuffle reductions (`fa37b7e`, Class B)

`preconComputeKernel` (15 accumulators) and `lambdaPrecAssembleKernel` (3
accumulators) drop their shared-memory binary trees for a **fixed**
`__shfl_down_sync` within-warp tree plus a fixed cross-warp combine by thread 0.
The tree is fixed, so the result stays deterministic; only the summation order
changes (Class B). `preconComputeKernel` no longer needs dynamic shared memory
(launch `smem` 30720 → 0), and `lambdaPrecAssembleKernel`'s per-block shared
shrinks from 6 KB to 192 B.

## 3. Key design decisions

1. **`rhs_count` as a compile-time template parameter on `pcrSolveKernel`, not
   a runtime loop bound.** The legacy kernel's `#pragma unroll` over the two RHS
   planes is part of the frozen codegen; a runtime bound would let the compiler
   emit a different (non-unrolled) loop and risk a `-use_fast_math` FMA
   contraction difference. `enqueue_solve` dispatches on `rhs_count == 1/2`
   (production always uses 2), preserving bitwise parity.
2. **Scale computed in a separate kernel, not inline in the PCR kernel.** Adding
   a block reduction to the hot PCR kernel would risk changing its register
   allocation/codegen and the frozen arithmetic. `preconScaleKernel` runs once
   per refresh; the PCR kernel only reads `scale[mode]`, so its staged
   arithmetic is untouched.
3. **Backends operate on raw strided pointers, not `SpectralView`.** The
   tridiagonal solve is a generic batched operator; it should not know the
   six-component spectral layout. `preconApply` builds the views from the
   residual slab (`rhs_stride = 3*mnmax*ns` for the (0,3)/(1,4) component pairs),
   keeping the spectral-layout knowledge at the preconditioner boundary.
4. **The breakdown status is detected but not yet folded into the control
   record.** `d_preconStatus` accumulates every pass and is reset by
   `preconApply`; folding it into the host `ControlRecord`/`NumericalStatus`
   decision is a follow-up. The frozen trajectories never trigger it, so there
   is no behavioral change yet — but the *detection* now exists and is gated by
   `test_tridiagonal`.
5. **The old absolute `1e-30` is gone entirely**, not kept alongside the
   scale-aware floor. The new guard is structurally identical (same ternary
   `hasL/hasR` + `invL = hasL ? 1/dL : 0`), only the floor literal becomes a
   runtime scale-derived value, so the healthy path is bitwise-identical while
   the near-singular path is reported instead of silently clamped.

## 4. Verification

### Class A bitwise gate (the 8.1 extraction)

The PCR extraction is a pure move: the Solovev trajectory reproduces the frozen
values exactly — stages **251 → 199 → 456**, final **FSQR = 9.583e-17**,
total 906 effective iterations (matching the Phase-5/6/7 handovers and the
CLAUDE.md baseline).

### Class B trajectory gate (the 8.3 reductions, `compare_runs.py` vs the phase-7 tag)

| Config | Restart sequence | Convergence | Final residual | Final state (interior) |
| ------ | ---------------- | ----------- | -------------- | ---------------------- |
| Solovev | identical (0 events) | 456 = 456 | identical (rel 0) | **bitwise-identical (0.00e+00)** |
| W7-X | identical (15 events) | 2011 = 2011 | identical (rel 0) | ≤ 1.25e-9 |

The W7-X restart sequence (5 BADP + 10 BADJ at identical iterations) is the
critical controller gate, and it is unchanged. The near-axis λ families
(lmnsc/lmncs) spread the most (1.25e-9), consistent with the known
near-degenerate λ-gauge amplification noted in the Phase 5–7 handovers (the
Phase-7 fusion spread was 4e-9, so this is tighter). Solovev happens to be
bitwise-identical because its `nZnT = 18` surface sums are small enough that the
tree/shuffle order produces identical double roundings.

### Test matrix

| Preset | Result |
| ------ | ------ |
| `verify` (double, both backends) | **32/32** (21 unit + 11 compute-sanitizer memcheck) |
| `float` | compiles; `test_tridiagonal` runs both legs (double + float) |

`test_tridiagonal.cu` (new, `unit;precon` + sanitizer variant) drives the public
`TridiagonalBackend` interface directly: CPU serial Thomas vs GPU Thomas vs GPU
PCR across `ns = {3, 17, 65, 99, 130, 257, 512}`, mixed m-parity jMin,
`rhs_count = 2`, plus a zero-diagonal breakdown case (both backends must report
`status > 0`). The existing `test_regression_kernels.cu` still exercises the
production `preconCompute` + `preconApply` path (now through `PcrBackend`) on
real geometry at the same row-count matrix.

## 5. Deferred (documented, not hidden)

- **Tiled PCR/Thomas hybrid and a library backend** (§8.9 "long-term,
  benchmark"). The current `PcrBackend` is grid-stride (arbitrary rows) and its
  `10·ns·sizeof(T)` shared memory fits the validated `ns ≤ 512` range. A tiled
  hybrid or a strided-batch library solver is a *measured-speedup* item that
  should follow the benchmark harness, not precede it.
- **Refresh-stream overlap** (§8.8 "after measurement"). The auxiliary-stream
  overlap of preconditioner/force-norm work is gated on the still-deferred
  `cumes_benchmark_fixed_iteration` harness (deferred since Phase 6).
- **Folding `d_preconStatus` into the control record.** The breakdown is
  detected and accumulated; wiring it into the host `ControlRecord`/status bits
  so the solver can act on it is a follow-up (the frozen trajectories never
  trigger it, so this is a dormant diagnostic today).
- **`PivotPolicy` as a runtime knob.** `PivotPolicy.kappa` is wired through the
  backend constructors, but production uses the default `kappa = 1.0`; a
  documented policy surface (e.g. a degenerate `kappa = 1e30` to reproduce the
  legacy absolute clamp in tests) is not needed until a second policy appears.

## 6. Next steps

1. Land `cumes_benchmark_fixed_iteration` (§8.1) — the prerequisite for the
   tiled-hybrid backend, the refresh-stream overlap, and every remaining
   measured-performance claim.
2. With the harness, benchmark the `ThomasBackend` for the small axisymmetric
   shape and the `PcrBackend` for W7-X, then decide whether a tiled PCR/Thomas
   hybrid earns its complexity.
3. Fold `d_preconStatus` into the combined `ControlRecord` so a singular
   preconditioner is a structured numerical failure, not a dormant diagnostic.
4. Retire the legacy `preconCompute`/`preconApply` free-function signatures once
   the `Preconditioner` operator class (still a Phase-5 stub) is implemented
   behind them.
