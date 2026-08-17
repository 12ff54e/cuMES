# cuMES Phase 11 close-out handover — `dynSharedBase()` removal landed, migration complete

Status date: 2026-08-16. Branch: `overhaul`. This handover continues
`docs/phase-11-tail-3-handover.md` and closes out **Phase 11** (the
strangler-fig migration, blueprint §11 / `docs/strangler-fig-migration-plan.md`):
the one deferred step-13 item — the `dynSharedBase()` dynamic-shared-memory
indirection (tail-3 §2, "item 4") — is now removed, and the change measured
**Class A bit-identical**, stronger than the Class B re-freeze the docs
predicted. The frozen baselines stand unchanged.

> **Post-closeout acceptance note (2026-08-17):** Phase 11 completed the
> strangler-fig migration, but it did not close every requirement in the
> original CUDA overhaul blueprint. The remaining numerical-safety, I/O,
> precision/runtime, and release-gate work is ordered with explicit exit
> criteria in [`overhaul-completion-plan.md`](overhaul-completion-plan.md).
> Do not interpret "Phase 11 is complete" below as final design acceptance.

## 1. What changed

The non-templated `dynSharedBase()` accessor (an `extern __shared__ unsigned
char[]` returned as `void*` and `static_cast` to `T*` at each use) is deleted
from `src/fourier_impl.cuh`, `src/geometry_impl.cuh`, and `src/precon_impl.cuh`.
Each consuming kernel now declares its dynamic shared memory directly —
legal per TU because the explicit instantiation split (`src/*_double.cu` /
`src/*_float.cu`) puts exactly one scalar type in each TU. Six kernel sites:

| File | Kernel | Declaration |
| ---- | ------ | ----------- |
| `fourier_impl.cuh` | `inverseAccumulateKernel<T, FuseRzCon>` | `extern __shared__ T sh[];` (4·mpol·kTile) |
| `fourier_impl.cuh` | `deAliasSynthesizeKernel<T>` | `extern __shared__ T sh[];` (2·(mpol-2)·kTile) |
| `geometry_impl.cuh` | `ncurr1FinalizeKernel<T>` | `extern __shared__ T s_buf[];` (2·blockDim.x) |
| `geometry_impl.cuh` | `computeNormPartialsKernel<T>` | `extern __shared__ T s_buf[];` (4·blockDim.x) |
| `precon_impl.cuh` | `pcrSolveKernel<T, R>` | `extern __shared__ T s_tri[];` ((6+2R)·ns) |
| `precon_impl.cuh` | `thomasSolveKernel<T>` | `extern __shared__ T s[];` ((1+rhs_count)·n) |

`src/constraint_impl.cuh` needed no change: its dyn-smem users moved into the
transform module with the de-alias bandpass during step 13.3, so it already
had no `dynSharedBase()`.

## 2. Verification — Class A bit-identical (not the predicted Class B)

The docs warned that the direct form changes `-use_fast_math` FMA fusion
(opaque function return vs. known shared-array aliasing) and perturbs the
trajectory at ~1e-10. It did not reproduce under the current layout: with the
instantiation split in place the compiler inlines the two-line accessor and
emits the same SASS, so the direct form is a pure source-level cleanup.

`scripts/compare_runs.py` old-vs-new (fresh baselines captured from the
pre-change binary immediately before the edit; `--max-iter-delta 0`):

| Config | Iterations | Final FSQR | Restart sequence | Full-precision state |
| ------ | ---------- | ---------- | ---------------- | -------------------- |
| Solovev (axisym default) | 251→199→456 | 9.583e-17 | identical (0 events) | **0.000e+00** max rel diff, all six families |
| W7-X (generic cuFFT) | 1877→1617→2011 | 9.778e-13 | identical (15 events, same iters) | **0.000e+00** max rel diff, all six families |
| Solovev generic (`CUMES_FORCE_GENERIC=1`) | 251→199→456 | 9.583e-17 | — | matches the frozen record |

Per-iteration printed residual rows match at 0.000e+00 everywhere (70/70 W7-X
rows), converged-iteration delta 0. The converged states are byte-identical
at full double precision after 5505 W7-X iterations through every changed
kernel (`inverseAccumulateKernel`, `deAliasSynthesizeKernel`,
`pcrSolveKernel`, `computeNormPartialsKernel`, `thomasSolveKernel`), so the
change is **Class A** — no re-freeze required, the frozen
`251→199→456` / `1877→1617→2011` baseline remains the regression oracle
unchanged.

Test matrix: **35/35 CTest** (double build, incl. the 12 compute-sanitizer
memcheck entries), **23/23 CTest** (float build), both builds clean.

## 3. Docs hygiene in the close-out

- `docs/architecture.md` §1 header + §2: the retention rationale is replaced
  with the measured result (direct `extern __shared__ T[]`, bit-identical).
- The three `*_impl.cuh` files carry a short comment stating the same
  measured result next to the kernels.
- `CLAUDE.md` (project instructions) was still describing the pre-overhaul
  world (`GridParams`, `SpectralState`, `fourier.cuh`, `MetricWorkspace`, the
  dynSharedBase caveat, "hot restart not yet", …). It is rewritten to the
  post-Phase-11 architecture: the operator library + `include/cumes` layout,
  the explicit-instantiation split, `ValidatedProblem`/`DeviceParams<T>`,
  the CLI (`--output-schema`, `--restart`, `--checkpoint`), and the updated
  implemented/omitted feature table. The stale untracked duplicate `AGENTS.md`
  (byte-identical to the old CLAUDE.md) is deleted.
- `docs/cuda-overhaul-blueprint.md` was never `git add`ed; it is committed
  now as the phase record it always was (the two downloaded CUDA programming
  guide dumps, `docs/cuda-asynchronous-execution.md` / `docs/cuda-graphs.md`
  + their `_files/` asset dirs, are reference material and remain untracked).

## 4. Definition of done (strangler plan §1, blueprint §11 Phase 10 exit gate)

- `solverRun` drives only `cumes` operator classes — **done** (step 12).
- The generic transform is a `SpectralOperator<T>` backend on equal footing
  with `AxisymmetricOperator`; no `axisym_active` branch — **done**.
- Transforms own only transform tables/plans/scratch — **done** (step 1).
- `include/*.cuh` legacy structs and `InputParams` deleted; kernel bodies
  survive only as operator implementations — **done** (step 13.1–13.3);
  the surviving `solver.cuh`/`output.cuh`/`vmec_types.h` are thin app shims
  over `cumes` types only.
- Full precise trajectories bit-identical to the frozen baseline — **done**,
  re-verified in §2 including the deferred `dynSharedBase()` item.
- The `dynSharedBase()` decision deferred in tail-3 §2 — **resolved**: removed,
  measured Class A.

## 5. Commits

```
383fdbc Phase 11 close-out: remove dynSharedBase() — direct extern __shared__ T[] (Class A)
```

Phase 11 is complete.
