# cuMES architecture

Status date: 2026-08-16 (Phase 11 step 13 in progress). This document describes
the architecture as it exists after Phase 10 — the tested operator boundaries,
the build/library split, and the state of the strangler-fig migration. The
normative numerical contracts live in `docs/cuda-overhaul-blueprint.md` §4; the
layout contracts in `docs/data-layout.md`; the measured performance in
`docs/performance.md`.

## 1. Two layers: legacy kernels + the `cumes` scaffold

cuMES is mid-migration (blueprint §1 "strangle behind tested operator
interfaces"). The code is split into two layers that share one build:

- **Legacy implementation layer** — `include/*.cuh` + `src/*_impl.cuh`, split
  into explicit `double`/`float` instantiation TUs (`src/*_double.cu` /
  `src/*_float.cu`). These are the *actual* production kernels: `fourier`,
  `geometry`, `forces`, `solver`, `profiles`, `precon`, `constraint`, `refine`,
  and `axisymmetric`. They own the raw device buffers (`FourierPlan`,
  `MetricWorkspace`, `ConstraintWorkspace`, `PreconWorkspace`, `RadialProfiles`)
  and reproduce the frozen VMEC trajectories bit-for-bit.
- **New `cumes` namespace** — `include/cumes/*` + `src/cumes/*`. Host-only
  validated config model (`ProblemSpec` → `ValidatedProblem`), RAII CUDA runtime
  (`DeviceBuffer`, `DeviceArena`, `Stream`, `Event`, `DeviceContext`), typed
  non-owning views (`SpectralView`, `RealFieldView`, the parity bundles), and
  the operator boundary headers (`SpectralOperator`, `GeometryOperator`,
  `ForceOperator`, `ConstraintOperator`, `TridiagonalBackend`,
  `IterationController`, …).

The production hot loop (`src/solver_impl.cuh::solverRun`) already runs on the
new typed-view plumbing and one explicit compute stream, but the *kernels* are
the legacy ones wrapped behind views. Full replacement of the legacy kernels by
the `cumes` operator classes is the long tail of the migration; it is **not**
required for normal execution and remains future work (see §5).

## 2. Build/library split (the CMake targets)

The single `cuMES` executable is composed from scoped libraries rather than one
monolithic compile (blueprint §9):

| Target | Kind | Contents |
| ------ | ---- | -------- |
| `cumes_core` | host C++ | `result.hpp`, `checked_size.hpp`, `grid_shape`, `mode_table` |
| `cumes_config_json` | host C++ | `validation_report`, `validated_problem`, `json_reader` |
| `cumes_io_host` | host C++ | `output_spec`, `run_report`, `equilibrium_snapshot`, binary v0/v1, checkpoint |
| `cumes_io` | host C++ (links cudart/cufft for D2H only) | output dispatcher + NetCDF/HDF5 adapters |
| `cumes_cuda_runtime` | host CUDA-runtime | `device_context`, centralized `check_cuda`/`check_cufft` |
| `cumes_cuda_double` / `cumes_cuda_float` | device | the nine `*_double.cu` / `*_float.cu` operator TUs |
| `cuMES` | executable | `main.cu`, links only the TU matching `Real` |
| `cumes_benchmark_fixed_iteration`, `cumes_benchmark_graph_overhead` | bench | §8.1 harness, graph microbenchmark |

The CUDA operator libraries are the explicit-instantiation split that makes the
non-templated `dynSharedBase()` shared-memory indirection unnecessary in new
code; it is retained only because switching to `extern __shared__ T[]` changes
`--use_fast_math` FMA fusion and perturbs the trajectory (~1e-10) — a Class B
change left for a future re-freeze.

## 3. Production per-iteration pipeline

The regular iteration (blueprint §7) remains mathematically sequential and is
enqueued on one compute stream until a single deliberate control fence:

1. `extrapolateAxisKernel` — copy the six `m=1` families + the `m=0` `Lcs`
   axis row from surface 1 to the axis;
2. `inverseDFTFused` — parity-split R/Z/λ + derivatives, *and* the fused
   `xmpq = m(m-1)`-weighted `rCon`/`zCon` (blueprint §8.4; the separate rzCon
   transform was retired in Phase 10);
3. `computeGeometry` — half-grid `√g`, covariant metric, `B`, current closure
   (`ncurr=0` first-pass only, `ncurr=1` every pass);
4. `computeJacobianStats` — device-side oriented-Jacobian reduction;
5. `constraintResetRzCon0` — LCFS-extrapolated reference, on `iter2 == iter1`;
6. on the `(iter2-iter1) % 25 == 0` cadence, `preconCompute` +
   `enqueueForceNorms`;
7. `computeForces` — the monolithic 16-family MHD force kernel;
8. `constraintCompute` — bandpass + add constraint force to `brmn`/`bzmn`;
9. `forwardDFT` — six spectral-force families;
10. odd-m decomposition + the `m=1` mixed gauge;
11. invariant + preconditioned residual reductions into one `ControlRecord`
    D2H copy, then the single host fence;
12. the pure `IterationController::advance` decision, then ordered apply
    (descent → post-descent checkpoint capture → post-descent restore+zero).

The `ijacob == 25/50` maintenance reset is handled by `next_schedule()` before
the DAG, without launching geometry or descent.

## 4. Dependency rule (blueprint §5.1)

The migration target is an acyclic graph. The rules already enforced in the
build and the boundary headers:

- transforms own only transform tables/plans/scratch — never force fields;
- constraint code reaches the transform service through the public
  `SpectralOperator` interface, never through a raw hidden Fourier pointer;
- output (`cumes_io_host`) consumes a host `EquilibriumSnapshot`/`RunReport`
  and never sees a device pointer;
- the controller (`IterationController`) is a pure host state machine with no
  CUDA calls;
- no library calls `exit()`; only `main.cu` maps `RunStatus` to an exit code.

## 5. What Phase 10 retired, and what remains

Retired in Phase 10 (dead code only — both configs re-verified bit-identical):

- the §8.10 force-split prototype (`computeForcesSplit`, `rzForcesKernel`,
  `lambdaForcesKernel`, `test_force_split.cu`); decision recorded in ADR-0002;
- the Phase-7 rzCon reference path (`constraintRzConCompute`,
  `rzConPackKernel`, `rzConAccumulateKernel`, the `plan_z2d_rz` compact
  round-trip, `test_rzcon_fusion.cu`); the fused `inverseDFTFused` is now the
  sole rCon/zCon producer and `test_axisym_backend`/`test_constraint_tcon`
  compare against it directly.

Still pending (blocked on consumers, blueprint §11 Phase 10 exit gate):

- the legacy `include/*.cuh` kernel structs `FourierPlan` and `SpectralState`
  are still the production implementation and cannot be deleted until the
  `cumes` operator classes fully replace them. Migration step 13.1/13.2 have
  landed (`GridParams` is now `DeviceParams<T>` in
  `include/cumes/config/device_params.hpp`; `InputParams` is deleted — the
  solver consumes `ValidatedProblem` directly), and step 13.3 parts 1–4 have
  deleted `RadialProfiles`, `MetricWorkspace`, `PreconWorkspace`, and
  `ConstraintWorkspace` (each operator now owns its buffers directly).

Emitted after Phase 10: `configs/schema-v1.json` freezes the `cumes-config-v1`
normalized-config schema (blueprint §6.1, `ValidatedProblem::normalize_to_json`,
pinned by the `tests/fixtures/*.normalized.json` goldens) and, under
`x-cumes-on-disk-contracts`, the legacy-v0 / versioned-v1 / checkpoint-v1
binary container layouts (blueprint §6.13).

Wired in after Phase 10 (see `docs/adr/0004`): `AxisymmetricOperator` (Phase
7.1) now runs the Solovev production path — `StageSolver::run` builds it when
`ntor=0/nzeta=1` and `solverRun` selects `enqueue_inverse`/`enqueue_rzcon`/
`enqueue_forward`/`constraintComputeAxisym` instead of the generic cuFFT calls
(`CUMES_FORCE_GENERIC=1` restores the generic backend). It is a Class B
trajectory member (ULP-equivalent, identical iteration counts), and a ~29%
median wall-time win on the submission-bound Solovev shape.
