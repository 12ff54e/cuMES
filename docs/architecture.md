# cuMES architecture

Status date: 2026-08-16 (Phase 11 complete — step 13 + the deferred
`dynSharedBase()` removal landed, both measured bit-identical). This document
describes the architecture as it exists after Phase 10 — the tested operator
boundaries, the build/library split, and the state of the strangler-fig
migration. The normative numerical contracts live in
`docs/cuda-overhaul-blueprint.md` §4; the layout contracts in
`docs/data-layout.md`; the measured performance in `docs/performance.md`.

## 1. One layer: the `cumes` operator library

The strangler-fig migration is complete (migration step 13): the legacy
`include/*.cuh` kernel structs and their free-function entry points are
deleted, and the production kernels live directly in the `cumes` operator
classes, which own their device buffers and reproduce the frozen VMEC
trajectories bit-for-bit.

- **Device operator modules** — `src/*_impl.cuh` + the explicit
  `double`/`float` instantiation TUs (`src/*_double.cu` / `src/*_float.cu`),
  one per operator: `fourier` (`ToroidalFftOperator` — the cuFFT plans,
  transform scratch and poloidal tables), `geometry`
  (`GeometryOperator`/`MagneticFieldOperator` — the half-grid metric/field
  buffers), `forces` (`ForceOperator`), `solver` (`EquilibriumOperator` +
  `solverRun`), `profiles` (`Profiles` — the 11 radial arrays), `precon`
  (`Preconditioner`), `constraint` (`ConstraintOperator`), `prolongation`
  (`Prolongation` — the grid-sequencing state interpolation), and
  `axisymmetric` (`AxisymmetricOperator`). Each operator owns its raw device
  buffers directly (arena-carved per stage) and exposes typed view bundles
  (`RadialProfileViews`, `BaseGeometryHalfViews`, `GeometryParityViews`, …).
- **Host-side `cumes` namespace** — `include/cumes/*` + `src/cumes/*`. Validated
  config model (`ProblemSpec` → `ValidatedProblem`), RAII CUDA runtime
  (`DeviceBuffer`, `DeviceArena`, `Stream`, `Event`, `DeviceContext`), typed
  non-owning views (`SpectralView`, `RealFieldView`, the parity bundles), the
  output/IO stack, and the host orchestration (`MultigridSolver`,
  `StageSolver`, `IterationController`, …).

The per-iteration DAG (`src/solver_impl.cuh::solverRun` via `EquilibriumOperator`)
runs on the typed-view plumbing and one explicit compute stream, driving every
transform through the unified `SpectralOperator` interface (the generic cuFFT
backend or the axisymmetric direct-poloidal backend).

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

The CUDA operator libraries are the explicit-instantiation split that made the
old non-templated `dynSharedBase()` shared-memory indirection removable: since
each TU instantiates exactly one scalar type, the kernels now declare their
dynamic shared memory directly as `extern __shared__ T[]`. The switch was
expected to be a Class B re-freeze but measured **bit-identical** on both
configs (2026-08-16), so the frozen trajectory baseline stands unchanged.

## 3. Production per-iteration pipeline

The regular iteration (blueprint §7) remains mathematically sequential and is
enqueued on one compute stream until a single deliberate control fence:

1. `extrapolateAxisKernel` — copy the six `m=1` families + the `m=0` `Lcs`
   axis row from surface 1 to the axis;
2. `ToroidalFftOperator::inverse_fused` — parity-split R/Z/λ + derivatives,
   *and* the fused `xmpq = m(m-1)`-weighted `rCon`/`zCon` (blueprint §8.4; the
   separate rzCon transform was retired in Phase 10);
3. `GeometryOperator::enqueue`/`MagneticFieldOperator::enqueue` — half-grid
   `√g`, covariant metric, `B`, current closure (`ncurr=0` first-pass only,
   `ncurr=1` every pass);
4. `GeometryOperator::jacobian_stats` — device-side oriented-Jacobian reduction;
5. `ConstraintOperator::reset_reference` — LCFS-extrapolated reference, on
   `iter2 == iter1`;
6. on the `(iter2-iter1) % 25 == 0` cadence, `Preconditioner::enqueue_compute` +
   `enqueueForceNorms`;
7. `ForceOperator::enqueue` — the monolithic 16-family MHD force kernel;
8. `ConstraintOperator::enqueue` — bandpass + add constraint force to
   `brmn`/`bzmn`;
9. `ToroidalFftOperator::forward` — six spectral-force families;
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

## 5. What Phase 10 and migration step 13 retired

Retired in Phase 10 (dead code only — both configs re-verified bit-identical):

- the §8.10 force-split prototype (`computeForcesSplit`, `rzForcesKernel`,
  `lambdaForcesKernel`, `test_force_split.cu`); decision recorded in ADR-0002;
- the Phase-7 rzCon reference path (`constraintRzConCompute`,
  `rzConPackKernel`, `rzConAccumulateKernel`, the `plan_z2d_rz` compact
  round-trip, `test_rzcon_fusion.cu`); the fused `inverseDFTFused` is now the
  sole rCon/zCon producer and `test_axisym_backend`/`test_constraint_tcon`
  compare against it directly.

Migration step 13 closed out the legacy-struct endgame (each part Class A
bit-identical, both configs re-verified):

- 13.1/13.2: `GridParams` is now `DeviceParams<T>`
  (`include/cumes/config/device_params.hpp`); `InputParams` and the legacy JSON
  parser are deleted — the solver consumes `ValidatedProblem` directly.
- 13.3 parts 1–7: `RadialProfiles`, `MetricWorkspace`, `PreconWorkspace`,
  `ConstraintWorkspace`, `FourierPlan`, and `SpectralState` are deleted, and
  the `interpolateState` free function folded into `Prolongation::enqueue` —
  each operator now owns its buffers directly and exposes typed views. The
  legacy free-function headers (`fourier.cuh`, `geometry.cuh`, `forces.cuh`,
  `precon.cuh`, `constraint.cuh`, `refine.cuh`) are all gone; no legacy structs
  remain.

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
