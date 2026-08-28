# cuMES architecture

This document describes the current post-overhaul architecture: the tested
operator boundaries, the build/library split,
the device safety predicates, the single-snapshot I/O path, and the precision
policies. The normative numerical contracts live in `docs/mathematics.md`;
the layout contracts in `docs/data-layout.md`; the verification tiers/gates
in `docs/verification.md`; the measured performance in
`docs/performance.md`; and the chronological execution and closeout record in
`docs/overhaul-history.md`.

## 1. One layer: the `cumes` operator library

The strangler-fig migration is complete (migration step 13): the legacy
`include/*.cuh` kernel structs and their free-function entry points are
deleted, and the production kernels live directly in the `cumes` operator
classes, which own their device buffers and reproduce the frozen
trajectories bit-for-bit.

- **Device operator modules** — `src/kernels/*_impl.cuh` + the explicit
  `double`/`float` instantiation TUs (`src/*_double.cu` / `src/*_float.cu`),
  one per operator: `fourier` (`ToroidalFftOperator` — the cuFFT plans,
  transform scratch and poloidal tables), `geometry`
  (`GeometryOperator`/`MagneticFieldOperator` — the half-grid metric/field
  buffers), `forces` (`ForceOperator`), `solver` (`EquilibriumOperator` +
  `solver_run`), `profiles` (`Profiles` — the 11 radial arrays), `precon`
  (`Preconditioner`), `constraint` (`ConstraintOperator`), `prolongation`
  (`Prolongation` — the grid-sequencing state interpolation), and
  `axisymmetric` (`AxisymmetricOperator`). Each operator owns its raw device
  buffers directly (arena-carved per stage) and exposes typed view bundles
  (`RadialProfileViews`, `BaseGeometryHalfViews`, `GeometryParityViews`, …).
- **Host-side `cumes` namespace** — `include/cumes/*` + `src/cumes/*`. Validated
  config model (`ProblemSpec` → `ValidatedProblem`), RAII CUDA runtime
  (`DeviceBuffer`, `PinnedBuffer`, `DeviceArena`, `Stream`, `Event`), typed
  non-owning views (`SpectralView`, `RealFieldView`, the parity bundles), the
  output/IO stack, and the host orchestration (`MultigridSolver`,
  `StageSolver`, `IterationController`, …).

The per-iteration DAG (`src/kernels/solver_impl.cuh::solver_run` via
`EquilibriumOperator`)
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
| `cumes_io_host` | host C++ | `output_spec`, `run_report`, `equilibrium_snapshot`, binary v1, checkpoint |
| `cumes_io` | host C++ (`output_print` links cudart for D2H) | the full `make_writer` dispatch + host-only NetCDF/HDF5 adapters (the ONLY target with the backend headers and defines) |
| `cumes_cuda_runtime` | header-only CUDA-runtime interface | centralized `check_cuda`/`check_cufft` and buffer/stream/event RAII; propagates only the CUDA runtime/cuFFT links |
| `cumes_cuda_double` / `cumes_cuda_float` | device | the nine `*_double.cu` / `*_float.cu` operator TUs |
| `cuMES` | executable | `main.cu`, links only the TU matching `Real` |
| `magnetic_coordinate` | standalone CUDA/C++ library | consumes schema-v8 equilibrium output and constructs PEST/Boozer coordinates |
| `cumes-boozer` | postprocessor executable | writes the mixed `(s, theta_b, zeta)` Boozer representation |
| `cumes_benchmark_fixed_iteration`, `cumes_benchmark_graph_overhead`, `cumes_benchmark_graph_realpass` | bench | §8.1 harness, graph microbenchmark, real-pass graph measurement |

The CUDA operator libraries are the explicit-instantiation split that made the
old non-templated `dynSharedBase()` shared-memory indirection removable: since
each TU instantiates exactly one scalar type, the kernels now declare their
dynamic shared memory directly as `extern __shared__ T[]`. The switch was
expected to be a Class B re-freeze but measured **bit-identical** on both
reference configurations, so the frozen trajectory baseline stands unchanged.

## 3. Production per-iteration pipeline

The regular iteration (blueprint §7) remains mathematically sequential and is
enqueued on one compute stream until a single deliberate control fence:

1. `extrapolate_axis_kernel` — copy the six `m=1` families + the `m=0` `Lcs`
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
   `enqueue_force_norms`;
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

**Transform implementation.** The cuFFT backend packs 12 slots per poloidal
mode (vmecpp `kBatch`: rmkcc/rmkss + ζ-derivative slots for R, Z, λ) into
batched 1D D2Z/Z2D transforms of length `nzeta` (batch `12·mpol·ns`) with
direct poloidal accumulation/reduction over θ from small per-mode tables —
structurally identical to vmecpp's `fft_toroidal.cc`. The same plans and
scratch serve the constraint module: rCon/zCon (xmpq-weighted reconstruction)
and the de-aliasing bandpass (full-grid sc/cs analysis → D2Z → normalized
coefficients → Z2D synthesis), with compact sub-batch plans (2·(mpol−2)·(ns−1)
elements for the bandpass, 4·mpol·ns for rCon/zCon). The original direct-sum
kernels were removed after A/B validation.

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

## 5. Free-boundary vacuum library (`deps/vacuum-field`)

`deps/vacuum-field` is a standalone CUDA C++ library — the port of vmecpp's
NESTOR vacuum-field algorithm — embedded as a git submodule (see
`adr/0005-vacuum-field-submodule.md`). It computes the vacuum magnetic field
on the LCFS from the boundary Fourier coefficients and an external-coil field:
the magnetic scalar potential via the boundary-element Laplace solve, and the
outputs `b_sq_vac` (|B|²/2, no mu0), the covariant components
`b_sub_u/b_sub_v`, the cylindrical `B_R/B_phi/B_Z`, and the surface-integral
scalars `b_sub_u_vac/b_sub_v_vac`.

The external-coil field can come from a MAKEGRID NetCDF file or from an
in-memory `MgridProvider::ResponseTable`. The optional host-only
`vfield::makegrid` component parses coils-dot geometry and grid-parameter JSON
and builds that table directly. cuMES constructs the table once when a
free-boundary run starts, then the existing CUDA interpolation and NESTOR
update path consumes it without a temporary file (see
`adr/0006-inline-makegrid.md`).

Dependency edge (the acyclic-graph rule): **cuMES → vfield only** — the
library never includes cuMES headers, owns its own runtime (a trimmed
`DeviceBuffer`/`check_cuda`), its optional NetCDF reader/writer
(`VFIELD_HAVE_NETCDF`), and its standalone tests. cuMES sets
`VFIELD_USE_FLOAT` from `CUMES_USE_FLOAT` and links the selected
`vfield::vfield` CUDA target plus `vfield::makegrid`
(`CUMES_USE_VACUUM_FIELD`, default ON, auto-off when the submodule is absent).
`FreeBoundaryOperator` owns the integration seam and applies the resulting
vacuum pressure and covariant field terms according to `nvacskip`.

## 6. Retired compatibility internals

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

`configs/schema-v1.json` freezes the `cumes-config-v1`
normalized-config schema (blueprint §6.1, `ValidatedProblem::normalize_to_json`,
pinned by the `tests/fixtures/*.normalized.json` goldens) and, under
`x-cumes-on-disk-contracts`, the versioned-v1 / checkpoint-v1
binary container layouts (blueprint §6.13).

`AxisymmetricOperator` now runs the Solovev production path (see
`docs/adr/0004`): `StageSolver::run` builds it when
`ntor=0/nzeta=1` and `solver_run` selects `enqueue_inverse`/`enqueue_rzcon`/
`enqueue_forward`/`constraintComputeAxisym` instead of the generic cuFFT calls
(`CUMES_FORCE_GENERIC=1` restores the generic backend). It is a Class B
trajectory member (ULP-equivalent, identical iteration counts), and a ~29%
median wall-time win on the submission-bound Solovev shape.
