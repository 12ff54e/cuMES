# Changelog

All notable changes to cuMES are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/).

## [1.8.0] - 2026-09-18

### Added

- Editable radial grid stages for every browser preset: stage count, radial
  grid points, step caps, and tolerances. Step caps and tolerances are folded
  by default, with an optional checkbox to synchronize them across stages.
  Custom settings persist across reloads and precision switches.
- Input JSON uploads for fixed and free boundaries, retaining boundary
  coefficients, profiles, resolution, and stage settings. Fixed uploads have
  a separate saved setup; Reset restores the original uploaded input.
- A complete, read-only input JSON preview with a Copy JSON button, available
  from either setup tab and during solves. Free-boundary previews include
  coil currents, the coil file, and field-grid parameters. Clipboard failures
  select the text for manual copying.

### Changed

- Restyled the browser interface with Biolinum/Libertine typography,
  teal/terracotta accents, warm light and grey dark themes, and matching
  boundary, residual, coil, and flux-surface plot colors. The theme toggle is
  icon-only, and the header links to the cuMES repository and WebGPU website.
- Split boundary setup and radial grid stages into keyboard-accessible tabs
  to reduce panel height while retaining edits and the selected tab.
- Replaced inline JSON editing with the shared preview. Profiles and angular
  resolution can be changed by uploading a revised input JSON file.

### Fixed

- Dashed target contours remain visible above the filled plasma boundary.
- Solovev result views retain a consistent height before and after solving.
- Loading a preset preserves the selected setup tab when saved stage
  settings need correction; Run still reveals invalid stage settings.
- Browser solves honor configured stage tolerances instead of replacing
  them with the precision defaults.

## [1.7.0] - 2026-09-17

### Added

- Non-stellarator-symmetric fixed- and free-boundary equilibria on CUDA and
  WebGPU through `lasym=true`, signed-n `rbs`/`zbc` boundary harmonics, and
  optional `raxis_s`/`zaxis_c` axis coefficients. All twelve spectral families
  participate in transforms, residuals, preconditioning, descent, and
  multigrid; vacuum coupling receives all eight R/Z edge families.
  The [qualification](docs/adr/0020-non-stellarator-symmetry.md)
  records native float/double and browser scalar/paired coverage; Newton
  corrections, tangents, and Boozer export still require stellarator symmetry.
- Twelve-family scientific output and restarts, with typed complementary
  input provenance in binary, NetCDF, and HDF5 containers. Asymmetric writes
  use binary v10 and checkpoint v8; symmetric writes retain binary v8 and
  checkpoint v6. Plotting and comparison readers handle all active families.
- Asymmetric tokamak inputs and a browser preset with Fourier and draggable
  contour editing. Both tokamak editors offer persistent contour point counts
  from 12 through 36 while preserving the selected poloidal resolution,
  profiles, and axis seeds.
- Reproducible original Fortran VMEC comparisons for
  [fixed boundaries](benchmarks/asymmetric_vmec/README.md) and
  [free boundaries](benchmarks/free_boundary_vmec/README.md), with pinned
  inputs, serial runners, physical comparison reports, and
  [QH gauge studies](docs/qh-gauge-convergence.md). Reports retain failed
  references and distinguish residual convergence from physical agreement.

### Changed

- Retained boundary tangent solves default to the device GMRES backend,
  keeping primal state, active maps, Krylov vectors, and JVP/preconditioner
  intermediates on the GPU across columns. HOST remains available as the
  reference backend; [measured benefits](docs/performance.md) depend on the
  workload and GPU.
- WebGPU requests supported adapter buffer limits and scans finite values
  with two-dimensional dispatches to support larger asymmetric grids.
  Browser validation captures large scientific outputs in bounded chunks.
- GitHub Pages builds and tests every `main` push, then selects deployments
  using a fingerprint derived from the WebGPU build and packaging inputs.

### Fixed

- Recovery checkpoints replace the preceding valid state only after geometry
  and finite-residual validation, preventing repeated restores of invalid
  post-descent states. CUDA and WebGPU diagnose invalid initial geometry
  immediately instead of retrying an unchanged initial checkpoint.
- Native iteration counts and final time steps are reported correctly when
  the iteration budget is exhausted by rejected passes. The persistent
  preconditioner pivot-scale cache is initialized before use.
- CUDA B-spline transfer matrices upload on their consuming solve stream,
  preventing corrupted geometry during multigrid refinement.
- WebGPU radial transfer copies endpoints exactly, preserving the prescribed
  fixed boundary across multigrid stages.
- Asymmetric plots include complementary boundary harmonics, average
  prescribed current over the full theta grid, and use consistent inner-face
  tangents in mixed radial metric terms.
- Vacuum sign and current-consistency guards apply once pressure coupling
  activates, allowing cold-axis relaxation during preliminary diagnostics.
  The vacuum dependency corrects singular Fourier branches and finite float
  tangent-pole limits.
- Browser symmetry controls lock during solves, and the shared vacuum test
  fixture is embedded at the path used by the Wasm bridge.
- CPU-only CI selects host tests by CMake label, excludes GPU recovery tests,
  and rejects an empty test selection. Fixed-only radius tests skip the
  optional vacuum fixture.

## [1.6.1] - 2026-09-09

### Changed

- Reuse host radial-profile evaluation, scalar interpolation references,
  constraint-filter coefficients, and Jacobian/finite-value checks across
  native and browser paths.
- Centralize WebGPU operator readback decoding for standalone and batched
  execution, retaining compact transfers and existing completion points.
- Share WebGPU operator setup, force/constraint residual phases, validation,
  and state updates. Completed batches are consumed directly without serial
  replay, preserving vacuum continuation, preconditioner caches, and Newton
  acceptance/rollback sequencing.
- Consolidate the independent CPU force reference used by native tests and
  share Chrome connection handling across browser harnesses.
- Share timestamp resources and readback capture between live iteration
  timing and diagnostic profiling while retaining their reporting policies.
  Expand regression coverage for timestamp ownership, capacity, mapping
  ranges, payload preservation, worker execution, and hook restoration;
  include the diagnostic profiler test in the WebGPU CTest suite.

## [1.6.0] - 2026-09-09

### Added

- A CUDA-free WebGPU/WebAssembly backend for axisymmetric and three-dimensional
  fixed- and free-boundary equilibria, with shared input validation, residual
  and geometry gates, controller recovery, multigrid, and schema-v8 scientific
  result downloads.
- A [hosted web solver](https://12ff54e.github.io/cuMES/) and reproducible
  GitHub Pages build/deployment workflow. A same-origin service worker enables
  the isolation required by threaded Wasm, with one reload on the first visit.
- A shared fixed/free-boundary editor with Solovev and W7-X fixed-boundary
  presets; Solovev, W7-X vacuum, and cth_like coil presets; coil uploads; and
  in-memory MAKEGRID generation. Fourier and axisymmetric contour controls,
  live residual/restart plots, iteration throughput, orbitable surfaces, coil
  overlays, and selectable toroidal cuts support interactive solves.
- Browser scalar-f32 and paired-f32 precision selection. The browser's Double
  option uses paired words rather than native WGSL f64; free-boundary and W7-X
  examples default to paired precision. Qualification remains specific to the
  case, precision, adapter, and browser; scalar free-boundary W7-X can stall.
- Opt-in paired-f32 vacuum kernels through `vacuum=webgpu`, with Wasm-double
  LU by default and resident pivoted LU through `vacuum_lu=webgpu` for systems
  of at most 256 unknowns. HOST/Wasm vacuum remains the default/reference;
  see the [vacuum backend qualification](docs/adr/0019-webgpu-vacuum-backend.md).
- Experimental `newton=1` corrections for paired, fixed-boundary axisymmetric
  browser solves, and `geometry=compensated-m1` for scalar three-dimensional
  geometry. Both are opt-in and have separate
  [Newton](docs/adr/0018-webgpu-newton-experiment.md) and
  [geometry](docs/adr/0017-webgpu-m1-geometry-compensation.md) qualification.
- Worker-based numerical conformance, Chrome and headless Firefox harnesses,
  controller-trace comparisons, and GPU/host profiling tools.

### Changed

- Browser iteration paths retain spectral state, velocity, reusable fields,
  constraints, and preconditioners on the GPU, with batched and compact
  readbacks. Free-boundary coupling batches work around vacuum updates,
  parallelizes MAKEGRID, and retains vacuum boundary-force corrections on GPU.
- Shared WGSL templates generate scalar and paired operators. Optional FFT
  transforms, GPU residual norms, and GPU Jacobian control provide additional
  execution paths with numerical checks and host fallbacks; direct Fourier
  projection remains the default.

### Fixed

- Zero prescribed-current inputs initialize without requiring a nonzero
  current-profile edge integral in either browser precision.
- Host and GPU Jacobian gates exempt the complete first half-grid angular
  surface (`ntheta * nzeta`) from the relative threshold in three dimensions,
  avoiding unnecessary checkpoint restores and step reductions.
- Paired shader rounding and compiler-sensitive array/projection operations
  preserve the qualified Chrome/D3D12 and Firefox numerical behavior.
- Versioned browser assets keep HTML, JavaScript, Wasm, and editor resources
  coherent across rebuilds and deployments.

## [1.5.0] - 2026-09-09

### Added

- `--newton` and `SolveRequest::enable_newton` for guarded Newton–Krylov
  corrections in fixed-boundary axisymmetric double solves (`ntor=0`,
  `nzeta=1`). The option is off by default, retains all configured multigrid
  stages, tolerances and iteration caps, and rejects unsupported requests
  before GPU setup. Benefits depend on the input; the
  [19-case qualification](docs/axisymmetric-newton-qualification.md) records
  both speedups and regressions on TITAN Xp and RTX 4090.
- Reusable CUDA GMRES, correction-coordinate maps and frozen-state Newton
  operators, with regression tests and optional block, Newton and coarse
  correction diagnostics.
- Pinned axisymmetric Newton and free-boundary benchmark matrices, paired
  runners, native-output and checkpoint validation, and reproducible
  performance reports. Resume checks preserve saved protocols and reject
  incompatible output directories before writing results.

### Changed

- Fourier transforms cache weighted forward bases once per stage and skip
  unused inverse constraint sums. Qualified W7-X steady-iteration latency
  falls by 5.19% on TITAN Xp and 6.50% on RTX 4090, with bit-identical
  trajectories within each architecture.
- Free-boundary solves parallelize axisymmetric vacuum source evaluation and
  small singular right-hand-side systems, and combine pinned host transfers
  with existing stream fences. Qualified solver-interval reductions are
  37.01% / 29.69% for precomputed-grid Solovev and 6.77% / 23.70% for the
  positive-flux W7-X case on TITAN Xp / RTX 4090. State, fields and numerical
  reports remain bit-identical within each architecture; the
  [qualification report](docs/free-boundary-performance.md) separates solver
  timing from process wall time and retains all failures.

### Fixed

- The free-boundary pressure-mismatch diagnostic starts at zero and reads
  device memory only after an edge-force evaluation, eliminating an
  uninitialized read before vacuum activation.

## [1.4.1] - 2026-09-07

### Changed

- Float `compensated` geometry now includes four m=1 toroidal R/Z sums and
  split odd scaling. Higher toroidal modes retain the native FFT path;
  double compensated reconstruction keeps its existing arithmetic.

### Fixed

- W7-X float ns=99 single-grid cold starts converge at `1e-5` with compensated
  geometry in 1,354 effective iterations (FSQR `9.785e-6`). The multigrid case
  converges in 149 → 277 → 311 iterations; both checkpoints converge again on
  their first replay iteration. A cold-start/replay regression test covers
  the single-grid case.

## [1.4.0] - 2026-09-07

### Added

- `CUMES_GEOMETRY_PRECISION=native|compensated` and library controls for
  selective odd R/Z reconstruction using float-float or double-double
  arithmetic in fixed-boundary three-dimensional solves. Native reconstruction
  remains the default.
- Precision and timing diagnostics for radius storage and odd R/Z
  reconstruction, with independent long-double references.
- A CTest audit that rejects FP64 instructions and PTX types in the cuMES and
  vacuum float CUDA libraries across all compiled GPU architectures.

### Changed

- Fixed-boundary three-dimensional float solves now use reference-plus-
  displacement radius storage by default, preserving small radial variations.
  The immutable angular reference is reconstructed once per stage before
  CUDA Graph capture. `CUMES_RADIUS_REFERENCE=0` restores absolute storage;
  checkpoints and output retain physical double coefficients.
- Float GPU kernels use float or float-float arithmetic throughout, including
  norm reductions, reference reconstruction, device control records, validity
  gates, and the vacuum dependency. The host controller and file formats
  retain double precision. Host and device convergence checks consume the
  same device-normalized residuals.

### Fixed

- W7-X float convergence at all-stage `1e-5` with the default radius reference
  and opt-in compensated reconstruction: the tested 33/66/99-grid case
  converges in 149 → 278 → 314 effective iterations, with final FSQR
  `9.288e-6`; checkpoint replay converges on its first iteration. Native double
  W7-X and Solovev retain their qualified checkpoints and telemetry unchanged.
- Restart imports reject inconsistent grid dimensions and spectral-family
  sizes before uploading state to the GPU.

## [1.3.0] - 2026-09-05

### Added

#### Library API

- An installable `cumes::solver` target and `cuMESConfig.cmake` package for
  `find_package(cuMES CONFIG REQUIRED)` and CMake `FetchContent` consumers.
- The in-process `cumes::EquilibriumSolver` facade, which accepts an immutable
  validated problem and returns an equilibrium snapshot, half-grid physical
  profiles, convergence report, and structured phase timings without writing
  files or exposing CUDA objects.
- Library solve controls for in-memory hot starts, quiet or diagnostic
  execution, process-environment isolation, and explicit radial-transfer
  selection.
- A deterministic `ProblemSpec` JSON writer for modifying and round-tripping
  optimizer-owned boundary inputs.

#### Forward sensitivities

- Fixed-boundary, stellarator-symmetric precise-double forward tangents through
  the spectral transforms, geometry, magnetic field, profiles, force, and
  constraint operators.
- The retained `cumes::EquilibriumLinearization` session for residual JVPs and
  repeated matrix-free boundary tangent solves using right-preconditioned
  restarted GMRES.
- Target-facing spectral, magnetic-field, geometry, flux, rotational-transform,
  and covariant-field derivatives. Current-density derivatives remain outside
  the qualified tangent interface.

#### Observability

- Per-solve setup, multigrid, stage setup/iteration/output/teardown, final-state
  transfer, and total wall-clock timings.

### Changed

- The CLI now delegates equilibrium calculation to the same supported solver
  facade used by embedding applications; optimizer parameterization and target
  functions remain owned by meow.
- Independent concurrent solvers coordinate CUDA graph capture without
  serializing their ordinary execution.
- Near-axisymmetric three-dimensional cold starts use a qualified coarse-grid
  shaping policy for analytic QA/QH optimization inputs.
- Boozer plotting and containers use the field-period toroidal-angle convention
  consistently, including the version-3 magnetic-coordinate schema.
- Zero-prescribed-current equilibria and plots are supported without requiring
  a nonzero current profile.

### Fixed

- Mixed-float packages now provide a linkable tangent API that reports the
  precise-double requirement explicitly instead of failing at final link.
- Retained tangent sessions no longer depend on the lifetime of the input
  `ValidatedProblem`, and reject non-finite linear-solver tolerances.
- Forward-dual elementary functions use conforming argument-dependent lookup
  instead of adding function overloads to `std`.
- Tangent constraint evaluation initializes and propagates all required
  scratch fields, including the magnetic-axis path.
- JSON serialization escapes every control character accepted in string
  values.

## [1.2.0] - 2026-08-31

### Added

#### Solver

- Per-stage and total CUDA device-time reporting in the executable's standard
  output.
- A direct, optional `BSplineInterpolation` header-only dependency for
  fixed-boundary multigrid transfer, with build-time and runtime fallbacks to
  the previous Catmull-Rom and linear interpolation paths.
- Diagnostic controls for cold-start shaping, axisymmetric lambda seeding,
  time-step recovery, prolongation selection, and free-boundary vacuum
  activation.

#### Plotting

- PyVista field lines rendered as three-dimensional tubes for improved depth
  and visibility.

### Changed

#### Convergence

- Fixed-boundary runs now use shaped cold starts, qualified stage-specific
  initial steps, and a conservative one-shot recovery after an early
  time-step reduction. Axisymmetric starts additionally seed lambda from the
  initial geometry.
- Free-boundary cold starts use qualified shaping and coarse-grid steps, and
  activate the vacuum edge force earlier once the predictor residual is low
  enough.
- Precise-double fixed-boundary multigrid continuation now applies a global
  cubic B-spline transfer matrix on the GPU. Matrix construction is prepared
  asynchronously on the host while the coarse GPU stage iterates, so the
  spectral state remains device-resident and stage transitions do not wait on
  interpolation setup.
- The qualified Solovev trajectory is reduced from 906 to 754 effective
  iterations and W7-X from 5505 to 4106, with all configured force-residual
  tolerances satisfied.

#### Maintenance

- Iteration-controller thresholds and tuning factors are centralized in the
  `cumes::control_policy` namespace.
- cuMES convergence is documented in terms of its own residual and validity
  gates; VMEC++ remains an independent diagnostic comparison.

### Fixed

- Restored typed CLI diagnostics when atomic output publication fails,
  including read-only and otherwise unwritable destinations.

## [1.1.0] - 2026-08-30

### Added

#### Main

- Derived magnetic-field and current-density output on the full and half
  radial grids, serialized consistently by the binary, NetCDF, and HDF5
  backends.
- Direct Boozer-equilibrium generation from the main executable through
  `--boozer-output`, backed by the integrated magnetic-coordinate transform
  and backend-neutral Boozer result containers.

#### Plotting

- An independent plotting package that consumes generated native and Boozer
  equilibrium files, including six-panel PEST and Boozer coordinate meshes,
  magnetic-field slices, and flux-surface field contours.
- Optional PyVista rendering, free-boundary coil overlays, JSON coil input,
  and full-grid equilibrium surfaces.
- Plotting `--output-dir` support for standard filenames without a shared
  prefix.

### Changed

- Native and Boozer result-output options are now explicitly mutually
  exclusive.
- Final scientific fields are captured only on the finest multigrid stage.
- Inline Makegrid generation is parallelized.
- The four comparison utilities are C++ executables with direct standalone
  builds; their common implementation is a single STB-style header.

### Fixed

#### Plotting

- Three-dimensional Matplotlib scenes globally depth-sort plasma and coil
  geometry for correct occlusion.

## [1.0.0] - 2026-08-26

- First versioned cuMES release, including the CUDA equilibrium solver,
  multigrid continuation, fixed- and free-boundary operation, checkpointing,
  versioned result containers, double and mixed-float precision policies, and
  the documented verification suite.
