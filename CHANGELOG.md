# Changelog

All notable changes to cuMES are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/).

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
