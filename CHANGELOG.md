# Changelog

All notable changes to cuMES are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/).

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
