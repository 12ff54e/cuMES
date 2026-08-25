# ADR-0005 — vacuum-field as a git submodule (adopted)

Status: adopted

## Context

cuMES is fixed-boundary only (`lfreeb=true` is a hard validation error). The
free-boundary work starts with the vacuum magnetic field on the LCFS: vmecpp's
NESTOR algorithm (coil mgrid field + magnetic-scalar-potential Green's-function
solve), which has no CUDA implementation anywhere. Porting it into cuMES
in-tree would drag the whole dependency question (NetCDF mgrid reading, the
golden test data, the boundary-element numerics) into the cuMES tree and its
Class-A trajectory discipline. The port needs its own iteration cadence and its
own Fortran-golden verification, independent of the cuMES solver.

## Decision

The port lives in its own repository, `12ff54e/vacuum-field` (private), a
CUDA C++ library in the cuMES house style (`template<class T>` operator
classes with an explicit double/float instantiation split, `vfield::`
namespace, `Real` double-default), added to cuMES as a git submodule at
`deps/vacuum-field`. cuMES builds and links it (`vfield::vfield`, the alias
selected by `VFIELD_USE_FLOAT` which cuMES propagates from `CUMES_USE_FLOAT`)
via `add_subdirectory`, guarded by `CUMES_USE_VACUUM_FIELD` (default ON,
auto-disabled when the submodule is not checked out, so hosted CI without
submodules keeps building). The library's own ctest suite runs inside cuMES's
ctest.

Design decisions inside the library, ported 1:1 from vmecpp v0.7.0 (MIT;
attribution in the library LICENSE):

- **Every accumulation is one-thread-per-output with vmecpp's exact summation
  order** (no atomics, no cuFFT — the analysis DFTs are smaller than FFT
  launch overhead), so vmecpp's own golden tolerances carry over unchanged:
  `vac1n_surface` 1e-12, `vac1n_bextern` 1e-10, `vac1n_analyt` 1e-9,
  `vac1n_greenf` 5e-10, `vac1n_fourp/fouri/solver` 1e-9, `vac1n_bsqvac`
  1e-10 (Fortran-VMEC checkpoint dumps of `cth_like_free_bdy`, vendored in the
  library's `tests/data/`).
- **The dense Laplace solve runs on the host in double** (`vfield::LuSolve`,
  LAPACK dgetrf/dgetrs semantics on vmecpp's flat layout — cross-validated
  against Fortran LAPACK via `potvac_out`): the system is
  `mnpd = (2*ntor+1)(mpol+2)` <= ~300, i.e. well under 0.1% of the update
  cost, and the potential is control-adjacent. `LuSolve` is the seam for a
  future cuSOLVER backend.
- **Outputs stay device-resident** (`const T*` accessors) so cuMES can consume
  them zero-copy when step 2 wires the solver; `update()` mirrors
  Nestor::update minus the OpenMP partitioning and the checkpoint machinery.

## Consequences

- cuMES's build now depends on the submodule for the vacuum-field targets;
  the fixed-boundary solver code is untouched (step 1 only builds and links —
  `lfreeb` still errors, `mgrid_file`/`extcur` stay ignored keys).
- The hosted CI keeps building without submodules (auto-disable); the library
  has its own CI (build + host-only tests; GPU/golden gates are local, like
  cuMES's).
- `git submodule update --init` is the only new contributor workflow step.

## Alternatives considered

- **FetchContent / vendoring the sources in-tree**: rejected — the library has
  its own release/verification cadence and ~79 MB of golden test data that
  should not bloat cuMES history; submodule pins a reviewed revision.
- **CPU-only first port, GPU later**: rejected — cuMES is pure-CUDA; the port
  was written CUDA-first with the host dense solve as the only deliberate
  exception (measured to be negligible).
- **cuSOLVER for the dense solve**: rejected for now — workspace management and
  launch latency for a <= 300-sized system buy nothing; the seam is in place.
