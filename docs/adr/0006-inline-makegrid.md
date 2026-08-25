# ADR-0006: Build external-coil response tables in vacuum-field

- Status: Accepted
- Date: 2026-08-25

## Context

Free-boundary runs originally required a precomputed MAKEGRID NetCDF file.
That forced users who already had coils-dot geometry to run a separate program,
manage a derived artifact, and require NetCDF even though the solver only needs
the magnetic-field response arrays. The MAKEGRID calculation is naturally part
of the external-field domain already owned by `deps/vacuum-field`.

The vacuum solver's iteration loop must remain independent of coil geometry:
generating a three-dimensional response grid is setup work, not an operation to
repeat on each vacuum update.

## Decision

Add an optional host-only `vfield::makegrid` library to `deps/vacuum-field`.
It parses MAKEGRID parameter JSON and coils-dot polygon or axis-aligned circular
filaments, then returns `MgridProvider::ResponseTable`. `VacuumFieldSolver`
accepts an owned response table as an alternative to an mgrid filename or a
fixed field.

cuMES free-boundary input accepts exactly one external-field source:

- `mgrid_file`; or
- `coils_file` together with either `makegrid_parameters_file` or the complete
  embedded parameter object `makegrid_paramters`.

When both parameter keys are present, cuMES emits a warning and the embedded
object takes precedence. This makes a free-boundary input self-contained while
preserving the file form for existing workflows.

For the second form, `FreeBoundaryOperator` generates the response table once
during construction and passes it to vacuum-field in memory. cuMES requires
raw-current mode (`normalize_by_currents=false`) so its `extcur` values retain
their established Ampere convention. A standalone `makegrid` CLI and NetCDF
writer remain optional interoperability features.

## Consequences

- Existing mgrid-file inputs and the CUDA vacuum update path are unchanged.
- In-memory generation does not require NetCDF and creates no temporary file.
- The dependency direction remains `cuMES -> vacuum-field`; vacuum-field has no
  cuMES or vmecpp dependency.
- Response-grid generation adds startup time proportional to coil complexity
  and grid size, but no work to the nonlinear iteration loop.
- The initial implementation generates magnetic-field response arrays only;
  it does not generate MAKEGRID vector-potential arrays.
