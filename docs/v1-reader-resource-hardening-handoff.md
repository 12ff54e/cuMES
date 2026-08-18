# V1 reader resource-hardening handoff

> **CLOSED (2026-08-18).** Commit `de265cd` implements every bounded repair in
> this handoff: HDF5 variable-length strings are rejected before read,
> per-family state storage has a documented pre-allocation cap, HDF5 integer
> objects use a signed native-int-width schema with endian conversion allowed,
> closed-range report values are validated, and report reconstruction is
> transactional. The expanded malformed fixtures pass in ordinary CTest and
> under ASan/UBSan. Every available-hardware exit gate passes and both frozen
> trajectories remain byte-identical. Final acceptance is restored; only the
> separately POSTPONED modern-GPU performance validation remains.

Status date: 2026-08-18. Implementation commit: `de265cd`.

## 1. Verified baseline

Do not re-freeze the numerical baseline while implementing this repair. At the
reviewed HEAD:

- `scripts/ci_gpu.sh`: 58/58 tests passed;
- sanitizer preset: 90/90 tests passed;
- float, no-backend, NetCDF-only, and HDF5-only presets: 30/30 each;
- Solovev: `251 -> 199 -> 456`, final FSQR `9.583e-17`;
- W7-X: `1877 -> 1617 -> 2011`, final FSQR `9.778e-13`;
- both legacy `.bin` final states were byte-identical to the previously
  accepted artifacts.

At review time, the exact-rank fixtures already proved scalar, vector, and
family ranks and extents before fixed-buffer reads; this handoff concerned only
resource limits and complete schema validation, not the solver trajectory.

### Closure results

- `scripts/ci_gpu.sh`: 58/58 passed;
- sanitizer preset: 90/90 passed, including the ASan/UBSan malformed reader;
- float, no-backend, NetCDF-only, and HDF5-only: 30/30 each;
- Solovev: `251 -> 199 -> 456`, FSQR `9.583e-17`;
- W7-X: `1877 -> 1617 -> 2011`, FSQR `9.778e-13`;
- both legacy `.bin` outputs are byte-identical to the frozen accepted states;
- no numerical baseline was re-frozen.

The state limit is `kMaxStateElementsPerFamily = 1 << 24`: 128 MiB per
double family, 768 MiB for the six-family snapshot, and 896 MiB peak state
storage in the HDF5 reader including its transpose slab.

## 2. P1 — reject HDF5 variable-length provenance strings

Location: `src/cumes/io/hdf5_writer.cpp`, `getStrAttr`, currently around
lines 98-108.

The helper accepts every `H5T_STRING` and uses `H5Tget_size()` as the stored
text width. For a variable-length string, HDF5 returns `sizeof(char*)`, not the
payload length. The existing 1 MiB check therefore succeeds, `H5Aread()` is
called with a character buffer where HDF5 expects pointer storage, and the
library may allocate an arbitrarily large payload. The result is bogus
provenance plus a leaked/unbounded allocation.

Required repair:

1. Call `H5Tis_variable_str(ty.get())` after proving the datatype class.
2. Treat a negative result as an HDF5 inquiry failure.
3. Reject a positive result before allocation or `H5Aread()`. Schema-v1 writers
   emit fixed-width strings, so supporting variable-length input is unnecessary
   and rejecting it is the safest contract.
4. Keep the existing scalar/one-element dataspace and fixed-width cap checks.

Required regression:

- add an HDF5 scalar provenance attribute whose datatype is
  `H5T_C_S1`/`H5T_VARIABLE`;
- require a typed reader failure with no crash, leak, or accepted report;
- run the fixture in ordinary CTest and the ASan/UBSan twin. Enable leak
  detection for this fixture if the platform sanitizer supports it.

Do not implement variable-length support with `std::string::data()`. If support
is ever desired, it requires `char*` pointer storage, explicit reclamation, and
a defensible pre-allocation policy; that is outside this bounded repair.

## 3. P1 — impose a practical state-allocation cap

Locations:

- `include/cumes/io/writer_helpers.hpp`, currently lines 40-46;
- `src/cumes/io/netcdf_writer.cpp`, state dimension checks around lines 630-652;
- `src/cumes/io/hdf5_writer.cpp`, state dimension checks around lines 611-637.

The readers currently require only positive dimensions, `INT_MAX` narrowing,
and non-overflowing multiplication. A sparse container with dimensions such as
`[1 << 30, 1]` passes those checks and attempts to resize an 8 GiB vector.
`std::bad_alloc` conversion is not a sufficient guard on an overcommitting
Linux host: the allocation can appear to succeed and the subsequent
value-initialization/read can trigger the OOM killer.

Required repair:

1. Add a documented reader limit such as `kMaxStateElements` or
   `kMaxSnapshotBytes`. A small `ReaderLimits` value object is also acceptable
   if the public reader API can carry it without widening this task.
2. Apply the cap after checked multiplication and before **every** state vector
   allocation or resize.
3. Account for peak resident storage, not only one family. HDF5 currently holds
   a transpose buffer plus the six destination families; NetCDF holds six
   destination families.
4. Return a typed error that distinguishes a resource-limit violation from
   integer overflow and library I/O failure.
5. Document the chosen limit and its units. Keep ordinary solver output well
   below the limit.

Required regressions for both backends:

- create a sparse v1 container with a dimension below `INT_MAX` but above the
  new resource cap;
- require rejection before any state payload read or enormous allocation;
- retain the existing beyond-`INT_MAX` fixture as a separate narrowing test;
- run both cases under the host sanitizer configuration.

## 4. P2 — finish the exact HDF5 datatype contract

Location: `src/cumes/io/hdf5_writer.cpp`, `datasetIsInteger`, currently around
lines 536-543, and the stage/restart vector readers around lines 577-585.

`datasetIsInteger` checks only `H5T_INTEGER`. It therefore accepts arbitrary
integer widths and signedness and asks HDF5 to convert them into native `int`.
That does not meet the closeout documents' claim that every object has an exact
schema datatype.

Required repair:

- define the portable schema explicitly (recommended: signed 32-bit integer;
  permit endian conversion if desired);
- check integer class, four-byte width, and signedness before `H5Dread()` for
  stage/restart datasets and scalar integer attributes;
- either make the floating-point rule equally explicit (IEEE/binary64 or the
  intentionally supported compatible set) or soften the documentation from
  “exact datatype” to the precise compatibility contract implemented.

Required regressions:

- HDF5 stage/restart arrays encoded as unsigned, 8-bit, and 64-bit integers;
- a scalar integer attribute with wrong signedness;
- each malformed schema must fail cleanly before populating `RunReport`.

NetCDF already requires exact `NC_INT`/`NC_DOUBLE`; preserve that behavior.

## 5. P2 — validate all closed-range serialized values

Locations:

- `src/cumes/io/netcdf_writer.cpp`, report reconstruction around lines 690-783;
- `src/cumes/io/hdf5_writer.cpp`, report reconstruction around lines 680-771.

Both readers still accept values that cannot be produced by a valid writer:

- `build_dirty` other than 0 or 1;
- `stage_converged` other than 0 or 1;
- nonpositive `stage_ns`;
- negative `restart_iteration`.

Required repair:

1. Validate these values before converting them to `bool` or constructing
   `StageReport`/`RestartEvent` objects.
2. Apply the same rules and error categories to both backends.
3. Keep the existing status, precision, total-iteration, stage-iteration, and
   restart-offset checks.

Add one focused malformed fixture per rule per backend. Table-driven mutation
cases are preferred to duplicating complete files.

## 6. Documentation reconciliation

While this handoff is open, it supersedes the final-acceptance statements in:

- `docs/overhaul-completion-plan.md`;
- `docs/verification.md`;
- `docs/reader-rank-hardening-handoff.md`.

After the implementation and all exit gates pass:

1. mark this document `CLOSED` with the fixing commit IDs and measured results;
2. restore final acceptance in the completion plan;
3. describe the actual resource caps and HDF5 compatibility rules in
   `docs/verification.md`;
4. avoid claiming “exact datatype” if any documented conversion remains;
5. keep modern-GPU performance validation clearly `POSTPONED`.

## 7. Exit gates

The repair is complete only when all of the following are true:

1. every new malformed fixture fails with a typed reader error;
2. ordinary and ASan/UBSan malformed-reader tests pass;
3. the full verify and sanitizer presets pass;
4. float, no-backend, NetCDF-only, and HDF5-only matrices pass;
5. `scripts/ci_gpu.sh` passes;
6. Solovev and W7-X reproduce the iteration decisions and residuals in §1;
7. legacy final states remain byte-identical to the frozen accepted artifacts;
8. `overhaul` is clean and synchronized with `origin/overhaul`;
9. the documentation in §6 states only measured, currently true claims.

No modern-GPU performance result is required for this repair.
