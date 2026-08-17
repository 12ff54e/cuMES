# V1 container reader rank-hardening handoff

> **Follow-up acceptance note (review of `611e8d7`):** this exact-rank task is
> closed, but the subsequent resource/schema audit found additional bounded
> work. Final acceptance now depends on
> [`v1-reader-resource-hardening-handoff.md`](v1-reader-resource-hardening-handoff.md).

> **CLOSED (2026-08-17).** The bounded repair below is implemented in
> `102c6ec` (readers + fixtures; docs reconciled in a follow-up commit on top):
> exact rank/type/extent checks precede every NetCDF/HDF5 read into a fixed or
> sized host buffer, `test_io_malformed_shapes` runs the malformed fixtures in
> ordinary CTest and under ASan/UBSan (the ASan twin caught a real fixture
> overread during development), and every exit gate of §7 passes: verify
> 58/58, sanitizer 90/90, float/no-backend/NetCDF-only/HDF5-only 30/30 each,
> `ci_gpu.sh` green, trajectories with accepted controller decisions, and the
> legacy `.bin` final states byte-identical to `dc0d0c4`. Modern-GPU
> performance validation remains separately POSTPONED. The sections below are
> retained as the review record.

Status date: 2026-08-17. Reviewed branch: `overhaul` at
`3a1f7b0` (`origin/overhaul` synchronized and the worktree clean at review
start).

## 1. Verdict and scope

The first post-overhaul follow-up is implemented and all available-hardware
gates pass. Final code acceptance is nevertheless reopened for one bounded
memory-safety class: the v1 NetCDF/HDF5 readers validate selected extents but
do not first prove the exact rank and scalar shape required by the fixed-size
buffers passed to the library APIs.

This handoff covers:

1. exact rank/type/extent validation for every v1 reader object;
2. checked dimension narrowing and allocation bounds;
3. malformed-rank ASan/UBSan fixtures;
4. reconciliation of the closeout documents after the fix.

Modern-GPU performance validation is not part of this task. It remains
explicitly POSTPONED until suitable hardware is available.

## 2. Finding - NetCDF v1 reader trusts variable shape

File: `src/cumes/io/netcdf_writer.cpp`, `NetcdfV1Reader::read`.

### Unsafe sites

- The six state variables are read with two-element `start`/`count` arrays
  without first requiring rank 2 and the exact `[ns, mnmax]` dimensions.
- The run-outcome helper calls `nc_get_var_int` into one `int` without proving
  that the variable is scalar. A malicious array variable can write more than
  one integer into that destination.
- The stage-array helpers pass `&dimid`, storage for one dimension ID, to
  `nc_inq_vardimid` without first requiring `nc_inq_varndims(...) == 1`.
- `ns` and `mnmax` are `size_t` dimensions narrowed to `int` after only a
  lower-bound check.

The NetCDF contract exposes `nc_inq_varndims` specifically to obtain the
number of dimensions before retrieving the dimension-ID list. A reader of an
untrusted container must never assume the rank implied by the variable name.

### Required NetCDF helper contracts

Introduce shared checked helpers rather than repeating ad-hoc inquiry calls:

- `read_scalar_int(name)`:
  - variable exists;
  - rank is exactly 0;
  - datatype is compatible with the declared schema (prefer exact `NC_INT`);
  - read exactly one value, preferably with an API whose requested element
    count is explicit;
- `read_vector<T>(name, expected_dim_id, expected_len)`:
  - rank exactly 1 before allocating the dimension-ID buffer;
  - dimension ID and extent exactly match the schema;
  - datatype matches the schema;
  - destination allocation uses a checked byte count;
- `read_state_family(name, ns_dim_id, mnmax_dim_id, ns, mnmax)`:
  - rank exactly 2;
  - dimension order and IDs exactly `[ns, mnmax]`;
  - extents exactly match the named dimensions;
  - datatype exactly `NC_DOUBLE` for schema v1;
  - read selections cannot consume more than the checked destination size.

Reject the file with a typed `Result` error before any read when an invariant
does not hold.

## 3. Finding - HDF5 v1 reader trusts dataspace and attribute rank

File: `src/cumes/io/hdf5_writer.cpp`, `Hdf5V1Reader::read` and
`getStrAttr`.

### Unsafe sites

- `getDim` accepts any rank >= 1 and calls `H5Sget_simple_extent_dims` with a
  caller-owned one- or two-element array. A higher-rank dataset requires a
  larger dimension array and can overwrite the caller's stack storage.
- The stage integer/double helpers use one-element `dims` arrays without
  first requiring rank 1.
- `getIntAttr` reads into one `int` without proving the attribute dataspace is
  scalar (or exactly one element).
- `getStrAttr` sizes the destination from datatype width only. It does not
  prove a scalar/one-element attribute dataspace, so an array of fixed-size
  strings can exceed the allocated buffer.
- Only `rmncc` supplies state dimensions; the other five state datasets are
  not independently required to have rank 2 and identical dimensions before
  `H5Dread`.
- HDF5 dimension values are narrowed to `int` before proving they fit.

### Required HDF5 helper contracts

Before calling `H5Sget_simple_extent_dims`, call
`H5Sget_simple_extent_ndims` and require the schema's exact rank. Then:

- scalar attributes: require scalar dataspace or exactly one point, the
  expected datatype class/size, and a destination sized for the complete
  selected element count;
- strings: require one element, a string datatype, a bounded nonzero width,
  and checked `width * npoints` before allocation/read;
- stage/restart datasets: require rank 1, exact extent, and expected datatype;
- every state family: require rank 2, dimensions exactly `[ns, mnmax]`, and
  `H5T_NATIVE_DOUBLE`-compatible stored type before reading;
- close every opened type/dataspace/dataset/attribute on every failure path,
  preferably through small RAII handle wrappers.

## 4. Checked dimensions and allocation limits

For both backends:

- reject `ns == 0` or `mnmax == 0`;
- reject either dimension above `INT_MAX` before conversion into
  `EquilibriumSnapshot` fields;
- use `checked_mul(ns, mnmax)` and a checked byte count before allocating;
- validate `nstages`, `nrestarts`, string widths, and attribute lengths before
  allocating host memory;
- catch or convert allocation failures into typed reader errors rather than
  allowing an exception to escape across the `Reader` interface;
- validate serialized enum/status and nonnegative iteration fields where the
  schema defines a closed range.

A reasonable implementation may impose documented schema/resource caps. A
sparse file declaring enormous dimensions must fail before allocating an
enormous vector.

## 5. Required regression tests

Extend `tests/test_io_restart_offsets.cpp` or add a focused host-only
`test_io_malformed_shapes.cpp`. Each backend must include:

### NetCDF fixtures

- a state family with rank 1 and rank 3;
- a state family with swapped or mismatched rank-2 dimensions;
- a scalar outcome variable encoded as a rank-1 array with multiple values;
- a stage/restart variable encoded as rank 0 and rank 2;
- correct rank but wrong datatype;
- `ns`/`mnmax` beyond the representable or documented resource bound.

### HDF5 fixtures

- `rmncc` with rank 1 and rank 3;
- one of the other five state families with a different rank or extent;
- a stage/restart dataset with rank 0 and rank 2;
- integer outcome attribute with multiple elements;
- string provenance attribute with multiple elements;
- correct rank but wrong datatype;
- dimensions beyond the representable or documented resource bound.

Every malformed file must return a typed failure without a crash, out-of-
bounds access, partial `RunReport`, or excessive allocation. Run the exact
fixture source in the ASan/UBSan host twin. Keep valid v1 round trips and the
negative/descending/oversized restart-offset cases as controls.

## 6. Documentation reconciliation

After the reader fix and tests pass:

- update the reviewed HEAD recorded in this document;
- change historical present-tense wording in
  `post-overhaul-follow-up.md` (for example, "does not yet satisfy") to
  explicit past tense or a clearly delimited archived-review section;
- remove or label the old 55/55 and mixed-float 26/27 counts, since the
  current suites are 57/57 and 29/29 respectively at `3a1f7b0`;
- update the stale "not design-complete" paragraph in
  `overhaul-completion-plan.md` after all current-hardware gates pass;
- only then restore a final-acceptance banner.

Do not change the separate statement that modern-GPU validation is postponed.

## 7. Exit gates

The next agent should not close this handoff until all of the following hold:

1. Exact rank/type/extent checks precede every NetCDF/HDF5 read into a fixed
   or sized host buffer.
2. All malformed-shape fixtures fail cleanly under ASan/UBSan.
3. `bash scripts/ci_gpu.sh` passes on the available TITAN Xp.
4. Mixed float, no-backend, NetCDF-only, and HDF5-only suites pass.
5. The full sanitizer preset passes, including the new ASan reader cases.
6. Full Solovev and W7-X trajectories retain accepted controller decisions,
   iteration counts, residuals, and all six state families.
7. The legacy `.bin` final states remain byte-identical to the accepted
   `dc0d0c4` outputs. Do not re-freeze a trajectory merely because it
   converges.
8. `git diff --check` passes and the full matrix leaves no scratch files.
9. The closeout documents contain no known contradiction with source or test
   behavior.

## 8. Review evidence before this handoff

At `3a1f7b0`, before the reader-rank repair:

- exact GPU release script: 57/57 and the capped-trajectory oracle passed;
- mixed float: 29/29;
- NetCDF-only: 29/29;
- HDF5-only: 29/29;
- no optional backends: 29/29;
- full sanitizer matrix: 88/88;
- Solovev: `251 -> 199 -> 456`, FSQR `9.583e-17`;
- W7-X: `1877 -> 1617 -> 2011`, FSQR `9.778e-13`;
- both final legacy binary state files were byte-identical to the previously
  reviewed `dc0d0c4` outputs.

These green results establish a narrow regression baseline; they do not cover
the malformed-rank memory-safety class described above.
