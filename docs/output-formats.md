# cuMES on-disk output formats

Human-readable description of every container the solver writes (and reads
back). The `CUMES_DUMP`-gated diagnostic files are documented separately in
`docs/dump-files.md`. The authoritative freeze lives in `configs/schema-v1.json` under
`x-cumes-on-disk-contracts` (blueprint §6.13); the implementations are
`src/cumes/io/` (`versioned_binary.cpp`, `checkpoint.cpp`,
`netcdf_writer.cpp`, `hdf5_writer.cpp`). All formats are little-endian.

## 1. Which container is written when

| Situation | Container | Magic |
| --- | --- | --- |
| `./build/cumes <in> --output out.bin` | versioned binary (schema v1, on-disk version 8) | `CUMES001` |
| a `.nc`/`.h5` suffix | NetCDF/HDF5 v1 (versioned, attributes) | — |
| `--checkpoint <path>` | versioned checkpoint (v6) | `CUMECKP1` |
| `--restart <path>` (read) | versioned checkpoint (v6; v1–v5 still read) | `CUMECKP1` |

The output path defaults to `$PWD/cumes-output.bin`; an unknown suffix is an
error. All writers publish atomically (write a temporary file in the target
directory, then rename over the destination; a failed write removes the
temporary and leaves the destination untouched).

## 2. The shared state payload

Every container above carries the same converged-state payload: the six
spectral coefficient families, always **double** on disk regardless of the
computation scalar type (the device→host copy converts `T` → double).

Family order (the `EquilibriumSnapshot::Component` order, data-layout.md §2):

```
rmncc, zmnsc, lmnsc, rmnss, zmncs, lmncs
```

Each family is `mnmax * ns` doubles, **mode-major, surface-contiguous**:

```
index(surface, mode) = surface + mode * ns
mode = m * (ntor + 1) + n        (folded n >= 0 basis; mnmax = mpol * (ntor+1))
```

The physical toroidal mode is `N = n * nfp`; the ζ grid covers one field
period. The families reconstruct real space as (kernels/fourier_impl.cuh):

```
R = rmncc·cos(mθ)cos(nζ) + rmnss·sin(mθ)sin(nζ)
Z = zmnsc·sin(mθ)cos(nζ) + zmncs·cos(mθ)sin(nζ)
λ = lmnsc·sin(mθ)cos(nζ) + lmncs·cos(mθ)sin(nζ)
```

The axis row (`j = 0`) is the constant-extrapolated row, which the
comparison scripts intentionally skip (compare_states.py).

### Scientific result fields

Solver outputs (but not restart checkpoints) also carry the final real-space
magnetic field, current density, and coordinate Jacobian. The point ordering
is theta-fast:

```
point(theta, zeta, surface) = theta + ntheta * (zeta + nzeta * surface)
```

The seven half-grid arrays have shape `[ns-1, nzeta, ntheta]`:

```
sqrtg, bsups, bsupu, bsupv, bsubs, bsubu, bsubv
```

They are respectively `sqrt(g)`, the contravariant components
`(B^s, B^theta, B^zeta)`, and the covariant components
`(B_s, B_theta, B_zeta)`. `B^s` is stored explicitly as zero. The remaining
magnetic arrays and `sqrtg` are the solver's final native half-grid values;
`B_s` is obtained by lowering the index with the final metric.

The six full-grid arrays have shape `[ns, nzeta, ntheta]`:

```
jsups, jsupu, jsupv, jsubs, jsubu, jsubv
```

They are `(J^s, J^theta, J^zeta)` and `(J_s, J_theta, J_zeta)`. cuMES derives
them after the last solver iteration from `J = curl(B)/mu0`. In flux
coordinates, the interior stencil is

```
J^s     = (d_theta B_zeta - d_zeta B_theta) / (mu0 sqrtg)
J^theta = (d_zeta B_s - d_s B_zeta) / (mu0 sqrtg)
J^zeta  = (d_s B_theta - d_theta B_s) / (mu0 sqrtg)
```

Angular derivatives use the periodic Fourier-collocation derivative on the
output grid; `d_zeta` includes the `nfp` factor because the stored grid spans
one field period. Radial derivatives are centered half-to-full differences.
`J^s`, naturally half-grid, is interpolated conservatively with `sqrtg`
weights to the full grid. All six current arrays use linear endpoint
extrapolation, then the covariant components are obtained with the full-grid
cylindrical metric.

This post-processing runs after convergence and cannot change the iteration
trajectory. `test_derived_fields` verifies the angular derivative, radial
staggering, `nfp` scaling, metric lowering, and component values against a
manufactured field using only project code. `test_io_golden` verifies exact
round trips through binary, NetCDF, and HDF5.

## 3. Versioned binary (schema v1, on-disk version 8)

```
magic     8 bytes  "CUMES001"
version   int32    = 8 (the current on-disk version)
ns        int32
mnmax     int32
families  6 * (mnmax*ns) doubles, mode-major     ← the §2 payload
fields    ntheta (int32), nzeta (int32), then the seven half-grid and
          six full-grid arrays above as f64 in their listed order;
          (0, 0) is an explicit absent marker for library-only snapshots
---- provenance trailer ----
precision int32    0 = double, 1 = float (the computation type)
status    int32    RunStatus (0 converged, 1 not converged, 2 numerical
                   failure, 3 validation failure, 4 output failure)
total_iter int32   total effective iterations over all stages
nstages   int32    number of stage records that follow
strings   build.revision, build.build_type, build.precision_policy,
          build.compile_flags, input.source_path, input.source_hash,
          runtime.gpu_name, runtime.driver, runtime.runtime,
          runtime.toolkit            (see string encoding below)
u8        build.dirty                (written between revision and build_type)
stages    per stage: ns (int32), iterations (int32), converged (u8),
          fsqr (f64), fsqz (f64), fsql (f64), nrestarts (int32),
          then nrestarts restart iteration indices (int32 each)
params    the embedded normalized-input record (see §5), the LAST trailer
          element; version 3 and up
```

The embedded input record (§5) mirrors
`ValidatedProblem::normalize_to_json()` field-for-field with the RESOLVED
values (the angular-grid defaults already applied): `mpol, ntor, nfp,
ntheta, nzeta, ncurr` (int32 each), `delt, phiedge, pres_scale,
adiabatic_index, spres_ped, bloat, curtor, tcon0` (f64 each), the `schema`
string, the `pmass_type, piota_type, pcurr_type` profile-type strings,
then the vectors `am, ac, ai, aphi, raxis_c, zaxis_s` (f64 each),
the input stages (`nstages_in` then per stage `ns` int32, `max_iter`
int32, `ftol` f64), the raw boundary (`rbc_m` int32, `rbc_n` int32,
`rbc_value` f64, then `zbs_*`), and the folded boundary `rbcc, rbss, zbsc,
zbcs` (f64 each), followed by `lfreeb`, `nvacskip`, `mgrid_file`, `extcur`,
`coils_file`, `makegrid_parameters_file`, and the optional embedded
`makegrid_parameters` object. Every vector is an int32 element count followed
by the payload; counts are capped at 2^20 by the reader.

Notes:

- **String encoding:** `int32` byte length followed by the bytes, no NUL
  terminator. The reader caps a length prefix at 2^24 before allocating.
- **No `scalar_type` string:** the precision tag is authoritative and the
  reader reconstructs `scalar_type` from it.
- **Historical versions 1 and 2** (written by earlier builds, still
  readable): version 1 carried a `scalar_type` string between `build_type`
  and `source_path` and no precision-policy pair (the reader consumes and
  discards it); versions 1 and 2 carry no input record and the reader
  reports a default-empty one. (Before 2026-08-20 the writer emitted
  `scalar_type` while the v2 reader expected the policy pair — a
  writer/reader sequence mismatch fixed together with the v1-compat path
  above; the state payload itself was never affected.) Version 3 added the
  input record, version 4 added profile-type strings, versions 5 and 6 are
  the free-boundary lineage, and version 7 combines the profile-type and
  complete free-boundary extensions. Version 8 added the scientific-result
  field block between the stable spectral payload and provenance trailer.
- The state payload is read and validated independently of the trailer, so
  a reader can stop after the state and stays forward-compatible with
  later trailer revisions. The trailer is parsed only when the caller
  requests the `RunReport`.

## 4. Versioned checkpoint v6 (`--checkpoint` / `--restart`)

```
magic     8 bytes  "CUMECKP1"
version   int32    = 6
precision int32    = 0 (the checkpoint is always double on disk)
ns        int32
mnmax     int32
families  6 * (mnmax*ns) doubles, mode-major     ← the §2 payload
params    the embedded normalized-input record (§5), version 2 and up
```

Version 2 added the input record, version 3 added the profile-type strings,
versions 4 and 5 are the free-boundary lineage, and version 6 combines the
profile-type and complete free-boundary extensions. There is no provenance
trailer beyond the input record—the restart path reads the state only and
never touches the record. Version-1 checkpoints (no record) remain readable.
Header mismatch, unsupported version/precision, or corruption is an error,
never a silent cold start.

## 5. The embedded input record / NetCDF-HDF5 layout (v1)

The embedded input record (§3/§4 in the binary formats) is represented
natively in NetCDF/HDF5:

- the scientific result variables listed in §2 use native 3-D double
  datasets with dimensions `[ns_half, n_zeta, n_theta]` or
  `[ns, n_zeta, n_theta]`; older files without any of these variables remain
  readable, while a partial field set is rejected;

- scalar variables (`mpol, ntor, nfp, ntheta, nzeta, ncurr, lfreeb,
  nvacskip` as int,
  `delt, phiedge, pres_scale, adiabatic_index, spres_ped, bloat, curtor,
  tcon0` as double) — HDF5: same names as root-group attributes;
- 1-D array variables `am, ac, ai, aphi, raxis_c, zaxis_s, extcur` (double) and
  the input stage arrays `stage_in_ns, stage_max_iter` (int),
  `stage_ftol` (double) — HDF5: same names as 1-D datasets;
- the `schema`, `mgrid_file`, `coils_file`, and
  `makegrid_parameters_file` string attributes and the `pmass_type`, `piota_type`,
  `pcurr_type` profile-type string attributes (absent in older
  containers -> read as "power_series"); embedded Makegrid fields use
  `makegrid_parameters_present` plus scalar `makegrid_*` attributes;
- the boundary is the pre-existing native pair: `rbc_m/rbc_n/rbc_value`
  and `zbs_m/zbs_n/zbs_value` (int/int/double over `nrbc`/`nzbs`) plus the
  folded 2-D matrices `rbcc/rbss/zbsc/zbcs` over `[n_mpol, n_ntorp1]`.

Empty profile vectors get no variable/dataset (classic NetCDF gives a
0-length dimension unlimited semantics and allows only one); a reader
treats an absent array as empty. Readers prove exact rank/datatype/extent
before every read into a fixed or sized buffer (see the reader-rank
hardening in overhaul-history.md); malformed shapes are rejected, never
tolerated. A container without the record fields (written before the
embedding) is read with a default-empty record.

The rest of the layout is the same provenance as the binary trailer —
scalar variables `precision`/`status`/`total_iterations`/`build_dirty`,
string attributes `revision`, `build_type`, `precision_policy`,
`compile_flags`, `source_path`, `source_hash`, `gpu_name`, `driver`,
`runtime`, `toolkit`, and per-stage variables `stage_ns`/
`stage_iterations`/`stage_converged`/`stage_fsqr`/`stage_fsqz`/`stage_fsql`
plus the restart metadata `restart_stage_offset`/`restart_iteration`. The
state families are 2-D datasets over `[ns, mnmax]` with the logical value
`family[surface, mode]` mapped to device offset `surface + mode * ns`.

## 6. Reading the state in Python

`scripts/compare_states.py` is the reference reader for the versioned
binary payload:

```python
magic, version, ns, mnmax = struct.unpack("<8siii", f.read(20))  # "CUMES001"
fams = {name: struct.unpack(f"<{ns*mnmax}d", f.read(8*ns*mnmax))
        for name in ("rmncc", "zmnsc", "lmnsc", "rmnss", "zmncs", "lmncs")}
```

The 20-byte header (magic + version + ns + mnmax) is followed directly by
the six families; only a consumer that wants provenance needs the trailer.
(compare_bitwise.py compares the state payload byte-wise, not the whole
file: the trailer embeds the git revision, so full-file bytes differ
across revisions by design.) `scripts/plot_equilibrium.py` walks the full trailer
and parses the embedded input record, so a converged equilibrium plots
without its input JSON (containers written before the record are rejected
with a "re-run the solver" message).

The plotter defaults to its dependency-light Matplotlib 3-D backend. For scenes
where plasma/coil occlusion matters, install the optional PyVista package and
select the VTK backend, which renders the 3-D panels with a z-buffer:

```bash
python -m pip install pyvista
python scripts/plot_equilibrium.py --state out.bin --backend pyvista \
  --coils path/to/coils.json --out equilibrium.png
```

Both backends write the same `_perspective.png`, `_top.png`, `_combined.png`,
and `_slices.png` files. The R-Z cross-sections and final figure composition
remain Matplotlib-based; with `--backend pyvista`, every embedded 3-D panel is
the VTK-rendered image. Displayed flux surfaces use the integer/full radial
grid. Their `|B|` colors are linearly interpolated from the staggered half grid;
the LCFS uses one-sided linear extrapolation from the last two half-grid values.
