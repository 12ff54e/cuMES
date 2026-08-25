# cuMES on-disk output formats

Human-readable description of every container the solver writes (and reads
back). The authoritative freeze lives in `configs/schema-v1.json` under
`x-cumes-on-disk-contracts` (blueprint §6.13); the implementations are
`src/cumes/io/` (`versioned_binary.cpp`, `checkpoint.cpp`,
`netcdf_writer.cpp`, `hdf5_writer.cpp`). All formats are little-endian.

## 1. Which container is written when

| Situation | Container | Magic |
| --- | --- | --- |
| `./build/cumes <in> out.bin` | versioned binary (schema v1, on-disk version 5) | `CUMES001` |
| a `.nc`/`.h5` suffix | NetCDF/HDF5 v1 (versioned, attributes) | — |
| `--checkpoint <path>` | versioned checkpoint (v4) | `CUMECKP1` |
| `--restart <path>` (read) | versioned checkpoint (v4; v1-v3 still read) | `CUMECKP1` |

An output path is always required; an unknown suffix is an error. All writers
publish atomically (write a temporary file in the target directory, then
rename over the destination; a failed write removes the temporary and leaves
the destination untouched).

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
period. The families reconstruct real space as (fourier_impl.cuh):

```
R = rmncc·cos(mθ)cos(nζ) + rmnss·sin(mθ)sin(nζ)
Z = zmnsc·sin(mθ)cos(nζ) + zmncs·cos(mθ)sin(nζ)
λ = lmnsc·sin(mθ)cos(nζ) + lmncs·cos(mθ)sin(nζ)
```

The axis row (`j = 0`) is the constant-extrapolated row, which the
comparison scripts intentionally skip (compare_states.py).

## 3. Versioned binary (schema v1, on-disk version 5)

```
magic     8 bytes  "CUMES001"
version   int32    = 5 (the current on-disk version)
ns        int32
mnmax     int32
families  6 * (mnmax*ns) doubles, mode-major     ← the §2 payload
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
          element; version 3+
```

The embedded input record (§5) mirrors
`ValidatedProblem::normalize_to_json()` field-for-field with the RESOLVED
values (the angular-grid defaults already applied): `mpol, ntor, nfp,
ntheta, nzeta, ncurr` (int32 each), `delt, phiedge, pres_scale,
adiabatic_index, spres_ped, bloat, curtor, tcon0` (f64 each), the `schema`
string, then the vectors `am, ac, ai, aphi, raxis_c, zaxis_s` (f64 each),
the input stages (`nstages_in` then per stage `ns` int32, `max_iter`
int32, `ftol` f64), the raw boundary (`rbc_m` int32, `rbc_n` int32,
`rbc_value` f64, then `zbs_*`), and the folded boundary `rbcc, rbss, zbsc,
zbcs` (f64 each), followed by the free-boundary fields `lfreeb`, `nvacskip`,
`mgrid_file`, `extcur`, `coils_file`, and `makegrid_parameters_file`. Every
vector is an int32 element count followed by the payload; counts are capped at
2^20 by the reader.

Notes:

- **String encoding:** `int32` byte length followed by the bytes, no NUL
  terminator. The reader caps a length prefix at 2^24 before allocating.
- **No `scalar_type` string:** the precision tag is authoritative and the
  reader reconstructs `scalar_type` from it.
- **Historical versions 1-4** remain readable: version 1 carried a
  `scalar_type` string between `build_type` and `source_path` and no
  precision-policy pair (the reader consumes and discards it); versions 1 and
  2 carry no input record, version 3 lacks free-boundary fields, and version 4
  lacks the inline-Makegrid source paths. Missing fields retain their defaults.
  (Before 2026-08-20 the writer emitted
  `scalar_type` while the v2 reader expected the policy pair — a
  writer/reader sequence mismatch fixed together with the v1-compat path
  above; the state payload itself was never affected.)
- The state payload is read and validated independently of the trailer, so
  a reader can stop after the state and stays forward-compatible with
  later trailer revisions. The trailer is parsed only when the caller
  requests the `RunReport`.

## 4. Versioned checkpoint v4 (`--checkpoint` / `--restart`)

```
magic     8 bytes  "CUMECKP1"
version   int32    = 4
precision int32    = 0 (the checkpoint is always double on disk)
ns        int32
mnmax     int32
families  6 * (mnmax*ns) doubles, mode-major     ← the §2 payload
params    the embedded normalized-input record (§5), version 2+
```

No provenance trailer beyond the input record — the restart path reads the
state only and never touches the record. Version 2 carries the base record,
version 3 adds free-boundary fields, and version 4 adds inline-Makegrid source
paths. Version-1 checkpoints (no record) remain readable. Header mismatch,
unsupported version/precision, or corruption is an error, never a silent cold
start.

## 5. The embedded input record / NetCDF-HDF5 layout (v1)

The embedded input record (§3/§4 in the binary formats) is represented
natively in NetCDF/HDF5:

- scalar variables (`mpol, ntor, nfp, ntheta, nzeta, ncurr` as int,
  `delt, phiedge, pres_scale, adiabatic_index, spres_ped, bloat, curtor,
  tcon0` as double) — HDF5: same names as root-group attributes;
- 1-D array variables `am, ac, ai, aphi, raxis_c, zaxis_s` (double) and
  the input stage arrays `stage_in_ns, stage_max_iter` (int),
  `stage_ftol` (double) — HDF5: same names as 1-D datasets;
- the `schema`, `mgrid_file`, `coils_file`, and
  `makegrid_parameters_file` string attributes;
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

## 6. Dump files (diagnostics, not a stable format)

`CUMES_DUMP`-gated per-step files (`step_*_rmncc.bin` etc. and the
`step_A_*` real-space arrays) are development diagnostics with no
stability contract; scripts that consume them (compare_bitwise.py) are
paired with the same dump producer. Everything stable is the containers
above.

## 7. Reading the state in Python

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
across revisions by design.) `scripts/plot_w7x.py` walks the full trailer
and parses the embedded input record, so a converged equilibrium plots
without its input JSON (containers written before the record are rejected
with a "re-run the solver" message).
