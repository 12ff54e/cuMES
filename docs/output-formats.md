# cuMES on-disk output formats

Human-readable description of every container the solver writes (and reads
back). The authoritative freeze lives in `configs/schema-v1.json` under
`x-cumes-on-disk-contracts` (blueprint §6.13); the implementations are
`src/cumes/io/` (`legacy_binary_v0.cpp`, `versioned_binary.cpp`,
`checkpoint.cpp`, `netcdf_writer.cpp`, `hdf5_writer.cpp`). All formats are
little-endian.

## 1. Which container is written when

| Situation | Container | Magic |
| --- | --- | --- |
| `./build/cuMES <in> out.bin` (default output schema) | legacy binary v0 | none |
| `./build/cuMES <in> out.bin --output-schema v1` | versioned binary (schema v1, on-disk version 2) | `CUMES001` |
| `--output-schema v1` + a `.nc`/`.h5` suffix | NetCDF/HDF5 v1 (versioned, attributes) | — |
| legacy `.nc`/`.h5` output (default schema) | NetCDF/HDF5 v0 (fixed capacities) | — |
| `--checkpoint <path>` | versioned checkpoint (v1) | `CUMECKP1` |
| `--restart <path>` (read) | versioned checkpoint (v1) | `CUMECKP1` |
| `--restart-legacy <path>` (read) | legacy six-family init payload | none |

All writers publish atomically (write a temporary file in the target
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
period. The families reconstruct real space as (fourier_impl.cuh):

```
R = rmncc·cos(mθ)cos(nζ) + rmnss·sin(mθ)sin(nζ)
Z = zmnsc·sin(mθ)cos(nζ) + zmncs·cos(mθ)sin(nζ)
λ = lmnsc·sin(mθ)cos(nζ) + lmncs·cos(mθ)sin(nζ)
```

The axis row (`j = 0`) is the constant-extrapolated row, which the
comparison scripts intentionally skip (compare_states.py).

## 3. Legacy binary v0 (`cumes_state.bin`)

```
int32 ns
int32 mnmax
double rmncc[mode][surface]
double zmnsc[mode][surface]
double lmnsc[mode][surface]
double rmnss[mode][surface]
double zmncs[mode][surface]
double lmncs[mode][surface]
```

No magic, no provenance — the state payload alone. The reader is strict:
truncated files are errors, trailing bytes after the six families are an
error, and the header dimensions are validated against the actual file size
before any allocation (a wrong-format file decodes as enormous dimensions;
this bound is what stops it).

## 4. Versioned binary (schema v1, on-disk version 2)

```
magic     8 bytes  "CUMES001"
version   int32    = 2 (the current on-disk version)
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
```

Notes:

- **String encoding:** `int32` byte length followed by the bytes, no NUL
  terminator. The reader caps a length prefix at 2^24 before allocating.
- **No `scalar_type` string:** the precision tag is authoritative and the
  reader reconstructs `scalar_type` from it.
- **Historical version 1** (written by earlier builds, still readable):
  the trailer carried a `scalar_type` string between `build_type` and
  `source_path` and no precision-policy pair; the reader consumes and
  discards it. (Before 2026-08-20 the writer emitted `scalar_type` while
  the v2 reader expected the policy pair — a writer/reader sequence
  mismatch fixed together with the v1-compat path above; the state payload
  itself was never affected.)
- The state payload is read and validated independently of the trailer, so
  a reader can stop after the state and stays forward-compatible with
  later trailer revisions. The trailer is parsed only when the caller
  requests the `RunReport`.

## 5. Versioned checkpoint v1 (`--checkpoint` / `--restart`)

```
magic     8 bytes  "CUMECKP1"
version   int32    = 1
precision int32    = 0 (the checkpoint is always double on disk)
ns        int32
mnmax     int32
families  6 * (mnmax*ns) doubles, mode-major     ← the §2 payload
```

No provenance trailer — the checkpoint records state only. Header
mismatch, unsupported version/precision, or corruption is an error, never
a silent cold start.

## 6. Legacy init payload (`--restart-legacy`)

The pre-overhaul `CUMES_LOAD_INIT` / `vmecpp_init.bin` six-family payload:

```
int32 ns
int32 mnmax
6 * (mnmax*ns) doubles
```

`convert_legacy_init` validates the header against the expected
`(ns, mnmax)` of the run before producing a checkpoint snapshot.

## 7. NetCDF / HDF5

**v0** (the default schema for `.nc`/`.h5`): the legacy fixed-capacity
layout — dimensions `ngrids=8`, `ncoeff=16`, `naxis=32`, `nbm=nbn=16`
with the active run using only part of them; the state families are
2-D datasets over `[ns, mnmax]` with the logical value
`family[surface, mode]` mapped to device offset `surface + mode * ns`.
Byte-exactness of the NetCDF v0 output is pinned by the full-run
compare_bitwise gate; the HDF5 contract is structural (libhdf5 embeds a
per-second timestamp).

**v1** (`--output-schema v1`): versioned containers carrying the same
provenance as the binary trailer — scalar variables
`precision`/`status`/`total_iterations`/`build_dirty`, string attributes
`revision`, `build_type`, `precision_policy`, `compile_flags`,
`source_path`, `source_hash`, `gpu_name`, `driver`, `runtime`, `toolkit`,
and per-stage variables `stage_ns`/`stage_iterations`/`stage_converged`/
`stage_fsqr`/`stage_fsqz`/`stage_fsql` plus the restart metadata
`restart_stage_offset`/`restart_iteration`. Readers prove exact
rank/datatype/extent before every read into a fixed or sized buffer
(see the reader-rank hardening in overhaul-history.md); malformed shapes
are rejected, never tolerated.

## 8. Dump files (diagnostics, not a stable format)

`CUMES_DUMP`-gated per-step files (`step_*_rmncc.bin` etc. and the
`step_A_*` real-space arrays) are development diagnostics with no
stability contract; scripts that consume them (compare_bitwise.py) are
paired with the same dump producer. Everything stable is the containers
above.

## 9. Reading the state in Python

`scripts/compare_states.py` is the reference reader for the v0 payload:

```python
ns, mnmax = struct.unpack("<ii", f.read(8))
fams = {name: struct.unpack(f"<{ns*mnmax}d", f.read(8*ns*mnmax))
        for name in ("rmncc", "zmnsc", "lmnsc", "rmnss", "zmncs", "lmncs")}
```

The same loop skips the 20-byte header (magic + version) and reads the
versioned binary's payload directly; only a consumer that wants
provenance needs the trailer.
