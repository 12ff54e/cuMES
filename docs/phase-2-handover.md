# cuMES Phase 2 handover — validated host model and versioned I/O

Status date: 2026-08-13. Branch: `overhaul` (Phase 0 `bd26857` + Phase 1
`12bcc44` + this Phase 2 work). This document records what Phase 2 of
`docs/cuda-overhaul-blueprint.md` delivered, how it was verified, and what was
deliberately deferred and why.

## 1. Scope

Phase 2 is the **validated host model and versioned I/O**: replace the
fixed-capacity, nvcc-compiled `InputParams` (`include/input.h`) and the ad-hoc
device-pointer writers with a typed, validated, host-only configuration model
and a versioned I/O layer — without touching the frozen numerical path. Per the
user's decision, this phase is **library + tests only**: `main.cu` and the CUDA
device path stay on the legacy code unchanged; integration lands in Phases 3–5.

The blueprint's exit gate:

> normalized Solovev/W7-X configuration goldens pass; new and legacy outputs
> round-trip; malformed-input and I/O-failure matrix passes.

That gate is met (see §4).

## 2. What changed

### New `namespace cumes` library (`include/cumes/`, `src/cumes/`)

| Layer | Files | Contents |
| ----- | ----- | -------- |
| core | `core/{result,scalar,checked_size,grid_shape,mode_table}.hpp` + `src/cumes/core/*.cpp` | `BasicResult<T,E>`/`Status`, `ScalarType`, checked size_t arithmetic, extents-only `GridShape` with resolved-shape validation, per-mode `ModeTable<T>` (physical_n, mn_scale, xmpq, parity, first_surface). |
| config | `config/{problem_spec,precision_policy,validation_report,solver_options,validated_problem,json_reader}.hpp` + `src/cumes/config/*.cpp` | dynamic `ProblemSpec`, `PrecisionPolicy` with tolerance floors, `ValidationReport` (collects all findings), four-stage `validate()` (parse→validate→fold→resolve), legacy `to_input_params()` bridge, `normalize_to_json()`, and the JSON reader. |
| io | `io/{output_spec,run_report,equilibrium_snapshot,writer,reader,checkpoint}.hpp` + `src/cumes/io/*.cpp` | `OutputSpec`/`OutputFormat`/`OutputSchema`, `RunReport` with full stage history + provenance, host `EquilibriumSnapshot`, `Writer`/`Reader` interfaces, legacy binary v0, versioned binary v1, versioned checkpoint + legacy init converter. |

### Build targets (added to `CMakeLists.txt`)

- `cumes_core` — core host model (static).
- `cumes_json` — the JsonParser implementation, extracted into a single TU
  shared by the legacy parser and the new reader (removes a double-definition
  when both `input_json.cpp` and `json_reader.cpp` instantiated the header).
- `cumes_config_json` — config model + JSON reader.
- `cumes_io_host` — host I/O (no CUDA).

### Tests (host-only `.cpp`, compiled by g++)

- `test_host_config` (unit;config) — normalization goldens, adapter parity,
  mode table, precision floor, malformed-input matrix.
- `test_host_io` (unit;io) — output-spec dispatch, binary v0 byte layout,
  binary v0/v1 round-trips, writer failure matrix.
- `test_checkpoint` (unit;io) — checkpoint round-trip + rejection, legacy init
  converter.

## 3. Key design decisions

1. **Dynamic model, no fixed capacities.** `ProblemSpec`/`ValidatedProblem`
   replace the fixed `InputParams` caps (8 stages, 16 coefficients, 256
   boundary entries, 32 axis entries) with `std::vector`. A consequence: an
   oversized schedule (>8 stages) is now *valid* in the new model (it was
   rejected by the legacy capacity); the legacy `to_input_params()` bridge
   reports an error only when the validated problem exceeds a legacy capacity,
   which the shipped configs never do.
2. **Adapter parity is the correctness anchor.** `ValidatedProblem::
   to_input_params()` must be field-for-field identical to the legacy
   `initInputParamsFromJson()`; `test_host_config` proves this for both shipped
   configs across every field (scalars, profiles, axis, raw + folded boundary,
   stage schedule).
3. **Findings are collected, not thrown.** The legacy parser threw on the first
   error; the new reader records every type error, integer narrowing,
   unsupported feature, and unknown key into a `ValidationReport`. Unknown keys
   are a warning in compatibility mode and an error in `strict_schema` mode.
4. **Byte-exact legacy binary v0.** The v0 container is the exact contract
   (`int32 ns`, `int32 mnmax`, six mode-major double families); the reader is
   strict about truncation and trailing data. The versioned v1 container is
   self-describing (magic + version + state payload + a provenance trailer the
   reader can skip, so it stays forward-compatible).
5. **Versioned checkpoint replaces `CUMES_LOAD_INIT`.** `write_checkpoint`/
   `read_checkpoint` validate magic/version/dimensions; `convert_legacy_init`
   reads the legacy six-family `vmecpp_init.bin` payload. The legacy env-var
   path in `main.cu` remains until Phase 3 wires `--restart`.

### Deferred (documented, not hidden)

- **NetCDF/HDF5 host adapters.** The existing device-pointer
  `output_netcdf.cpp`/`output_hdf5.cpp` remain the legacy NetCDF/HDF5 path. The
  host-snapshot NetCDF/HDF5 writers/readers are deferred to Phase 3, because a
  faithful v0 writer needs the config (padded provenance) which the
  snapshot-only `Writer` interface does not carry, and the contiguous
  device-state snapshot bridge lands in Phase 3. The `OutputFormat::kNetCdf`/
  `kHdf5` enums and `output_format_available()` are in place for that wiring.
- **`RuntimeCapabilities` probing** — deferred to Phase 3 (no device probing in
  the host model yet).
- **`DeviceParams<T>` packing** — Phase 3.

## 4. Verification

### Test matrix

| Preset | Result |
| ------ | ------ |
| `verify` (double, both backends) | **17/17** (12 unit + 5 sanitizer) |
| `nobackend` (binary only)        | **12/12** |
| `float` (new host tests)         | test_host_config / test_host_io / test_checkpoint all pass |

The new host tests are type-agnostic (all host data is double), so they are
identical across the float/nobackend matrix; the NetCDF/HDF5 availability
preflight degrades correctly to binary-only in `nobackend`.

### Config gate

- `normalize_to_json()` matches the checked-in `tests/fixtures/{solovev,w7x}.normalized.json`
  goldens (generated once by `test_host_config --emit-golden`).
- `to_input_params()` is field-identical to `initInputParamsFromJson()` for both
  shipped configs.
- The malformed-input matrix (nonzero gamma, out-of-range boundary modes
  skipped with a warning, empty/mismatched/non-monotonic schedules, integer
  narrowing, wrong-type aux/asym keys, unsupported physics, unknown-key
  strict/compat) all pass.

### I/O gate

- Legacy binary v0: exact byte layout + round-trip.
- Versioned binary v1: round-trip + bad-magic rejection.
- Checkpoint: round-trip + magic/truncation rejection; legacy init converter +
  header-mismatch rejection.
- Writer failure matrix: open failure and rename-over-directory return errors
  and leave the target untouched.

### Class A safety

No device/solver source changed. The frozen trajectories and
`scripts/compare_bitwise.py` baseline are untouched by construction; the only
legacy source touched is `src/input_json.cpp` (removed one `#define`, now
linking the shared `cumes_json` parser — build-only, no behavior change,
re-verified by the existing `test_input_json` passing unchanged).

### Adversarial review

An adversarial three-reviewer pass over the new code found and fixed: a
negative `niter_array`/`ns_array` entry wrapping to `size_t` and bypassing
validation; readers `bad_alloc`-ing on a mismatched-format/corrupt header
(bounded against the file size); a present-but-empty `raxis_c`/`zaxis_s` being
silently zero-padded instead of rejected; dropped validation warnings; a
signed-overflow in `GridShape::modes()`; and unbounded `read_string`. Each has
a regression test.

## 5. Next steps (Phase 3)

Phase 3 is **RAII buffers, typed views, and exact current layouts**: `DeviceContext`,
buffer/stream/event RAII, typed `SpectralView`/`RealFieldsView`, contiguous
state/velocity slabs, and a state-only checkpoint slab, with legacy kernels
wrapped behind views. It can now also:
- wire `main.cu` to parse via `read_and_validate` and to write via the versioned
  writers + checkpoint reader (replacing `CUMES_LOAD_INIT`);
- add the host NetCDF/HDF5 v0/v1 writers once the snapshot bridge exists.
