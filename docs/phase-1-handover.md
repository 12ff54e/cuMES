# cuMES Phase 1 handover — build and library split

Status date: 2026-08-13. Branch: `overhaul` (Phase 0 `bd26857` + this Phase 1
work). This document records what Phase 1 of `docs/cuda-overhaul-blueprint.md`
delivered, how it was verified, and what was deliberately deferred and why.

## 1. Scope

Phase 1 is the **build and library split**: restructure the monolithic nvcc
build into target-scoped libraries without changing any numerical result. The
blueprint's exit gate:

> no numerical source changes; Class A bitwise equivalence; all optional-backend
> build combinations compile and test.

That gate is met: the full dump-set tree (final state + per-iteration trajectory
record + initial-state snapshots + every `dump/cuMES/*` file — 1175 files across
four configs) is **byte-identical** to the Phase 0 baseline, and all four
NetCDF/HDF5 build combinations compile and pass their tests.

## 2. What changed

| Area | Change |
| ---- | ------ |
| Host-only sources | `src/input_json.cu`, `src/output.cu`, `src/output_netcdf.cu`, `src/output_hdf5.cu` → `.cpp`, now compiled by the C++ compiler. The output writers use only the CUDA runtime API (`cudaMemcpy`/`cudaMemcpy2D`), so they need `CUDA::cudart` + the CUDA include path, not nvcc. `vmec_types.h` is now self-contained for `M_PI` (nvcc pre-defined it; g++ does not). |
| CUDA instantiation split | Each of the 8 device modules (`fourier geometry forces solver profiles precon constraint refine`) split into `src/<mod>_impl.cuh` (template definitions) + `src/<mod>_double.cu` / `src/<mod>_float.cu` (one explicit-instantiation TU per scalar type). |
| Build targets | `cumes_config` (JSON parser), `cumes_io` (writers + optional backends), `cumes_cuda_double` / `cumes_cuda_float` (device operators), `cumes_test_support` (shared test helpers), `cuMES` (CLI). The CLI links only the TU matching `Real`; tests link both. |
| cmake/ modules | `CumesOptions`, `CumesCudaArchitectures`, `CumesDependencies`, `CumesWarnings`, `CumesTest`, `CumesSanitizers`. |
| CMakePresets.json | `verify` (double/default), `float`, `debug`, `sanitizer`, `nobackend`, `profiling`. |
| Test support | `tests/support/cumes_test_support.cuh` — the `checkCuda`/`cc` CUDA error check that every kernel-driving test duplicated. |

The net effect on the CLI binary: it now links only one scalar type, so
`build/cuMES` dropped from ~4.8 MiB to ~2.4 MiB and the device modules are
compiled once into a static library instead of once per consuming target.

## 3. Key finding — `dynSharedBase()` retained (removal is Class B)

The blueprint lists "removes the `dynSharedBase()` workaround" as a Phase 1
intent, because the explicit instantiation split makes it *unnecessary* (one
scalar type per TU means a direct `extern __shared__ T smem[]` is legal). I
implemented and A/B-tested that removal. **It is a Class B change, so it was
reverted and the workaround retained.**

- With `dynSharedBase()` removed (direct `extern __shared__ T smem[]`), the
  Solovev trajectory shifts by up to ~1.5e-10 — a `--use_fast_math` FMA-fusion
  reordering, not a logic bug. The opaque `void*` return of `dynSharedBase()`
  gives nvcc a conservative aliasing view; the direct known-array form lets it
  fuse multiply-adds differently.
- With `dynSharedBase()` **retained** (and the TU split in place), all four
  configs are byte-identical to the baseline — the split itself is Class A.
- Phase 1's gate is "no numerical source changes", so the removal belongs in a
  later Class B phase with a re-frozen baseline. The four impl headers carry a
  NOTE comment recording this so it is not "cleaned up" by accident.

## 4. Verification

### 4.1 Class A bitwise gate (full dump set)

`scripts/capture_baseline.sh` re-captured the baseline at `bd26857`, then the
candidate after Phase 1. `scripts/compare_bitwise.py` (default `--essentials`
mode, which also verifies every `dump/cuMES/*` file against the manifest):

| Config | state | trajectory | step_0 | dump manifest |
| ------ | ----- | ---------- | ------ | ------------- |
| double/solovev | OK | OK | OK (10) | OK (235 files) |
| double/w7x    | OK | OK | OK (10) | OK (526 files) |
| float/solovev | OK | OK | OK (10) | OK (224 files) |
| float/w7x     | OK | OK | OK (10) | OK (190 files) |

All `PASS: byte-identical`.

### 4.2 Optional-backend matrix (none / netcdf / hdf5 / both)

| Combination | ctest | suffix dispatch |
| ----------- | ----- | --------------- |
| both (default `build`)          | 14/14 (9 unit + 5 sanitizer) | `.nc`/`.h5`/`.bin` all produced |
| nobackend (`build-nobackend`)   | 9/9 | `.nc` rejected before solve (exit 1) |
| netcdf-only (`build-netcdf-only`) | 9/9 | `.nc` produced, `.h5` rejected |
| hdf5-only (`build-hdf5-only`)   | 9/9 | `.h5` produced, `.nc` rejected |

### 4.3 Tests

- Double build: **14/14** (9 unit + 5 compute-sanitizer memcheck variants).
- Float build: **9/9** unit (sanitizer variants off there).

## 5. Rebuild instructions

```bash
cmake --preset verify && cmake --build build -j            # double (default)
cmake --preset float  && cmake --build build-float -j      # single precision
cmake --preset nobackend && cmake --build build-nobackend -j  # binary-only
ctest --test-dir build                                     # run tests
```

## 6. Notes and deferred items

- **Stale `build-float` cache.** `build-float/CMakeCache.txt` carried
  `CMAKE_CUDA_ARCHITECTURES=52` from a long-forgotten manual configure. The old
  CMakeLists *forced* the arch each configure, hiding it; the new
  user-overridable default (61;75;80;86;89) respects the cache, which surfaced
  it as an `atomicAdd(double)` overload error on sm_52. Reconfigure with
  `-DCMAKE_CUDA_ARCHITECTURES="61;75;80;86;89"` (or wipe the dir). Fresh
  presets are unaffected.
- **`dynSharedBase()` removal** — deferred to a Class B phase (see §3).
- **Test-support scope** — only the bit-for-bit-identical `checkCuda`/`cc` was
  extracted. The CPU scalar references (`cpuInvDFT`, `thomasSolve`,
  `cpuDealiasBandpass`, …) are each used by exactly one test, so there is
  nothing to share yet; they move into `cumes_test_support` when a second
  consumer appears.
- **Not committed (by design)** — the untracked `AGENTS.md` symlink and the
  downloaded CUDA-guide markdown under `docs/` (per the Phase 0 handover §7),
  plus the gitignored `.verify-scratch/` trees.

## 7. Next steps (Phase 2)

Phase 2 is the validated host model + versioned I/O (`ProblemSpec`,
`ValidatedProblem`, `GridShape`, `ModeTable`, `PrecisionPolicy`, `OutputSpec`,
`RunReport`, `EquilibriumSnapshot`, versioned writers and legacy adapters). It
now builds on a frozen, byte-comparable Phase 1 baseline with the same
`compare_bitwise.py` Class A gate.
