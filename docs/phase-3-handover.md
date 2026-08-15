# cuMES Phase 3 handover — RAII buffers, typed views, exact current layouts

Status date: 2026-08-15. Branch: `overhaul` (Phase 0 `bd26857` + Phase 1 `12bcc44`
+ Phase 2 `168170a` + this Phase 3 work). This document records what Phase 3 of
`docs/cuda-overhaul-blueprint.md` delivered, how it was verified, and what was
deliberately deferred and why.

## 1. Scope

Phase 3 is the **first device-side change with no arithmetic change**: replace raw
owning device pointers with RAII, introduce the typed views, and co-locate the
spectral state/velocity into contiguous slabs with a state-only checkpoint. Per the
user's decision, this pass is **runtime + views + slabs only**: `main.cu` stays on
the legacy parse/output path (config/I-O wiring is Phase 4–5), the legacy kernels
keep their raw-pointer signatures (full `SpectralView` migration is Phase 5), and
error handling is centralized into a shared `check_cuda`/`check_cufft` that throws
`CumesError`.

The blueprint's exit gate:

> Class A bitwise equivalence, zero hot-loop allocations, Compute Sanitizer clean.

That gate is met (see §4).

## 2. What changed

### New `cumes` runtime + views + state slab (header-only where templated)

| Layer | Files | Contents |
| ----- | ----- | -------- |
| runtime | `include/cumes/runtime/{cuda_status,device_buffer.cuh,pinned_buffer,stream,event,device_context}.hpp/.cuh` + `src/cumes/runtime/device_context.cpp` | `CumesError` + throwing `check_cuda`/`check_cufft` (with a cuFFT enum→string table; `cufftGetErrorString` does not exist); movable non-copyable `DeviceBuffer<T>`/`PinnedBuffer<T>`; `Stream`/`Event`/`DeviceContext` RAII. |
| core | `include/cumes/core/tensor_view.cuh` | `SpectralComponent` enum, domain tags, `SpectralView<T,Domain>` (`[component][mode][surface]`, surface-contiguous) and `RealFieldView<T>` (`[surface][zeta][theta]`, theta-contiguous), both `__host__ __device__`. |
| state | `include/cumes/state/spectral_storage.hpp` | `SpectralStorage<T>`: owns two 6·mnmax·ns slabs (state + velocity) and exposes `legacy_view()`, `physical()`/`velocity()` views, and the owning buffers for the checkpoint. |

### Build target

- `cumes_cuda_runtime` (STATIC), linked by `cumes_cuda_double`/`cumes_cuda_float`
  and `cuMES`.

### Migration of the allocating call sites

- `src/main.cu`: `initState` returns a `SpectralStorage<Real>` (cold-start/loadInit
  logic unchanged; the 12 `cudaMalloc` + 6 velocity `cudaMemset` are replaced by the
  zero-initialized slab); `freeState` deleted. The stage loop owns a move-assigned
  `SpectralStorage`, and the solve+output body is wrapped in `try/catch (CumesError)`.
- `include/refine.cuh` + `src/refine_impl.cuh`: `interpolateState` returns a
  `SpectralStorage` (allocates + zeroes the new slab once, launches the unchanged
  `interpolateStateKernel`).
- `include/solver.cuh` + `src/solver_impl.cuh`: `solverRun(SpectralStorage<T>&, …)`
  derives `SpectralState<T> st = storage.legacy_view()` once (all kernels/consumers
  keep their pointer arithmetic). The six `d_bk_*` backup arrays become one
  `DeviceBuffer<T> checkpoint`, so `backupState`/`restoreState` are one
  `cudaMemcpy` + one velocity `cudaMemset`. The remaining scratch (`d_f_spec`,
  `d_sq`, `d_psum`, `d_rzsum`, `d_jac_stats`) and pinned mirrors (`h_jac_stats`,
  `h_sq_i_pin`, `h_sq_pin`) are `DeviceBuffer`/`PinnedBuffer`.
- `tests/test_force_verify.cu`: constructs a `SpectralStorage` for `solverRun`.

### Centralized error handling

- All twelve modules' duplicated `checkCuda`/`cc`/`ccf`/`checkCufft` helpers are
  replaced with `cumes::check_cuda`/`cumes::check_cufft` (throw `CumesError`); `main`
  catches it at the boundary. Error-path-only, so Class A safe.
- `tests/support/cumes_test_support.cuh` is intentionally left as-is (test infra).

### New test

- `tests/test_runtime.cu` (`unit;runtime`): `DeviceBuffer`/`PinnedBuffer` alloc/zero/
  copy/move; `Stream`/`Event`/`DeviceContext`; `SpectralStorage` slab-offset layout
  (the 12 legacy pointers == slab + {0..5}·mnmax·ns); a device round-trip through
  `SpectralView`; `check_cuda`/`check_cufft` error injection. Plus a `static_assert`
  that `SpectralComponent` and `EquilibriumSnapshot::Component` share one order.

## 3. Key design decisions

1. **`SpectralState<T>` stays a non-owning 12-pointer view.** `vmec_types.h` must
   stay CUDA-free (it is transitively included by host-only `cumes_config` via
   `input.h`), so the owning slab lives in `cumes::SpectralStorage` instead of in
   `SpectralState`. `legacy_view()` points the 12 pointers into the slabs, so every
   existing consumer and kernel keeps its indexing and arithmetic unchanged —
   bitwise identical, and the whole phase compiles against 16 consumer files with
   zero kernel-signature churn.
2. **Contiguous slab order is the contract.** The slab concatenation is exactly
   `Rcc Zsc Lsc Rss Zcs Lcs` — the same order as `d_f_spec` and
   `EquilibriumSnapshot` — so the six old per-family copies become one, and the
   component-major layout matches the forward-DFT residual slab.
3. **Single-copy checkpoint.** `backupState`/`restoreState` collapse from 6×
   `cudaMemcpy` (+ 6× `cudaMemset`) to one copy + one memset, because the slab order
   matches the old six-copy order.
4. **Error handling is centralized but keeps `main` as the only catch.** No library
   calls `exit`; a CUDA/cuFFT failure throws `CumesError`, caught once at the CLI.
5. **Streams are introduced, not wired.** The solver keeps the legacy default stream
   and `cudaStreamSynchronize(0)`; `DeviceContext::compute_stream()` exists and is
   tested but not consumed (that is Phase 6A). `cufftSetStream` is deliberately not
   added.

### Diagnostic-determinism fix (required for a clean dump-set gate)

`step_A_l_real_iter_1.bin` dumps `fp.d_l_real` *before* `fourierCombineParity`
produces it — a latent dump of uninitialized memory. Its bytes depended on the CUDA
allocator state, so the slab re-allocation (this phase's very purpose) shifted them
for Solovev. The nine combined `*_real` buffers are now zero-initialized in
`fourierCreate`, making that dump deterministic. This is a diagnostic-only change
(the combined buffers are never read by the numerical kernels, which use the parity
arrays); the numerics are unaffected.

## 4. Verification

### Test matrix

| Preset | Result |
| ------ | ------ |
| `verify` (double, both backends) | **19/19** (13 unit + 6 compute-sanitizer memcheck) |
| `float`  | **13/13** |
| `nobackend` | **13/13** |
| `netcdf-only` | **13/13** |
| `hdf5-only` | **13/13** |

### Class A bitwise gate (the critical gate)

Captured a fresh baseline at HEAD **plus only the diagnostic fix** (`.verify-scratch/
baseline-phase3`) and compared against the full Phase-3 tree (`.verify-scratch/
candidate-phase3`) with `scripts/compare_bitwise.py` (default `--essentials` mode,
which also verifies every `dump/cuMES/*` file against the manifest):

| Config | state | trajectory | step_0 | dump manifest |
| ------ | ----- | ---------- | ------ | ------------- |
| double/solovev | OK | OK | OK (10) | OK (235 files) |
| double/w7x    | OK | OK | OK (10) | OK (526 files) |

Both `PASS: byte-identical`. Against the pre-fix Phase-0 baseline
(`.verify-scratch/baseline-head`, `bd26857`), the float configs reproduce the same
picture: `cumes_state.bin` + `per_iter_residuals_cumes.bin` + `step_0` byte-identical,
and only the formerly-uninitialized `step_A_l_real_iter_1.bin` differs (now zeros) —
float/w7x is byte-identical across all 190 files.

### No hot-loop allocation

`grep cudaMalloc|cudaFree|cudaMallocHost|cudaFreeHost src/solver_impl.cuh` → none;
all allocations are RAII objects constructed before the iteration loop.

## 5. Deferred (documented, not hidden)

- Wiring `main.cu` to `read_and_validate` + versioned writers + checkpoint reader
  (Phase 4–5).
- Full kernel-signature migration to `SpectralView` (Phase 5); `DeviceArena` and the
  `*Create`/`*Free` workspace structs → RAII (Phase 5).
- Explicit nonblocking-stream execution / `cufftSetStream` / one combined control
  fence (Phase 6A).
- `dynSharedBase()` removal (Class B, per Phase-1 handover §3).

## 6. Next steps (Phase 4)

Phase 4 is the **pure controller and observers**: deterministic `IterationController`
+ `ControlRecord`, `StageSolver`/`MultigridSolver` separation, structured scalar
telemetry, and versioned lazy snapshots — with the recorded residual histories
reproducing the exact restart/damping decisions, and observers provably unable to
change state hashes or the iteration count.
