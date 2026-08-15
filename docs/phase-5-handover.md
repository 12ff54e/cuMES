# cuMES Phase 5 handover — operator/workspace boundaries

Status date: 2026-08-15. Branch: `overhaul` (Phase 0 `bd26857` + Phase 1
`12bcc44` + Phase 2 `168170a` + Phase 3 `c21564c` + Phase 4 `1b0d099` + this
Phase 5 work, followed by the config/I-O wiring completion in §7 and the
kernel-signature migration + scalar references in §8). This document records
what Phase 5 of `docs/cuda-overhaul-blueprint.md` delivered, how it was
verified, what was deliberately deferred and why, and how the deferred items
were subsequently landed.

## 1. Scope

Phase 5 is the **operator/workspace boundaries**: turn the five per-stage
workspaces (profiles, Fourier, metric, preconditioner, constraint) into one
arena allocation with a reported liveness/peak, and introduce the typed
real-space views + operator interface contracts that the kernels will migrate
onto. The numerical path stays bit-for-bit unchanged throughout — every commit
is Class A.

This phase delivers three of the blueprint's four Phase-5 deliverables plus the
`DeviceArena`:

| Blueprint deliverable | Status |
| --------------------- | ------ |
| stage arena with reported liveness/peak memory | **Done** (DeviceArena + arena-backed workspaces + StageSolver report) |
| transform-only `SpectralOperator` | **Contract declared** (abstract interface; backends are Phase 7) |
| profiles/geometry/B/force/constraint/residual/preconditioner/descent interfaces | **Contracts declared** (header-only boundary types) |
| scalar CPU references + old/new dual-run hooks | **Done** (§8) |

The remaining Phase-5 items from the Phase-4 handover — full `SpectralView`
kernel-signature migration, config/I-O wiring in `main.cu` — are deferred with
specific guidance (§5); the config/I-O wiring has since been landed (§7).

## 2. What changed

### 2.1 `DeviceArena` (`include/cumes/runtime/device_arena.cuh`)

A host-side linear arena over one `cudaMalloc`'d backing store that carves
**named, aligned** subspans (`alloc_span<T>(name, count, align)`) and reports
per-category liveness/peak (`spans()`, `used_bytes()`, `peak_bytes()`,
`total_bytes()`). It throws `CumesError` on overflow (a too-small plan is a
loud setup error, never silent aliasing) and rejects non-power-of-two
alignments. Move resets the source.

### 2.2 Arena-backed workspaces

Every workspace array now allocates through `alloc_span` behind a
`DeviceArena* arena = nullptr` default on each `*Create`:

- `arena == nullptr` → the legacy per-array `cudaMalloc` path (all six tests
  that call `*Create` directly are unchanged).
- `arena != nullptr` → named subspans of one stage allocation; each struct
  carries `arena_backed = true` so its `*Free` frees only the non-arena
  resources (cuFFT plans, the pinned `h_faccon`) and resets the struct.

`StageSolver` now plans (`stage_arena_bytes<T>`, blueprint §6.5
`StageWorkspace::plan`), allocates one arena for the whole stage (profiles +
Fourier + metric + preconditioner + constraint), and reports the peak/liveness
after the solve:

```
stage arena: 122 spans, peak 71366160 bytes (68.06 MiB), reserved 71431688 bytes
```

One `cudaMalloc` per stage replaces ~110 per-array allocations.

### 2.3 Typed real-space views (`include/cumes/state/real_fields.cuh`)

`ReducedThetaView<T>` (a **distinct** reduced-theta quadrature view — never an
integer reinterpretation of a full-grid view, blueprint §4.1) plus the
aggregate bundles `GeometryParityViews` / `RadialProfileViews` /
`BaseGeometryHalfViews` / `MagneticFieldViews` / `ForceParityViews`. All are
trivially-copyable, `__host__ __device__`, and index bit-for-bit like the
legacy `surface*nZnT + zeta*ntheta + theta` layout.

### 2.4 Operator interface contracts

Header-only boundary declarations that establish the acyclic dependency DAG
(blueprint §5.1):

- `cumes/transforms/spectral_operator.hpp` — abstract `enqueue_inverse`/
  `enqueue_forward` (the Axisymmetric/ToroidalFft backends land in Phase 7).
- `cumes/physics/{profiles,geometry_operator,magnetic_field_operator,
  force_operator,constraint_operator}.hpp`.
- `cumes/numerics/{residual_operator,preconditioner,tridiagonal_backend,
  descent_operator,prolongation}.hpp`.

`tests/test_operator_views.cu` includes every interface header (the no-cycle
gate), round-trips a `ReducedThetaView` on device, and `static_assert`s the view
bundles are trivially copyable.

## 3. Key design decisions

1. **`arena_backed` flag, not a signature-only change.** A defaulted
   `DeviceArena* arena = nullptr` keeps the legacy path (and all six direct test
   call sites) bit-for-bit, while `StageSolver` opts into the arena. `*Free`
   branches on the flag so it can skip device `cudaFree` for arena-backed
   structs while still destroying cuFFT plans and the pinned `h_faccon`.
2. **`stage_arena_bytes` is the host-side plan.** The arena must be sized before
   the `*Create` calls carve it; the plan sums each module's exact span counts
   (mirroring the `alloc_span` calls) plus a 64 KiB alignment slack.
   `alloc_span` throws on overflow, so a miscount fails loudly at setup rather
   than corrupting the trajectory.
3. **cuFFT scratch is explicitly 16-byte aligned.** This was the one real bug
   this phase surfaced: `cudaMalloc`'s 256-byte alignment had masked cuFFT's
   16-byte requirement (`cufftDoubleComplex` is a 16-byte vector, and the real
   input of a double D2Z is processed as `double2` chunks). On the W7-X
   prescribed-current `deAlias d2z`, the 8-byte-aligned arena `double*` input
   returned `CUFFT_INVALID_VALUE`. Aligning the zeta scratch (both real and
   complex) to 16 bytes restores it; the other kernels have no alignment
   requirement beyond `alignof(T)`.
4. **Interfaces are contracts, not dead code.** The typed views are concrete and
   device-tested; the operator headers are compiled together in
   `test_operator_views.cu` (the acyclic-DAG gate). The kernels migrate onto
   these signatures in the follow-up (§5) — no numerical path depends on them
   yet, which is what keeps this phase Class A.

## 4. Verification

### Class A bitwise gate (the critical gate)

Fresh baseline at `overhaul/phase-4` (`1b0d099`), captured to
`.verify-scratch/baseline-phase5`; candidate trees captured from the Phase-5
tree and compared with `scripts/compare_bitwise.py`:

| Config | state | trajectory | step_0 | dump manifest |
| ------ | ----- | ---------- | ------ | ------------- |
| double/solovev | OK | OK | OK (10) | OK (235 files) |
| double/w7x    | OK | OK | OK (10) | OK (526 files) |

Both `PASS: byte-identical`. Effective-iteration counts reproduce the baseline
exactly (Solovev 251 → 199 → 456; W7-X 1877 → 1617 → 2011).

### Test matrix

| Preset | Result |
| ------ | ------ |
| `verify` (double, both backends) | **24/24** (17 unit + 7 compute-sanitizer memcheck) |
| `float` | **15/15** |

The float build compiles and its solver runs the same arena path; the double
build is the verification configuration (blueprint §1).

### No hot-loop allocation

The five workspaces now allocate once per stage (one arena `cudaMalloc`); the
solver's internal `d_f_spec`/`d_sq`/`d_psum`/`d_jac_stats`/`checkpoint` buffers
were already RAII `DeviceBuffer`s from Phase 3. `grep cudaMalloc src/*_impl.cuh`
now shows only the `arena == nullptr` legacy fallback branches, never a hot-loop
call.

## 5. Deferred (documented, not hidden)

Phase 5 is delivered as the **boundary layer**; the deferred items that complete
it — config/I-O wiring (§7), the full kernel-signature migration and the scalar
CPU references + dual-run hooks (§8) — have since landed. The one item that
remains **out of scope** for Phase 5:

- **`dynSharedBase()` removal** is still a Class B change (Phase-1 handover §3),
  not part of this phase.

## 6. Next steps (Phase 5 completion)

1. ~~Golden-test the versioned legacy-v0 writer against `outputSaveBinary`, then
   wire `main.cu` to `read_and_validate` + versioned writers + checkpoint
   reader (config/I-O wiring).~~ **Done** — see §7.
2. ~~Migrate the kernels onto the operator signatures module-by-module, with
   Class A bitwise verification after each, and add the scalar CPU references +
   dual-run hooks as the per-boundary gate.~~ **Done** — see §8.
3. Then Phase 6 (control-path performance): explicit nonblocking streams,
   one combined control fence, one-copy checkpoint (already one copy since
   Phase 3), fixed-iota update skip.

## 7. Config/I-O wiring (landed 2026-08-15)

The first §6.1 deferred item is delivered: `main.cu` now runs the Phase 2 host
layer end-to-end, with the numerical path and the on-disk `cumes_state.bin`
bytes unchanged (Class A).

- **Snapshot bridge** (`include/cumes/io/snapshot_bridge.cuh`): a single D2H
  copy of the contiguous `SpectralStorage` state slab converted T→double into a
  host `EquilibriumSnapshot` — the "Phase 3 snapshot bridge" the writers were
  designed to consume.
- **Golden byte-identity test** (`tests/test_io_golden.cu`): proves
  `LegacyBinaryV0Writer` ≡ `outputSaveBinary` byte-for-byte (both precisions),
  the prerequisite the handover demanded before the swap.
- **Config**: `read_and_validate` → `to_input_params()` replaces
  `initInputParams`; the float build declares `kMixedFloat` so validation
  rejects impossible tolerances.
- **Output**: binary goes through the versioned writers (`legacy-v0` default,
  `v1` via `--output-schema v1`); `.nc`/`.h5` keep the legacy device-reading
  backends.
- **Checkpoint**: `--restart <ckpt>` (v1) and `--restart-legacy <init>`
  (six-family) replace the removed `CUMES_LOAD_INIT` env path, with
  dimension/mode validation; `--checkpoint <path>` writes a v1 checkpoint after
  the solve for a resumable run.
- **Provenance**: the v1 writer records git revision/dirty/build-type, the input
  source hash, and GPU/driver/runtime/toolkit.

Verification: `test_io_golden` passes (both precisions); the full test matrix is
25/25; and `scripts/compare_bitwise.py` reports **byte-identical** for both
Solovev and W7-X (state, trajectory, step_0, and the full 235/526-file dump
manifest) against the Phase-5 baseline.

## 8. Kernel-signature migration + scalar references (landed 2026-08-15)

The remaining two deferred items are delivered, completing Phase 5's kernel
side. Every kernel now takes the typed views/bundles its operator contract
names, while keeping the flat-index arithmetic bit-for-bit unchanged (Class A).

- **Spectral side** (`SpectralView<T, Domain>`): the inverse/forward DFT
  (`inverseDFT`/`forwardDFT`), the constraint reconstruction
  (`constraintRzConCompute`), the preconditioner apply (`preconApply`/
  `tridiagSolveKernel`), and the solver's `extrapolateAxis`/`rzNorm`/`scalxc`/
  `m1Constraint`/`m1PreconScale`/`computeResiduals`/`descentStep` kernels all
  read/write through `SpectralComponent::Rcc..Lcs` and the domain tags
  (`PhysicalStateDomain`/`DecomposedVelocityDomain`/`DecomposedResidualDomain`).
- **Real-space side** (`GeometryParityViews`/`RadialProfileViews`/
  `BaseGeometryHalfViews`/`MagneticFieldViews`/`ForceParityViews`): `geometryKernel`
  and `forcesKernel` take the typed bundles (the kernels re-derive their raw
  pointers from the bundles, so the `surface*nZnT + zeta*ntheta + theta`
  arithmetic is unchanged).
- **Scalar CPU reference + dual-run** (blueprint §10.1): `tests/
  test_force_reference.cu` adds the missing local scalar oracle for the force
  operator (the transforms already had `cpuInvDFT` and the tridiagonal solve a
  Thomas reference) and a dual-run gate comparing the GPU kernel against the
  CPU reference on a frozen input across all 16 force families.

Verification: each migration commit is Class A — `scripts/compare_bitwise.py`
reports **byte-identical** on Solovev (235 dump files) and W7-X (526 dump files)
after every module — and the full test matrix is 26/26 (17 unit + 8 memcheck +
1 force reference). The force reference agrees to 5.7e-14 (axisymmetric) /
9.3e-10 (3D, fast-math vs IEEE) in double.
