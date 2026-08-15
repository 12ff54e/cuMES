# cuMES Phase 6 handover — low-risk control-path performance

Status date: 2026-08-15. Branch: `overhaul` (Phase 0 `bd26857` + Phase 1
`12bcc44` + Phase 2 `168170a` + Phase 3 `c21564c` + Phase 4 `1b0d099` + Phase 5
`2b9aaf8` + this Phase 6 work). This document records what Phase 6 of
`docs/cuda-overhaul-blueprint.md` delivered, how it was verified, and what was
deliberately deferred and why.

## 1. Scope

Phase 6 is the **low-risk control-path performance** milestone: move the
solver onto an explicit nonblocking stream and collapse the five ordinary-path
host barriers per iteration into one deliberate control fence, without
changing a single arithmetic result. The blueprint splits the phase into

- **6A — existing plans/order** (bitwise where execution order is preserved):
  explicit nonblocking stream + cuFFT stream binding, remove timing fences,
  one combined residual/control fence, one-copy state checkpoint, fixed-iota
  update skip, complete recover writes (no force memset);
- **6B — Class B candidates** (per-operator/trajectory bounds): device-only
  force-normalization reduction, explicit/replanned shared cuFFT work area,
  event-DAG scratch reuse.

This phase delivers **all six 6A items**; 6B is deferred with specific
guidance (§5). Every commit is Class A — bitwise-identical trajectory.

| Blueprint 6A deliverable | Status |
| ------------------------ | ------ |
| Explicit nonblocking stream + cuFFT stream binding | **Done** (commit `945b776`) |
| Remove timing fences | **Done** (commit `8e47ce8`) |
| One combined residual/control fence | **Done** (commit `36403ad`) |
| One-copy state checkpoint | **Already done** since Phase 3 (re-verified, made stream-ordered) |
| Fixed-iota update skip | **Done** (commit `8a53ced`) |
| Complete recover writes (no force memset) | **Done** (commit `bb1281f`) |

## 2. What changed

### 2.1 Explicit nonblocking compute stream + cuFFT binding (`945b776`)

A single `cumes::Stream` (nonblocking) is created once in `main.cu` and
threaded through the whole run: `MultigridSolver::run` → `StageSolver::run` →
`solverRun` → every hot-loop operator (`computeGeometry`, `computeJacobianStats`,
`computeForceNormPartials`, `inverseDFT`/`forwardDFT`/`fourierCombineParity`,
`computeForces`, the three constraint functions, `preconCompute`/`preconApply`,
`interpolateState`). Each operator gained a defaulted `cudaStream_t stream = 0`
parameter, so the six tests that call operators directly are unchanged; the
explicit-instantiation `.cu` files were updated to match.

- **cuFFT plans are bound with `cufftSetStream`** once per stage (the Fourier
  z2d/d2z plans and the constraint deAlias d2z/z2d + rCon/zCon z2d plans), so
  the batched ζ-transforms execute in stream order with the surrounding kernels.
- **Stage setup stays synchronous** on the default stream (`profilesCreate`,
  `fourierCreate`, `metricCreate` are blocking H2D/plan work); they complete
  before the solve, so the compute stream reads them safely. `interpolateState`
  (prolongation between stages) runs on the same compute stream so it is
  ordered before the next stage.
- **State checkpoint/restore became stream-ordered**: `DeviceBuffer` gained
  `zero_async` / `copy_from_async`, and the solver's `backupState`/`restoreState`
  now use `cudaMemcpyAsync`/`cudaMemsetAsync` on the compute stream (they run
  after the descent kernel on the same stream, so a blocking default-stream
  copy would have raced).
- **Dump/print reads sync first**: `dumpDeviceArray` (compile+run-time gated)
  and `axisRAtZeta0` synchronize the compute stream before their default-stream
  D2H reads, preserving the exact dump/telemetry bytes.

The fence *count* is unchanged in this commit — it is a pure order-preserving
scheduling change (same kernels, same order, new stream).

### 2.2 Remove timing fences (`8e47ce8`)

The two `cudaEventSynchronize` host barriers that bracketed `inverseDFT` and
`forwardDFT` are gone. The four timing events (`ev0/ev1_inv`, `ev0/ev1_fwd`)
are still recorded on the compute stream every pass, but their elapsed times
are sampled only at the already-required control fence (both transforms precede
it on the same stream). Two host barriers per pass → zero.

### 2.3 Complete recover writes — no force memset (`bb1281f`)

`forwardRecoverKernel` now writes **all six** spectral-force families for every
`(mode, surface)`, including explicit zeros for the axis (m>0 rows and the four
non-`frcc`/`fzcs` families) and the LCFS (non-λ families) that the old pre-zero
used to cover. The forward DFT's `cudaMemset` of the full `6·mnmax·ns` residual
slab is removed — one full-slab device write per iteration eliminated. The
recover output is bit-identical (the memset and the explicit zeros write the
same values).

### 2.4 Fixed-iota update skip (`8a53ced`)

`computeGeometry` gained an `update_iota_chi` gate. For `ncurr=1` the half-grid
`iotaH`/`chipH` evolve through the current closure every pass, so the full-grid
`iotaF`/`chipF` update keeps running each iteration. For `ncurr=0` the half-grid
profiles are fixed (only `ncurr1FinalizeKernel` mutates them), so
`updateIotaChipFKernel` is idempotent and runs only on the first pass — the
compatibility proof is the bitwise-identical Solovev run. Removes one kernel
launch per iteration on the fixed-iota path.

### 2.5 One combined residual/control fence (`36403ad`)

The three per-pass host barriers (oriented-Jacobian gate, invariant residual,
preconditioned residual) are folded into one device `ControlRecord` of 10
elements — `[0..3]` Jacobian stats, `[4..6]` invariant residual,
`[7..9]` preconditioned residual — reduced on the compute stream and delivered
by a single `cudaMemcpyAsync` + one `cudaStreamSynchronize`.

- `computeJacobianStats` is now **device-only** (reduces the oriented Jacobian
  stats into the record; no host copy or fence); `test_geometry_ncurr` copies
  the four values out itself.
- The invariant and preconditioned residual reductions write into the record's
  slots; the **invariant reduction still precedes the in-place preconditioner
  by stream order**, so the reducer and preconditioner never race on the
  residual slab.
- The host decision order is preserved after the fence: **Jacobian-invalid →
  nonfinite → converged → `decide_restart`**. On an invalid-geometry pass the
  downstream kernels now run on the degenerate geometry, which is safe because
  (a) the Jacobian gate is checked *first* so its garbage/nonfinite residuals
  are ignored, (b) the descent is skipped, and (c) the re-anchor
  (`iter1 = iter2`) makes the next pass a refresh+reset pass that rebuilds the
  preconditioner/constraint caches from the restored geometry — the persistent
  caches are self-healing.

Net: **five ordinary-path host barriers → one** per effective iteration (the
pre-6A loop had inverse-timing, Jacobian-copy, forward-timing, invariant, and
preconditioned syncs). The refresh-cadence `computeForceNorms` sync (every 25
passes) remains; eliminating it is the 6B "device-only force-normalization
reduction" item.

## 3. Key design decisions

1. **Defaulted `stream = 0` on operators, not the `*Create` functions.** The
   hot-loop operators take the stream; the stage setup (`profilesCreate`,
   `fourierCreate`, `metricCreate`) stays synchronous on the default stream
   because it is blocking and completes before the solve — threading a stream
   through it would be churn with no ordering benefit. The one async op between
   stages (`interpolateState`) *does* take the stream so prolongation is
   ordered before the next stage.
2. **`cufftSetStream` at plan binding in `solverRun`, not in `*Create`.** The
   Fourier plans are created in `StageSolver`, the constraint plans in
   `solverRun`; binding all five once at the top of `solverRun` (before any
   transform runs) is exactly once per plan per stage and keeps the `*Create`
   signatures/test call sites unchanged.
3. **"Self-healing" one-fence, not the full §6.9 device terminal-predicate.**
   The blueprint's §6.9/§7 form guards every cache-mutating kernel with a
   device status word so invalid-geometry passes no-op on the device. The
   variant delivered here enqueues the whole DAG and lets the host decide after
   one fence, relying on the Jacobian-gate-first order + the restart re-anchor
   (which forces the next pass to rebuild the caches) for correctness. It is
   bit-identical on the frozen trajectories and much smaller; the full
   device-side status-bit + guarded-kernel form remains available as a follow-up
   if a non-self-healing cache is ever introduced.
4. **Bitwise-equivalence gate as the acceptance criterion.** Each commit was
   captured (`scripts/capture_baseline.sh`) and byte-compared against the
   Phase-6 baseline (`scripts/compare_bitwise.py`): `cumes_state.bin`,
   `per_iter_residuals_cumes.bin`, the `step_0_*` set, and the full dump
   manifest (235 Solovev / 526 W7-X files) are byte-identical.

## 4. Verification

### Class A bitwise gate (the critical gate)

Baseline captured at `overhaul/phase-5` (`2b9aaf8`) into
`.verify-scratch/baseline-phase6`; candidate trees captured after each commit
and compared with `scripts/compare_bitwise.py`:

| Config | state | trajectory | step_0 | dump manifest |
| ------ | ----- | ---------- | ------ | ------------- |
| double/solovev | OK | OK | OK (10) | OK (235 files) |
| double/w7x    | OK | OK | OK (10) | OK (526 files) |

Both `PASS: byte-identical` after every commit. Effective-iteration counts
reproduce the frozen trajectory exactly (Solovev 251 → 199 → 456; W7-X
1877 → 1617 → 2011).

### Test matrix

| Preset | Result |
| ------ | ------ |
| `verify` (double, both backends) | **26/26** (18 unit + 8 compute-sanitizer memcheck) |
| `float` | **18/18** |

`test_geometry_ncurr` was updated for the device-only `computeJacobianStats`
and still passes; the `test_fourier`/`test_force_verify`/`test_force_reference`
operators exercise the defaulted `stream = 0` path unchanged.

## 5. Deferred (documented, not hidden)

The **6B** items (blueprint §11) are out of scope for this phase:

- **Device-only force-normalization reduction** (§6.9/§8.8). `computeForceNorms`
  still does `cudaDeviceSynchronize` + several D2H copies on the refresh cadence
  (every 25 passes). Move the `E_mag`/`E_therm`/`V`/`S_RZ`/`S_L` partial sums and
  `rzNorm` into one device reduction + one record.
- **Explicit/replanned shared cuFFT work area** (§6.6/§8.7). The plans use
  cuFFT's automatic workspace; query work sizes, disable auto-allocation, and
  reuse one maximum-sized area for plans whose event-DAG lifetimes cannot
  overlap.
- **Event-DAG scratch reuse** (§6.5/§8.7). De-alias/rCon-zCon scratch leasing
  and overlap.
- **Full §6.9 device terminal-predicate + guarded kernels.** The delivered
  one-fence form is host-decision-after-one-fence (§3.3); the device
  `status_bits` + no-op-guarded field/force/constraint/preconditioner kernels
  and the §6.9 `ControlRecord` with `status_bits` remain as a follow-up if the
  self-healing argument is ever invalidated by a new persistent cache.

## 6. Next steps (Phase 7 — transform specialization)

Blueprint §11 Phase 7: axisymmetric transform + constraint/bandpass backend,
weighted R/Z constraint accumulation fused into the inverse DFT, bounded
theta/zeta/mode tiles, pack/recover transpose experiments, generalized de-alias
coverage. The explicit-stream foundation and the one-fence control path from
this phase are the substrate those transform-specialization kernels enqueue on.
