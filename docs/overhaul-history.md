# cuMES overhaul history

> The overhaul is complete. This file consolidates the former tuning sessions,
> phase handovers, migration plans, code-review record, completion plan, and
> reader-hardening handoffs into one chronological archive. Current operational contracts live in
> `architecture.md`, `mathematics.md`, `data-layout.md`, `performance.md`, and
> `verification.md`; when a historical statement conflicts with a later entry or
> a current contract, the later/current document wins.

## Consolidation status (2026-08-19)

- Phases 0–11 and the strangler-fig migration are complete; the legacy owning
  structures and `dynSharedBase()` indirection were removed without changing
  the frozen numerical trajectories.
- The whole-branch review recorded 58 surviving findings. The subsequent fix
  campaign fixed 57 and retained one controller behavior with an explicit
  compatibility rationale.
- The numerical-safety, configuration/I/O, runtime-policy, release-gate,
  post-overhaul, reader-rank, and reader-resource work recorded below is closed.
- GPU-only sanitizer, trajectory, and cross-architecture performance work is
  intentionally manual/postponed until suitable hardware is available. Hosted
  CI covers builds and host-only tests; `verification.md` is authoritative for
  the current gate matrix.

## Provenance and ordering

Entries are ordered by the **committer timestamp of the commit that first
tracked the former file**, recovered with:

```bash
git log --follow --diff-filter=A --format='%cI %H' -- <former-path>
```

This deliberately ignores filename dates, filesystem timestamps, and embedded
status dates. Phase 0 is the only handover entry whose author and committer timestamps
differ materially: it was authored at 2026-08-13 01:55 +08:00 and first
tracked at 10:14 +08:00; the latter controls its position here. The handover,
plan, and review entries retain the final text of each former file immediately
before consolidation, including later closure banners and corrections. The two
earlier tuning-session entries preserve their measurements, adopted/reverted
experiments, environment caveats, and conclusions in condensed form; the
current performance contract supersedes them. Later entries supersede earlier
forward-looking plans without erasing the original decision record.

## Chronology

| First tracked (+08:00) | Archived entry | Creation commit | Former path |
| --- | --- | --- | --- |
| 2026-08-03 16:42:11+08:00 | [cuMES optimization pass](#optimization-pass) | [`b4a24f7`](https://github.com/12ff54e/cuMES/commit/b4a24f7f81ab73851310e7ed5c0f5470fb6cd30a) | `docs/optimization-2026-08-03.md` |
| 2026-08-05 15:34:26+08:00 | [cuMES profiling session — RTX 4090](#profiling-session-rtx-4090) | [`d62ae9d`](https://github.com/12ff54e/cuMES/commit/d62ae9df465e00874209ebe23b43bd8051e5cf7c) | `docs/profiling-2026-08-05.md` |
| 2026-08-13 10:14:42+08:00 | [cuMES Phase 0 handover — overhaul/phase-0 branch](#phase-0-handover) | [`bd26857`](https://github.com/12ff54e/cuMES/commit/bd268574d258cf1889318a57b8081ce77673395a) | `docs/phase-0-handover.md` |
| 2026-08-13 11:32:02+08:00 | [cuMES Phase 1 handover — build and library split](#phase-1-handover) | [`12bcc44`](https://github.com/12ff54e/cuMES/commit/12bcc44b337a61f5e824bed37f32d5cde738d9f1) | `docs/phase-1-handover.md` |
| 2026-08-13 14:08:50+08:00 | [cuMES Phase 2 handover — validated host model and versioned I/O](#phase-2-handover) | [`168170a`](https://github.com/12ff54e/cuMES/commit/168170a7b7218cd62f11d687ae044c10e66860aa) | `docs/phase-2-handover.md` |
| 2026-08-15 19:48:11+08:00 | [cuMES Phase 3 handover — RAII buffers, typed views, exact current layouts](#phase-3-handover) | [`c21564c`](https://github.com/12ff54e/cuMES/commit/c21564c0291d1c5a4c871db65092cd4a0686020a) | `docs/phase-3-handover.md` |
| 2026-08-15 20:24:43+08:00 | [cuMES Phase 4 handover — pure controller and observers](#phase-4-handover) | [`1b0d099`](https://github.com/12ff54e/cuMES/commit/1b0d09995323e50266735195ea1b87041fe6c2b9) | `docs/phase-4-handover.md` |
| 2026-08-15 21:08:24+08:00 | [cuMES Phase 5 handover — operator/workspace boundaries](#phase-5-handover) | [`5d19e3b`](https://github.com/12ff54e/cuMES/commit/5d19e3b189e48e88dde7a251f76d9b901f82ca90) | `docs/phase-5-handover.md` |
| 2026-08-15 23:42:33+08:00 | [cuMES Phase 6 handover — low-risk control-path performance](#phase-6-handover) | [`721abd4`](https://github.com/12ff54e/cuMES/commit/721abd458dce2fbf5b7227feb75b5e6d14a6ce97) | `docs/phase-6-handover.md` |
| 2026-08-16 00:51:31+08:00 | [cuMES Phase 7 handover — transform specialization](#phase-7-handover) | [`a59fff1`](https://github.com/12ff54e/cuMES/commit/a59fff1078c19d4d74f52e75ff38113b48deb991) | `docs/phase-7-handover.md` |
| 2026-08-16 09:53:02+08:00 | [cuMES Phase 8 handover — scalable preconditioner and reductions](#phase-8-handover) | [`1069939`](https://github.com/12ff54e/cuMES/commit/1069939eb6061a8d208f5bba14f938c5f7604df9) | `docs/phase-8-handover.md` |
| 2026-08-16 10:38:31+08:00 | [cuMES Phase 9 handover — graphs and high-risk fusion](#phase-9-handover) | [`bda03f8`](https://github.com/12ff54e/cuMES/commit/bda03f88e7bec349018668d5b02fa835fbefe9df) | `docs/phase-9-handover.md` |
| 2026-08-16 11:06:09+08:00 | [cuMES Phase 10 handover — retire compatibility internals](#phase-10-handover) | [`c624eb6`](https://github.com/12ff54e/cuMES/commit/c624eb603b0c063bf208b2f5bce1c23856d3c5c6) | `docs/phase-10-handover.md` |
| 2026-08-16 11:34:58+08:00 | [Strangler-fig migration plan — legacy kernels → `cumes` operators](#strangler-fig-migration-plan) | [`35416b0`](https://github.com/12ff54e/cuMES/commit/35416b061d71792a0ad9c2b08dab15e2905fa290) | `docs/strangler-fig-migration-plan.md` |
| 2026-08-16 12:53:51+08:00 | [cuMES Phase 11 handover — strangler-fig migration (owning operators + FourierPlan split)](#phase-11-handover) | [`426e76a`](https://github.com/12ff54e/cuMES/commit/426e76a2a1c159df41b86bfbeb421af3382d1ae1) | `docs/phase-11-handover.md` |
| 2026-08-16 14:24:37+08:00 | [cuMES Phase 11 tail handover — operator unification + legacy-deletion begin](#phase-11-tail-handover) | [`585cb49`](https://github.com/12ff54e/cuMES/commit/585cb49f7fd897c0ecaa7a73a2db15d9205d446d) | `docs/phase-11-tail-handover.md` |
| 2026-08-16 15:22:55+08:00 | [cuMES Phase 11 tail #2 handover — steps 1–4 landed, step 13 remaining](#phase-11-tail-2-handover) | [`4d1aef6`](https://github.com/12ff54e/cuMES/commit/4d1aef645185a0bf3aa3886aab2533c1e543ffde) | `docs/phase-11-tail-2-handover.md` |
| 2026-08-16 15:58:47+08:00 | [cuMES Phase 11 tail #3 handover — step 13 items 1–2 landed, 3–6 remaining](#phase-11-tail-3-handover) | [`446b25e`](https://github.com/12ff54e/cuMES/commit/446b25ea9a8c49a833e20325bf76a0005d682966) | `docs/phase-11-tail-3-handover.md` |
| 2026-08-16 19:15:11+08:00 | [cuMES Phase 11 close-out handover — `dynSharedBase()` removal landed, migration complete](#phase-11-closeout-handover) | [`9a3a019`](https://github.com/12ff54e/cuMES/commit/9a3a0199b4e857ab2ab7042587b205753b8809c6) | `docs/phase-11-closeout-handover.md` |
| 2026-08-16 23:58:59+08:00 | [cuMES Code Review Log — branch `overhaul` (main...HEAD)](#cumes-code-review-2026-08-16) | [`261d6dd`](https://github.com/12ff54e/cuMES/commit/261d6dd553818fcaa3ad1662a8a54f6b196cf26b) | `docs/cuMES-code-review-2026-08-16.md` |
| 2026-08-17 20:13:23+08:00 | [cuMES overhaul completion plan](#overhaul-completion-plan) | [`48713b2`](https://github.com/12ff54e/cuMES/commit/48713b232be915cec7a8f7afaf2f5b05c5715e67) | `docs/overhaul-completion-plan.md` |
| 2026-08-17 23:07:42+08:00 | [Post-overhaul follow-up handoff](#post-overhaul-follow-up) | [`3a1f7b0`](https://github.com/12ff54e/cuMES/commit/3a1f7b0e14e2aeb3de966a3ec929a1b2f0e9bde6) | `docs/post-overhaul-follow-up.md` |
| 2026-08-17 23:46:16+08:00 | [V1 container reader rank-hardening handoff](#reader-rank-hardening-handoff) | [`611e8d7`](https://github.com/12ff54e/cuMES/commit/611e8d7929431ab4579249362ba5bef1febaf096) | `docs/reader-rank-hardening-handoff.md` |
| 2026-08-18 00:05:59+08:00 | [V1 reader resource-hardening handoff](#v1-reader-resource-hardening-handoff) | [`ac7f94e`](https://github.com/12ff54e/cuMES/commit/ac7f94ea2fb75388bf70f6dea5ad9fe1b7e17a60) | `docs/v1-reader-resource-hardening-handoff.md` |

---

<a id="optimization-pass"></a>

## 2026-08-03 16:42:11+08:00 — cuMES optimization pass

**Former path:** `docs/optimization-2026-08-03.md`

**First tracked:** [`b4a24f7`](https://github.com/12ff54e/cuMES/commit/b4a24f7f81ab73851310e7ed5c0f5470fb6cd30a) at 2026-08-03T16:42:11+08:00

This profile-driven pass reduced a W7-X run on a TITAN Xp (sm_61, CUDA 12.1,
the then-current fast-math build) from 7.786 s to 4.984 s, or 2.63 to
1.68 ms/effective iteration, while retaining the convergence point and printed
residual trajectory. The baseline and candidate states were compared across all
six spectral families; rounding-order changes stayed within the then-current
tolerance gate.

The pass used Nsight Systems timelines, `cuobjdump --dump-resource-usage`, and
targeted microbenchmarks because the installed Nsight Compute no longer
supported Pascal. The dominant original costs were a serial Thomas
tridiagonal solve, uncoalesced inverse/forward Fourier accumulation, cuFFT,
and the force/geometry kernels.

Adopted changes were:

- parallel cyclic reduction for the preconditioner solve, reducing roughly
  600 µs to 73 µs per iteration on the measured W7-X shape;
- theta-fast coalesced Fourier access and deterministic shuffle-tree forward
  reduction;
- splitting inverse accumulation into R/Z/lambda slot groups to reduce shared
  memory per launch;
- removing redundant per-iteration synchronizations, producing parity-combined
  fields only for diagnostics, and removing a dead memset;
- compact constraint cuFFT sub-batches for de-aliasing and rCon/zCon;
- larger launch blocks and parallelized small kernels;
- pinned asynchronous residual copies and parallel de-alias analysis.

Important debugging results included the PCR row-coverage failure at larger
`ns`, compact-FFT slot indexing and accidental spectrum clearing, and an atomic
contention regression caused by the initial theta-dimension swap. Each was
corrected before the final measurement. Experiments involving force/geometry
launch bounds, a surface-major forward-reduction m-loop, and partial Thomas
parallelization measured flat or negative and were reverted.

The final recorded per-iteration budget was approximately 397 µs inverse
accumulation, 391 µs forward reduction, 296 µs main cuFFT work, 167 µs force,
115 µs geometry, 78/74 µs rCon/PCR, and roughly 250 µs in smaller kernels.
These are historical measurements, not the current acceptance baseline; the
current harness, precision policy, and performance gate are documented in
`performance.md`.

---

<a id="profiling-session-rtx-4090"></a>

## 2026-08-05 15:34:26+08:00 — cuMES profiling session — RTX 4090

**Former path:** `docs/profiling-2026-08-05.md`

**First tracked:** [`d62ae9d`](https://github.com/12ff54e/cuMES/commit/d62ae9df465e00874209ebe23b43bd8051e5cf7c) at 2026-08-05T15:34:26+08:00

This follow-up session ran the then-current solver on an RTX 4090 (sm_89,
CUDA 12.2) to obtain a modern-GPU profile. Nsight Compute hardware counters
were unavailable because the host enforced admin-only profiling, so the
session again used Nsight Systems, resource dumps, and microbenchmarks.

The measured solver loop was about 0.534 ms/iteration and approximately 95%
GPU-busy. The largest costs were forward reduction (110 µs), the three inverse
accumulation launches (106 µs total), cuFFT (about 55 µs), force (41 µs),
geometry (33 µs), and the tridiagonal/residual kernels (about 27 µs each).
The converged result passed the historical cross-machine comparison; small
state differences were attributed to architecture-specific cuFFT selection and
remained below the active tolerance at that time.

Every tested kernel-level change was reverted after measuring flat or negative:
inverse-accumulation m/k splitting, constant/shared-memory basis tables,
register-capping launch bounds for force, and the previously rejected
surface-major forward-reduction restructuring. The durable conclusion was that
the measured loop was near the structural traffic/latency floor of that code
state.

The host also imposed about 4.9 s of CUDA context-creation latency per process,
dominating its 6.6 s end-to-end W7-X runtime. Isolated probes localized this to
driver primary-context creation rather than cuFFT/cuBLAS setup or JIT caching;
it was an environment property requiring administrator investigation, not a
solver optimization target.

This session predates the completed overhaul and therefore does not satisfy the
current two-architecture acceptance matrix. It remains historical evidence;
modern-GPU validation is postponed until suitable hardware can run the current
commit and fixed harness described in `performance.md`.

---

<a id="phase-0-handover"></a>

## 2026-08-13 10:14:42+08:00 — cuMES Phase 0 handover — overhaul/phase-0 branch

**Former path:** `docs/phase-0-handover.md`
**First tracked:** [`bd26857`](https://github.com/12ff54e/cuMES/commit/bd268574d258cf1889318a57b8081ce77673395a) at 2026-08-13T10:14:42+08:00

Status date: 2026-08-13. Branch: `overhaul/phase-0`, based on `631a16a`
(the blueprint's named baseline). This document records what Phase 0 work was
completed, how it was verified, and what is intentionally deferred and why.

### 1. Scope agreed

Per the user's choice, this session executed **Full Phase 0** of
`docs/cuda-overhaul-blueprint.md` on a dedicated branch. The blueprint's Phase 0
exit gate is:

> all supported small paths are sanitizer-clean; invalid/unsupported inputs fail
> before allocation; Solovev baseline remains correct; W7-X short trajectory is
> frozen.

That gate is met: the suite is 14 tests (9 unit + 5 Compute Sanitizer memcheck
variants), all passing; every invalid input has a registered negative test; and
the Solovev (906) / W7-X (5506) no-dump trajectories are byte-identical to the
pre-change baseline.

### 2. Commits (631a16a..HEAD, 14 total)

| Commit | What it does |
| ------ | ------------ |
| `a6eabd1` | **Verification anchor tooling.** `scripts/capture_baseline.sh` regenerates a byte-comparable run tree on demand (double+float × Solovev+W7-X, `CUMES_DUMP=1`, per-config float ftol); `scripts/compare_bitwise.py` byte-compares final state + per-iteration trajectory record + step_0 snapshots + full dump-set SHA-256 manifest. `.verify-scratch/` is gitignored. |
| `c0f2dbb` | **JSON negative-test matrix** (test_input_json): nonzero gamma, negative/m>=mpol boundary skip, empty/oversized/mismatched/non-monotonic schedules, integer narrowing, wrong-type aux/asym keys, lasym/lfreeb/spline rejection, unknown-key stderr warning. 52 checks. |
| `fa0932c` | **Output failure-injection matrix** (test_output_failure) + **close-safe cleanup**. Caught a real bug: all three writers double-closed the failing handle on close failure. |
| `85708df` | **ncurr=0/1 memcheck regression** (test_geometry_ncurr) at the OOB ns=33 size; registered under `CUMES_ENABLE_SANITIZER_TESTS`. |
| `240cc18` | **Non-unit tcon0 scaling regression** (test_constraint_tcon): constraint force linear in tcon0 (0/1/2), pinned the tcon0 propagation fix. |
| `11dc16e` | **De-alias theta coverage + PCR row coverage** (test_regression_kernels): CPU-reference tests for ntheta>32 and ns>129, **mutation-verified** (reintroducing each bug fails only the intended cases). |
| `3253ff3` | **Unique per-process scratch path** in test_input_json (parallel-CTest collision). |
| `3c6a2b3` | **test_forces → numerical gate** (was print-only, returned 0 unconditionally). |
| `cb338cf` | **Self-contained fixtures**: test_geometry_iso (no more dump/step_0_*.bin) and test_force_verify (no more absent vmecpp_init.bin — runs the solver to convergence in-test). Both now registered. |
| `61d4ad5` | **Atomic writer publication** (temp-file + fsync + rename) for all three writers; verified byte-identical re-capture. |
| `4ea7746` | **Observer isolation**: removed the dump-driven preconditioner refresh so diagnostic mode no longer changes the trajectory. |
| `44fb554` | **Inactive-lane reduction identity fix** in jacobianStatsKernel. |
| `8172767` | **Oriented Jacobian tracking** (min signJ·√g, not fabs(√g)) so a genuine sign-flip collapse fails validity. |
| `2b2cc75` | Removed dead `h_tcon` buffer; corrected stale module docs. |

### 3. How the verification anchor works

The baseline is **not committed as data** — it is regenerated on demand (per the
user's direction). The durable artifacts are the recipe + tools:

```bash
# Reproduce the baseline tree at any revision:
scripts/capture_baseline.sh --build build --float-build build-float \
    --out .verify-scratch/baseline --schema

# Compare a candidate build's tree against it (Class A gate):
scripts/compare_bitwise.py .verify-scratch/baseline/double/solovev \
    <candidate>/double/solovev
```

Critical comparability requirement: **both sides must run with `CUMES_DUMP=1`**
(or both without), because the dump machinery writes per-iteration records. The
capture script fixes the knobs; a candidate tree must use the identical recipe
(revision, build, inputs, `CUMES_DUMP`, same toolchain/flags/GPU).

Two independent captures of the same revision were verified byte-identical
(527-file manifest, Solovev + W7-X), proving determinism.

### 4. Trajectory verification (what was checked)

Every solver-core change was verified trajectory-neutral on the **no-dump
production path** (the physics that matters) by byte-comparing `cumes_state.bin`:

- Observer isolation (`4ea7746`): no-dump Solovev 906 / W7-X 5506 byte-identical;
  and with the coupling removed, a `CUMES_DUMP=1` run now produces the **same**
  state as no-dump (diagnostic mode no longer changes physics).
- Reduction identity (`44fb554`) and oriented Jacobian (`8172767`): both
  byte-identical on the frozen trajectories.
- Atomic writers (`61d4ad5`): re-captured baselines byte-identical.
- Dead-buffer removal (`2b2cc75`): byte-identical.

The W7-X no-dump trajectory (1878 → 1617 → 2011 = 5506) matches the blueprint's
documented reference exactly.

### 5. Test suite (14 registered)

| Test | Labels | Verifies |
| ---- | ------ | -------- |
| test_fourier | unit;fourier | cuFFT transform vs CPU reference |
| test_input_json | unit;input | JSON mapping + negative matrix |
| test_forces | unit;forces | force/geometry chain numerical invariants |
| test_output_failure | unit;output | writer open/rename/truncation failure contracts |
| test_geometry_ncurr | unit;geometry | ncurr=0/1 geometry at OOB ns=33 |
| test_constraint_tcon | unit;constraint | tcon0 linear scaling |
| test_regression_kernels | unit;constraint;precon | de-alias theta + PCR row coverage |
| test_geometry_iso | unit;geometry | bsupu/bsubu write coverage |
| test_force_verify | unit;forces | converged-equilibrium force balance |
| sanitizer_* (5) | sanitizer | memcheck variants of the kernel-driving tests |

`CUMES_ENABLE_SANITIZER_TESTS=ON` is required for the memcheck variants
(compute-sanitizer must be on PATH).

### 6. Intentionally deferred (and why)

These are listed in the blueprint's Phase 0 remaining-work table but are
genuinely **later-phase** items that require infrastructure this branch does not
introduce. Deferring them avoids destabilizing the now-frozen baseline with
partial changes.

| Item | Blueprint phase | Why deferred here |
| ---- | --------------- | ----------------- |
| **Device Jacobian status chain** (reset→reduce→finalize + event gating + dependent-kernel no-op) | Phase 5 (operator boundaries, §6.7) | Requires the `JacobianStatusDevice` / `ControlRecord` architecture and CUB/CAS reductions that are Phase 5 deliverables. The synchronous host check remains the containment. The concrete sub-bugs (reduction identity, orientation) are fixed. |
| **Scale-aware denominator policies + `NumericalStatus`** | Phase 2–3 | The current `1e-30` floors are functional containment (exact-rounding-preserving). Scale-aware policies are Class B numerical changes that risk the frozen trajectory, and the structured-status type is a Phase 3 deliverable. |
| **Deep-`exit` removal from lower-level helpers** | Phase 3 (centralized error handling) | `cc()`/`ccf()`/`checkCuda()` exit(1); converting to structured errors requires the RAII/Result boundary that is Phase 3. |
| **Timing fences (cudaEventRecord/Synchronize)** | Phase 6A (control-path perf) | Observability artifacts; removing them without the NVTX/event-ring replacement would lose the timing report. |
| **PrecisionPolicy (float-ftol gate)** | Phase 2 (config) | The hardcoded `1e-6` float gate is functional; `PrecisionPolicy` is a Phase 2 validated-host-model deliverable. |

### 7. Next steps (Phase 1)

Phase 1 is the build/library split, and it can now proceed against a frozen,
byte-comparable baseline using `compare_bitwise.py` as its Class A gate:

- target-scoped CMake + `CMakePresets.json` + `cmake/` module dir;
- host-only `.cpp` parser/output targets (input_json, output_* are host-only but
  compiled by nvcc today);
- explicit double/float CUDA instantiation TUs (removes the `dynSharedBase()`
  workaround);
- `cumes_test_support` shared test-support library;
- no numerical source changes; Class A bitwise equivalence via the anchor.

The blueprint also notes `main` is a known baseline to branch from; do not
commit the untracked `AGENTS.md` symlink or the downloaded CUDA guide markdown
files under `docs/`.

### 8. Reproducibility notes

- Float W7-X prescribed-current does **not** converge at ftol ≤ 1e-3 (float
  floor ~3e-3); the capture recipe uses per-config float ftol (solovev=1e-6,
  w7x=1e-2).
- `DUMP_CUMES_VERIFY` is hardcoded at `src/solver.cu:9`; the dump machinery is
  compiled in but runtime-gated by `CUMES_DUMP=1`.
- The forensic `cumes_state.bin` stays double on disk regardless of T (the
  binary writer converts T→double).

---

<a id="phase-1-handover"></a>

## 2026-08-13 11:32:02+08:00 — cuMES Phase 1 handover — build and library split

**Former path:** `docs/phase-1-handover.md`
**First tracked:** [`12bcc44`](https://github.com/12ff54e/cuMES/commit/12bcc44b337a61f5e824bed37f32d5cde738d9f1) at 2026-08-13T11:32:02+08:00

Status date: 2026-08-13. Branch: `overhaul` (Phase 0 `bd26857` + this Phase 1
work). This document records what Phase 1 of `docs/cuda-overhaul-blueprint.md`
delivered, how it was verified, and what was deliberately deferred and why.

### 1. Scope

Phase 1 is the **build and library split**: restructure the monolithic nvcc
build into target-scoped libraries without changing any numerical result. The
blueprint's exit gate:

> no numerical source changes; Class A bitwise equivalence; all optional-backend
> build combinations compile and test.

That gate is met: the full dump-set tree (final state + per-iteration trajectory
record + initial-state snapshots + every `dump/cuMES/*` file — 1175 files across
four configs) is **byte-identical** to the Phase 0 baseline, and all four
NetCDF/HDF5 build combinations compile and pass their tests.

### 2. What changed

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

### 3. Key finding — `dynSharedBase()` retained (removal is Class B)

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

### 4. Verification

#### 4.1 Class A bitwise gate (full dump set)

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

#### 4.2 Optional-backend matrix (none / netcdf / hdf5 / both)

| Combination | ctest | suffix dispatch |
| ----------- | ----- | --------------- |
| both (default `build`)          | 14/14 (9 unit + 5 sanitizer) | `.nc`/`.h5`/`.bin` all produced |
| nobackend (`build-nobackend`)   | 9/9 | `.nc` rejected before solve (exit 1) |
| netcdf-only (`build-netcdf-only`) | 9/9 | `.nc` produced, `.h5` rejected |
| hdf5-only (`build-hdf5-only`)   | 9/9 | `.h5` produced, `.nc` rejected |

#### 4.3 Tests

- Double build: **14/14** (9 unit + 5 compute-sanitizer memcheck variants).
- Float build: **9/9** unit (sanitizer variants off there).

### 5. Rebuild instructions

```bash
cmake --preset verify && cmake --build build -j            # double (default)
cmake --preset float  && cmake --build build-float -j      # single precision
cmake --preset nobackend && cmake --build build-nobackend -j  # binary-only
ctest --test-dir build                                     # run tests
```

### 6. Notes and deferred items

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

### 7. Next steps (Phase 2)

Phase 2 is the validated host model + versioned I/O (`ProblemSpec`,
`ValidatedProblem`, `GridShape`, `ModeTable`, `PrecisionPolicy`, `OutputSpec`,
`RunReport`, `EquilibriumSnapshot`, versioned writers and legacy adapters). It
now builds on a frozen, byte-comparable Phase 1 baseline with the same
`compare_bitwise.py` Class A gate.

---

<a id="phase-2-handover"></a>

## 2026-08-13 14:08:50+08:00 — cuMES Phase 2 handover — validated host model and versioned I/O

**Former path:** `docs/phase-2-handover.md`
**First tracked:** [`168170a`](https://github.com/12ff54e/cuMES/commit/168170a7b7218cd62f11d687ae044c10e66860aa) at 2026-08-13T14:08:50+08:00

Status date: 2026-08-13. Branch: `overhaul` (Phase 0 `bd26857` + Phase 1
`12bcc44` + this Phase 2 work). This document records what Phase 2 of
`docs/cuda-overhaul-blueprint.md` delivered, how it was verified, and what was
deliberately deferred and why.

### 1. Scope

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

### 2. What changed

#### New `namespace cumes` library (`include/cumes/`, `src/cumes/`)

| Layer | Files | Contents |
| ----- | ----- | -------- |
| core | `core/{result,scalar,checked_size,grid_shape,mode_table}.hpp` + `src/cumes/core/*.cpp` | `BasicResult<T,E>`/`Status`, `ScalarType`, checked size_t arithmetic, extents-only `GridShape` with resolved-shape validation, per-mode `ModeTable<T>` (physical_n, mn_scale, xmpq, parity, first_surface). |
| config | `config/{problem_spec,precision_policy,validation_report,solver_options,validated_problem,json_reader}.hpp` + `src/cumes/config/*.cpp` | dynamic `ProblemSpec`, `PrecisionPolicy` with tolerance floors, `ValidationReport` (collects all findings), four-stage `validate()` (parse→validate→fold→resolve), legacy `to_input_params()` bridge, `normalize_to_json()`, and the JSON reader. |
| io | `io/{output_spec,run_report,equilibrium_snapshot,writer,reader,checkpoint}.hpp` + `src/cumes/io/*.cpp` | `OutputSpec`/`OutputFormat`/`OutputSchema`, `RunReport` with full stage history + provenance, host `EquilibriumSnapshot`, `Writer`/`Reader` interfaces, legacy binary v0, versioned binary v1, versioned checkpoint + legacy init converter. |

#### Build targets (added to `CMakeLists.txt`)

- `cumes_core` — core host model (static).
- `cumes_json` — the JsonParser implementation, extracted into a single TU
  shared by the legacy parser and the new reader (removes a double-definition
  when both `input_json.cpp` and `json_reader.cpp` instantiated the header).
- `cumes_config_json` — config model + JSON reader.
- `cumes_io_host` — host I/O (no CUDA).

#### Tests (host-only `.cpp`, compiled by g++)

- `test_host_config` (unit;config) — normalization goldens, adapter parity,
  mode table, precision floor, malformed-input matrix.
- `test_host_io` (unit;io) — output-spec dispatch, binary v0 byte layout,
  binary v0/v1 round-trips, writer failure matrix.
- `test_checkpoint` (unit;io) — checkpoint round-trip + rejection, legacy init
  converter.

### 3. Key design decisions

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

#### Deferred (documented, not hidden)

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

### 4. Verification

#### Test matrix

| Preset | Result |
| ------ | ------ |
| `verify` (double, both backends) | **17/17** (12 unit + 5 sanitizer) |
| `nobackend` (binary only)        | **12/12** |
| `float` (new host tests)         | test_host_config / test_host_io / test_checkpoint all pass |

The new host tests are type-agnostic (all host data is double), so they are
identical across the float/nobackend matrix; the NetCDF/HDF5 availability
preflight degrades correctly to binary-only in `nobackend`.

#### Config gate

- `normalize_to_json()` matches the checked-in `tests/fixtures/{solovev,w7x}.normalized.json`
  goldens (generated once by `test_host_config --emit-golden`).
- `to_input_params()` is field-identical to `initInputParamsFromJson()` for both
  shipped configs.
- The malformed-input matrix (nonzero gamma, out-of-range boundary modes
  skipped with a warning, empty/mismatched/non-monotonic schedules, integer
  narrowing, wrong-type aux/asym keys, unsupported physics, unknown-key
  strict/compat) all pass.

#### I/O gate

- Legacy binary v0: exact byte layout + round-trip.
- Versioned binary v1: round-trip + bad-magic rejection.
- Checkpoint: round-trip + magic/truncation rejection; legacy init converter +
  header-mismatch rejection.
- Writer failure matrix: open failure and rename-over-directory return errors
  and leave the target untouched.

#### Class A safety

No device/solver source changed. The frozen trajectories and
`scripts/compare_bitwise.py` baseline are untouched by construction; the only
legacy source touched is `src/input_json.cpp` (removed one `#define`, now
linking the shared `cumes_json` parser — build-only, no behavior change,
re-verified by the existing `test_input_json` passing unchanged).

#### Adversarial review

An adversarial three-reviewer pass over the new code found and fixed: a
negative `niter_array`/`ns_array` entry wrapping to `size_t` and bypassing
validation; readers `bad_alloc`-ing on a mismatched-format/corrupt header
(bounded against the file size); a present-but-empty `raxis_c`/`zaxis_s` being
silently zero-padded instead of rejected; dropped validation warnings; a
signed-overflow in `GridShape::modes()`; and unbounded `read_string`. Each has
a regression test.

### 5. Next steps (Phase 3)

Phase 3 is **RAII buffers, typed views, and exact current layouts**: `DeviceContext`,
buffer/stream/event RAII, typed `SpectralView`/`RealFieldsView`, contiguous
state/velocity slabs, and a state-only checkpoint slab, with legacy kernels
wrapped behind views. It can now also:
- wire `main.cu` to parse via `read_and_validate` and to write via the versioned
  writers + checkpoint reader (replacing `CUMES_LOAD_INIT`);
- add the host NetCDF/HDF5 v0/v1 writers once the snapshot bridge exists.

---

<a id="phase-3-handover"></a>

## 2026-08-15 19:48:11+08:00 — cuMES Phase 3 handover — RAII buffers, typed views, exact current layouts

**Former path:** `docs/phase-3-handover.md`
**First tracked:** [`c21564c`](https://github.com/12ff54e/cuMES/commit/c21564c0291d1c5a4c871db65092cd4a0686020a) at 2026-08-15T19:48:11+08:00

Status date: 2026-08-15. Branch: `overhaul` (Phase 0 `bd26857` + Phase 1 `12bcc44`
+ Phase 2 `168170a` + this Phase 3 work). This document records what Phase 3 of
`docs/cuda-overhaul-blueprint.md` delivered, how it was verified, and what was
deliberately deferred and why.

### 1. Scope

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

### 2. What changed

#### New `cumes` runtime + views + state slab (header-only where templated)

| Layer | Files | Contents |
| ----- | ----- | -------- |
| runtime | `include/cumes/runtime/{cuda_status,device_buffer.cuh,pinned_buffer,stream,event,device_context}.hpp/.cuh` + `src/cumes/runtime/device_context.cpp` | `CumesError` + throwing `check_cuda`/`check_cufft` (with a cuFFT enum→string table; `cufftGetErrorString` does not exist); movable non-copyable `DeviceBuffer<T>`/`PinnedBuffer<T>`; `Stream`/`Event`/`DeviceContext` RAII. |
| core | `include/cumes/core/tensor_view.cuh` | `SpectralComponent` enum, domain tags, `SpectralView<T,Domain>` (`[component][mode][surface]`, surface-contiguous) and `RealFieldView<T>` (`[surface][zeta][theta]`, theta-contiguous), both `__host__ __device__`. |
| state | `include/cumes/state/spectral_storage.hpp` | `SpectralStorage<T>`: owns two 6·mnmax·ns slabs (state + velocity) and exposes `legacy_view()`, `physical()`/`velocity()` views, and the owning buffers for the checkpoint. |

#### Build target

- `cumes_cuda_runtime` (STATIC), linked by `cumes_cuda_double`/`cumes_cuda_float`
  and `cuMES`.

#### Migration of the allocating call sites

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

#### Centralized error handling

- All twelve modules' duplicated `checkCuda`/`cc`/`ccf`/`checkCufft` helpers are
  replaced with `cumes::check_cuda`/`cumes::check_cufft` (throw `CumesError`); `main`
  catches it at the boundary. Error-path-only, so Class A safe.
- `tests/support/cumes_test_support.cuh` is intentionally left as-is (test infra).

#### New test

- `tests/test_runtime.cu` (`unit;runtime`): `DeviceBuffer`/`PinnedBuffer` alloc/zero/
  copy/move; `Stream`/`Event`/`DeviceContext`; `SpectralStorage` slab-offset layout
  (the 12 legacy pointers == slab + {0..5}·mnmax·ns); a device round-trip through
  `SpectralView`; `check_cuda`/`check_cufft` error injection. Plus a `static_assert`
  that `SpectralComponent` and `EquilibriumSnapshot::Component` share one order.

### 3. Key design decisions

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

#### Diagnostic-determinism fix (required for a clean dump-set gate)

`step_A_l_real_iter_1.bin` dumps `fp.d_l_real` *before* `fourierCombineParity`
produces it — a latent dump of uninitialized memory. Its bytes depended on the CUDA
allocator state, so the slab re-allocation (this phase's very purpose) shifted them
for Solovev. The nine combined `*_real` buffers are now zero-initialized in
`fourierCreate`, making that dump deterministic. This is a diagnostic-only change
(the combined buffers are never read by the numerical kernels, which use the parity
arrays); the numerics are unaffected.

### 4. Verification

#### Test matrix

| Preset | Result |
| ------ | ------ |
| `verify` (double, both backends) | **19/19** (13 unit + 6 compute-sanitizer memcheck) |
| `float`  | **13/13** |
| `nobackend` | **13/13** |
| `netcdf-only` | **13/13** |
| `hdf5-only` | **13/13** |

#### Class A bitwise gate (the critical gate)

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

#### No hot-loop allocation

`grep cudaMalloc|cudaFree|cudaMallocHost|cudaFreeHost src/solver_impl.cuh` → none;
all allocations are RAII objects constructed before the iteration loop.

### 5. Deferred (documented, not hidden)

- Wiring `main.cu` to `read_and_validate` + versioned writers + checkpoint reader
  (Phase 4–5).
- Full kernel-signature migration to `SpectralView` (Phase 5); `DeviceArena` and the
  `*Create`/`*Free` workspace structs → RAII (Phase 5).
- Explicit nonblocking-stream execution / `cufftSetStream` / one combined control
  fence (Phase 6A).
- `dynSharedBase()` removal (Class B, per Phase-1 handover §3).

### 6. Next steps (Phase 4)

Phase 4 is the **pure controller and observers**: deterministic `IterationController`
+ `ControlRecord`, `StageSolver`/`MultigridSolver` separation, structured scalar
telemetry, and versioned lazy snapshots — with the recorded residual histories
reproducing the exact restart/damping decisions, and observers provably unable to
change state hashes or the iteration count.

---

<a id="phase-4-handover"></a>

## 2026-08-15 20:24:43+08:00 — cuMES Phase 4 handover — pure controller and observers

**Former path:** `docs/phase-4-handover.md`
**First tracked:** [`1b0d099`](https://github.com/12ff54e/cuMES/commit/1b0d09995323e50266735195ea1b87041fe6c2b9) at 2026-08-15T20:24:43+08:00

Status date: 2026-08-15. Branch: `overhaul` (Phase 0 `bd26857` + Phase 1
`12bcc44` + Phase 2 `168170a` + Phase 3 `c21564c` + this Phase 4 work). This
document records what Phase 4 of `docs/cuda-overhaul-blueprint.md` delivered,
how it was verified, and what was deliberately deferred and why.

### 1. Scope

Phase 4 is the **pure controller and observers**: extract the solver's host-side
fixed-point control state machine into a deterministic, CUDA-free
`IterationController<T>`, separate the multigrid orchestration into
`StageSolver`/`MultigridSolver`, and make the per-pass telemetry a typed record —
while leaving the numerical path bit-for-bit unchanged. `main.cu` stays on the
legacy parse/output path (config/I-O wiring remains Phase 5); the legacy kernels
keep their raw-pointer signatures (full `SpectralView` migration remains Phase 5).

The blueprint's exit gate:

> recorded residual histories reproduce exact restart/damping decisions;
> enabling observers cannot change state hashes or iteration count.

That gate is met (see §4): the controller-driven solver reproduces the frozen
`per_iter_residuals_cumes.bin` byte-for-byte, and a `CUMES_DUMP=0` run writes a
state file identical to the `CUMES_DUMP=1` run.

### 2. What changed

#### New `cumes/solver` headers (pure host, templated on T)

| File | Contents |
| ---- | -------- |
| `control_record.hpp` | `RestartReason` (0/1/2, part of the frozen telemetry contract), `JacobianStatus<T>`, `InvariantVerdict<T>`, `Damping<T>`, `RestartDecision<T>`, and the aggregate `ControlRecord<T>` (the single-fence target; see §5). |
| `iteration_controller.hpp` | `IterationController<T>` + `kPreconInterval = 25`: owns iter2/iter1/log-anchor, the ten-sample 1/tau history, the running-minimum `res0`, `ijacob`, `delt`, and `ftol`. Methods `next_schedule()` (ijacob 25/50 maintenance), `jacobian_invalid()`, `classify_invariant()`, `decide_restart()`, `after_descent()`, plus pass-invariant accessors (`reset_constraint_reference`, `refresh_preconditioner`, `fsqz_prev`, …). |
| `pass_record.hpp` | `PassRecord`: the typed 15-column per-pass telemetry record (standard-layout 15 doubles, `static_assert`ed) whose field order is the frozen on-disk contract. |
| `stage_solver.hpp` | `StageSolver<T>::run`: one stage's `profilesCreate`/`fourierCreate`/`metricCreate` + `solverRun` + free. State stays owned by the caller. |
| `multigrid_solver.hpp` | `MultigridSolver<T>::run` + `MultigridOutcome<T>`: owns the state across stages, prolongs via `interpolateState`, validates the schedule, emits a `RunReport` with per-stage history, and reports a failing stage without calling `exit()`. |

#### Migration

- `src/solver_impl.cuh`: the ~70 lines of inline control state and the
  maintenance / bad-Jacobian / nonfinite / convergence / damping / restart
  decision logic are replaced by one `IterationController<T>` and the five
  per-fence calls in the exact frozen order. Every kernel launch, fence, and
  dump block is unchanged. `recordPass` now takes `iter2`/`iter1` explicitly so
  the recover branches record the *pre*-update values exactly as before.
- `src/main.cu`: the stage loop is replaced by `MultigridSolver<Real>::run`; the
  CLI cold-starts stage 0, maps the outcome to an exit code, and prints the
  stage-failure FATAL.
- `CMakeLists.txt`: `test_controller` (host-only, `unit;solver`) added.

#### Already-complete Phase-4 deliverables (re-verified, not re-done)

- **Delete dump-driven solver branches.** The only dump-mode-dependent control
  branch (the `iter2 == kDumpIter` extra preconditioner refresh) was removed in
  Phase 0 (`4ea7746`). A grep confirms every remaining `dumpEnabled()`/`kDumpIter`
  reference is observability only — none feeds a decision.
- **Unproduced combined-force buffers.** Removed in Phase 3; `grep` confirms no
  combined-force buffer/dump remains.

### 3. Key design decisions

1. **Multi-method state machine, not a single `advance()`.** The current solver
   fences twice (Jacobin gate, then invariant/preconditioned residuals), and the
   recover/terminal branches skip preconditioning. A single `advance(ControlRecord)`
   presupposes the Phase 6A one-fence DAG, so Phase 4 exposes the five per-fence
   methods instead; `ControlRecord<T>` is declared as the target shape.
2. **Controller templated on `T`, not `double`.** The legacy control arithmetic
   uses `T` throughout (including `fsq_prev`, the 1/tau history, and the
   `log(fsq/fsq_prev)` ratio). Templating preserves float-build arithmetic
   bit-for-bit; the blueprint's "double accumulation" is a Phase 6/8 concern.
3. **`recordPass` takes `iter2`/`iter1` explicitly.** The controller applies the
   restart bookkeeping (`iter1 = iter2`, `delt *= 0.9`) *inside* the fence calls,
   so the solver snapshots the pre-update values before calling and passes them
   to the record — keeping the frozen per-iteration bytes identical.
4. **`PassRecord` serialization walks standard-layout fields.** The dump writer
   writes the 15 fields column-major via `&r.invariant_fsqr + c`, byte-identical
   to the legacy `double[15]` layout, with a `static_assert(sizeof == 15*8)`.
5. **`StageSolver`/`MultigridSolver` bridge the legacy bridge.** They consume the
   legacy `InputParams`/`GridParams` and the move-only `SpectralStorage<T>`; the
   config model (`ValidatedProblem`) and versioned writers are not wired (Phase 5).
   No library code calls `exit()` — the stage failure is a `failed_stage` index
   the CLI maps to `EXIT_FAILURE`.
6. **`dtau_floor` comparison.** The controller's `dtau_floor > T(0)` replaces the
   legacy `kDtauFloor > 0.0` (double) test. Identical for the frozen paths
   (`CUMES_DTAU_FLOOR` defaults to 0); a float build with a subnormal-tiny floor
   would differ, which is outside any golden.

### 4. Verification

#### Class A bitwise gate (the critical gate)

`scripts/compare_bitwise.py` against the Phase-3 baseline (`.verify-scratch/
baseline-phase3`, tag `overhaul/phase-3`) after each of the four commits:

| Config | state | trajectory | step_0 | dump manifest |
| ------ | ----- | ---------- | ------ | ------------- |
| double/solovev | OK | OK | OK (10) | OK (235 files) |
| double/w7x    | OK | OK | OK (10) | OK (526 files) |

Both `PASS: byte-identical` — the controller-driven solver replays the frozen
trajectory exactly (same restarts, damping, effective-iteration counts, and
final state).

#### Test matrix

| Preset | Result |
| ------ | ------ |
| `verify` (double, both backends) | **20/20** (14 unit + 6 compute-sanitizer memcheck) |
| `float` | **14/14** |

#### Controller unit test (`test_controller`)

Drives scripted residual sequences through every branch and asserts the exact
historical semantics: convergence, nonfinite recovery (delt ×0.9 + re-anchor),
the anchor-pass 1/tau history (otav/`dtau`/`b1`/`fac` values), the bad-Jacobian
restart (delt ×0.9, `ijacob`++), the bad-progress restart (delt ÷1.03), the
`ijacob == 25` maintenance reset (delt = 0.98·delt0), the refresh predicate
(age > 10), and effective-iteration / restart-anchor bookkeeping.

#### Observer isolation (the second exit-gate clause)

A `CUMES_DUMP=0` Solovev run writes a `cumes_state.bin` **byte-identical** to the
`CUMES_DUMP=1` run — enabling the dump observer changes neither the state hash
nor the iteration count. (This was established in Phase 0; Phase 4 re-verified
it and made the telemetry a typed record so the property is structural.)

### 5. Deferred (documented, not hidden)

- **Single-fence `advance(ControlRecord)`** and the device terminal-predicate
  kernel (§6.9) — the Phase 6A one-control-fence DAG.
- **Observer/SnapshotManager** with producer events, a bounded in-flight ring,
  and version stamps (§6.12). The isolation *property* is delivered; the
  versioned-snapshot *machinery* (NVTX ranges, sampled event records, lazy
  materialization policy) belongs to Phase 5–6.
- **Config/I-O wiring** — `main.cu` still parses via the legacy `initInputParams`
  and writes via the legacy `outputSave`; `read_and_validate` + versioned writers
  + the checkpoint reader remain Phase 5 (per the Phase 3 handover).
- **Full `SpectralView` kernel-signature migration** and `DeviceArena` (Phase 5).

### 6. Next steps (Phase 5)

Phase 5 is **operator/workspace boundaries**: transform-only `SpectralOperator`,
profiles/geometry/B/force/constraint/residual/preconditioner/descent interfaces,
a stage arena with reported liveness/peak memory, and scalar CPU references at
every boundary — plus wiring `main.cu` to `read_and_validate` + versioned writers
+ the checkpoint reader.

---

<a id="phase-5-handover"></a>

## 2026-08-15 21:08:24+08:00 — cuMES Phase 5 handover — operator/workspace boundaries

**Former path:** `docs/phase-5-handover.md`
**First tracked:** [`5d19e3b`](https://github.com/12ff54e/cuMES/commit/5d19e3b189e48e88dde7a251f76d9b901f82ca90) at 2026-08-15T21:08:24+08:00

Status date: 2026-08-15. Branch: `overhaul` (Phase 0 `bd26857` + Phase 1
`12bcc44` + Phase 2 `168170a` + Phase 3 `c21564c` + Phase 4 `1b0d099` + this
Phase 5 work, followed by the config/I-O wiring completion in §7 and the
kernel-signature migration + scalar references in §8). This document records
what Phase 5 of `docs/cuda-overhaul-blueprint.md` delivered, how it was
verified, what was deliberately deferred and why, and how the deferred items
were subsequently landed.

### 1. Scope

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

### 2. What changed

#### 2.1 `DeviceArena` (`include/cumes/runtime/device_arena.cuh`)

A host-side linear arena over one `cudaMalloc`'d backing store that carves
**named, aligned** subspans (`alloc_span<T>(name, count, align)`) and reports
per-category liveness/peak (`spans()`, `used_bytes()`, `peak_bytes()`,
`total_bytes()`). It throws `CumesError` on overflow (a too-small plan is a
loud setup error, never silent aliasing) and rejects non-power-of-two
alignments. Move resets the source.

#### 2.2 Arena-backed workspaces

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

#### 2.3 Typed real-space views (`include/cumes/state/real_fields.cuh`)

`ReducedThetaView<T>` (a **distinct** reduced-theta quadrature view — never an
integer reinterpretation of a full-grid view, blueprint §4.1) plus the
aggregate bundles `GeometryParityViews` / `RadialProfileViews` /
`BaseGeometryHalfViews` / `MagneticFieldViews` / `ForceParityViews`. All are
trivially-copyable, `__host__ __device__`, and index bit-for-bit like the
legacy `surface*nZnT + zeta*ntheta + theta` layout.

#### 2.4 Operator interface contracts

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

### 3. Key design decisions

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

### 4. Verification

#### Class A bitwise gate (the critical gate)

Fresh baseline at `overhaul/phase-4` (`1b0d099`), captured to
`.verify-scratch/baseline-phase5`; candidate trees captured from the Phase-5
tree and compared with `scripts/compare_bitwise.py`:

| Config | state | trajectory | step_0 | dump manifest |
| ------ | ----- | ---------- | ------ | ------------- |
| double/solovev | OK | OK | OK (10) | OK (235 files) |
| double/w7x    | OK | OK | OK (10) | OK (526 files) |

Both `PASS: byte-identical`. Effective-iteration counts reproduce the baseline
exactly (Solovev 251 → 199 → 456; W7-X 1877 → 1617 → 2011).

#### Test matrix

| Preset | Result |
| ------ | ------ |
| `verify` (double, both backends) | **24/24** (17 unit + 7 compute-sanitizer memcheck) |
| `float` | **15/15** |

The float build compiles and its solver runs the same arena path; the double
build is the verification configuration (blueprint §1).

#### No hot-loop allocation

The five workspaces now allocate once per stage (one arena `cudaMalloc`); the
solver's internal `d_f_spec`/`d_sq`/`d_psum`/`d_jac_stats`/`checkpoint` buffers
were already RAII `DeviceBuffer`s from Phase 3. `grep cudaMalloc src/*_impl.cuh`
now shows only the `arena == nullptr` legacy fallback branches, never a hot-loop
call.

### 5. Deferred (documented, not hidden)

Phase 5 is delivered as the **boundary layer**; the deferred items that complete
it — config/I-O wiring (§7), the full kernel-signature migration and the scalar
CPU references + dual-run hooks (§8) — have since landed. The one item that
remains **out of scope** for Phase 5:

- **`dynSharedBase()` removal** is still a Class B change (Phase-1 handover §3),
  not part of this phase.

### 6. Next steps (Phase 5 completion)

1. ~~Golden-test the versioned legacy-v0 writer against `outputSaveBinary`, then
   wire `main.cu` to `read_and_validate` + versioned writers + checkpoint
   reader (config/I-O wiring).~~ **Done** — see §7.
2. ~~Migrate the kernels onto the operator signatures module-by-module, with
   Class A bitwise verification after each, and add the scalar CPU references +
   dual-run hooks as the per-boundary gate.~~ **Done** — see §8.
3. Then Phase 6 (control-path performance): explicit nonblocking streams,
   one combined control fence, one-copy checkpoint (already one copy since
   Phase 3), fixed-iota update skip.

### 7. Config/I-O wiring (landed 2026-08-15)

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

### 8. Kernel-signature migration + scalar references (landed 2026-08-15)

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

---

<a id="phase-6-handover"></a>

## 2026-08-15 23:42:33+08:00 — cuMES Phase 6 handover — low-risk control-path performance

**Former path:** `docs/phase-6-handover.md`
**First tracked:** [`721abd4`](https://github.com/12ff54e/cuMES/commit/721abd458dce2fbf5b7227feb75b5e6d14a6ce97) at 2026-08-15T23:42:33+08:00

Status date: 2026-08-15. Branch: `overhaul` (Phase 0 `bd26857` + Phase 1
`12bcc44` + Phase 2 `168170a` + Phase 3 `c21564c` + Phase 4 `1b0d099` + Phase 5
`2b9aaf8` + this Phase 6 work). This document records what Phase 6 of
`docs/cuda-overhaul-blueprint.md` delivered, how it was verified, and what was
deliberately deferred and why.

### 1. Scope

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

This phase delivers **all six 6A items** plus the 6B **device force-norm
reduction** and **shared cuFFT work area**; the 6B **event-DAG scratch reuse**
is deferred (§5). The 6A commits are Class A (bitwise-identical); the 6B
force-norm reduction is Class B (ULP-level, verified by trajectory bounds).

| Blueprint 6A deliverable | Status |
| ------------------------ | ------ |
| Explicit nonblocking stream + cuFFT stream binding | **Done** (commit `945b776`) |
| Remove timing fences | **Done** (commit `8e47ce8`) |
| One combined residual/control fence | **Done** (commit `36403ad`) |
| One-copy state checkpoint | **Already done** since Phase 3 (re-verified, made stream-ordered) |
| Fixed-iota update skip | **Done** (commit `8a53ced`) |
| Complete recover writes (no force memset) | **Done** (commit `bb1281f`) |

| Blueprint 6B deliverable | Status |
| ------------------------ | ------ |
| Device-only force-normalization reduction | **Done** (commit `e29c713`, Class B) |
| Explicit/shared cuFFT work area | **Done** (commit `e43a936`, bitwise-neutral) |
| Event-DAG scratch reuse | **Deferred** (§5) |

### 2. What changed

#### 2.1 Explicit nonblocking compute stream + cuFFT binding (`945b776`)

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

#### 2.2 Remove timing fences (`8e47ce8`)

The two `cudaEventSynchronize` host barriers that bracketed `inverseDFT` and
`forwardDFT` are gone. The four timing events (`ev0/ev1_inv`, `ev0/ev1_fwd`)
are still recorded on the compute stream every pass, but their elapsed times
are sampled only at the already-required control fence (both transforms precede
it on the same stream). Two host barriers per pass → zero.

#### 2.3 Complete recover writes — no force memset (`bb1281f`)

`forwardRecoverKernel` now writes **all six** spectral-force families for every
`(mode, surface)`, including explicit zeros for the axis (m>0 rows and the four
non-`frcc`/`fzcs` families) and the LCFS (non-λ families) that the old pre-zero
used to cover. The forward DFT's `cudaMemset` of the full `6·mnmax·ns` residual
slab is removed — one full-slab device write per iteration eliminated. The
recover output is bit-identical (the memset and the explicit zeros write the
same values).

#### 2.4 Fixed-iota update skip (`8a53ced`)

`computeGeometry` gained an `update_iota_chi` gate. For `ncurr=1` the half-grid
`iotaH`/`chipH` evolve through the current closure every pass, so the full-grid
`iotaF`/`chipF` update keeps running each iteration. For `ncurr=0` the half-grid
profiles are fixed (only `ncurr1FinalizeKernel` mutates them), so
`updateIotaChipFKernel` is idempotent and runs only on the first pass — the
compatibility proof is the bitwise-identical Solovev run. Removes one kernel
launch per iteration on the fixed-iota path.

#### 2.5 One combined residual/control fence (`36403ad`)

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

### 3. Key design decisions

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

### 4. Verification

#### Class A bitwise gate (the critical gate)

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

#### Test matrix

| Preset | Result |
| ------ | ------ |
| `verify` (double, both backends) | **26/26** (18 unit + 8 compute-sanitizer memcheck) |
| `float` | **18/18** |

`test_geometry_ncurr` was updated for the device-only `computeJacobianStats`
and still passes; the `test_fourier`/`test_force_verify`/`test_force_reference`
operators exercise the defaulted `stream = 0` path unchanged.

### 5. Phase 6B (Class B)

#### 5.1 Device-only force-normalization reduction (`e29c713`)

`computeForceNorms` is split into `enqueueForceNorms` (device) and
`finalizeForceNorms` (host). The device part reduces the per-surface partial
sums (`sRZ`/`sL`/`sMag`/`eTherm`/`vol`) with a new `forceNormReduceKernel` and
the `rzNorm` with the existing kernel, writing six scalars into the combined
control record's `[10..15]` slots; the host finalize derives
`fNormRZ`/`fNormL`/`fNorm1` after the single control fence.
`computeForceNormPartials` no longer synchronizes the stream.

This removes the refresh-cadence `cudaDeviceSynchronize` and the three
synchronous D2H copies (`psum` 4·nH, `dVdsH` nH, `presH` nH) plus the host
sequential sum, folding the force-norm transfer into the one combined control
copy. **Class B**: the device tree reduction (and fast-math FMA in the eTherm
dot-product) changes summation order, moving the force-norm factors by 2–4 ULP.

#### 5.2 Explicit/shared cuFFT work area (`e43a936`)

cuFFT auto-allocation is disabled and one max-sized work area is shared per
module — the two Fourier plans (`z2d`/`d2z`) reuse one buffer, the three
constraint plans (`d2z_da`/`z2d_da`/`z2d_rz`) reuse another. All five
transforms are sequential on the one compute stream, so their work-area
lifetimes never overlap. For the W7-X double shape this replaces cuFFT's five
per-plan auto allocations (`~4.1 MB × 2 + 0.55 MB × 2 + 1.37 MB ≈ 10.4 MB`)
with two buffers (`4.1 MB + 1.37 MB ≈ 5.5 MB`). Bitwise-neutral (cuFFT results
are deterministic independent of the work-area location).

#### 5.3 Verification

| Item | Gate | Result |
| ---- | ---- | ------ |
| Force-norm reduction | `compare_runs.py` (Class B) | PASS: restart sequence identical (W7-X 15 events), convergence identical (Solovev 456 / W7-X 2011), converged **state bitwise-identical**, residual/force-norm records differ ≤ 4 ULP |
| cuFFT work area | `compare_bitwise.py` vs pre-change tree | PASS: byte-identical (bitwise-neutral) |

Full test matrix re-verified: double `verify` **26/26**, `float` **18/18**.

#### 5.4 Benchmark deltas

These are host-serialization-removal and memory changes, not a measurable
wall-clock speedup on this GPU-bound workload (W7-X solve ≈ 7.2 s either way);
the "no regression" evidence is the identical trajectory (same iteration
counts → same GPU work) plus strictly fewer host barriers/copies.

- **Host-blocking:** the refresh-cadence fence + 3 synchronous D2H copies are
  removed (folded into the one per-pass control copy). Per effective iteration
  the host blocks once (the control fence) instead of 5 times (pre-6A).
- **Submission:** per refresh pass, 4 D2H copies replaced by 1 async copy of 6
  scalars + 1 reduction kernel launch.
- **Memory:** cuFFT work area ≈ 10.4 MB → ≈ 5.5 MB for the W7-X double shape.

### 6. Deferred (documented, not hidden)

- **Event-DAG scratch reuse** (§6.5/§8.7). The de-alias and rCon/zCon ζ-scratch
  buffers could alias (they are sequential on the single stream, and rCon/zCon
  is the larger), saving ~1.2 MB for W7-X. Deferred: the aliasing would be
  *unsafe* under the future multi-stream/graph architecture (§6.8
  `ScratchLease`), so it is better introduced together with the event-DAG
  lifetime machinery (Phase 7+) than as a hardcoded alias that must later be
  undone.
- **Full §6.9 device terminal-predicate + guarded kernels.** The delivered
  one-fence form is host-decision-after-one-fence (§3.3); the device
  `status_bits` + no-op-guarded field/force/constraint/preconditioner kernels
  and the §6.9 `ControlRecord` with `status_bits` remain as a follow-up if the
  self-healing argument is ever invalidated by a new persistent cache.
- **`cumes_benchmark_fixed_iteration`** (§8.1). A fixed-iteration benchmark
  harness with median/p95 wall microseconds, host-blocking counts, arena/cuFFT
  bytes, and residual/state hashes was not built; the deltas above are
  analytical. It should precede the Phase 7 transform-specialization work,
  where measured speedups are the acceptance criterion.

### 7. Next steps (Phase 7 — transform specialization)

Blueprint §11 Phase 7: axisymmetric transform + constraint/bandpass backend,
weighted R/Z constraint accumulation fused into the inverse DFT, bounded
theta/zeta/mode tiles, pack/recover transpose experiments, generalized de-alias
coverage. The explicit-stream foundation and the one-fence control path from
this phase are the substrate those transform-specialization kernels enqueue on.

---

<a id="phase-7-handover"></a>

## 2026-08-16 00:51:31+08:00 — cuMES Phase 7 handover — transform specialization

**Former path:** `docs/phase-7-handover.md`
**First tracked:** [`a59fff1`](https://github.com/12ff54e/cuMES/commit/a59fff1078c19d4d74f52e75ff38113b48deb991) at 2026-08-16T00:51:31+08:00

Status date: 2026-08-16. Branch: `overhaul` (Phase 0 `bd26857` + Phase 1
`12bcc44` + Phase 2 `168170a` + Phase 3 `c21564c` + Phase 4 `1b0d099` + Phase 5
`2b9aaf8` + Phase 6 `759f933` + this Phase 7 work). This document records what
Phase 7 of `docs/cuda-overhaul-blueprint.md` delivered, how it was verified, and
what was deliberately deferred and why.

### 1. Scope

Phase 7 is **transform specialization** (§8.4–§8.7): specialize the spectral
transforms for the axisymmetric case and fold the constraint's duplicate
transform work into the main inverse accumulator. The blueprint lists five
deliverables:

| Blueprint § | Deliverable | Status |
| ----------- | ----------- | ------ |
| §8.5 | Axisymmetric transform + constraint/bandpass backend | **Done** (commit `26b66d0`) |
| §8.4 | Weighted R/Z constraint accumulation fused into inverse | **Done** (commit `05a59d1`) |
| §8.6 | Bounded theta/zeta/mode tiles | **Deferred** (§5) |
| §8.7 | Pack/recover transpose experiments | **Deferred** (§5) |
| §8.7 | Generalized de-alias coverage | **Already done** (Phase 0 containment, §6) |

The two delivered items are the structurally-driven transform specializations
with concrete differential/bitwise gates. The two deferred items are
performance experiments whose acceptance criterion (§8.1) is a measured speedup
on a benchmark harness that is still deferred from Phase 6.

### 2. What changed

#### 2.1 Axisymmetric transform backend (`26b66d0`)

For `ntor = 0, nzeta = 1` the toroidal direction is a single point and every
folded mode has n = 0, so the product basis collapses to
`R = Σ rmncc·cos(mθ)`, `Z = Σ zmnsc·sin(mθ)`, `λ = Σ lmnsc·sin(mθ)` with zero
toroidal derivatives and no sin(nζ) families. The generic backend produces
exactly this after its length-one Z2D/D2Z, so `AxisymmetricOperator` performs
the same poloidal synthesis/projection directly and never creates or executes a
length-one cuFFT plan.

- **`include/cumes/transforms/axisymmetric_operator.hpp`** + `src/
  axisymmetric_impl.cuh` + `_double.cu`/`_float.cu`: a concrete
  `SpectralOperator` backend owning its per-mode trigonometric tables. It
  implements `enqueue_inverse` (18 parity-split geometry arrays) and
  `enqueue_forward` (6 spectral families), plus the axisymmetric constraint
  helpers `enqueue_rzcon` (xmpq-weighted rCon/zCon) and `enqueue_dealias`
  (the bandpass as a direct poloidal sum).
- **Interface refinement**: `SpectralOperator::enqueue_forward` gains a
  `ConstraintForceViews` parameter — the forward DFT folds `xmpq·frcon/fzcon`
  into the R/Z projections (blueprint §4.8), so the abstract contract now
  names the real input. `ConstraintForceViews` was added to
  `real_fields.cuh`.
- **`constraintDealiasBandpass` extracted** from `constraintCompute` so the
  bandpass is testable in isolation (the axisymmetric backend replaces exactly
  that step). This is a pure code move — the Solovev dump tree is byte-identical
  (Class A).

#### 2.2 Weighted R/Z fusion (`05a59d1`)

The main inverse poloidal accumulator already stages every per-m R/Z ζ-signal,
so it now accumulates the `xmpq = m(m-1)`-weighted rCon/zCon sums at the same
time — removing the constraint's separate pack + zeta inverse + accumulation
(`constraintRzConCompute`'s rzCon path) from the hot loop.

- **`inverseAccumulateKernel` gains a `FuseRzCon` non-type parameter** (guarded
  by `if constexpr`): the `FuseRzCon=false` geometry path is bit-identical to
  the pre-change kernel (verified by byte-comparison), while the `=true` path
  additionally writes rCon (from the R-slot launch) and zCon (from the
  Z-slot launch) as full real-space fields.
- **`inverseDFTFused`** is the fused entry point; the solver calls it once per
  pass and drops the separate `constraintRzConCompute` launch. The reference
  rzCon path (function + `plan_z2d_rz` + `d_zeta_real_rz`/`d_zeta_spectra_rz`)
  is retained for the differential test and the two constraint tests that still
  read it.

### 3. Key design decisions

1. **Axisymmetric backend as a separate, selectable operator — not wired into
   the production hot loop.** The blueprint §8.5 gate is "runs both backends
   and compares every transform product"; the acceptance is a differential test
   against the retained generic backend, not an immediate swap-in. Wiring
   Solovev through `AxisymmetricOperator` is a Class B trajectory change
   (re-freeze) left as the follow-up (§5).
2. **`FuseRzCon` as a compile-time `if constexpr`, not a runtime branch.** The
   concern is real: adding live registers/statements to the hottest kernel can
   change `--use_fast_math` FMA contraction of the existing expressions and
   perturb the geometry at ULP level (the same hazard the `dynSharedBase()`
   indirection guards against). The template parameter makes the non-fused
   instantiation identical to the old kernel, so the Class A bitwise gate holds
   for the plain `inverseDFT` path while the fused path is a clearly-scoped
   Class B change.
3. **Retained reference paths, not eager deletion.** Both `constraintRzConCompute`
   and the axisymmetric constraint helpers keep their cuFFT/reference
   counterparts available for differential tests. The ~2.8 MB rzCon scratch is
   still allocated (the reference path owns it); retiring it belongs with the
   broader reference-retirement in Phase 10, not a Phase 7 cleanup that would
   remove the very oracle the differential tests compare against.

### 4. Verification

#### Differential tests (the per-intermediate gate, §8.5/§8.4)

| Test | What it compares | Result |
| ---- | ---------------- | ------ |
| `test_axisym_backend` | `AxisymmetricOperator` vs the generic cuFFT backend on frozen axisymmetric inputs: inverse 18 arrays, forward 6 families, rCon/zCon, gCon | PASS — double ≤ 5e-16, float ≤ 3e-8 |
| `test_rzcon_fusion` | fused rCon/zCon vs `constraintRzConCompute`; fused geometry bitwise vs `inverseDFT` | PASS — double 1.4e-14, float 7.6e-6; geometry bitwise |

#### Class A bitwise gate

The `FuseRzCon=false` geometry path and the `constraintDealiasBandpass`
extraction are pure moves: the full Solovev dump tree (235 files) is
byte-identical to the Phase-6 baseline after both commits.

#### Class B trajectory gate (`compare_runs.py` vs pre-change baselines)

| Config | Restart sequence | Convergence | Final residual | Final state (interior) |
| ------ | ---------------- | ----------- | -------------- | ---------------------- |
| Solovev | identical (0 events) | 456 = 456 | identical (rel 0) | ≤ 9e-14 |
| W7-X | identical (15 events) | 2011 = 2011 | identical (rel 0) | ≤ 4e-9 |

The W7-X restart sequence (5 BADP + 10 BADJ at identical iterations) is the
critical controller gate: the fusion changes the constraint force at the ULP
level but does not change a single branch decision. The near-axis λ state
spreads the most (4e-9), consistent with the known near-degenerate λ-gauge
amplification noted in the Phase 5/6 handovers.

#### Test matrix

| Preset | Result |
| ------ | ------ |
| `verify` (double, both backends) | **30/30** (20 unit + 10 compute-sanitizer memcheck) |
| `float` | compiles; both new tests run the float leg |

### 5. Deferred (documented, not hidden)

- **Bounded mode tiles (§8.6).** The kernels are already ζ-tiled (`computeKTile`
  from the Phase 0 containment) and the de-alias analyze loops over 32-point
  θ-groups, so the *bounded-launch* part is done; the remaining gap is **mode
  tiles** (the per-thread m-loop is still serial `O(mpol)`). This is a
  shared-memory tiling rewrite whose acceptance (§8.1) is a *measured* speedup.
  It should follow the benchmark harness below, not precede it.
- **Pack/recover transpose experiments (§8.7).** Shared-memory tiled transpose
  vs `cufftPlanMany` embedding/stride vs the current pack/recover — a
  benchmark comparison, not a code change to land without evidence.
- **`cumes_benchmark_fixed_iteration` (§8.1, deferred since Phase 6).** The
  fixed-iteration harness (median/p95 wall µs, host-blocking counts, arena/cuFFT
  bytes, residual/state hashes, per-kernel regs/spills/occupancy) is the
  prerequisite for the two deferred items above; it should be the first task of
  the next phase.
- **Production wiring of `AxisymmetricOperator`.** Solovev still runs the
  generic cuFFT backend; swapping in the axisymmetric operator is a Class B
  trajectory change (re-freeze) once the operator has a short-trajectory gate
  beyond the component differential test.
- **rzCon plan/scratch retirement.** The `plan_z2d_rz` + ~2.8 MB compact
  scratch are still allocated (the reference `constraintRzConCompute` owns
  them); retiring them belongs with Phase 10 reference-retirement after the
  differential tests become permanent regressions.

### 6. Generalized de-alias coverage (already delivered)

The Phase 0 containment already generalized the de-alias analysis to loop over
32-point θ-groups (so `ntheta > 32` is fully summed, not silently truncated);
`test_regression_kernels` and its sanitizer variant cover the awkward angular
shapes. No further Phase 7 work is required for this item.

### 7. Next steps

1. Land `cumes_benchmark_fixed_iteration` (§8.1) — the measurement harness that
   gates every remaining performance claim.
2. With the harness in place, attempt the §8.6 mode tiles and the §8.7
   pack/recover transpose, selecting by measured shape as the blueprint
   requires.
3. Wire `AxisymmetricOperator` into the Solovev production path behind the
   retained generic backend, with a short-trajectory + full-regression
   re-freeze (Class B).
4. Retire the reference rzCon path (plan/scratch) once the differential tests
   are permanent regressions (Phase 10 scope).

---

<a id="phase-8-handover"></a>

## 2026-08-16 09:53:02+08:00 — cuMES Phase 8 handover — scalable preconditioner and reductions

**Former path:** `docs/phase-8-handover.md`
**First tracked:** [`1069939`](https://github.com/12ff54e/cuMES/commit/1069939eb6061a8d208f5bba14f938c5f7604df9) at 2026-08-16T09:53:02+08:00

Status date: 2026-08-16. Branch: `overhaul` (Phase 0 `bd26857` + Phase 1
`12bcc44` + Phase 2 `168170a` + Phase 3 `c21564c` + Phase 4 `1b0d099` + Phase 5
`2b9aaf8` + Phase 6 `759f933` + Phase 7 `a59fff1` + this Phase 8 work). This
document records what Phase 8 of `docs/cuda-overhaul-blueprint.md` delivered,
how it was verified, and what was deliberately deferred and why.

### 1. Scope

Phase 8 is **scalable preconditioner and reductions** (§8.8, §8.9, §4.9): make
the radial tridiagonal solve backend-neutral and scale-aware, and replace the
shared-memory reduction trees with warp shuffles. The blueprint lists four
deliverables:

| Blueprint § | Deliverable | Status |
| ----------- | ----------- | ------ |
| §8.9 | Backend-neutral batched tridiagonal API + scalable implementation | **Done** (`f58acd6`) |
| §4.9 | Scale-aware pivot/breakdown status | **Done** (`f58acd6`) |
| §8.8 | Warp/CUB reductions with deterministic verification mode | **Done** (`fa37b7e`, Class B) |
| §8.8 | Optional refresh-stream overlap after measurement | **Deferred** (§5) |

The first three are the self-contained numerical contracts of the phase. The
fourth is gated on the benchmark harness (`cumes_benchmark_fixed_iteration`,
deferred since Phase 6), so it stays deferred.

### 2. What changed

#### 2.1 Backend-neutral tridiagonal solve + scale-aware pivot (`f58acd6`)

- **`include/cumes/numerics/tridiagonal_backend.hpp`** (fleshed out from the
  Phase-5 stub): a concrete `StridedBatchTridiagonalView` carrying the
  `lower/diagonal/upper` coefficients, a `rhs` strided batch (`rhs_count` RHS
  planes sharing one elimination, `rhs_stride` between planes), the per-mode
  `first_surface` (jMin) and `scale`, and a shared `last_surface` (the excluded
  LCFS row). `TridiagonalStatus` and `PivotPolicy` name the pivot contract.
- **Two concrete backends** in `src/precon_impl.cuh`:
  - `PcrBackend` — the production 128-thread grid-stride PCR, extracted
    **bit-for-bit** from the legacy `tridiagSolveKernel` (the `rhs_count` loop
    is a compile-time template parameter so the `#pragma unroll` and the
    `-use_fast_math` FMA contraction of the staged arithmetic are preserved).
  - `ThomasBackend` — a serial Thomas reference (one thread per system), the
    small-batch fallback named in §8.9.
- **`preconApply` routes through `PcrBackend`**: the R (comps 0,3) and Z
  (comps 1,4) systems are two `enqueue_solve` calls with `rhs_count=2`, and the
  boundary zeroing + lambda-diagonal finishing move to a separate
  `preconBoundaryKernel`. The legacy `tridiagSolveKernel` is gone; the legacy
  `preconCompute`/`preconApply` signatures are unchanged so the solver and the
  regression tests keep their call sites.

#### 2.2 Scale-aware pivot/breakdown (`f58acd6`)

The legacy **absolute** `1e-30` pivot clamp (which silently flipped negative
pivots to `+1e-30` before the Phase-0 copysign fix, and still silently clamped
near-singular diagonals) is replaced by the blueprint §4.9 scale-aware floor:

```
floor = kappa * eps_T * scale[mode],   scale = max |lower|,|diagonal|,|upper|
```

- `preconScaleKernel` computes `scale[mode]` once per preconditioner refresh
  (the matrix changes only on refresh) into a new `PreconWorkspace::d_preconScale`.
- The solve kernels read it and **guard with `copysign` AND count the breakdown**
  into a device `d_preconStatus` accumulator (`atomicAdd`), instead of silently
  clamping. A genuinely sub-scale diagonal is now *reported*, not absorbed.
- For the frozen trajectories the diagonals are O(1)..O(m²), far above the
  floor, so the guard never fires and the solve is byte-identical (Class A).

#### 2.3 Warp-shuffle reductions (`fa37b7e`, Class B)

`preconComputeKernel` (15 accumulators) and `lambdaPrecAssembleKernel` (3
accumulators) drop their shared-memory binary trees for a **fixed**
`__shfl_down_sync` within-warp tree plus a fixed cross-warp combine by thread 0.
The tree is fixed, so the result stays deterministic; only the summation order
changes (Class B). `preconComputeKernel` no longer needs dynamic shared memory
(launch `smem` 30720 → 0), and `lambdaPrecAssembleKernel`'s per-block shared
shrinks from 6 KB to 192 B.

### 3. Key design decisions

1. **`rhs_count` as a compile-time template parameter on `pcrSolveKernel`, not
   a runtime loop bound.** The legacy kernel's `#pragma unroll` over the two RHS
   planes is part of the frozen codegen; a runtime bound would let the compiler
   emit a different (non-unrolled) loop and risk a `-use_fast_math` FMA
   contraction difference. `enqueue_solve` dispatches on `rhs_count == 1/2`
   (production always uses 2), preserving bitwise parity.
2. **Scale computed in a separate kernel, not inline in the PCR kernel.** Adding
   a block reduction to the hot PCR kernel would risk changing its register
   allocation/codegen and the frozen arithmetic. `preconScaleKernel` runs once
   per refresh; the PCR kernel only reads `scale[mode]`, so its staged
   arithmetic is untouched.
3. **Backends operate on raw strided pointers, not `SpectralView`.** The
   tridiagonal solve is a generic batched operator; it should not know the
   six-component spectral layout. `preconApply` builds the views from the
   residual slab (`rhs_stride = 3*mnmax*ns` for the (0,3)/(1,4) component pairs),
   keeping the spectral-layout knowledge at the preconditioner boundary.
4. **The breakdown status is detected but not yet folded into the control
   record.** `d_preconStatus` accumulates every pass and is reset by
   `preconApply`; folding it into the host `ControlRecord`/`NumericalStatus`
   decision is a follow-up. The frozen trajectories never trigger it, so there
   is no behavioral change yet — but the *detection* now exists and is gated by
   `test_tridiagonal`.
5. **The old absolute `1e-30` is gone entirely**, not kept alongside the
   scale-aware floor. The new guard is structurally identical (same ternary
   `hasL/hasR` + `invL = hasL ? 1/dL : 0`), only the floor literal becomes a
   runtime scale-derived value, so the healthy path is bitwise-identical while
   the near-singular path is reported instead of silently clamped.

### 4. Verification

#### Class A bitwise gate (the 8.1 extraction)

The PCR extraction is a pure move: the Solovev trajectory reproduces the frozen
values exactly — stages **251 → 199 → 456**, final **FSQR = 9.583e-17**,
total 906 effective iterations (matching the Phase-5/6/7 handovers and the
CLAUDE.md baseline).

#### Class B trajectory gate (the 8.3 reductions, `compare_runs.py` vs the phase-7 tag)

| Config | Restart sequence | Convergence | Final residual | Final state (interior) |
| ------ | ---------------- | ----------- | -------------- | ---------------------- |
| Solovev | identical (0 events) | 456 = 456 | identical (rel 0) | **bitwise-identical (0.00e+00)** |
| W7-X | identical (15 events) | 2011 = 2011 | identical (rel 0) | ≤ 1.25e-9 |

The W7-X restart sequence (5 BADP + 10 BADJ at identical iterations) is the
critical controller gate, and it is unchanged. The near-axis λ families
(lmnsc/lmncs) spread the most (1.25e-9), consistent with the known
near-degenerate λ-gauge amplification noted in the Phase 5–7 handovers (the
Phase-7 fusion spread was 4e-9, so this is tighter). Solovev happens to be
bitwise-identical because its `nZnT = 18` surface sums are small enough that the
tree/shuffle order produces identical double roundings.

#### Test matrix

| Preset | Result |
| ------ | ------ |
| `verify` (double, both backends) | **32/32** (21 unit + 11 compute-sanitizer memcheck) |
| `float` | compiles; `test_tridiagonal` runs both legs (double + float) |

`test_tridiagonal.cu` (new, `unit;precon` + sanitizer variant) drives the public
`TridiagonalBackend` interface directly: CPU serial Thomas vs GPU Thomas vs GPU
PCR across `ns = {3, 17, 65, 99, 130, 257, 512}`, mixed m-parity jMin,
`rhs_count = 2`, plus a zero-diagonal breakdown case (both backends must report
`status > 0`). The existing `test_regression_kernels.cu` still exercises the
production `preconCompute` + `preconApply` path (now through `PcrBackend`) on
real geometry at the same row-count matrix.

### 5. Deferred (documented, not hidden)

- **Tiled PCR/Thomas hybrid and a library backend** (§8.9 "long-term,
  benchmark"). The current `PcrBackend` is grid-stride (arbitrary rows) and its
  `10·ns·sizeof(T)` shared memory fits the validated `ns ≤ 512` range. A tiled
  hybrid or a strided-batch library solver is a *measured-speedup* item that
  should follow the benchmark harness, not precede it.
- **Refresh-stream overlap** (§8.8 "after measurement"). The auxiliary-stream
  overlap of preconditioner/force-norm work is gated on the still-deferred
  `cumes_benchmark_fixed_iteration` harness (deferred since Phase 6).
- **Folding `d_preconStatus` into the control record.** The breakdown is
  detected and accumulated; wiring it into the host `ControlRecord`/status bits
  so the solver can act on it is a follow-up (the frozen trajectories never
  trigger it, so this is a dormant diagnostic today).
- **`PivotPolicy` as a runtime knob.** `PivotPolicy.kappa` is wired through the
  backend constructors, but production uses the default `kappa = 1.0`; a
  documented policy surface (e.g. a degenerate `kappa = 1e30` to reproduce the
  legacy absolute clamp in tests) is not needed until a second policy appears.

### 6. Next steps

1. Land `cumes_benchmark_fixed_iteration` (§8.1) — the prerequisite for the
   tiled-hybrid backend, the refresh-stream overlap, and every remaining
   measured-performance claim.
2. With the harness, benchmark the `ThomasBackend` for the small axisymmetric
   shape and the `PcrBackend` for W7-X, then decide whether a tiled PCR/Thomas
   hybrid earns its complexity.
3. Fold `d_preconStatus` into the combined `ControlRecord` so a singular
   preconditioner is a structured numerical failure, not a dormant diagnostic.
4. Retire the legacy `preconCompute`/`preconApply` free-function signatures once
   the `Preconditioner` operator class (still a Phase-5 stub) is implemented
   behind them.

---

<a id="phase-9-handover"></a>

## 2026-08-16 10:38:31+08:00 — cuMES Phase 9 handover — graphs and high-risk fusion

**Former path:** `docs/phase-9-handover.md`
**First tracked:** [`bda03f8`](https://github.com/12ff54e/cuMES/commit/bda03f88e7bec349018668d5b02fa835fbefe9df) at 2026-08-16T10:38:31+08:00

Status date: 2026-08-16. Branch: `overhaul` (Phase 0 `bd26857` + Phase 1
`12bcc44` + Phase 2 `168170a` + Phase 3 `c21564c` + Phase 4 `1b0d099` + Phase 5
`2b9aaf8` + Phase 6 `759f933` + Phase 7 `a59fff1` + Phase 8 `1069939` + this
Phase 9 work). This document records what Phase 9 of
`docs/cuda-overhaul-blueprint.md` delivered, how it was verified, and what was
deliberately deferred and why.

### 1. Scope

Phase 9 is **graphs and high-risk fusion** (§8.10–§8.12). Its exit gate is
unusual among the phases: *"each backend has an ADR, differential tests, and
measured benefit on named hardware. Unsuccessful experiments are removed rather
than becoming maintenance paths."* The four deliverables and their outcomes:

| Blueprint § | Deliverable | Status |
| ----------- | ----------- | ------ |
| §8.1 | Fixed-iteration benchmark harness | **Done** (`a01fd37`, `67625a1`) |
| §8.12 / §8.8 | Mixed-float double reductions | **Done, adopted** (`4769fda`, `85fda96`) |
| §8.10 | Force split prototype (R/Z vs lambda) | **Done, measured, not adopted** (`2d05071`) |
| §8.11 | CUDA Graph variants | **Measured, integration deferred** (`e6602eb`) |
| §8.10 / §8.3 | Fused descent/checkpoint + force/projection | **Deferred** (§5) |

Three of the four were measured on the TITAN Xp (sm_61, 12 GB) — the named
hardware the acceptance policy requires — and one (double reductions) is adopted
into the float build.

### 2. What changed

#### 2.1 Benchmark harness (`a01fd37`, `67625a1`, §8.1)

`cumes_benchmark_fixed_iteration` runs one radial stage at a config's final
shape (Solovev ns=55, W7-X ns=99) with `ftol=0` (never converges), discards a
warm-up, and emits JSON: GPU identity, shape, arena/cuFFT bytes, setup vs solve
time, median/p95 wall µs/iter, and an FNV final-state hash.

- The seed construction (`init_params`/`init_state`/`restart_state`) was moved
  verbatim out of `main.cu` into `include/cumes/state/seed_state.hpp` so the
  harness and CLI share one copy (a pure relocation; Solovev bit-identical).
- A new opt-in `cumes::SolverBench` observer is threaded through `solverRun`
  and sampled at the single control fence. It is pure host observability —
  `bench == nullptr` in production leaves the hot loop byte-identical.
- Steady-state TITAN Xp: **Solovev ~213 µs/iter, W7-X ~2.11 ms/iter**.
  Registered as a 2-pass CTest smoke gate (`performance;smoke`).

#### 2.2 Mixed-float double accumulation (`4769fda`, `85fda96`, §8.8/§8.12)

`cumes::NormAccum<T>` (`float → double`, `double → double`) now drives the
accumulator of the three control-feeding reductions — `computeResidualsKernel`,
`rzNormKernel`, `forceNormReduceKernel`. The *terms* keep their `T` arithmetic;
only the summation width changes. ADR-0001.

- **Class A** for the double build (`NormAccum<double> == double`). After the
  refinement commit (`85fda96`) the term/result expressions are byte-for-byte
  the originals, so bit-identity is by construction, not by compiler identity-
  cast elision.
- **Class B** for the float build. `test_accumulation.cu` shows a ~20× lower
  summation error on a 1M-term dynamic-range case (4.6e1 → 2.3e0). It does
  **not** unlock a lower float `ftol` (the float-state floor dominates); a full
  double `ControlRecord` is the larger follow-up (ADR-0001).

#### 2.3 Force split prototype (`2d05071`, §8.10)

`computeForcesSplit` splits the monolith into `rzForcesKernel` (12 families) +
`lambdaForcesKernel` (4 families), copied verbatim, behind the retained
monolith. ADR-0002.

- ptxas (sm_61): monolith **108 registers, 0 spills**; split **82 / 54**.
- `test_force_split.cu`: **bit-identical** (max |diff| = 0 across all sixteen
  families, double + float), but **1.20–1.45× slower** (the kernel is
  input-traffic-bound, so doubling the geometry/field loads dominates the
  occupancy win). Not adopted; retained only as the differential gate.

#### 2.4 CUDA Graph primitive + measurement (`e6602eb`, §8.11)

- `cumes::CudaGraph` (capture/instantiate/launch RAII) in
  `include/cumes/runtime/cuda_graph.hpp`.
- `test_cuda_graph.cu` proves a three-kernel DAG **and a cuFFT inverse
  transform** replay bit-identically to stream execution — cuFFT-in-graph works
  on CUDA 12.1 / sm_61 with the Phase-6B explicit work area.
- `cumes_benchmark_graph_overhead` (empty-kernel microbenchmark): enqueue
  1.94 µs/kernel, graph launch 6.43 µs/pass for 24 kernels, **~40 µs/pass
  savings upper bound**; capture+instantiate 436 µs/variant. That is ~19% of
  Solovev (submission-bound) vs ~2% of W7-X (GPU-bound). ADR-0003 defers the
  four-variant solver integration until the Phase-7 `AxisymmetricOperator` is
  wired and a real-kernel re-measure confirms the bound.

### 3. Key design decisions

1. **"Measured, then adopted or removed."** Each §8.10/§8.11 backend was built
   behind the production path (never wired into `solverRun`), given a
   differential test, measured on the TITAN Xp, and then either adopted (double
   reductions) or documented as not-adopted (force split, graphs) — matching the
   exit gate. The one *adopted* change is the cheapest and only Class-A-for-
   double one.
2. **Bit-identity by construction, not by assumption.** The 9.5 refinement
   (`85fda96`) removed the identity `A(...)`/`T(...)` casts so the double build's
   term expressions are verbatim — an explicit `double(x)` cast on an already-
   double expression *should* be elided, but removing it removes any doubt that
   `-use_fast_math` FMA contraction is preserved. This was driven by the W7-X
   stage-1 check (below).
3. **The force split is a negative result kept as a gate, not a backend.** Its
   value is that it empirically classifies the force path as input-traffic-bound
   (not register-bound), which redirects §8.10's remaining experiments (radial
   tile, force+projection fusion) away from register-reduction strategies.
4. **Graphs are measured but not integrated.** The empty-kernel savings are an
   upper bound and only meaningful for the submission-bound axisymmetric shape;
   the higher-leverage change for that shape is the already-built (Phase 7)
   `AxisymmetricOperator`, so wiring graphs before it would optimize the wrong
   thing.

### 4. Verification

#### Class A bitwise gate (the critical gate)

Both configs reproduce the frozen trajectory byte-for-byte against the Phase-8
baseline `f83a3ca`:

| Config | Effective iterations | Final residual | Final state |
| ------ | -------------------- | -------------- | ----------- |
| Solovev | 251 → 199 → 456 | FSQR 9.583e-17 | bit-identical |
| W7-X | 1877 → 1617 → 2011 (5505) | FSQR 9.778e-13 | bit-identical (`compare_states.py` 0.0) |

**Note on the recorded W7-X count.** `CLAUDE.md` and `README.md` still say
"1878 → 1617 → 2011 (5506)", but the *actual* frozen baseline — recorded in the
Phase 5 and Phase 6 handovers ("byte-identical … W7-X 1877 → 1617 → 2011") and
reproduced here by building `f83a3ca` directly — is **1877 → 1617 → 2011
(5505)**. The stage-1 count moved 1878 → 1877 with the Phase-6B device-only
force-norm reduction (a Class B reorder); `CLAUDE.md`/`README` were not updated.
This Phase-9 handover is the corrected record.

#### Test matrix

| Preset | Result |
| ------ | ------ |
| `verify` (double, sanitizers ON) | **39/39** (24 unit + 14 compute-sanitizer memcheck + 1 smoke) |
| `float` | compiles; `test_accumulation`, `test_tridiagonal`, `test_force_split` run both legs |

New tests: `test_accumulation` (double-accum policy), `test_force_split`
(monolith ≡ split + timing), `test_cuda_graph` (graph + cuFFT-in-graph), and the
`cumes_benchmark_smoke` CTest gate.

### 5. Deferred (documented, not hidden)

- **Four-variant solver graph integration** (ADR-0003). Deferred until (a) the
  Phase-7 `AxisymmetricOperator` is wired into the Solovev production path, and
  (b) a real-kernel (non-empty) re-measure confirms the ~40 µs upper bound. It
  also needs a per-pass kernel-parameter update for the `m=1` gauge `zeroZ`
  scalar and the `ncurr=0` first-pass `update_iota_chi` case, so it is not a
  plain four-variant capture.
- **Fused descent/checkpoint** (§8.3 option 2, Phase 9 "optional"). Writes the
  post-descent state to the checkpoint from registers; a Class B change to the
  delicate post-descent checkpoint ordering. The benchmark harness now exists to
  measure it.
- **Force-plus-projection fusion** (§8.10, high risk). The force-split result
  (input-traffic-bound) makes this unlikely to pay; needs its own measured gate.
- **Full double `ControlRecord`** (ADR-0001). Thread double invariants through
  the host controller, not just the device accumulation.
- **Retirement of the force-split prototype and the Phase-7 rzCon reference
  path** — Phase 10 scope, once the differential tests are permanent regressions.

### 6. Next steps

1. Wire `AxisymmetricOperator` into the Solovev production path (Phase 7
   follow-up) — the highest-leverage change for the submission-bound shape, and
   the prerequisite for re-measuring graph benefit.
2. With that in place, re-measure graph submission savings with real kernels and
   decide on the four-variant integration (ADR-0003).
3. Land the §8.3 fused descent/checkpoint behind the benchmark harness.
4. Phase 10: retire the now-superseded prototypes (force split, rzCon reference
   path, `computeForcesSplit`), correct the stale W7-X count in
   `CLAUDE.md`/`README.md`, and publish the ADR-backed performance/architecture
   docs from these tested contracts.

---

<a id="phase-10-handover"></a>

## 2026-08-16 11:06:09+08:00 — cuMES Phase 10 handover — retire compatibility internals

**Former path:** `docs/phase-10-handover.md`
**First tracked:** [`c624eb6`](https://github.com/12ff54e/cuMES/commit/c624eb603b0c063bf208b2f5bce1c23856d3c5c6) at 2026-08-16T11:06:09+08:00

Status date: 2026-08-16. Branch: `overhaul` (Phase 0 `bd26857` + Phase 1
`12bcc44` + Phase 2 `168170a` + Phase 3 `c21564c` + Phase 4 `1b0d099` + Phase 5
`2b9aaf8` + Phase 6 `759f933` + Phase 7 `a59fff1` + Phase 8 `1069939` + Phase 9
`bda03f8` + this Phase 10 work). This document records what Phase 10 of
`docs/cuda-overhaul-blueprint.md` delivered, how it was verified, and what was
deliberately left for the tail of the migration.

### 1. Scope

Phase 10 is **retire compatibility internals** (blueprint §11, last phase). Its
exit gate is: *"source dependency graph matches section 5, all release gates
pass from a clean clone, and no legacy code is required for normal execution."*
The concrete work this phase delivered is the retirement of the two
measured-but-not-adopted Phase 9 prototypes, the correction of a stale
documented iteration count, and the publication of the tested contracts as
standalone docs:

| Item | Status |
| ---- | ------ |
| Retire the §8.10 force-split prototype | **Done** (`c9caeec`) |
| Retire the Phase-7 rzCon reference path | **Done** (`c2b467e`) |
| Correct the stale W7-X count in CLAUDE.md/README.md | **Done** (`c52a84f`) |
| Publish architecture/data-layout/performance docs | **Done** (`7b60910`) |
| Delete fixed-capacity `InputParams` / raw owning workspaces / old kernels | **Blocked** (§4) |
| Freeze `configs/schema-v1.json` artifact | **Deferred** (§4) |

### 2. What changed

#### 2.1 Force-split prototype retired (`c9caeec`)

Removed `computeForcesSplit`, `rzForcesKernel`, `lambdaForcesKernel`, the
`computeForcesSplit` declaration and double/float explicit instantiations,
`test_force_split.cu`, and all CMake references. `computeForces` (the monolith)
remains the sole production force path, pinned by `test_force_reference.cu`
(scalar CPU reference) and the frozen trajectories. ADR-0002 now records the
retirement; its durable conclusion — the force kernel is input-traffic-bound,
so register-reduction strategies do not pay — is preserved.

#### 2.2 rzCon reference path retired (`c2b467e`)

Removed `constraintRzConCompute`, `rzConPackKernel`, `rzConAccumulateKernel`,
the `ConstraintWorkspace` rzCon round-trip (`d_zeta_real_rz`,
`d_zeta_spectra_rz`, `plan_z2d_rz`) and its cuFFT work-area accounting, the
double/float instantiations, `test_rzcon_fusion.cu`, and all CMake references.

Since Phase 7.2 the fused `inverseDFTFused` has been the production rCon/zCon
producer; the reference path was retained only as the oracle for the
differential tests. `test_constraint_tcon` and `test_axisym_backend` now
produce rCon/zCon via `inverseDFTFused` (Class B ULP-equivalent), so the
reference is no longer needed. This also drops ~2.8 MB of W7-X-double constraint
scratch (`plan_z2d_rz` + the 4-slot compact batch) and one cuFFT plan per stage.

#### 2.3 Stale W7-X count corrected (`c52a84f`)

CLAUDE.md and README.md recorded the W7-X run as `1878 → 1617 → 2011 (5506)`.
The frozen baseline (Phase 6B onward) is `1877 → 1617 → 2011 (5505)` — the
stage-1 count moved with the Phase-6B device-only force-norm reduction and the
docs were never updated. Corrected in both files.

#### 2.4 Docs published (`7b60910`)

- `docs/architecture.md` — the two-layer (legacy kernels + `cumes` scaffold)
  reality, the build/library split, the production per-iteration pipeline, the
  dependency rules, and the remaining strangler-fig tail.
- `docs/data-layout.md` — the spectral/real/half-grid layouts, parity and
  mode numbering, the reduced-theta quadrature, and the legacy binary v0
  contract.
- `docs/performance.md` — the measured TITAN Xp numbers, the Phase 9
  adopted/not-adopted outcomes (ADR-0001..0003), and the §10.7 acceptance
  policy.
- `docs/adr/0002-force-split.md` — status-update section recording the
  retirement.

### 3. Verification

Both retirements are **pure dead-code removals** — the production `solverRun`
path never called `computeForcesSplit` or `constraintRzConCompute`, so no
numerical expression, kernel, launch, or scheduling decision changed. The
trajectory was re-run against the frozen baseline:

| Config | Effective iterations | Final FSQR | Restarts | Verdict |
| ------ | -------------------- | ---------- | -------- | ------- |
| Solovev | 251 → 199 → 456 | 9.583e-17 | 0 | bit-identical |
| W7-X | 1877 → 1617 → 2011 (5505) | 9.778e-13 | 15 (10 BADJ + 5 BADP) | bit-identical |

Test matrix:

| Preset | Result |
| ------ | ------ |
| `verify` (double, sanitizers ON) | **35/35** (23 unit + 13 compute-sanitizer memcheck + 1 smoke) — two fewer tests than Phase 9 because `test_force_split` and `test_rzcon_fusion` were removed |
| `float` | compiles cleanly (the float CLI and the float test legs) |

The W7-X restart sequence was re-confirmed as 5 BAD PROGRESS + 10 BAD JACOBIAN
(the 3 additional "invalid √g" log lines are initial invalid-geometry passes,
not restart events), matching the Phase 6/7/8 handovers.

### 4. Deferred / blocked (documented, not hidden)

The blueprint's full Phase 10 exit gate — *"no legacy code is required for
normal execution"* — is **not** met this phase, and cannot be met until the
strangler-fig migration replaces the legacy kernels outright:

- **`include/*.cuh` kernel structs (`FourierPlan`, `MetricWorkspace`,
  `InputParams`, …) are still the production implementation.** Deleting them
  is explicitly gated on "only after their consumers are gone" (blueprint §11);
  the `cumes` operator classes do not yet own the production hot loop, so this
  is a larger multi-phase migration, not a Phase 10 cleanup.
- **`AxisymmetricOperator` (Phase 7.1) is not wired into the Solovev production
  path.** It is built and gated (`test_axisym_backend`) but swapping it in is a
  Class B trajectory change (re-freeze) and remains the highest-leverage
  submission-bound-shape follow-up. It is also the prerequisite for re-measuring
  graph benefit (ADR-0003).
- **`configs/schema-v1.json` is not emitted.** The schema-v1 and checkpoint
  compatibility contracts are specified (blueprint §6.13) and exercised by the
  host I/O tests, but the standalone schema artifact is future work.
- **`docs/mathematics.md` / `docs/verification.md`** are not written separately;
  the normative mathematics is blueprint §4 and the verification tiers are
  blueprint §10, now cross-referenced from `docs/architecture.md`.

### 5. Next steps

1. Wire `AxisymmetricOperator` into the Solovev production path behind the
   retained generic backend (Class B re-freeze), then re-measure graph
   submission savings (ADR-0003).
2. Land the §8.3 fused descent/checkpoint behind the benchmark harness.
3. Continue the strangler-fig migration toward the §5 target: replace the
   legacy kernel structs with the `cumes` operator classes, then delete the
   now-unused `include/*.cuh` legacy headers and `InputParams`.
4. Emit `configs/schema-v1.json` and freeze the checkpoint compatibility policy
   as a versioned artifact.

---

<a id="strangler-fig-migration-plan"></a>

## 2026-08-16 11:34:58+08:00 — Strangler-fig migration plan — legacy kernels → `cumes` operators

**Former path:** `docs/strangler-fig-migration-plan.md`
**First tracked:** [`35416b0`](https://github.com/12ff54e/cuMES/commit/35416b061d71792a0ad9c2b08dab15e2905fa290) at 2026-08-16T11:34:58+08:00

Status date: 2026-08-16. This is the plan for the final leg of the
`docs/cuda-overhaul-blueprint.md` (§11 Phase 10 exit gate): *"source dependency
graph matches section 5, … and no legacy code is required for normal
execution."* It is the handover's "next step 3" and the one remaining large item
after the Phase 10 tail (axisymmetric wiring, schema v1).

**Progress (2026-08-16):** all five owning operators are landed and wired into
the solver — `Profiles`, `ToroidalFftOperator`, `GeometryOperator`,
`Preconditioner`, `ConstraintOperator` (commits `87cec03`, `75dcc65`) — each
owns its workspace (RadialProfiles / FourierPlan / MetricWorkspace /
PreconWorkspace / ConstraintWorkspace) and `solverRun`/`StageSolver` construct
and drive the operators instead of the raw structs. The §4 step-1 `FourierPlan`
split landed (`a692200`: transform scratch vs stage-owned `RealSpaceStorage`).
Then the `SpectralOperator` unification landed (`d03640f`): the interface grew
rCon/zCon + `enqueue_dealias`, `ToroidalFftOperator` became a
`SpectralOperator<T>` peer, the compact de-alias scratch moved into the
FourierPlan (transform-owned), and `solverRun` drives a single
`SpectralOperator<T>*` with no `axisym_active` branch. The four stateless
operators landed (`3296ae1`): `ForceOperator`/`ResidualOperator`/
`DescentOperator`/`Prolongation` as thin wrappers over `computeForces`/
`computeResidualsKernel`/`descentStepKernel`/`interpolateState`. All verified
Class A bit-identical (Solovev `251→199→456` FSQR 9.583e-17, W7-X
`1877→1617→2011` FSQR 9.778e-13, 35/35 CTest, float build clean).

Legacy-struct deletion then began with three ownership slices (all Class A):
`53ed57a` extracted the folded-mode table out of `FourierPlan` into a stage-owned
`cumes::DeviceModeTable` (`FourierBasis` deleted from `vmec_types.h`; the
transform free functions + `preconCompute` now take `const int* xm/xn`);
`67ff2fd` moved the cuFFT stream binding into `ToroidalFftOperator::bind_stream`;
`b8fa3a0` moved `m1PreconScale` into `Preconditioner::enqueue_m1_scale`.

Remaining: step 5 (base-geometry vs B/pressure split — Class B), step 12
(`EquilibriumOperator` composition), and step 13 (legacy-struct deletion),
gated on `solverRun` naming no legacy struct. Steps 5 and 12 are the blocking
predecessors of step 13. After the three slices `solverRun`'s remaining legacy
naming is:

- `FourierPlan` — only the dump-gated `inverseDFT`/`fourierCombineParity`
  observability path (the hot loop is sealed behind the operator +
  `bind_stream`); `FourierBasis`/`fp.basis` and `fp.plan_*` are already gone.
- `PreconWorkspace` / `MetricWorkspace` — named as arguments (`constraint.enqueue`
  takes `pw`; `precon.enqueue_compute`/`enqueueForceNorms` take `mw`); their
  `.d_*` field reads are now dump-only.
- `ConstraintWorkspace` / `RadialProfiles` — still read in the hot loop (the
  `RealFieldView`s over `cw.d_rCon/d_zCon/d_frcon/d_fzcon`, and `rp.d_sqrtS_F`/
  `rp.delta_s`/`rp.d_dVds_H`/`rp.d_pres_H`).
- `GridParams<T>` everywhere (needs the `DeviceParams<T>` replacement) and
  `InputParams` in `StageSolver`/`main` (needs `ValidatedProblem`).

No `DeviceParams<T>` replacement exists yet.

The term "strangler fig" is from blueprint §1: wrap the current implementation
behind tested operator interfaces and let the operators *replace* the legacy
structs one at a time, rather than rewriting the solver in one pass. Every step
below is Class A (bit-identical) unless explicitly marked Class B, so the frozen
Solovev `251→199→456` / W7-X `1877→1617→2011` trajectories are the regression
oracle at each step.

### 1. Definition of done (blueprint §14, §11 Phase 10 exit gate)

- `solverRun` drives only `cumes` operator classes; it holds no `FourierPlan`,
  `MetricWorkspace`, `PreconWorkspace`, `ConstraintWorkspace`, or `RadialProfiles`.
- The generic transform is a `SpectralOperator<T>` backend (`ToroidalFftOperator`)
  on equal footing with `AxisymmetricOperator`; `solverRun` picks one pointer, no
  `if (axisym_active)` branches.
- Transforms own only transform tables/plans/scratch — never geometry, force, or
  diagnostic arrays (blueprint §5.1).
- `include/*.cuh` legacy structs and `InputParams` are deleted; `src/*_impl.cuh`
  kernel bodies survive only as the operator implementations (renamed/moved into
  `src/cumes/` or left in place, not as free-function entries the solver calls).
- Full precise trajectories are bit-identical to the frozen baseline (or a
  documented Class B re-freeze), all CTest + Compute Sanitizer entries pass, and
  the optional-backend matrix (none/netcdf/hdf5) still compiles.

### 2. Current state (2026-08-16)

#### 2.1 Legacy layer — the *production* implementation

| Module | Header | Impl | Owned struct(s) | Solver entry points |
| ------ | ------ | ---- | --------------- | ------------------- |
| transforms | `include/fourier.cuh` | `src/fourier_impl.cuh` | `FourierPlan<T>` (transform scratch + 18 geometry + 16 force + 9 combined arrays + cuFFT plans) | `inverseDFTFused`, `forwardDFT`, `fourierCombineParity` |
| geometry | `include/geometry.cuh` | `src/geometry_impl.cuh` | `MetricWorkspace<T>` | `computeGeometry`, `computeJacobianStats`, `computeForceNormPartials` |
| forces | `include/forces.cuh` | `src/forces_impl.cuh` | — (writes `FourierPlan` force arrays) | `computeForces` |
| profiles | `include/profiles.cuh` | `src/profiles_impl.cuh` | `RadialProfiles<T>` | `profilesCreate/Free` |
| preconditioner | `include/precon.cuh` | `src/precon_impl.cuh` | `PreconWorkspace<T>` | `preconCreate/Free/Compute/Apply` |
| constraint | `include/constraint.cuh` | `src/constraint_impl.cuh` | `ConstraintWorkspace<T>` | `constraintCreate/Free/Compute/ComputeAxisym/ResetRzCon0/DealiasBandpass` |
| solver | `include/solver.cuh` | `src/solver_impl.cuh` | — (owns the loop) | `solverRun` |
| refine | `include/refine.cuh` | `src/refine_impl.cuh` | — | `interpolateState` |
| axisymmetric | — | `src/axisymmetric_impl.cuh` | `cumes::AxisymmetricOperator<T>` (already a `cumes` class) | `enqueue_*` |

Each module is split into `src/<mod>_double.cu` / `src/<mod>_float.cu` explicit
instantiation TUs (one scalar type each) under `cumes_cuda_double` /
`cumes_cuda_float`.

#### 2.2 `cumes` layer — boundaries declared, mostly unimplemented

| Boundary | Header | Implemented? |
| -------- | ------ | ------------ |
| `SpectralOperator<T>` (interface) | `transforms/spectral_operator.hpp` | interface only |
| `AxisymmetricOperator<T>` | `transforms/axisymmetric_operator.hpp` | **yes** (Phase 7.1, wired Phase 10 tail) |
| `ToroidalFftOperator<T>` | *(conceptual — no class yet)* | **no** |
| `GeometryOperator<T>` | `physics/geometry_operator.hpp` | no (`enqueue` declared only) |
| `MagneticFieldOperator<T>` | `physics/magnetic_field_operator.hpp` | no |
| `ForceOperator<T>` | `physics/force_operator.hpp` | no |
| `ConstraintOperator<T>` | `physics/constraint_operator.hpp` | no |
| profiles host build | `physics/profiles.hpp` | no (`RadialProfilesResult` struct only) |
| `ResidualOperator<T>` | `numerics/residual_operator.hpp` | no |
| `Preconditioner<T>` | `numerics/preconditioner.hpp` | no |
| `TridiagonalBackend<T>` (+ `PcrBackend`/`ThomasBackend`) | `numerics/tridiagonal_backend.hpp` | **yes** (Phase 8, in `src/precon_impl.cuh`) |
| `DescentOperator<T>` | `numerics/descent_operator.hpp` | no |
| `Prolongation<T>` | `numerics/prolongation.hpp` | no |
| `IterationController<T>` | `solver/iteration_controller.hpp` | **yes** (Phase 4, pure host) |
| host config / I/O / runtime | `config/*`, `io/*`, `runtime/*` | **yes** (Phases 2–6) |

So the migration is: **implement the unimplemented `enqueue` methods as thin
wrappers over the legacy kernels, split the mixed-ownership structs, then rewire
`solverRun` to drive the operators and delete the legacy structs.**

### 3. Principles

1. **One operator per step; bit-identical before the next step.** Each step is
   independently committable and re-verified against the frozen trajectory.
2. **Wrap first, decompose second.** An operator's first implementation calls the
   *exact* legacy kernel with the *exact* launch config; any kernel split that
   changes arithmetic order is a separate Class B step with a differential test.
3. **Ownership moves with the operator.** The `FourierPlan` split (step 1) is
   prerequisite: transforms must stop owning physics/diagnostic arrays before a
   transform operator is meaningful.
4. **The solver's residual/velocity/state slabs stay as-is.** `SpectralStorage`,
   the component-major slabs, and the typed domain views are already correct and
   do not change.
5. **No new dependency edges.** Constraint reaches the transform only through
   `SpectralOperator`; output never sees a device pointer; the controller stays
   pure host.

### 4. Step plan (dependency order)

#### Step 1 — Split `FourierPlan` into transform-only vs storage (Class A)

`FourierPlan<T>` currently mixes: (a) transform state (`plan_z2d/d2z`,
`d_zeta_spectra/real`, the four poloidal tables, `d_fwd_w`, `d_cufft_work`);
(b) real-space geometry (18 parity + 9 combined arrays); (c) force arrays (16).

- Introduce `RealSpaceStorage<T>` (or reuse a `StageWorkspace`-owned
  `GeometryParityViews` + `ForceParityViews` + combined arrays) holding (b) and
  (c), backed by the existing `DeviceArena` spans (no layout change).
- Shrink `FourierPlan<T>` to (a) only. The `inverseDFT/forwardDFT` signatures
  change to take the geometry/force views instead of reaching into `fp.d_r_e` etc.
- `MetricWorkspace`, `PreconWorkspace`, `ConstraintWorkspace`, `RadialProfiles`
  are left untouched this step.

*Verify:* `test_fourier`, `test_operator_views`, full Solovev/W7-X bit-identical.

#### Step 2 — `ToroidalFftOperator` (Class A)

Create `include/cumes/transforms/toroidal_fft_operator.hpp` + an impl TU: a
`SpectralOperator<T>` backend that owns the (now transform-only) `FourierPlan<T>`
and wraps `inverseDFTFused`/`forwardDFT` + the generic de-alias. Same `enqueue_*`
signature family as `AxisymmetricOperator` (inverse/forward + rzCon/de-alias).

- `enqueue_inverse` → `inverseDFTFused(..., rCon, zCon, stream)`.
- `enqueue_forward` → `forwardDFT(...)`.
- `enqueue_dealias` → `constraintDealiasBandpass(...)`.

*Verify:* `test_axisym_backend`-style differential test — generic operator vs the
legacy free functions, bit-identical (same kernels, so should be exact).

#### Step 3 — Unify the solver transform selection (Class A)

`solverRun` takes a `SpectralOperator<T>*` (either backend) instead of
`AxisymmetricOperator<T>*` + the `axisym_active` branches. The three call sites
collapse to `op->enqueue_inverse/rzcon`, `op->enqueue_forward`, and the
constraint path via `op->enqueue_dealias` (or a small backend-neutral
`constraintCompute` that takes a `SpectralOperator&`).

*Verify:* Solovev (axisym) and W7-X (toroidal) both bit-identical to the
committed `9d18e4e` state; `CUMES_FORCE_GENERIC=1` still forces the toroidal
backend on Solovev.

#### Step 4 — Host profile build (Class A)

Implement `profiles.hpp`: a `profilesCreateTyped(p, ip, arena)` that fills
`RadialProfileViews` + `delta_s` + `lamscale` by calling the existing
`profilesCreate` internals, returning `RadialProfilesResult<T>`. `StageSolver`
switches to it; the raw `RadialProfiles<T>` struct is still the internal storage
for one step, then folded in.

*Verify:* profile arrays bit-identical (`test_fourier`-adjacent profile check or
the existing trajectory).

#### Step 5 — Split base geometry from B/pressure (Class A, then B)

`computeGeometry` currently produces base geometry (`tau`, `√g`, covariant
metric), then the field (B^u/B^v/B_u/B_v, total pressure) and current closure,
all in one entry point.

- Implement `GeometryOperator::enqueue` = the base-geometry + metric kernels
  (no `1/√g`), writing `BaseGeometryHalfViews`.
- Implement `MagneticFieldOperator::enqueue` = the `1/√g` field + total-pressure
  + `ncurr=0/1` closure kernels, reading base geometry + profiles.
- Split `computeJacobianStats` into the `reset→reduce→finalize` device chain
  described in blueprint §6.7 (this is the Class B part if the reduction tree or
  status finalization changes summation order; otherwise Class A).

*Verify:* pointwise geometry/B equality on manufactured cases
(`test_geometry_iso`, `test_geometry_ncurr`); full trajectory bit-identical if
Class A, ULP + identical controller decisions if Class B.

#### Step 6 — `ForceOperator` (Class A)

Implement `ForceOperator::enqueue` wrapping `computeForces` verbatim. Add the
scalar CPU reference from blueprint §4.7/§10.3 as `test_force_reference`
(already exists; extend it to be the gate). No kernel change.

#### Step 7 — `ConstraintOperator` (Class A)

Implement `ConstraintOperator::enqueue` wrapping the `constraintComputeHead` +
dealias (generic or axisym via `SpectralOperator`) + `constraintComputeTail`
split already in place. Move `ConstraintState` (versioned reference) into the
operator and thread the reset cadence from `IterationController`.
`ConstraintWorkspace` storage becomes the operator's private arena-backed spans.

#### Step 8 — `ResidualOperator` (Class A)

Implement `enqueue_invariant`/`enqueue_preconditioned` wrapping
`computeResidualsKernel` + the host `plainPerEl`/`fNorm*` scaling into a device
`ControlRecord` (blueprint §6.9). The force-norm `enqueueForceNorms` +
`finalizeForceNorms` pair moves here or into a `ForceNormOperator`.

#### Step 9 — `Preconditioner` (Class A)

Implement `enqueue_compute` (→ `preconCompute`) and `enqueue_apply` (→
`preconApply`, which already calls the Phase 8 `PcrBackend`/`ThomasBackend`).
`PreconWorkspace` becomes the operator's private storage.

#### Step 10 — `DescentOperator` + checkpoint (Class A)

Implement `DescentOperator::enqueue` wrapping `descentStepKernel` + the
`backupState`/`restoreState` single-copy checkpoint, driven by `DescentAction`
(already declared). Folds the ordered apply (descent → post-descent checkpoint →
post-descent restore+zero) into one operator call.

#### Step 11 — `Prolongation` (Class A)

Implement `Prolongation::enqueue` wrapping `interpolateState`; `MultigridSolver`
calls it instead of the free function.

#### Step 12 — `EquilibriumOperator` + stage/multigrid rewire (Class A)

Compose the operators into `EquilibriumOperator::enqueue` (the per-iteration
DAG from blueprint §7) and rewrite `solverRun` to be a thin loop over the pure
`IterationController` + `EquilibriumOperator`. `StageSolver::run`/`MultigridSolver::run`
construct operators instead of the five legacy workspaces. This is the payoff
step: `solverRun` no longer names any legacy struct.

*Verify:* full precise trajectories bit-identical; `cumes_benchmark_fixed_iteration`
and the smoke gate still run.

#### Step 13 — Delete legacy internals (Class A, deletion only)

Delete `include/*.cuh` (`fourier`, `geometry`, `forces`, `profiles`, `precon`,
`constraint`, `refine`, `solver`, `input`, `input_json`, `output`, `vmec_types`
where superseded), `src/*_impl.cuh` free functions no longer referenced, and
`InputParams`/`GridParams` legacy bridges (keeping the `cumes` `GridShape`/
`ValidatedProblem`/`DeviceParams`). Rename the surviving `src/cumes/*` to drop
the "legacy wrapper" comments. Update `CMakeLists.txt` targets and the
`docs/architecture.md` §1 "two layers" description to a single layer.

*Verify:* clean-clone build, full CTest + sanitizer matrix, both configs
bit-identical, `docs` scaffold (blueprint §5) matches the source tree.

### 5. Verification per step

- **Class A gate:** `./build/cuMES inputs/{solovev,w7x}.json` bit-identical to
  the frozen state files (`compare_converged_state.py` / `test_io_golden`), plus
  the full CTest + Compute Sanitizer matrix.
- **Class B gate (steps 5/6 if decomposed):** per-operator ULP bounds on frozen
  inputs, identical finite/status classification, identical controller decisions
  (iteration counts and restart sequence) — via `test_controller` replay and the
  fixed-iteration benchmark's `state_hash`.
- Every operator gets a `test_<operator>.cu` differential test against the legacy
  free function it wraps, in the `test_axisym_backend` style.

### 6. Risks and open questions

- **`FourierPlan` split (step 1) is the linchpin.** It touches every transform
  consumer and the dump machinery; do it first and re-verify before anything else.
- **`computeGeometry` kernel boundaries (step 5).** Whether the `1/√g` division
  is already a separate kernel or fused into base geometry is to be confirmed in
  `src/geometry_impl.cuh`; a fused division forces a Class B split with a
  differential proof (blueprint §6.7 explicitly forbids assuming a grid-wide
  barrier).
- **The `dynSharedBase()` workaround.** `constraint_impl.cuh` retains the
  non-templated dynamic-shared-memory indirection because switching to
  `extern __shared__ T[]` perturbs `--use_fast_math` FMA (~1e-10). When the
  explicit instantiation split is fully in place this indirection *can* be
  removed, but only as a Class B re-freeze — decide at step 13 whether to fold it
  in or leave it.
- **`vmec_types.h` / `GridParams`.** `DeviceParams<T>` (blueprint §6.1) is the
  eventual replacement for `GridParams<T>`; if the migration stalls, the legacy
  `GridParams` can remain as the DeviceParams backing store without violating the
  exit gate, as long as `InputParams` and the raw owning structs are gone.
- **The dump machinery (`DUMP_CUMES_VERIFY`)** reaches into `fp.d_*` arrays; the
  step-1 split must keep the dump path working (it is observability-only and must
  not change the trajectory) or gate it behind the same views.

### 7. Ordering rationale

The dependency graph (blueprint §5.1) forces this order: transforms (steps 1–3)
must be owned before constraint (step 7) can reach them through `SpectralOperator`;
base geometry (step 5) must be split before B/pressure (step 5b) and force (step
6); residual (step 8) and preconditioner (step 9) consume geometry/field; descent
(step 10) consumes residual + controller; prolongation (step 11) is independent;
the equilibrium operator (step 12) composes everything; deletion (step 13) is
last and gated on every consumer. Steps 4 and 11 can be done in parallel with
steps 5–10.

---

<a id="phase-11-handover"></a>

## 2026-08-16 12:53:51+08:00 — cuMES Phase 11 handover — strangler-fig migration (owning operators + FourierPlan split)

**Former path:** `docs/phase-11-handover.md`
**First tracked:** [`426e76a`](https://github.com/12ff54e/cuMES/commit/426e76a2a1c159df41b86bfbeb421af3382d1ae1) at 2026-08-16T12:53:51+08:00

Status date: 2026-08-16. Branch: `overhaul` (… + Phase 10 tail `9d18e4e` + this
Phase 11 work). This document records what the strangler-fig migration
(`docs/strangler-fig-migration-plan.md`) delivered in its first landing, how it
was verified, and what was deliberately left for the tail.

### 1. Scope

Phase 11 begins the blueprint §11 Phase 10 exit gate: *"source dependency graph
matches section 5, … and no legacy code is required for normal execution."* It
is the migration's first committed, verified landing — **wrap the legacy
workspaces in owning operator classes and split the `FourierPlan`** so the
transform operator owns only transform scratch. It is deliberately **not** the
full exit gate (see §4).

The concrete work, in four commits:

| Commit | What |
| ------ | ---- |
| `87cec03` | four owning operators (`Preconditioner`, `ConstraintOperator`, `GeometryOperator`, `Profiles`) + solver rewire |
| `75dcc65` | `ToroidalFftOperator` owns the `FourierPlan` (all 5 workspaces owned) |
| `25bcb97` | docs: record the progress in the migration plan |
| `a692200` | split `FourierPlan` — transform scratch vs stage-owned `RealSpaceStorage` |

### 2. What changed

#### 2.1 Five owning operators (`87cec03`, `75dcc65`)

Each legacy workspace moved out of `solverRun`/`StageSolver` into a `cumes`
operator that owns it and wraps the existing free function/kernel unchanged:

| Operator | Owns | Wraps |
| -------- | ---- | ----- |
| `Profiles` | `RadialProfiles` | `profilesCreate` |
| `ToroidalFftOperator` | `FourierPlan` | `inverseDFTFused` / `forwardDFT` |
| `GeometryOperator` | `MetricWorkspace` | `computeGeometry` + `computeJacobianStats` + `computeForceNormPartials` |
| `Preconditioner` | `PreconWorkspace` | `preconCompute` / `preconApply` (which already routes the solve through the Phase 8 `PcrBackend`/`ThomasBackend`) |
| `ConstraintOperator` | `ConstraintWorkspace` | `constraintCompute` (generic or axisym de-alias) |

Implemented in each module's `*_impl.cuh` (the `PcrBackend`/`ThomasBackend`
pattern), explicitly instantiated per scalar type. `StageSolver::run` constructs
them (RAII); `solverRun` calls their `enqueue` methods in the hot loop.

#### 2.2 `FourierPlan` split (`a692200`)

`FourierPlan<T>` no longer owns the 43 real-space arrays (18 parity-split
geometry + 9 combined + 16 force). Those move to a new stage-owned
`cumes::RealSpaceStorage<T>` (`include/cumes/state/real_space_storage.hpp`,
`realSpaceCreate`/`realSpaceFree` in `fourier.cuh`/`fourier_impl.cuh`), so the
transform operator owns only transform scratch (plans, ζ buffers, poloidal
tables, work area, mode tables). The transform/geometry/force/constraint/
preconditioner functions now take `RealSpaceStorage&` (geometry/force) alongside
`FourierPlan&` (tables). This satisfies the blueprint §5.1 dependency rule —
*transforms no longer own force/geometry fields*.

### 3. Verification

Every commit is **pure Class A** (bit-identical): the operators wrap the exact
legacy kernels/launch configs, and the `RealSpaceStorage` pointers are bit-for-bit
the `FourierPlan`'s old members. Re-verified at each commit:

| Config | Effective iterations | Final FSQR | Verdict |
| ------ | -------------------- | ---------- | ------- |
| Solovev (generic, `CUMES_FORCE_GENERIC=1`) | 251 → 199 → 456 | 9.583e-17 | bit-identical |
| Solovev (axisymmetric, default) | 251 → 199 → 456 | 9.583e-17 (ULP-equiv) | fast path intact |
| W7-X | 1877 → 1617 → 2011 (5505) | 9.778e-13 | bit-identical |

Test matrix: **35/35 CTest** (23 unit + 12 compute-sanitizer memcheck + 1 smoke),
float build clean. The `CUMES_FORCE_GENERIC=1` knob (added in the Phase 10 tail)
remains the A/B switch for the transform backend.

### 4. Deferred / remaining (documented, not hidden)

The blueprint's full Phase 10 exit gate — *"no legacy code is required for normal
execution"* — is **not** met; the legacy `include/*.cuh` structs and `InputParams`
are still the production kernels, and the operators still *name* the legacy
structs they read (via `fourier_plan()`/`workspace()` aliases) in their
transitional form. Remaining, in dependency order:

1. **`SpectralOperator` unification.** `ToroidalFftOperator` is not yet a
   `SpectralOperator<T>` peer of `AxisymmetricOperator`; `solverRun` still has
   the `axisym_active` branch. Unifying requires resolving the fused-vs-split
   rCon/zCon impedance (the generic backend fuses rCon/zCon into the inverse via
   `inverseDFTFused`; the axisymmetric backend splits them into `enqueue_inverse`
   + `enqueue_rzcon`). The §6.6 interface currently declares only
   `enqueue_inverse`/`enqueue_forward`; it must grow rCon/zCon (and the
   de-alias) to cover both backends.
2. **Stateless operators.** `ForceOperator`, `ResidualOperator`,
   `DescentOperator`, `Prolongation` — thin wrappers over `computeForces`,
   `computeResidualsKernel`, `descentStepKernel` + checkpoint, and
   `interpolateState`. They are declaration-only in `include/cumes/*`.
3. **Legacy-struct deletion.** `include/*.cuh` (`fourier`, `geometry`, `forces`,
   `profiles`, `precon`, `constraint`, `refine`, `solver`, `input`, `input_json`,
   `output`, `vmec_types` where superseded) + `InputParams`, gated on `solverRun`
   naming no legacy struct. Also the `dynSharedBase()` non-templated shared-memory
   indirection (removable only as a Class B re-freeze, per the `--use_fast_math`
   FMA note in `constraint_impl.cuh`/`geometry_impl.cuh`).

### 5. Next steps

1. Land `SpectralOperator` unification (collapse the `axisym_active` branch).
2. Land the four stateless operators.
3. Delete the legacy structs, then re-check the §5 dependency graph and the
   blueprint §14 definition of done from a clean clone.

---

<a id="phase-11-tail-handover"></a>

## 2026-08-16 14:24:37+08:00 — cuMES Phase 11 tail handover — operator unification + legacy-deletion begin

**Former path:** `docs/phase-11-tail-handover.md`
**First tracked:** [`585cb49`](https://github.com/12ff54e/cuMES/commit/585cb49f7fd897c0ecaa7a73a2db15d9205d446d) at 2026-08-16T14:24:37+08:00

Status date: 2026-08-16. Branch: `overhaul`. This document records the work
that continues `docs/phase-11-handover.md` (the strangler-fig migration's first
landing): the `SpectralOperator` unification and the four stateless operators
(the handover's next-steps 1–2), and the first three ownership slices of the
legacy-struct deletion (next-step 3). Every commit is **Class A bit-identical**
to the frozen trajectory.

### 1. Scope

This is the continuation of Phase 11's strangler-fig migration. The prior
handover ended with five owning operators and a split `FourierPlan`, but left
three items open. This handover delivers:

1. **`SpectralOperator` unification** — `solverRun` drives a single
   `SpectralOperator<T>*` with no `axisym_active` branch.
2. **The four stateless operators** — `ForceOperator`, `ResidualOperator`,
   `DescentOperator`, `Prolongation`.
3. **The first legacy-deletion slices** — mode-table extraction, cuFFT stream
   binding, and `m1PreconScale` all move out of `solverRun`'s naming.

### 2. What changed

#### 2.1 `SpectralOperator` unification (`d03640f`)

The interface grew to the full transform surface the solver needs, and
`ToroidalFftOperator` became a `SpectralOperator<T>` peer of
`AxisymmetricOperator`:

| Interface method | Toroidal (generic) | Axisymmetric |
| ---------------- | ------------------ | ------------ |
| `enqueue_inverse(coeff, geometry, rCon, zCon, stream)` | fused `inverseDFTFused` (rCon/zCon in the accumulate) | direct poloidal synthesis + rzCon kernel |
| `enqueue_forward(force, constraint_force, residual, stream)` | `forwardDFT` (constraint force as 4 raw ptrs) | direct reduced-θ projection |
| `enqueue_dealias(gConEff, tcon, faccon, gCon, stream)` | compact cuFFT round trip | direct poloidal bandpass |

The fused-vs-split rCon/zCon impedance was resolved by folding rCon/zCon into a
unified `enqueue_inverse` (the axisymmetric backend runs its rzCon kernel right
after its synthesis, in the same stream order the solver previously used). The
compact de-alias scratch + plans moved out of `ConstraintWorkspace` into the
`FourierPlan` (transform-owned), so the constraint reaches the bandpass through
`op->enqueue_dealias` instead of naming `FourierPlan`. The dead
`constraintComputeAxisym` free function was removed.

`solverRun`, `StageSolver`, and `benchmarks/fixed_iteration.cu` now build one
`SpectralOperator<T>*` (`nullptr` → generic) and call the three methods
unconditionally; the `if (axisym_active)` branches are gone.

#### 2.2 Stateless operators (`3296ae1`)

`ForceOperator`, `ResidualOperator`, `DescentOperator`, `Prolongation` were
declaration-only; they are now thin wrappers wired into the drivers:

- `ForceOperator::enqueue(rs, p, rp, mw, stream)` → `computeForces`.
- `ResidualOperator::enqueue(residual, ns, mnmax, sq_out, stream)` →
  `computeResidualsKernel` (one launch, 3-group output; the invariant vs
  preconditioned distinction stays host-side).
- `DescentOperator::enqueue(state, velocity, residual, xm, xn, ns, mnmax,
  action, stream)` → `descentStepKernel` under a `DescentAction`; the
  single-copy checkpoint capture/restore stays with the solver's state slab
  (blueprint §6.10 keeps the checkpoint a distinct operator).
- `Prolongation::enqueue(p_new, state_old, p_old, stream)` → `interpolateState`.

#### 2.3 Legacy-deletion ownership slices (`53ed57a`, `67ff2fd`, `b8fa3a0`)

Three Class-A moves began peeling the legacy structs out of `solverRun`:

| Commit | Move | Legacy naming removed from `solverRun` |
| ------ | ---- | -------------------------------------- |
| `53ed57a` | folded-mode table → `cumes::DeviceModeTable` (new `cumes/state/mode_table.cuh`); `FourierBasis` deleted | `fp.basis.d_xm/d_xn` |
| `67ff2fd` | cuFFT stream binding → `ToroidalFftOperator::bind_stream` | `fp.plan_z2d/d2z/d2z_da/z2d_da` |
| `b8fa3a0` | `m1PreconScaleKernel` → `Preconditioner::enqueue_m1_scale` | `pw.d_ard/d_brd/azd/bzd` |

The transform free functions (`inverseDFT`/`inverseDFTFused`/`forwardDFT`) and
`preconCompute` now take `const int* xm, const int* xn` instead of reaching into
`fp.basis`; `ToroidalFftOperator` holds a non-owning mode-table pointer and
exposes `xm()`/`xn()`.

### 3. Verification

Re-verified Class A bit-identical at every commit:

| Config | Effective iterations | Final FSQR | Verdict |
| ------ | -------------------- | ---------- | ------- |
| Solovev (axisym, default) | 251 → 199 → 456 | 9.583e-17 | bit-identical |
| Solovev (generic, `CUMES_FORCE_GENERIC=1`) | 251 → 199 → 456 | 9.583e-17 | bit-identical |
| W7-X | 1877 → 1617 → 2011 (5505) | 9.778e-13 | bit-identical |

Test matrix: **35/35 CTest** (23 unit + 12 compute-sanitizer memcheck + 1
smoke), float build clean.

### 4. Remaining legacy naming in `solverRun` (grepped, precise)

After `b8fa3a0`, the legacy structs `solverRun` still names are:

- **`FourierPlan`** — only the dump-gated `inverseDFT`/`fourierCombineParity`
  observability path (the hot loop is sealed behind the operator + `bind_stream`).
- **`PreconWorkspace` / `MetricWorkspace`** — named as arguments
  (`constraint.enqueue` takes `pw`; `precon.enqueue_compute`/`enqueueForceNorms`
  take `mw`); their `.d_*` field reads are now dump-only.
- **`ConstraintWorkspace` / `RadialProfiles`** — still read in the hot loop: the
  `RealFieldView`s over `cw.d_rCon/d_zCon/d_frcon/d_fzcon`, and
  `rp.d_sqrtS_F`/`rp.delta_s`/`rp.d_dVds_H`/`rp.d_pres_H`.
- **`GridParams<T>`** — everywhere (needs the `DeviceParams<T>` replacement).
- **`InputParams`** — in `StageSolver`/`main` (needs `ValidatedProblem`).

`dynSharedBase()` (the non-templated shared-memory indirection) is still
retained in `fourier_impl.cuh`/`constraint_impl.cuh`/`geometry_impl.cuh`; it is
removable only as a Class B re-freeze (the `--use_fast_math` FMA note), to be
decided at step 13.

### 5. Next steps (dependency order)

1. **Seal `FourierPlan` from the solver.** Give `ToroidalFftOperator` a
   dump-only `combine_parity()`/`inverse_dump()` accessor (or move the dump
   path behind a diagnostic operator) so `solverRun` drops the
   `transform.fourier_plan()` alias entirely.
2. **View accessors for `Profiles` / `ConstraintOperator`.** Add typed
   `RadialProfileViews`/`ConstraintForceViews`/rCon-zCon accessors so
   `solverRun` stops building raw `RealFieldView`s over `cw.d_*` and reading
   `rp.d_*` directly.
3. **Migration step 5** — split base geometry from B/pressure (Class B; the
   `1/√g` division is fused into `geometryKernel`, so a split is a re-freeze
   with a differential proof).
4. **Migration step 12** — `EquilibriumOperator` composition: `solverRun`
   becomes a thin loop over `IterationController` + `EquilibriumOperator`.
5. **Migration step 13** — delete `include/*.cuh` legacy structs + `InputParams`
   after `solverRun` names none of them; replace `GridParams` with `DeviceParams`
   and `InputParams` with `ValidatedProblem` (the `to_input_params()` bridge
   already exists); decide `dynSharedBase()` removal.

### 6. Commits (this handover)

```
f9a1b0d docs: record the three legacy-deletion ownership slices
b8fa3a0 Phase 11: move m1PreconScale into the Preconditioner operator
67ff2fd Phase 11: move cuFFT stream binding into ToroidalFftOperator
53ed57a Phase 11: extract the mode table out of FourierPlan (DeviceModeTable)
0bb4a91 docs: record strangler-fig progress (SpectralOperator unification + stateless operators)
3296ae1 Phase 11: land the four stateless operators (force/residual/descent/prolongation)
d03640f Phase 11: SpectralOperator unification — collapse the axisym_active branch
```

---

<a id="phase-11-tail-2-handover"></a>

## 2026-08-16 15:22:55+08:00 — cuMES Phase 11 tail #2 handover — steps 1–4 landed, step 13 remaining

**Former path:** `docs/phase-11-tail-2-handover.md`
**First tracked:** [`4d1aef6`](https://github.com/12ff54e/cuMES/commit/4d1aef645185a0bf3aa3886aab2533c1e543ffde) at 2026-08-16T15:22:55+08:00

Status date: 2026-08-16. Branch: `overhaul`. This handover continues
`docs/phase-11-tail-handover.md` (whose §5 listed five next steps). It records a
critical correctness fix plus the first four of those five steps — all landed
and **Class A bit-identical** — and scopes the one remaining item (migration
step 13, the legacy-struct deletion).

### 1. Critical fix: prolongation stale-read (commit `8b25458`)

The default (axisymmetric) Solovev path read an **all-zero ns=55 stage** while
the generic backend was fine (stages ns=5/ns=11 converged; ns=55 failed with
`√g=0` everywhere and `Rax=0`). Root cause: the coarse state is written on the
nonblocking compute stream, and `cudaStreamSynchronize(stream)` at the prior
stage's exit does **not** make those writes visible to `interpolateState`'s
kernel — only a full `cudaDeviceSynchronize()` orders them (the stage's
synchronous default-stream memsets/memcpys interact with the nonblocking
stream). Fix: `cudaDeviceSynchronize()` before the prolongation kernel (one
fence per stage, never in the hot loop) plus `cudaStreamSynchronize(stream)`
after, so the coarse buffer is not freed (via move-assignment of the returned
`SpectralStorage`) while the kernel still reads it.

Verified: Solovev axisym **and** generic `251→199→456` FSQR 9.583e-17; W7-X
`1877→1617→2011` FSQR 9.778e-13.

### 2. Steps landed (handover §5, items 1–4)

| Handover §5 | Migration step | Commit | What |
| ----------- | -------------- | ------ | ---- |
| 1 | — (seal) | `ed6e091` | `FourierPlan` sealed from `solverRun` via `ToroidalFftOperator::enqueue_inverse_dump`/`combine_parity` (dump-only). `solverRun` no longer holds `transform.fourier_plan()`. |
| 2 | — (views) | `bbbf4cd` | `Profiles::profile_views()/delta_s()` + `ConstraintOperator::rcon_view()/zcon_view()/constraint_force_views()`; `solverRun` reads typed `RadialProfileViews`/views, not raw `rp.d_*`/`cw.d_*`. |
| 3 | step 5 (base geom split) | `0a2e55c` | `geometryKernel` split into `baseGeometryKernel` (interpolation/Jacobian/metric, no 1/√g) + `magneticFieldKernel` (1/√g B + covariant B + total pressure); `computeGeometry` = full-pipeline wrapper; solver drives `GeometryOperator` (base) + stateless `MagneticFieldOperator` (field), ordered base → Jacobian stats → field. |
| 4 | step 12 (EquilibriumOperator) | `a9e198a` | Per-iteration device DAG (axis extrapolation → … → preconditioned residual, incl. interleaved dump machinery) extracted into `cumes::EquilibriumOperator::enqueue` (`include/cumes/solver/equilibrium_operator.hpp`). `solverRun` is now a thin loop: `next_schedule()` → `EvaluationSchedule` → `enqueue` → one fence → Jacobian/invariant/restart decisions → descent → post-descent capture/restore. |

Each commit is Class A bit-identical: Solovev axisym + generic `251→199→456`
FSQR 9.583e-17; W7-X `1877→1617→2011` FSQR 9.778e-13; 35/35 CTest; `CUMES_DUMP=1`
dump path and the float build clean.

### 3. What `solverRun` still names (post step 12)

After `a9e198a`, `solverRun`'s only legacy-struct naming is the dump
observability (NOT the hot loop): `SpectralState<T> st = storage.legacy_view()`
and `const RadialProfiles<T>& rp = profiles.workspace()` — both feed
`axisRAtZeta0`/`printIterRow`/the step-0 dumps. Plus `const GridParams<T>& p`
(everywhere) and `InputParams` in `StageSolver`/`main`. The hot-loop operators
are already behind `EquilibriumOperator`/typed views.

### 4. Remaining: migration step 13 — delete legacy structs (handover §5, item 5)

This is the whole legacy-deletion endgame and is **not** a single-session change.
It is deliberately left as the next unit of work, in dependency order:

1. **Create `DeviceParams<T>`** (`include/cumes/config/…`, blueprint §6.1). It
   must be the compact trivially-copyable stage+scalar pack that replaces
   `GridParams<T>` field-for-field: `ns, mnmax, ntheta, nzeta, nfp, nZnT, mpol,
   ntor, ncurr, delt, ftol, max_iter, tcon0, lamscale` plus the constants
   `kSignJacobian`/`kMu0`. `GridParams` is referenced in **65 files**
   (all `src/*_impl.cuh`, the `include/cumes/{physics,numerics,solver,state,
   transforms}` operator headers, `include/*.cuh`, `main.cu`, `output*.cpp`,
   and ~15 tests). The mechanical move is: `typedef`/`using` swap first, then
   shrink `GridParams` into `DeviceParams` and delete `GridParams`.

2. **`InputParams` → `ValidatedProblem`.** `main.cu` currently does
   `vr.value().to_input_params()` and threads `InputParams` through
   `init_params`/`init_state`/`restart_state` (`seed_state.hpp`),
   `Profiles`/`profilesCreate`, `StageSolver`/`MultigridSolver`, and
   `outputSave`/`outputPrint`. Delete `InputParams` (and `to_input_params()`)
   once these consumers take `ValidatedProblem` directly; `GridShape`/
   `ValidatedProblem` already carry the extents/folding.

3. **Delete the raw owning structs + free functions** once the operators own
   everything with typed views: `FourierPlan` (fourier.cuh), `MetricWorkspace`
   (geometry.cuh), `RadialProfiles` (vmec_types.h), `PreconWorkspace`
   (precon.cuh), `ConstraintWorkspace` (constraint.cuh), `SpectralState`
   (vmec_types.h), and their `*Create/*Free` + `compute*/inverseDFT/forwardDFT/
   preconCompute/preconApply/constraintCompute/interpolateState` entry points.
   The `src/*_impl.cuh` kernel bodies survive as operator implementations.

4. **Decide `dynSharedBase()` removal** (Class B re-freeze, per the
   `--use_fast_math` FMA note in `constraint_impl.cuh`/`geometry_impl.cuh`).
   This can be folded in or left as a documented follow-up; it is the only
   non-deletion arithmetic question in step 13.

5. **Update the tests** that still construct legacy structs directly
   (`test_fourier`, `test_forces`, `test_geometry_iso`, `test_geometry_ncurr`,
   `test_constraint_tcon`, `test_force_reference`, `test_force_verify`,
   `test_axisym_backend`, `test_regression_kernels`) to drive the operators.

6. **Update `CMakeLists.txt`** (drop the legacy `.cu` TUs once folded into
   `src/cumes/`) and `docs/architecture.md` §1 (two layers → one layer).

Verification bar per step: `./build/cuMES inputs/{solovev,w7x}.json` bit-identical
(`251→199→456` / `1877→1617→2011`), 35/35 CTest + the optional-backend matrix
(none/netcdf/hdf5), float build clean.

### 5. Commits (this handover)

```
a9e198a Phase 11 step 12: EquilibriumOperator — solverRun becomes a thin loop
0a2e55c Phase 11 step 5: split base geometry from the magnetic field
bbbf4cd Phase 11: typed view accessors for Profiles/ConstraintOperator
ed6e091 Phase 11: seal FourierPlan from the solver (dump-only transform accessors)
8b25458 fix: prolongation stale-read of the coarse state on the axisymmetric path
```

---

<a id="phase-11-tail-3-handover"></a>

## 2026-08-16 15:58:47+08:00 — cuMES Phase 11 tail #3 handover — step 13 items 1–2 landed, 3–6 remaining

**Former path:** `docs/phase-11-tail-3-handover.md`
**First tracked:** [`446b25e`](https://github.com/12ff54e/cuMES/commit/446b25ea9a8c49a833e20325bf76a0005d682966) at 2026-08-16T15:58:47+08:00

Status date: 2026-08-16. Branch: `overhaul`. This handover continues
`docs/phase-11-tail-2-handover.md`, whose §4 scoped migration step 13 (the
legacy-struct deletion endgame) as six dependency-ordered items. Items 1 and 2
have now landed, both **Class A bit-identical**; items 3–6 remain and are
re-scoped below with the exact current state.

**UPDATE (same day): step 13 is COMPLETE.** Parts 5–7 have landed — `FourierPlan`,
`SpectralState`, and the `interpolateState` free function are deleted (each
operator owns its buffers directly and exposes typed views; `fourier.cuh` and
`refine.cuh` are gone), the kernel tests drive the operators, and the CMake/docs
close-out is in (`CUMES_CUDA_MODULES` now names `prolongation` instead of
`refine`; `docs/architecture.md` §1 is "one layer" and §5's pending list is
empty). All parts Class A: Solovev `251→199→456` FSQR 9.583e-17, W7-X
`1877→1617→2011` FSQR 9.778e-13, 35/35 CTest, float build clean.

### 1. Items landed (this handover)

| Item | Migration step | Commit | What |
| ---- | -------------- | ------ | ---- |
| 1 | step 13.1 | `a7c0030` | `GridParams<T>` renamed to `DeviceParams<T>` and moved to `include/cumes/config/device_params.hpp` (blueprint §6.1 four-stage config pipeline). Pure rename, global namespace retained so all 65 kernel/operator headers resolve unchanged. |
| 2 | step 13.2 | `22e71d0` | `InputParams` deleted. `init_params`/`init_state`/`restart_state` (`seed_state.hpp`), `Profiles`/`profilesCreate`, `StageSolver`/`MultigridSolver`, and `outputSave`/NetCDF/HDF5 now consume `ValidatedProblem` directly. The legacy JSON parser (`input.h`/`input_json.h`/`src/input_json.cpp`) and `to_input_params()` are gone; the NetCDF/HDF5 v0 writers keep their padded layout via a new `cumes::io::LegacyInputProvenance`. Tests/benchmark drive `read_and_validate`/`validate` through a `cumes_test_support` `loadValidated`/`validateSpec` helper. |
| 5–7 | step 13.3 | `bf2d065` `33246e3` `e697a89` | Parts 5–7: `FourierPlan` deleted (ToroidalFftOperator owns the cuFFT plans/scratch/tables; `inverseDFT`/`inverseDFTFused`/`forwardDFT`/`fourierCombineParity`/`constraintDealiasBandpass` become operator methods `inverse`/`inverse_fused`/`forward`/`combine_parity`/`dealias_bandpass`; the de-alias kernels move into the transform module). `SpectralState` deleted (`SpectralStorage::legacy_view()` → `family_ptr`/`velocity_family_ptr`; the output writers now take `const SpectralStorage<T>&`). `interpolateState` deleted (`Prolongation::enqueue` owns the body; `refine.cuh` → the `prolongation` module). All kernel tests drive the operators. | |

Both are Class A: Solovev `251→199→456` FSQR 9.583e-17, W7-X `1877→1617→2011`
FSQR 9.778e-13, 35/35 CTest, float + none/netcdf/hdf5 backend builds clean.

### 2. Item 4 (decided): `dynSharedBase()` is a deferred Class B follow-up

The non-templated dynamic-shared-memory base accessor (`dynSharedBase()` in
`fourier_impl.cuh`/`geometry_impl.cuh`/`constraint_impl.cuh`) is **retained**.
Switching to a direct `extern __shared__ T[]` changes `--use_fast_math` FMA
fusion in the consumers (opaque function return vs. known shared-array aliasing)
and perturbs the trajectory at ~1e-10 — a Class B change, not the Class A
bitwise equivalence the library split must preserve. Decision: leave it in
place and re-freeze only in a dedicated Class B pass (already documented in the
`.cuh` comment and `docs/architecture.md` §2). No code change in step 13.

### 3. Migration step 13 items 3, 5, 6 — LANDED (see the UPDATE at the top)

For the record, this is the endgame `tail-2` §4 described, now complete.
Item 1 and 2 were the mechanical preconditions; the rest is the coupled
deletion of the owning structs **and** the tests that construct them. In
dependency order:

#### 3.1 Delete the six legacy owning structs + their free functions

The `cumes` operators **already own** the structs (strangler-fig): `Profiles`
owns `RadialProfiles`, `ToroidalFftOperator` owns `FourierPlan`,
`GeometryOperator` owns `MetricWorkspace`, `Preconditioner` owns
`PreconWorkspace`, `ConstraintOperator` owns `ConstraintWorkspace`, and
`SpectralStorage::legacy_view()` materializes `SpectralState`. Deleting them
means the operators own the raw `DeviceBuffer<T>`s directly and the
`src/*_impl.cuh` kernel bodies become the operators' method implementations
(the build/library split is already done — `cumes_cuda_double`/`_float` are the
nine `*_double.cu`/`*_float.cu` TUs).

Delete, leaf-first:

- `SpectralState` (vmec_types.h) — consumers are `outputSaveBinary`/`outputPrint`
  (`output.cpp`), `outputSaveNetcdf`/`outputSaveHdf5`, `seed_state.hpp`'s
  `init_state`/`restart_state` (they use `storage.legacy_view()` only for the
  six upload `cudaMemcpy`s), and the `solver_impl.cuh` dump machinery (`st.d_*`).
  Replace with `SpectralStorage` slab accessors; `legacy_view()` goes away.
- `RadialProfiles` (vmec_types.h) — `Profiles` should own `DeviceBuffer<T>`
  fields and expose `RadialProfileViews` directly (the view struct already
  exists in `real_fields.cuh`).
- `MetricWorkspace` (geometry.cuh) — `GeometryOperator` owns the buffers; the
  base-geometry/field split (step 5, `0a2e55c`) already put
  `computeBaseGeometry`/`computeMagneticField` behind the two operators.
- `PreconWorkspace` (precon.cuh) — `Preconditioner` owns the buffers.
- `ConstraintWorkspace` (constraint.cuh) — `ConstraintOperator` owns the buffers.
- `FourierPlan` (fourier.cuh) — `ToroidalFftOperator` owns the cuFFT plans +
  scratch + poloidal tables directly; `inverseDFT`/`inverseDFTFused`/
  `forwardDFT`/`fourierCombineParity`/`constraintDealiasBandpass` become its
  methods (the `RealSpaceStorage` split is already done).

The `*Create`/`*Free` + `compute*`/`inverseDFT`/`forwardDFT`/`preconCompute`/
`preconApply`/`constraintCompute`/`interpolateState` free-function entry points
are deleted alongside their struct. `interpolateState` (`refine.cuh`) already
has a `Prolongation` operator wrapping it.

#### 3.2 Update the tests that construct legacy structs directly

`test_fourier`, `test_forces`, `test_geometry_iso`, `test_geometry_ncurr`,
`test_constraint_tcon`, `test_force_reference`, `test_force_verify`,
`test_axisym_backend`, `test_regression_kernels` all call `fourierCreate`/
`metricCreate`/`preconCreate`/`constraintCreate`/`realSpaceCreate`/
`modeTableCreate` + the free functions. They must drive the operators
(`ToroidalFftOperator`, `GeometryOperator`, `Preconditioner`,
`ConstraintOperator`, `Prolongation`, …) instead. This is coupled to 3.1 — a
struct cannot be deleted until its tests stop constructing it.

#### 3.3 Update `CMakeLists.txt` + `docs/architecture.md`

- Drop the legacy `.cu` TUs from `CUMES_CUDA_MODULES` once each `*_impl.cuh` is
  folded into its operator's `src/cumes/` TU (the `cumes_cuda_double`/`_float`
  targets remain; their source lists shrink).
- `docs/architecture.md` §1 "Two layers" → "one layer", and §5's "still
  pending" list (which still names `FourierPlan`, `MetricWorkspace`,
  `InputParams`, …) shrinks to empty. `InputParams` is already gone as of
  `22e71d0`; the remaining entries are the owning structs named in 3.1.

Verification bar per step (unchanged): `./build/cuMES inputs/{solovev,w7x}.json`
bit-identical (`251→199→456` / `1877→1617→2011`), 35/35 CTest + the
optional-backend matrix (none/netcdf/hdf5), float build clean.

### 4. Commits (this handover)

```
e697a89 Phase 11 step 13.3 (part 7): delete interpolateState free function (Prolongation owns it)
33246e3 Phase 11 step 13.3 (part 6): delete SpectralState + SpectralStorage::legacy_view()
bf2d065 Phase 11 step 13.3 (part 5): delete FourierPlan + fourier transform free functions
0ef9e82 Phase 11 step 13.3 (part 4): delete ConstraintWorkspace + constraint free functions
ba88d05 Phase 11 step 13.3 (part 3): delete PreconWorkspace + preconCreate/preconFree/preconApply
c9da234 Phase 11 step 13.3 (part 2): delete MetricWorkspace + geometry/forces/precon free functions
b483428 Phase 11 step 13.3 (part 1): delete RadialProfiles + profilesCreate/profilesFree
22e71d0 Phase 11 step 13.2: InputParams -> ValidatedProblem (delete the legacy parser)
a7c0030 Phase 11 step 13.1: GridParams<T> -> DeviceParams<T> (per-stage param pack)
```

Step 13.3 is now COMPLETE: `RadialProfiles`, `MetricWorkspace`,
`PreconWorkspace`, `ConstraintWorkspace`, `FourierPlan`, and `SpectralState`
are deleted and `interpolateState` folded into `Prolongation::enqueue` (each
operator owns its buffers directly and exposes typed views/accessors; the
matching `*Create`/`*Free` + `compute*`/`inverseDFT`/`forwardDFT`/`precon*`/
`constraint*` free functions are gone), the kernel tests drive the operators,
and the CMake/docs close-out is in. All Class A bit-identical (see the UPDATE
at the top).

---

<a id="phase-11-closeout-handover"></a>

## 2026-08-16 19:15:11+08:00 — cuMES Phase 11 close-out handover — `dynSharedBase()` removal landed, migration complete

**Former path:** `docs/phase-11-closeout-handover.md`
**First tracked:** [`9a3a019`](https://github.com/12ff54e/cuMES/commit/9a3a0199b4e857ab2ab7042587b205753b8809c6) at 2026-08-16T19:15:11+08:00

Status date: 2026-08-16. Branch: `overhaul`. This handover continues
`docs/phase-11-tail-3-handover.md` and closes out **Phase 11** (the
strangler-fig migration, blueprint §11 / `docs/strangler-fig-migration-plan.md`):
the one deferred step-13 item — the `dynSharedBase()` dynamic-shared-memory
indirection (tail-3 §2, "item 4") — is now removed, and the change measured
**Class A bit-identical**, stronger than the Class B re-freeze the docs
predicted. The frozen baselines stand unchanged.

> **Post-closeout acceptance note (2026-08-17):** Phase 11 completed the
> strangler-fig migration, but it did not close every requirement in the
> original CUDA overhaul blueprint. The remaining numerical-safety, I/O,
> precision/runtime, and release-gate work is ordered with explicit exit
> criteria in [`overhaul-completion-plan.md`](#overhaul-completion-plan).
> Do not interpret "Phase 11 is complete" below as final design acceptance.
>
> **EXECUTED (2026-08-17):** the four closure steps are landed —
> `48713b2` numerical safety predicates, `4363e71` config/I-O contracts,
> `d602d2c` runtime/performance policy, and the release-gate commit
> (warnings-as-errors, initcheck, CI, event-DAG tests, docs). Every step
> re-verified the frozen Solovev `251→199→456` / W7-X `1877→1617→2011`
> trajectories **Class A byte-identical** (full dump manifests) — including
> the removal of `--use_fast_math`, which proved codegen-neutral — so the
> branch now satisfies the original overhaul definition of done with no
> re-freeze.

### 1. What changed

The non-templated `dynSharedBase()` accessor (an `extern __shared__ unsigned
char[]` returned as `void*` and `static_cast` to `T*` at each use) is deleted
from `src/fourier_impl.cuh`, `src/geometry_impl.cuh`, and `src/precon_impl.cuh`.
Each consuming kernel now declares its dynamic shared memory directly —
legal per TU because the explicit instantiation split (`src/*_double.cu` /
`src/*_float.cu`) puts exactly one scalar type in each TU. Six kernel sites:

| File | Kernel | Declaration |
| ---- | ------ | ----------- |
| `fourier_impl.cuh` | `inverseAccumulateKernel<T, FuseRzCon>` | `extern __shared__ T sh[];` (4·mpol·kTile) |
| `fourier_impl.cuh` | `deAliasSynthesizeKernel<T>` | `extern __shared__ T sh[];` (2·(mpol-2)·kTile) |
| `geometry_impl.cuh` | `ncurr1FinalizeKernel<T>` | `extern __shared__ T s_buf[];` (2·blockDim.x) |
| `geometry_impl.cuh` | `computeNormPartialsKernel<T>` | `extern __shared__ T s_buf[];` (4·blockDim.x) |
| `precon_impl.cuh` | `pcrSolveKernel<T, R>` | `extern __shared__ T s_tri[];` ((6+2R)·ns) |
| `precon_impl.cuh` | `thomasSolveKernel<T>` | `extern __shared__ T s[];` ((1+rhs_count)·n) |

`src/constraint_impl.cuh` needed no change: its dyn-smem users moved into the
transform module with the de-alias bandpass during step 13.3, so it already
had no `dynSharedBase()`.

### 2. Verification — Class A bit-identical (not the predicted Class B)

The docs warned that the direct form changes `-use_fast_math` FMA fusion
(opaque function return vs. known shared-array aliasing) and perturbs the
trajectory at ~1e-10. It did not reproduce under the current layout: with the
instantiation split in place the compiler inlines the two-line accessor and
emits the same SASS, so the direct form is a pure source-level cleanup.

`scripts/compare_runs.py` old-vs-new (fresh baselines captured from the
pre-change binary immediately before the edit; `--max-iter-delta 0`):

| Config | Iterations | Final FSQR | Restart sequence | Full-precision state |
| ------ | ---------- | ---------- | ---------------- | -------------------- |
| Solovev (axisym default) | 251→199→456 | 9.583e-17 | identical (0 events) | **0.000e+00** max rel diff, all six families |
| W7-X (generic cuFFT) | 1877→1617→2011 | 9.778e-13 | identical (15 events, same iters) | **0.000e+00** max rel diff, all six families |
| Solovev generic (`CUMES_FORCE_GENERIC=1`) | 251→199→456 | 9.583e-17 | — | matches the frozen record |

Per-iteration printed residual rows match at 0.000e+00 everywhere (70/70 W7-X
rows), converged-iteration delta 0. The converged states are byte-identical
at full double precision after 5505 W7-X iterations through every changed
kernel (`inverseAccumulateKernel`, `deAliasSynthesizeKernel`,
`pcrSolveKernel`, `computeNormPartialsKernel`, `thomasSolveKernel`), so the
change is **Class A** — no re-freeze required, the frozen
`251→199→456` / `1877→1617→2011` baseline remains the regression oracle
unchanged.

Test matrix: **35/35 CTest** (double build, incl. the 12 compute-sanitizer
memcheck entries), **23/23 CTest** (float build), both builds clean.

### 3. Docs hygiene in the close-out

- `docs/architecture.md` §1 header + §2: the retention rationale is replaced
  with the measured result (direct `extern __shared__ T[]`, bit-identical).
- The three `*_impl.cuh` files carry a short comment stating the same
  measured result next to the kernels.
- `CLAUDE.md` (project instructions) was still describing the pre-overhaul
  world (`GridParams`, `SpectralState`, `fourier.cuh`, `MetricWorkspace`, the
  dynSharedBase caveat, "hot restart not yet", …). It is rewritten to the
  post-Phase-11 architecture: the operator library + `include/cumes` layout,
  the explicit-instantiation split, `ValidatedProblem`/`DeviceParams<T>`,
  the CLI (`--output-schema`, `--restart`, `--checkpoint`), and the updated
  implemented/omitted feature table. The stale untracked duplicate `AGENTS.md`
  (byte-identical to the old CLAUDE.md) is deleted.
- `docs/cuda-overhaul-blueprint.md` was never `git add`ed; it is committed
  now as the phase record it always was (the two downloaded CUDA programming
  guide dumps, `docs/cuda-asynchronous-execution.md` / `docs/cuda-graphs.md`
  + their `_files/` asset dirs, are reference material and remain untracked).

### 4. Definition of done (strangler plan §1, blueprint §11 Phase 10 exit gate)

- `solverRun` drives only `cumes` operator classes — **done** (step 12).
- The generic transform is a `SpectralOperator<T>` backend on equal footing
  with `AxisymmetricOperator`; no `axisym_active` branch — **done**.
- Transforms own only transform tables/plans/scratch — **done** (step 1).
- `include/*.cuh` legacy structs and `InputParams` deleted; kernel bodies
  survive only as operator implementations — **done** (step 13.1–13.3);
  the surviving `solver.cuh`/`output.cuh`/`vmec_types.h` are thin app shims
  over `cumes` types only.
- Full precise trajectories bit-identical to the frozen baseline — **done**,
  re-verified in §2 including the deferred `dynSharedBase()` item.
- The `dynSharedBase()` decision deferred in tail-3 §2 — **resolved**: removed,
  measured Class A.

### 5. Commits

```
383fdbc Phase 11 close-out: remove dynSharedBase() — direct extern __shared__ T[] (Class A)
```

Phase 11 is complete.

---

<a id="cumes-code-review-2026-08-16"></a>

## 2026-08-16 23:58:59+08:00 — cuMES Code Review Log — branch `overhaul` (main...HEAD)

**Former path:** `docs/cuMES-code-review-2026-08-16.md`
**First tracked:** [`261d6dd`](https://github.com/12ff54e/cuMES/commit/261d6dd553818fcaa3ad1662a8a54f6b196cf26b) at 2026-08-16T23:58:59+08:00

**Date:** 2026-08-16 · **Reviewed commit:** 13f4a2d
**Scope:** the entire `overhaul` branch vs `main` — 186 changed files, ~31.8k diff lines
(the full CUDA overhaul, phases 0–11).
**Method:** 12 finder agents (line-by-line, removed-behavior, cross-file, language
pitfalls, wrapper, reuse, simplification, efficiency, altitude, conventions angles)
over 6 diff slices → 13 verifier agents (1-vote, 3-state) → 1 gap sweep → verify.

**Totals:** 58 findings survived verification (49 CONFIRMED, 9 PLAUSIBLE),
4 candidates REFUTED (appendix).

**Verdict legend:**
- **CONFIRMED** — triggering inputs/state named; wrong output or crash demonstrated
  (several reproduced empirically, e.g. `cufftPlanMany(batch=0)`, the
  `capture_baseline.sh` failures).
- **PLAUSIBLE** — mechanism proven real from the code; trigger is uncertain
  (no current caller/config reaches it), or the claim's framing was partly wrong.

Line numbers refer to the files at HEAD at review time.

---

### Fix Status (2026-08-17)

All 58 findings addressed in an 11-agent fix campaign; 57 fixed (mostly
with a failing repro added first) and 1 kept with documentation (3.1).
Verification: full ctest 35/35, float build 23/23, asan 4/4, and
`compare_runs.py` vs the pre-fix baseline — Solovev and W7-X both PASS
with **zero relative diff** in every residual row, all restart events,
convergence, and all six state families (states bit-identical).

| # | Status | Commit |
|---|--------|--------|
| 1.1 | fixed (+theta-32 repro) | `d755f37` |
| 1.2 | fixed (+mpol=2 repro + end-to-end solve) | `d755f37` |
| 1.3 | fixed (strictly-increasing validation + CumesError throw) | `a7834af` |
| 1.4 | fixed (+corrupt-header repro, ASan before/after) | `149ea79` |
| 1.5 | fixed (+typo-key repro) | `b2ae453` |
| 1.6 | fixed (v1 rejected for nc/h5 at preflight) | `b2ae453` |
| 1.7 | fixed (flags implemented, 9-case matrix) | `b2ae453` |
| 1.8 | fixed (stage cap at kMaxGrids, 9-stage rejected) | `a7834af` |
| 2.1 | fixed (iter-0 dump synced via dumpDeviceArray) | `feec400` |
| 2.2 | fixed (+scratch repro: reported 5 vs true 9) | `6f89bfd` |
| 2.3 | fixed (restart events plumbed; v1 container carries them) | `feec400` |
| 2.4 | fixed (+symlink-swap TOCTOU repro) | `b2ae453` |
| 2.5 | fixed (independent CPU gate + permanent negative control) | `243a239` |
| 2.6 | fixed (JSON-aware ftol rewrite, end-to-end float capture) | `4063e85` |
| 2.7 | fixed (+absolute-path repro, exit 127 before) | `4063e85` |
| 2.8 | fixed (copy reports instead of throwing; fail path runs) | `149ea79` |
| 2.9 | fixed (RAII staging in all three writers) | `149ea79` |
| 2.10 | fixed (+memcheck repro: 6.7 MB in 602 allocs → 0) | `243a239` |
| 2.11 | fixed (test-side memset; gate sensitivity demonstrated) | `243a239` |
| 2.12 | fixed (cc() routing) | `243a239` |
| 2.13 | fixed (toolkit-gated arch list; gates simulated via cmake -P) | `4063e85` |
| 2.14 | fixed (single-TU impl; nm 82 defs in exactly one object) | `4063e85` |
| 3.1 | **kept + documented** — see note below | `feec400` |
| 3.2 | fixed (moves deleted on all 5 classes; memcheck repro 15 errors) | `6f89bfd` `feec400` |
| 3.3 | fixed (+one-null repros: illegal access before, PASS after) | `6f89bfd` |
| 3.4 | fixed (return-before-write; no live OOB confirmed) | `149ea79` |
| 3.5 | fixed (typed enum; unknown tag = write failure) | `149ea79` |
| 3.6 | fixed (+non-aliasing-view repros) | `d755f37` |
| 3.7 | fixed (moved-from sync throws; +test) | `feec400` |
| 3.8 | fixed (copies deleted, nulling moves; +compile probes) | `77f1440` |
| 4.1 | fixed (one tempPathFor/publishAtomic for all writers) | `149ea79` |
| 4.2 | fixed (single inline def in real_fields.cuh; byte-identity pre-verified) | `6f89bfd` |
| 4.3 | fixed (shared FamilyStage copy helper) | `149ea79` |
| 4.4 | fixed (shared payload readers/writers; golden byte-exact) | `149ea79` |
| 4.5 | fixed (manufacturedState builder, 4 consumers; envelopes preserved) | `243a239` |
| 4.6 | fixed (bench_common.cuh; state-hash A/B identical) | `95a395b` |
| 4.7 | fixed (cc wraps check_cuda; exit-based UX kept) | `243a239` |
| 5.1 | fixed | `feec400` |
| 5.2 | fixed (ControlRecord deleted, InvariantVerdict de-templated) | `feec400` |
| 5.3 | fixed (DeviceContext removed; cumes_cuda_runtime now INTERFACE) | `feec400` `4063e85` |
| 5.4 | fixed (xm/xn removed; shim dropped once the test moved to the primary signature) | `feec400` |
| 5.5 | fixed (shared source lists; asan 4/4, link order verified) | `4063e85` |
| 5.6 | fixed (174 artifacts, not 83; pure `*`→`_`, 3 code cross-checks) | `cc3c03b` |
| 5.7 | fixed (kMaxAxis/kMaxM/kMaxN) | `149ea79` |
| 5.8 | fixed (one slab upload in state_slab() order; bit-identical) | `77f1440` |
| 6.1 | fixed (tail-zero folded into inversePackKernel; an intermediate offset bug was caught by the existing inverse tests) | `d755f37` |
| 6.2 | fixed (PinnedBuffer; ~1.0–1.6 µs pageable penalty isolated) | `95a395b` |
| 6.3 | fixed (dataspace hoisted out of the mode loop) | `149ea79` |
| 6.4 | fixed (three buffers carved from the stage arena) | `feec400` |
| 7.1 | fixed (9 dump-window helpers; enqueue 468→191 lines; dump md5 identical) | `feec400` |
| 7.2 | fixed (measuring arena via a virtual carve_span seam; exact byte totals) | `feec400` |
| 7.3 | fixed (std::vector sized p.ntor+1) | `feec400` |
| 8.1 | fixed (events through check_cuda) | `feec400` |
| 8.2 | fixed (snake_case in fourier_impl.cuh + plain_per_el) | `d755f37` `feec400` |
| 8.3 | fixed (ScopedRealSpace/ScopedModeTable RAII) | `feec400` |
| 8.4 | fixed (h_ prefixes in both seed functions) | `77f1440` |
| 8.5 | fixed (DeviceBuffer in both tests) | `243a239` |
| 8.6 | fixed (checked_mul in snapshot_bridge) | `149ea79` |

Bonus fixes found en route: `CUMES_HAVE_NETCDF/HDF5` defines now reach
`cumes_io_host`, so `output_format_available()` no longer reports `.nc`/`.h5`
unavailable in stock builds (the nc/h5 writer path was silently dead —
`4063e85`); `test_regression_kernels` now calls the primary `enqueue_apply`
signature; the toroidal header's stale aliasing contract comment refreshed.

**3.1 note (review claim corrected):** the review's PLAUSIBLE verdict said
the absolute `min_oriented <= 0` gate "never fired in the frozen
trajectories". The baseline logs prove the opposite — the frozen W7-X
trajectory fires the branch 3× in stage 1 (min(signJ·√g) ≈ −9.9e−1/−8.2e−1/
−1.2e−1 at interior jH: genuine sign flips of the early transient, recovered
by the standard restore + delt×0.9). The legacy relative |√g| gate cannot
express a sign flip at all, so restoring it would have changed the frozen
trajectory (Class B). The gate is kept, with a precise deviation comment in
`iteration_controller.hpp`.

---

### 1. Correctness — silent wrong results / crashes on validated inputs

#### 1.1 `src/fourier_impl.cuh:999` — forwardReduceKernel drops theta points when nThetaRed > 16 — CONFIRMED
`forwardReduceKernel` launches with a fixed 16-lane `blockDim.x` and one
reduced-theta point per thread (`int l = threadIdx.x;`, kernel body ~848-851), so any
config with `nThetaRed = ntheta/2+1 > 16` silently drops points `l>=16` from the
forward force quadrature. Validation accepts `mpol <= 16` and auto-resolves
`ntheta = 2*mpol+6`, so mpol ≥ 13 (or explicit ntheta ≥ 32) gives nThetaRed ≥ 17:
the reduced-grid trapezoid points (including the half-weight θ=π endpoint) are never
summed → wrong spectral forces, wrong converged state, no error. Shipped W7-X
(mpol=12 → nThetaRed=16) fits exactly, so the frozen regressions never exercise the
truncation. The sibling `deAliasAnalyzeKernel` got exactly this fix (theta-coverage
loop, lines 624-626); `forwardReduceKernel` did not.
**Fix:** stride loop over `l += blockDim.x` (or tiled launch) like the de-alias kernel.

#### 1.2 `src/fourier_impl.cuh:188` — mpol=2 makes the de-alias plan batch 0 → startup crash — CONFIRMED
`batchDa = 2*(mpol-2)*(ns-1)` is 0 for the validated minimum `mpol=2`
(validated_problem.cpp:89: `mpol must be in [2, 16]`), and the
`ToroidalFftOperator` is constructed **unconditionally** per stage
(stage_solver.hpp:92 — even when the axisymmetric backend is selected afterwards).
`cufftPlanMany(batch=0)` at fourier_impl.cuh:196-199 returns
`CUFFT_INVALID_SIZE` (empirically probed on CUDA 12.1) → `check_cufft` throws at
stage setup. If a cuFFT version tolerated batch=0, the de-alias launch
`gridDim.x = mpol-2 = 0` would be an illegal launch anyway.
**Fix:** skip plan creation / de-alias launches when `mpol <= 2` (there are no
m≥2 modes to de-alias), or raise the validation floor to mpol=3.

#### 1.3 `src/prolongation_impl.cuh:99` — exit(EXIT_FAILURE) mid-solve on validated equal-ns stages — CONFIRMED
`Prolongation::enqueue` calls `exit(EXIT_FAILURE)` when
`p_new.ns <= p_old.ns || p_new.mnmax != p_old.mnmax || p_old.ns < 3`, but
validation only requires `ns_array` to be **monotonically non-decreasing**
(validated_problem.cpp:136-138), so `ns_array {11,11,55}` passes validation and
then dies inside the library: RAII device buffers are not destroyed, the
`--checkpoint` file is never written, and the exit code bypasses the CLI's
run-report mapping — contradicting the caller's documented contract
(multigrid_solver.hpp:8: "It never calls exit() or writes output").
**Fix:** return a failure value to `MultigridSolver`, or validate strictly
increasing ns at config time.

#### 1.4 `src/cumes/io/io_common.hpp:72` — checkStateDimensions cast wraps negative → std::terminate on corrupt header — CONFIRMED
`static_cast<long long>(*needed) > *sz` casts a checked-`size_t` byte count; a
corrupt int32 header with `ns*mnmax` in [2^63, 2^64) (e.g. ns=2147483647,
mnmax=95000000) wraps negative, the file-size bound silently passes, and the
readers then execute `fam.resize(~2e17)` → `std::bad_alloc`. main.cu:358 catches
only `cumes::CumesError`, so the process `std::terminate`s instead of producing the
intended "dimensions implausible" error (checkpoint.cpp:91-95,
legacy_binary_v0.cpp:76-82).
**Fix:** compare in `size_t` / `long double`, or bound ns/mnmax against sane
maxima before the multiply.

#### 1.5 `src/main.cu:210` — validation warnings are never printed — CONFIRMED
Unknown input keys (json_reader.cpp:359-362) and skipped out-of-range boundary
harmonics (validated_problem.cpp:158-164) are collected into `vp.warnings()`
explicitly "so the caller can report them" (validated_problem.cpp:228-230), but
main.cu never iterates `warnings()`. A typo'd key (`"n_theta": 24`) or an
`m >= mpol` rbc harmonic runs silently with defaults/dropped harmonics — the legacy
parser printed `cuMES: WARNING: unknown input key ... ignored` and
`skipping mode m=... n=...` to stderr.
**Fix:** print `vr.value().warnings()` to stderr after successful validation.

#### 1.6 `src/main.cu:310` — `--output-schema v1` silently ignored for .nc/.h5 — CONFIRMED
The flag is parsed and validated for any output (main.cu:129-138) but applied only
in the binary branch (main.cu:305-311); the nc/h5 branch (326-331) calls
`outputSave<Real>()` without the schema, and the NetCDF/HDF5 writers are hard-wired
to the v0 fixed-capacity layout (output_netcdf.cpp:27-33). `--output-schema v1
out.nc` silently writes v0 with no warning or error.
**Fix:** reject the combination, or implement v1 for nc/h5.

#### 1.7 `src/main.cu:145` — `--input`/`--output` documented but unhandled — CONFIRMED
The file header documents `--input <path>` (line 4) and `--output <path>` (line 9),
but the parse loop handles only `--output-schema`/`--restart`/`--restart-legacy`/
`--checkpoint`; `cuMES --input x.json out.bin` hits the "unknown option" branch and
exits EXIT_FAILURE. The diff both advertises and fails to implement the flags.
**Fix:** implement the flags or delete them from the header comment.

#### 1.8 `include/cumes/io/legacy_provenance.hpp:62` — 9+-stage configs produce self-inconsistent v0 provenance — CONFIRMED
`from_validated` stores `p.n_grids = s.stages.size()` uncapped while the stage
arrays truncate at `kMaxGrids=8` (line 20). The legacy parser hard-failed >8 stages
("entries exceed the 8-entry capacity", main:src/input_json.cu:104-107); the new
validation has no stage-count cap. A 9-stage JSON now solves all 9 stages while the
v0 writers emit scalar `n_grids=9` against an `ngrids=8` dimension
(output_netcdf.cpp:67, output_hdf5.cpp:81/128) — stages 9+ silently dropped from
the provenance.
**Fix:** cap or validate the stage count against `kMaxGrids` again.

---

### 2. Correctness — diagnostics, provenance, CLI/tooling contracts

#### 2.1 `src/solver_impl.cuh:609` — dump iter-0 diagnostic reads d_r_e unsynchronized — CONFIRMED
With `CUMES_DUMP=1`, the iter-0 block does a synchronous `cudaMemcpy` of
`rs.d_r_e` on the legacy default stream right after the inverse transform was
enqueued on the `cudaStreamNonBlocking` compute stream (stream.hpp:17) — the
legacy stream does not wait for it. The legacy code synced
(main:src/solver.cu:675-679: `cudaEventRecord(ev1); cudaEventSynchronize(ev1);`),
and the file's own `dumpDeviceArray` documents the hazard and calls
`cudaDeviceSynchronize()` first (solver_impl.cuh:469-474). Result: stale/garbage
`[loop diag]` print and `dump/cuMES/debug_r_e.bin`.
**Fix:** add `cudaDeviceSynchronize()` (or route through `dumpDeviceArray`).

#### 2.2 `src/geometry_impl.cuh:535` — jacobianStatsKernel vmax under-reports on sign-flipped √g — CONFIRMED
The new-min branch (`else if (ov < vmin) { vmin = ov; ... }`) never updates
`vmax = fmax(vmax, a)`, so a flipped element (ov = -a) taking that branch excludes
its magnitude from `max|√g|`. Legacy reduced both over `fabs(g)` where
`vmax >= vmin` made the skip harmless. Impact: only the BAD-JACOBIAN diagnostic
printf (solver_impl.cuh:1290-1292) under-reports max|√g| — the validity decision
is unaffected because `min_oriented <= 0` short-circuits the `||` chain first, and
no-flip runs are exactly legacy-equivalent.
**Fix:** update `vmax` unconditionally in both branches.

#### 2.3 `include/cumes/io/run_report.hpp:35` — StageReport.restarts is never populated — CONFIRMED
The only construction site (multigrid_solver.hpp:65-72) never touches `restarts`,
yet versioned_binary.cpp:84-85 serializes it into every v1 container — the restart
history (which the frozen regression bar tracks, e.g. W7-X's restart events) is
always recorded as 0. The controller knows the events
(iteration_controller.hpp:165-190, `RestartReason::kBadJacobian/kBadProgress`) but
nothing plumbs them into the report.
**Fix:** carry restart events in `SolverResult<T>` and fill `StageReport.restarts`,
or drop the field and the container slot.

#### 2.4 `src/main.cu:83` — fill_provenance re-hashes the input file after the solve — CONFIRMED
`read_and_validate` parses the file at startup (no raw bytes retained), then
post-solve `fill_provenance` re-opens and FNV-1a-hashes the path. A file modified
or replaced mid-solve records a `source_hash` of bytes the solver never consumed
(TOCTOU), plus the unconditional whole-file double read.
**Fix:** hash the bytes once at read time and carry them into the report.

#### 2.5 `tests/test_force_verify.cu:154` — converged-force gate is tautological — CONFIRMED
The recompute path (lines 110-129) runs the same production
Geometry/MagneticField/Force/ToroidalFft operators the solver just used, so a
broken force formula that still permits convergence (e.g. zero forces on m≥2)
makes solverRun converge AND the recomputed residuals ~1e-16 — all three
`kFailThresh=1e-4` CHECKs pass. Contradicts the header's claim "a broken force
formula shows up as O(1) residuals" (the file's own comment at 150-152 admits the
shared kernels).
**Fix:** recompute forces via an independent CPU reference (as
test_force_reference.cu does) instead of the production kernels.

#### 2.6 `scripts/capture_baseline.sh:157` — float-capture sed hardcodes 3 ftol entries — CONFIRMED (empirical)
The sed rewrite always produces `[ft, ft, ft]`. A 2- or 4-stage `--configs` entry
fails validation (`ftol_array length must match ns_array`, json_reader.cpp:291);
a multi-line `ftol_array` silently escapes the line-based sed and the float build
then hard-errors on the double-tuned ftols below the 1e-6 floor
(precision_policy.hpp:33). Shipped configs are single-line 3-stage, so the default
path works.
**Fix:** JSON-aware substitution, or a solver-side per-stage ftol override env
(like `CUMES_MAX_ITER`/`CUMES_DELT0`).

#### 2.7 `scripts/capture_baseline.sh:101` — absolute --build path mangled via $OLDPWD — CONFIRMED (empirical)
`"$OLDPWD/$build/cuMES"` concatenates into `<repo-root>//<abs>/cuMES` for an
absolute `--build`; the preflight `[ -x "$BUILD/cuMES" ]` (line 89) passes but the
launch dies with exit 127 under `set -euo pipefail` (the only diagnostic is buried
in run.log). Same mangle for `--float-build` (line 159) and the `--schema` reruns
(116-119).
**Fix:** normalize `--build` to an absolute path once at startup.

#### 2.8 `src/output.cpp:112` — check_cuda throw bypasses the atomic-publish contract — CONFIRMED
`cumes::check_cuda(cudaMemcpy(buf, d, nb, D2H), tag)` inside `writeFam` throws
`CumesError`; the documented failure contract (close temp, remove temp, return
false — the `fail` lambda at output.cpp:94-101) is only wired to fwrite failures.
A GPU fault during the final T→double copy leaves `<out>.tmp.<pid>` on disk and
`fp` unclosed (main catches the error and exits). Same throw point at
output_hdf5.cpp:153 and output_netcdf.cpp:154, which additionally skip
H5Fclose/nc_close + remove(tmp).
**Fix:** wrap the copy in try/catch and run the fail path, or use unchecked
`cudaMemcpy` + manual error handling consistent with the writers' contract.

#### 2.9 `src/output_hdf5.cpp:179` — staging-buffer leak on write failure — CONFIRMED
`H5_CHECK` returns false on any per-mode `H5Dwrite` failure before
`delete[] dbuf; delete[] buf;` (185-186). NetCDF twin identical (NC_CHECK at
165-170, deletes at 171-172). One-shot leak of ~2·ns·mnmax·8 bytes (~270 KB for
W7-X) immediately before exit; visible only under LeakSanitizer.
**Fix:** delete before the macro's return-false, or RAII the buffers.

#### 2.10 `tests/test_regression_kernels.cu:435` — testPcr leaks a 43-buffer RealSpaceStorage — CONFIRMED
`testPcr` creates `RealSpaceStorage` via `realSpaceCreate(p)` (43 cudaMallocs with
the default `arena=nullptr`) and frees only `d_f`/`mt`; the sibling `testDealias`
calls `realSpaceFree(rs)` (line 417). 7 ns values × double/float = 14 leaked
43-buffer sets, reclaimed only at process exit.
**Fix:** add `realSpaceFree(rs)`.

#### 2.11 `tests/test_geometry_iso.cu:116` — coverage check relies on uninitialized memory — PLAUSIBLE
The kernel-coverage gate counts exact `0.0` entries in `bsupu`/`bsubu`, but those
buffers are never zero-initialized (geometry_impl.cuh:64-65 allocates via
cudaMalloc/arena with no memset). On fresh allocations the driver's zeroed pages
would make a skipped point read 0.0 and the check fire; on reused non-zero memory
it would not — the test's ability to catch the launch-shape regression it exists
for depends on allocator state. The codebase itself memsets similar buffers "so a
diagnostic dump … is deterministic" (fourier_impl.cuh:252-266).
**Fix:** memset the buffers (or use a sentinel fill) before the kernel.

#### 2.12 `tests/test_geometry_iso.cu:91` — unchecked cudaMemcpy in the m=1 block — PLAUSIBLE (framing)
The copies at 91/96 are indeed raw `cudaMemcpy` (failure → uninitialized `hcc`,
test still exits 0), but they are NOT a regression (identical in
main:tests/test_geometry_iso.cu) and NOT unique (lines 112/115/121 are also raw).
**Fix:** route through `cc()` for consistency.

#### 2.13 `cmake/CumesCudaArchitectures.cmake:18` — default arch list breaks CUDA 11.0–11.7 — CONFIRMED
`61;75;80;86;89` is set with no toolkit-version conditional, while CLAUDE.md
documents "CUDA Toolkit >= 11" and CumesDependencies.cmake:14 checks
`find_package(CUDAToolkit 11 REQUIRED)`. On CUDA 11.0–11.7 `nvcc` errors
"Unsupported gpu architecture compute_89" on the default `cmake -B build` (11.8
is the first toolkit supporting sm_89). The comment pointing to
docs/performance.md for the compat policy is stale — that doc contains no such
text.
**Fix:** gate sm_89 on `CUDAToolkit_VERSION >= 11.8` (and sm_86 on >= 11.1),
or document the override.

#### 2.14 `src/json_parser.cpp:8` + `src/cumes/config/json_reader.cpp:13` — ZQ_JSON_PARSER_IMPLEMENTATION defined twice — CONFIRMED
Both TUs define the macro (each file's comment claims to be "the only TU").
`nm` on the built objects shows 82 identical external `json::` symbols in both —
a genuine ODR violation. The current link graph avoids a collision only because
every consumer links `cumes_config_json` before `cumes_json` (PUBLIC at
CMakeLists.txt:78), so `json_parser.o` is never pulled from the archive —
"object-pulling luck". An LTO/unity/whole-archive/shared build that pulls both
members fails with duplicate definitions.
**Fix:** define the implementation in exactly one TU (or make the header
inline-only) and delete the other define.

---

### 3. Correctness — latent (mechanism real, no current trigger)

#### 3.1 `include/cumes/solver/iteration_controller.hpp:97` — jacobian_invalid adds an absolute min_oriented <= 0 gate — PLAUSIBLE
The new gate adds `s.min_oriented <= T(0)` (absolute, row-agnostic, computed over
all `nHalf = (ns-1)*nZnT` entries including the jH=0 axis row) where legacy gated
on `gbad>0 || gmax<=0 || (gmin < 1e-12*gmax && gminIdx >= nZnT)` — a relative
min-|√g| test that could not detect a sign flip and excluded the axis row. A
sign-flipped √g anywhere (most plausibly at jH=0 where |√g|→0) now forces
restoreState + delt×0.9 where the legacy trajectory continued. The frozen
trajectories are Class A bit-identical (the branch never fired in them), and
geometry_impl.cuh:495-499 marks the change trajectory-neutral there; a float
build or harsh `--restart` initial guess is the plausible trigger. Would confirm
by instrumenting the branch vs a legacy run.

#### 3.2 Defaulted moves + freeing destructors on 5 operator classes — PLAUSIBLE
`ToroidalFftOperator` (toroidal_fft_operator.hpp:37), `GeometryOperator`
(geometry_operator.hpp:28), `Profiles` (profiles.hpp:28), `Preconditioner`
(preconditioner.hpp:28), `ConstraintOperator` (constraint_operator.hpp:39) all
declare defaulted move ctor/assignment next to deleted copies, while their
destructors unconditionally `cudaFree` raw pointers / `cufftDestroy` plan handles
without nulling (e.g. fourier_impl.cuh:220-229, geometry_impl.cuh:86-95). Any
move (factory return, container storage, std::move) double-frees. Nothing moves
them today (StageSolver builds stack locals passed by reference;
`AxisymmetricOperator` is safe — DeviceBuffer moves null the source).
**Fix:** `= delete` the moves, or implement transfer-and-null.

#### 3.3 `src/axisymmetric_impl.cuh:278` — rzconKernel launched on OR but writes both outputs unconditionally — PLAUSIBLE
The interface documents "rCon/zCon may be null views to skip that output"
(spectral_operator.hpp:33-37); the generic backend guards each pointer
(fourier_impl.cuh:455-456) and launches per-slot with `(rCon, nullptr)`/
`(nullptr, zCon)`. The axisymmetric backend launches on
`rCon.data() != nullptr || zCon.data() != nullptr` into a kernel that writes both
`rCon[idx]` and `zCon[idx]` with no null check → device illegal-address fault for
one-null callers. No current caller passes exactly one null (solver passes both,
tests pass both null).

#### 3.4 `src/cumes/io/versioned_binary.cpp:57` — short-family guard still writes — PLAUSIBLE
`if (fam.size() != n) ok = false;` then proceeds to `write_f64_array(fp,
fam.data(), n)` — an OOB read of a short host vector (same pattern
checkpoint.cpp:48-49). No in-tree trigger: every producer resizes families to
exactly `family_size()` (snapshot_bridge.cuh:42, checkpoint.cpp:92/127,
versioned_binary.cpp:129), and even when tripped, `ok=false` still removes the
temp and returns an error — the defect is the OOB read itself.
**Fix:** return before the write when the size mismatches.

#### 3.5 `src/cumes/io/versioned_binary.cpp:62` — precision tag from string-compare — PLAUSIBLE
`precision = (report.build.scalar_type == "float") ? 1 : 0` is a stringly-typed
convention for a format discriminator. All in-repo producers write exactly
"double"/"float" from the same sizeof ternary (main.cu:81, test_io_golden.cu:149),
and the reader ignores the trailer field entirely, so nothing misfires today — a
foreign producer writing "single"/"fp32" would silently record 0=double.
**Fix:** a typed enum on RunReport.

#### 3.6 `src/fourier_impl.cuh:1038` — enqueue_inverse/enqueue_forward ignore the passed view bundles — PLAUSIBLE
`(void)geometry` / `(void)real_force` and the captured `rs_` is used instead,
while `AxisymmetricOperator` honors the passed views (axisymmetric_impl.cuh
266-292). Divergent contracts of the same `SpectralOperator` interface: a caller
passing non-aliasing views gets correct results on the axisymmetric backend and
wrong results on the generic one. Today's only caller wraps views over the same
`rs` the operator was constructed with (solver_impl.cuh:519-535), and the
toroidal header notes the aliasing — no wrong output in the current codebase.

#### 3.7 `include/cumes/runtime/stream.hpp:44` — synchronize() on a moved-from Stream syncs the default stream — CONFIRMED (use-after-move only)
Moves null the source (`other.stream_ = nullptr`), and `synchronize()` calls
`cudaStreamSynchronize(stream_)` with no null guard; `cudaStreamSynchronize(0)`
is defined behavior — it waits on the **legacy default stream**, which does not
sync nonblocking-stream work — and returns cudaSuccess. A use-after-move thus
reports success while never waiting for the moved-away stream's work. No current
call site moves a Stream and then synchronizes the source.
**Fix:** assert/throw on `stream_ == nullptr`.

#### 3.8 `include/cumes/state/mode_table.cuh:35` — DeviceModeTable implicit copy → double-free — PLAUSIBLE
Raw owning aggregate (d_xm/d_xn + arena_backed) with no deleted copy ops and
`modeTableFree` freeing unconditionally (35-38). Every call site uses
guaranteed-elision copy-init and frees once, so no copy occurs today; a
pass-by-value or explicit copy followed by modeTableFree on both would
double-cudaFree.
**Fix:** delete the copy ops (or RAII-own the pointers).

---

### 4. Cleanup — reuse (duplicated implementations)

#### 4.1 `src/output.cpp:23` — tempPathFor/publishAtomic re-implemented at four sites — CONFIRMED
`cumes::io_detail::tempPathFor`/`publishAtomic` exist in
src/cumes/io/io_common.hpp:20/27 (used by checkpoint.cpp and legacy_binary_v0.cpp);
output.cpp:23/33 re-implements the same protocol with bool+stderr instead of the
reason-string convention, and output_hdf5.cpp:53 / output_netcdf.cpp:38 hand-roll
the temp-path string inline. Four sites, two conventions — a fix to the shared
helper will not reach the three writers.

#### 4.2 `src/forces_impl.cuh:28` — geometryParityViews byte-identical to geometry_impl.cuh:46 — CONFIRMED
Verified byte-identical (`sed -n 27,39p` vs `45,57p` diff clean); the header
comment says "mirror of geometry_impl.cuh's helpers". Both TUs already include
`cumes/state/real_fields.cuh` — the natural single home for one inline
definition. Any view-bundle change must be applied twice or the force kernel's
views silently diverge.

#### 4.3 `src/output_hdf5.cpp:153` — D2H + T→double block copy-pasted into three writers — CONFIRMED
output_hdf5.cpp:153-154, output_netcdf.cpp:154-155, output.cpp:112-113 are
identical modulo the error tag. The on-disk-double contract (the Python compare
scripts parse doubles regardless of T) is re-implemented three times; one shared
helper next to io_common.hpp's serialization primitives is mechanically feasible.

#### 4.4 `src/cumes/io/checkpoint.cpp:58` — read_checkpoint duplicates VersionedBinaryReader — CONFIRMED
checkpoint.cpp:67-96 vs versioned_binary.cpp:108-135: same magic/version/ns/mnmax
header shape, same `checkStateDimensions`, same six-family f64 loop — differing
only in magic string ("CUMECKP1" vs "CUMES001") and error labels; writer halves
likewise (checkpoint.cpp:42-49 vs versioned_binary.cpp:52-59, the v1 writer then
appending its provenance trailer). Any layout evolution must be edited and
re-tested in both.

#### 4.5 `tests/test_geometry_ncurr.cu:53` — manufactured-state fixture copy-pasted across 4 tests — CONFIRMED
The six-family manufactured state appears at test_geometry_ncurr.cu:52-76,
test_constraint_tcon.cu:53-84 (quadratic envelope — deliberate per its comment),
test_geometry_iso.cu:34-60 (fillState), and a reduced variant at
test_cuda_graph.cu:95-109. tests/support/cumes_test_support.cuh hosts no state
builder, though its comment says helpers move in "when a second consumer
appears" — there are now 3-4.

#### 4.6 `benchmarks/graph_realpass.cu:76` — benchmark harness copy-pasted across 3 files — CONFIRMED
`now_us` (fixed_iteration.cu:55-58 vs graph_realpass.cu:76-79) byte-identical;
`median`, the `need` CLI lambda, the read_and_validate block, and the operator
stack + CUMES_FORCE_GENERIC selection structurally verbatim across
fixed_iteration.cu / graph_realpass.cu / graph_overhead.cu (only stderr prefixes
differ). Every harness change must land in three files or silently diverge.

#### 4.7 `tests/support/cumes_test_support.cuh:19` — cc() re-implements cumes::check_cuda — CONFIRMED (trivial)
cuda_status.hpp:52-58 throws `CumesError`; the test helper does the identical
check as fprintf+exit. The duplication is literal (cuda_status.hpp's own header
comment says check_cuda replaces "the per-file checkCuda/cc/ccf helpers that used
to exit(1)"), though the exit-based form is a defensible test-UX choice (an
uncaught throw → std::terminate loses the tag message).

---

### 5. Cleanup — simplification (dead code / derivable state)

#### 5.1 `include/cumes/solver/equilibrium_operator.hpp:96` — member op_ written, never read — CONFIRMED
Only the declaration and `op_(op)` in the constructor exist; `enqueue` uses the
duplicate `transform_op_` (solver_impl.cuh:515). Delete `op_`.

#### 5.2 `include/cumes/solver/control_record.hpp:62` — ControlRecord<T> is dead; InvariantVerdict<T> templates on unused T — CONFIRMED
`ControlRecord` has no non-comment reference in any TU (the header's own comment
admits it was the unlanded Phase 6A single-fence form). `InvariantVerdict` holds
only two bools — the T parameter exists purely to spell
`InvariantVerdict<double>` at the call site. Delete ControlRecord; de-template
InvariantVerdict.

#### 5.3 `include/cumes/runtime/device_context.hpp:26` — DeviceContext is dead infrastructure — CONFIRMED
Only tests/test_runtime.cu constructs it; production main.cu:275 creates its own
`cumes::Stream`. The header comment ("the solver keeps the legacy default stream
until the Phase 6A scheduling work") is stale — Phase 6A is landed.

#### 5.4 `src/precon_impl.cuh:982` — enqueue_apply retains dead xm/xn parameters — CONFIRMED
`(void)xm; (void)xn; // legacy signature` — the body never touches them, yet the
sole caller (solver_impl.cuh:975) still plumbs `transform.xm()/xn()` and the
header retains them. Remove the parameters and update the call site.

#### 5.5 `CMakeLists.txt:257` — CUMES_HOST_SANITIZERS re-declares all host targets — CONFIRMED
The asan block duplicates every source list and link setup of the four host
libraries + four test executables (257-316). The block's own comment explains the
deliberate reason (sanitized runtime FIRST in each asan executable's library
list), so this is a maintainability cost, not a bug: a new TU in `cumes_core`
must be added to two source lists or the sanitized build silently diverges.

#### 5.6 `docs/mathematics.md:21` — normative contract corrupted by markdown artifacts — CONFIRMED
83 occurrences of `*{`/`\sum*`-style artifacts: `s*j` for `s_j`, `\sum*{m,n}`,
`R^{cc}*{mn}`, `S*{mn}=m*{\mathrm{scale}}n*{\mathrm{scale}}`, `\mu*0` for
`\mu_0` (line 198). The file declares itself the normative numerical contract
("a change that alters any expression below is at least Class B") yet the
mixed-gauge formulas (e.g. `(f_Rss ± f_Zcs)/√2`, lines 91-92) are semantically
garbled for a contract document.

#### 5.7 `src/output_hdf5.cpp:130` — provenance dims hardcoded 32/16/16 — CONFIRMED
`d32[1]={32}`/`d16x16[2]={16,16}` (and output_netcdf.cpp:69-71) hardcode what
`LegacyInputProvenance::kMaxAxis=32, kMaxM=16, kMaxN=16`
(legacy_provenance.hpp:22-24) already defines — the same function uses
kMaxGrids/kMaxCoeff for the other dims. A capacity change would silently truncate
the writes.

#### 5.8 `include/cumes/state/seed_state.hpp:178` — six separate uploads instead of the slab — CONFIRMED
restart_state stages six host arrays and issues six H2D memcpys in the exact
`state_slab()` order (Rcc Zsc Lsc Rss Zcs Lcs; spectral_storage.hpp:7-8,52), which
was built precisely so "the six old per-family copies become one". init_state
(lines 108-119) has the same six-upload block. One 6·mnmax·ns staging buffer +
one memcpy replaces both (the double→T conversion is the only reason for host
staging).

---

### 6. Cleanup — efficiency

#### 6.1 `src/fourier_impl.cuh:510` — full 12·mpol·ns·nz2 memset every inverse pass — CONFIRMED
`inversePackKernel` writes every bin n<=ntor of all 12 slots each pass, so only
the tail bins ntor+1..nz2-1 are stale; the memset clears the whole buffer
(12·12·99·19·16 ≈ 4.3 MB for W7-X, ~29 MB for nzeta=256) every iteration. The
tail-zero can be folded into inversePackKernel (threads with n==ntor zero the
tail of their slots) — a one-time construction memset does not work because the
forward D2Z overwrites the bins each pass. Correctness is unaffected.

#### 6.2 `benchmarks/graph_realpass.cu:203` — pageable staging makes cudaMemcpyAsync synchronous — CONFIRMED
`new Real[16]` is pageable, so the benchmark's `cudaMemcpyAsync` degrades to a
synchronous 2-hop copy, while the production path it measures uses
`cumes::PinnedBuffer<double>(16)` (solver_impl.cuh:1087; pinned_buffer.hpp:3-5
documents exactly why). Both branches use the same h_ctl, so the
graph-vs-direct saving estimate mostly cancels — the biased part is the
absolute per-pass wall. Use PinnedBuffer like production.

#### 6.3 `src/output_hdf5.cpp:165` — H5Dget_space inside the per-mode loop — CONFIRMED
`H5Dget_space(ds)`/`H5Sclose(fs)` run once per mode per family (6·mnmax handle
pairs, e.g. ~800-1800 for W7-X) though the file dataspace `ds` is invariant
across modes. Fetch once before the loop.

#### 6.4 `src/solver_impl.cuh:509` — d_f_spec_/d_control_/d_psum_ outside the stage arena — CONFIRMED
The constructor receives the stage DeviceArena and carves precon_/constraint_
from it, but allocates the solver's own three buffers as standalone cudaMallocs —
the stage was consolidated so "one cudaMalloc per stage instead of ~110" and the
arena's peak/liveness report understates the real footprint. (Note: d_f_spec_ is
the largest of the operator's own buffers, but the arena's d_zeta_real is ~5.5×
larger for W7-X.)

---

### 7. Cleanup — altitude (bandaids / fragile decoupling)

#### 7.1 `src/solver_impl.cuh:607` — ~278 lines of dump machinery interleaved in enqueue — CONFIRMED
8 `#ifdef DUMP_CUMES_VERIFY` blocks (623-678, 702-719, 748-804, 810-866,
875-903, 920-937, 959-964, 977-1021) interleave the per-iteration DAG builder;
`dumpDeviceArray` allocates a fresh `new T[nelem]` and full
`cudaDeviceSynchronize()` per call, and `dumpEnabled()` re-reads the env var at
every one of ~6 entry points per iteration. The blueprint (§6.12) wants
arithmetic-only enqueue with observers as subscribers — extracting the dump
windows into the observer/record mechanism would shrink enqueue by a third.

#### 7.2 `include/cumes/solver/stage_solver.hpp:37` — stage_arena_bytes hand-sums every module's sizes — CONFIRMED
The function's own comment admits "keep the two in sync" with the modules'
authoritative `alloc_span` calls (43 arrays, 15 half-grid,
25·nH+9·ns+7·mnmax·ns+…, 64 KiB slack). Any buffer added/removed in a module
silently drifts the plan — an underestimate fails only at runtime, an
overestimate wastes memory silently. (Correction to the original claim: the two
benchmarks reuse `stage_arena_bytes` itself; they duplicate the surrounding
operator-stack setup, not the byte arithmetic.)
**Fix:** per-module size-report functions or a measuring-arena dry run.

#### 7.3 `src/solver_impl.cuh:1128` — h_ax[64] magic bound decoupled from the ntor cap — CONFIRMED
`T h_ax[64]; // ntor+1 <= 64 for the hardcoded inputs` followed by a
`cudaMemcpy2D` of `p.ntor + 1` elements. The inputs are no longer hardcoded; the
real guarantee is the validator's `ntor in [0, 15]`
(validated_problem.cpp:92-93). Raise that cap without touching this line and the
stack array overflows in the production printIterRow path.
**Fix:** size from `p.ntor + 1` (or a small std::vector).

---

### 8. Cleanup — conventions (CLAUDE.md exact rules)

#### 8.1 `src/solver_impl.cuh:541` — four cudaEventCreate calls unchecked — CONFIRMED
Breaks the CLAUDE.md rule that CUDA calls are "error-checked through the
centralized cumes::check_cuda/check_cufft" (the four `cudaEventDestroy` calls at
553-554 are likewise unchecked). A failed creation leaves an uninitialized handle;
the later `cudaEventRecord` is also unchecked, so the failure surfaces as garbage
timing instead of a CumesError at the true failure point.

#### 8.2 `src/fourier_impl.cuh:519` — kTile/nKTiles/kTileA/kTileS/nKTilesA/nKTilesS/blkX/invSmem are camelCase — CONFIRMED
CLAUDE.md: "**Variables:** `snake_case`". Runtime variables, not constants (the
kCamelCase escape), repeated at lines 387/403/633/734; `plainPerEl` at
solver_impl.cuh:1265 likewise.

#### 8.3 `include/cumes/solver/stage_solver.hpp:90` — realSpaceCreate/Free + modeTableCreate/Free non-RAII — CONFIRMED
Breaks CLAUDE.md: "xCreate/xFree replaced by RAII classes owning their buffers."
The pairing is manual (90-91, 113-114) and a throw between them skips the frees —
harmless on the always-arena stage path (both frees no-op when arena_backed) but
a full leak of ~30 cudaMallocs for any nullptr-arena caller (e.g.
test_constraint_tcon.cu).

#### 8.4 `include/cumes/state/seed_state.hpp:65` — host staging buffers missing h_ prefix — CONFIRMED
`c`, `s`, `zsc`, `zcs`, `lsc`, `lcs` (lines 65-67) and again in restart_state
(138-143). CLAUDE.md: "Host pointers: `h_` prefix (e.g., `h_rmnc`, `h_cos`)."

#### 8.5 `tests/test_geometry_ncurr.cu:122` — raw cudaMalloc/cudaFree for the stats probe — CONFIRMED (style-only)
Breaks CLAUDE.md: "All device allocations via RAII (DeviceBuffer/DeviceArena)".
The free is present and checked, so no leak — a `DeviceBuffer<T> stats(4)` is a
drop-in replacement. (test_regression_kernels.cu:475 also raw-mallocs `d_f`.)

#### 8.6 `include/cumes/io/snapshot_bridge.cuh:27` — bare size products vs the checked_mul mandate — CONFIRMED
checked_size.hpp:3-6: "Every derived element-count product … must go through
these helpers, not a bare `a * b`." snapshot_bridge.cuh:27-29 computes
`ns() * mnmax()` and `kCount * one` bare. No realistic overflow today (mpol
capped at 16, validated ints) — a mandate-conformance gap, not a hazard.

---

### Appendix — REFUTED candidates (checked, found wrong or guarded)

1. **scripts/capture_baseline.sh:153** — "`local` at top level breaks bash 4.x":
   the line is `local_scratch=...`, a plain variable assignment (the token is the
   variable name, not the `local` builtin). Works on all bash versions.
2. **src/solver_impl.cuh:1127** — "axisRAtZeta0's cudaStreamSynchronize is
   redundant": it is necessary — the print path runs after the descent kernels
   are enqueued, and the nonblocking compute stream does not order against the
   legacy default stream used by the copy; removing it would read stale data.
3. **benchmarks/graph_realpass.cu:203** — "Real[16] truncates the 16-double
   control record in float builds": CUMES_USE_FLOAT applies only to the cuMES
   executable; the benchmark is pinned to `cumes_cuda_double`
   (CMakeLists.txt:356-361), so Real is always double there.
4. **src/prolongation_impl.cuh:115** — "cudaDeviceSynchronize masks a
   new-slab zeroing race": `DeviceBuffer::zero()` uses synchronous cudaMemset
   (device_buffer.cuh:66-70), which completes before the constructor returns;
   the barrier's documented purpose is ordering the previous stage's
   coarse-state writes, which it does fully.

---

<a id="overhaul-completion-plan"></a>

## 2026-08-17 20:13:23+08:00 — cuMES overhaul completion plan

**Former path:** `docs/overhaul-completion-plan.md`
**First tracked:** [`48713b2`](https://github.com/12ff54e/cuMES/commit/48713b232be915cec7a8f7afaf2f5b05c5715e67) at 2026-08-17T20:13:23+08:00

> **CI AVAILABILITY CORRECTION (2026-08-18).** The permanently queued
> self-hosted GPU job has been removed because no such runner is available.
> Hosted CI now uses Ubuntu 22.04 and pins `Jimver/cuda-toolkit` v0.2.35 by
> immutable commit (the nonexistent `@v0.2` reference caused every hosted job
> to fail during setup). Its package inputs also keep CUDA-prefixed packages
> (`nvcc`, `cudart-dev`) separate from `libcufft-dev`, as required by the
> action's network installer. Hosted jobs compile every supported matrix and run
> CPU-only plus ASan/UBSan tests. `scripts/ci_gpu.sh`, Compute Sanitizer,
> trajectories, and GPU performance remain documented manual gates, postponed
> until suitable hardware is available; they no longer keep Actions runs
> queued indefinitely.

> **FINAL ACCEPTANCE RESTORED (2026-08-18).** Commit `de265cd` closes the final
> hostile-container resource/schema audit recorded in
> [`v1-reader-resource-hardening-handoff.md`](#v1-reader-resource-hardening-handoff):
> fixed-width-only HDF5 provenance, bounded state allocation, signed
> native-int-width HDF5 schema checks, closed-range values, transactional
> reports, and the expanded ordinary/ASan malformed fixtures. All available
> gates pass (verify 58/58, sanitizer 90/90, every precision/backend matrix
> 30/30, `ci_gpu.sh`, accepted trajectories, and byte-identical legacy states).
> Modern-GPU validation remains separately POSTPONED and is the only
> outstanding acceptance item.

> **POST-IMPLEMENTATION REVIEW (2026-08-17).** The four implementation
> commits below are landed, but a latest-HEAD acceptance review found a small
> set of correctness, release-gate, I/O, build-matrix, and documentation
> issues. Treat the overhaul as pending follow-up rather than finally accepted.
> The bounded work and its verification criteria are recorded in
> [`post-overhaul-follow-up.md`](#post-overhaul-follow-up). Modern-GPU
> performance validation is explicitly postponed until suitable hardware is
> available; it is not part of the immediately actionable closure work.
>
> **FINAL ACCEPTANCE RESTORED (2026-08-17).** The malformed-container
> memory-safety class named by the latest re-review is closed: the v1
> NetCDF/HDF5 readers now prove the exact rank, datatype, and extent of every
> object before reading into a fixed or sized host buffer
> (`docs/reader-rank-hardening-handoff.md`), with malformed-rank fixtures
> passing under ASan/UBSan. Every available-hardware exit gate passes
> (verify 58/58, sanitizer 90/90, float/no-backend/NetCDF-only/HDF5-only
> 30/30 each, `ci_gpu.sh` green, trajectories with accepted decisions, legacy
> `.bin` states byte-identical to `dc0d0c4`). Modern-GPU performance
> validation remains separately POSTPONED and is the only outstanding item.
>
> **FIRST FOLLOW-UP IMPLEMENTED (2026-08-17).** Every originally actionable
> item of `post-overhaul-follow-up.md` (§§2–5) was implemented on top of the
> four commits: the oriented-Jacobian first-sample fix with a production-path
> regression, validated v1 restart offsets with corrupted fixtures, the
> refresh-pass terminal contract closed via device-side force-norm
> finalization, the repaired `scripts/ci_gpu.sh` oracle and precision-aware
> CLI fixtures, the complete optional-backend preset/CI matrix, target-scoped
> precision flags, the checked library-publication chain with fault-injection
> tests, and the docs/repository-hygiene reconciliation. Both frozen
> trajectories were re-verified **Class A byte-identical** against the frozen
> `dc0d0c4` baseline (`scripts/compare_bitwise.py`, full dump manifests). Only
> the hardware-dependent modern-GPU performance validation remains, and it
> stays explicitly POSTPONED — implementation commits landed and acceptance
> gates passed are distinct claims; the modern-GPU gate is neither. The later
> reader-rank finding above must be closed before final code acceptance.

> **EXECUTED (2026-08-17).** All four steps are landed as separately
> reviewable commits on the `overhaul` branch:
>
> - Step 1 — numerical safety and error boundaries: `48713b2` — profile
>   normalization validation before CUDA init, zero library `exit()` calls,
>   the typed trivially-copyable `ControlRecord` with validity bits, the
>   device reset→reduce→finalize Jacobian status, status-guarded no-ops, the
>   on-device terminal classification, and the manufactured
>   inverted/collapsed/nonfinite/converged cases (memcheck + initcheck clean).
> - Step 2 — configuration and I/O contracts: `4363e71` — strict schema-v1
>   default + named `--compatibility`, the single-snapshot host-only
>   NetCDF/HDF5 writers (v0 byte/layout-exact + v1 with complete provenance
>   and round-tripping readers), the durable publication protocol (unique
>   temp, fsync, atomic rename, directory fsync), and the build-dependency
>   isolation (parser C++20 host-only, backend headers/defines confined to
>   the adapter library).
> - Step 3 — runtime and performance policy: `d602d2c` — named precision
>   policies (verify-double/fast-double/mixed-float/debug-double; no global
>   `--use_fast_math`), the single-construction stage arena (one
>   allocation, one module construction per stage), the fence-delivered
>   telemetry (one deliberate control fence per pass, no per-print barriers
>   or allocations), the dump machinery behind a build option, and the
>   TITAN Xp re-measurement (docs/performance.md).
> - Step 4 — release gate: warnings-as-errors in the verification presets,
>   compute-sanitizer initcheck, the original CI workflow (hosted matrix plus
>   a self-hosted gate, subsequently removed by the availability correction
>   above), the event-DAG poison/delay tests, and the docs
>   brought to measured reality (architecture/performance/verification/
>   CLAUDE.md).
>
> Every step passed its exit gate with the frozen Solovev
> `251→199→456` and W7-X `1877→1617→2011` trajectories **Class A
> byte-identical** (compare_bitwise over the full dump manifests) — no
> re-freeze was needed anywhere, including the fast-math removal.

Status date: 2026-08-17. Reviewed branch: `overhaul` at
`b9d3429bb36bf56be0e5498c4cf01b7356514e1e`.

Phase 11 completed the structural migration and the two reference problems
retain their frozen numerical trajectories. The review at the time found
unfinished safety predicates, host/device I/O boundaries, precision/build
policy, and acceptance gates; the remaining work was split into the four
ordered steps below. Those steps AND the two follow-up reviews that came after
(the first post-overhaul follow-up and the reader-rank hardening) are now all
closed — see the status banners at the top of this document and
`docs/post-overhaul-follow-up.md` / `docs/reader-rank-hardening-handoff.md`.
The historical step descriptions below are retained as the plan record.

This plan is deliberately a closure plan. It does not add free-boundary physics,
scientific `wout` output, or speculative CUDA Graph integration.

### Step 1 - Close numerical safety and error boundaries

This step comes first because later performance and I/O measurements are not
meaningful while invalid inputs or invalid device states can continue through
the operator DAG.

#### Work

1. Extend `ValidatedProblem` validation to reject non-finite, zero, and
   unreasonably ill-scaled profile normalizations before any CUDA allocation:

   - toroidal-flux edge normalization
     \(T_\mathrm{edge}=T(1)\), used by
     \(\Phi_\max=\mathrm{signJ}\,\Phi_\mathrm{edge}/(2\pi T_\mathrm{edge})\);
   - prescribed-current edge integral
     \(C_\mathrm{edge}=J_C(\min(|\mathrm{bloat}|,1))\), independent of
     \(T(1)\), used by
     \(I_\mathrm{tor}=\mathrm{signJ}\,\mu_0 I_\mathrm{edge}/(2\pi C_\mathrm{edge})\).

2. Remove the remaining `exit()` from `Profiles`. Constructors and operators
   must return a typed error or throw `CumesError`; only `main` maps a final
   status to a process exit code.

3. Replace the raw 16-double controller buffer with a trivially-copyable typed
   `ControlRecord`. It must contain explicit validity/status bits for:

   - finalized oriented-Jacobian validity;
   - invariant residual nonfinite/converged/continue state;
   - whether preconditioned residuals and force norms were evaluated.

4. Implement the device predicates required by the blueprint:

   - `reset -> reduce -> finalize` the global Jacobian status;
   - make magnetic-field, constraint-cache, preconditioner-refresh, force, and
     every `1/sqrt(g)` consumer no-op when geometry is invalid;
   - classify the invariant residual on device before in-place preconditioning;
   - make preconditioning and its residual reduction no-op on nonfinite or
     converged passes and mark their fields not evaluated.

5. Add manufactured inverted, collapsed, nonfinite, and already-converged
   cases that assert both the reported status and the absence of downstream
   cache/state writes.

#### Exit gate

- Invalid profile normalizations fail during host validation, before CUDA
  context/stage construction.
- No library code calls `exit()`.
- Invalid/terminal passes perform no forbidden state or cache mutation.
- Solovev and W7-X preserve their accepted controller decisions and final-state
  equivalence class.
- The new cases pass ordinary CTest and Compute Sanitizer memcheck/initcheck.

### Step 2 - Finish configuration and I/O contracts

#### Work

1. Make the CLI's input policy explicit:

   - strict schema is the default and rejects unknown keys;
   - a named compatibility option enables VMEC-style warn-and-ignore behavior;
   - output defaults/fallbacks are available only through the named
     compatibility policy.

2. Complete the single-snapshot output path. Construct one
   `EquilibriumSnapshot`/`RunReport` operation and make every backend consume
   host memory only. NetCDF and HDF5 must not accept `SpectralStorage`, include
   CUDA runtime headers, or perform private D2H copies.

3. Preserve current file compatibility explicitly:

   - keep byte/layout-exact binary, NetCDF, and HDF5 legacy-v0 adapters;
   - add schema-v1 NetCDF/HDF5 writers with active dimensions, source hash,
     raw boundary harmonics, build/runtime provenance, per-stage outcomes,
     restart history, total iterations, and terminal status;
   - keep checkpoint/restart separate from scientific result output.

4. Use one durable publication protocol for every backend: unique
   same-directory temporary file, checked write/flush/close, file `fsync`,
   atomic rename, and directory `fsync` where supported. Failures must preserve
   the previous target and return failure to `main`.

5. Isolate build dependencies completely. Apply the JSON parser's C++20
   requirement to its host target only; do not set the CUDA language mode
   because of the parser. Keep NetCDF/HDF5 headers and availability definitions
   confined to their adapter targets.

#### Exit gate

- Binary/NetCDF/HDF5 all use the same host snapshot and `Writer` interface.
- Legacy-v0 golden files retain their exact documented layouts.
- Schema-v1 round trips retain the complete `RunReport` and restart metadata.
- The none/NetCDF-only/HDF5-only/both backend build matrix passes, including
  unwritable paths, interrupted publication, unknown suffixes, and float-to-disk
  conversion.
- Strict and compatibility CLI behavior is covered end to end.

### Step 3 - Finish runtime and performance policy

#### Work

1. Wire the declared precision policies into target-scoped build presets:

   - `verify-double`: precise double math;
   - `fast-double`: selected, attributable fast intrinsics;
   - `mixed-float`: float state/FFT with the documented double reductions;
   - `debug-double`: precise math plus device checks.

   Remove global `--use_fast_math`; the executable and benchmark provenance
   must report the policy and effective flags that produced the binary.

2. Replace side-effectful arena measurement. Stage sizing must not allocate a
   temporary device arena or construct/upload profiles, FFT plans, geometry,
   preconditioner, and constraints twice. Use authoritative module requirement
   descriptors or a pointer-free allocation-planning pass, followed by one real
   stage allocation and construction.

3. Remove sampled production-path barriers and allocations. Add the displayed
   axis/boundary values to the existing control/telemetry delivery or copy them
   asynchronously into persistent pinned storage at an already-required fence.
   Compile dump machinery behind an actual build option and keep lazy diagnostic
   materialization versioned.

4. Re-measure the retained axisymmetric and generic backends. CUDA Graph work
   remains deferred unless the real-pass benchmark demonstrates a benefit after
   the synchronization and setup cleanup.

#### Exit gate

- A normal pass has one deliberate control transfer/fence and no allocation;
  sampled console output does not add a device-wide or stream-wide fence.
- One stage performs one arena allocation and one construction of each module.
- `verify-double` reports precise math; fast math appears only in opt-in builds.
- Repeated thermally stable runs report setup/output separately from iteration
  time, median, p95, 95% confidence interval, clocks, and noise floor on both a
  Pascal GPU and a modern GPU. The other primary workload's upper regression
  bound remains at or below 2% unless separately justified.
  **Status: the modern-GPU half is POSTPONED** (no second GPU available); the
  TITAN Xp numbers stay the measured baseline and no cross-architecture claim
  is made (see `performance.md` §4).

### Step 4 - Establish the release gate and close the documents

#### Work

1. Make warnings-as-errors part of the verification presets and fix the current
   missing-field, optimizer-option redefinition, and host array-bounds warnings.
2. Add Compute Sanitizer `initcheck`; retain memcheck/racecheck/synccheck without
   treating racecheck as proof of inter-kernel ordering.
3. Add CI for host ASan/UBSan, precise double, float, optional-backend matrices,
   small CUDA sanitizer tests, and frozen short trajectories.
4. Add explicit event-DAG poison/delay tests and an Nsight Systems/API-trace
   audit before any multi-stream or CUDA Graph production variant is enabled.
5. Update `architecture.md`, `performance.md`, and `verification.md` from
   aspirational statements to measured current behavior. Record every accepted
   arithmetic/scheduling change using the existing Class A/B/C rules.

#### Exit gate

- The warning-clean CI and sanitizer matrix pass from a clean checkout.
- Full Solovev and W7-X trajectory/state gates pass with recorded build, GPU,
  driver, toolkit, input hash, and output schema provenance.
- Documentation contains no known contradiction with source behavior.
- Only after Steps 1–4 pass should the branch be described as satisfying the
  original overhaul definition of done.

### Ordering and ownership

```text
Step 1: numerical safety
    -> Step 2: config and I/O contracts
        -> Step 3: runtime/performance closure
            -> Step 4: release acceptance
```

Steps should land as separately reviewable commits. Within a step, tests should
land with the behavior they guard. Do not re-freeze a trajectory merely because
it converges: classify the change and apply the corresponding numerical gate.

---

<a id="post-overhaul-follow-up"></a>

## 2026-08-17 23:07:42+08:00 — Post-overhaul follow-up handoff

**Former path:** `docs/post-overhaul-follow-up.md`
**First tracked:** [`3a1f7b0`](https://github.com/12ff54e/cuMES/commit/3a1f7b0e14e2aeb3de966a3ec929a1b2f0e9bde6) at 2026-08-17T23:07:42+08:00

> **FIRST FOLLOW-UP IMPLEMENTED; BOTH RE-REVIEWS CLOSED (2026-08-17).** The
> actionable items below (§§2–5) were implemented and their available-hardware
> gates pass. A later adversarial review found one remaining malformed-file
> memory-safety class in the v1 NetCDF/HDF5 readers; that bounded handoff
> ([`reader-rank-hardening-handoff.md`](#reader-rank-hardening-handoff)) is
> now closed too, restoring final code acceptance. The frozen trajectories
> remain Class A byte-identical to the `dc0d0c4` baseline, and the legacy
> `.bin` final states are byte-identical. Hardware-dependent §6 remains
> separately postponed. The historical sections below describe the state at
> the time of the first review and must not be read as current acceptance
> status.

Status date: 2026-08-17. Reviewed branch: `overhaul` at
`dc0d0c46f9bd00840cc80389092ae0a957d3700e` (four commits ahead of
`origin/overhaul` at review time).

### 1. Acceptance status (archived review record)

> This section is the HISTORICAL state at the time of the first review. It
> described the branch as not yet satisfying the completion plan's definition
> of done; that assessment has since been superseded — the items below were
> implemented, and the only remaining reopen (`reader-rank-hardening-
> handoff.md`) was closed afterwards. See the banner above for current status.

The structural CUDA overhaul was substantially implemented at the reviewed
commit and the two frozen reference trajectories passed, but the branch did
not yet satisfy the completion plan's full definition of done at that time.
This document was the bounded handoff for the remaining work.

Modern-architecture performance validation is **postponed** because no second
GPU is currently available. Do not represent that gate as passed or failed.
Keep the existing TITAN Xp measurements, record the hardware limitation, and
avoid cross-architecture performance claims until the deferred work in section
6 can be run.

Historical review evidence at the reviewed commit (superseded counts; the
current suites at the closure of the first follow-up, `3a1f7b0`, were verify
57/57, mixed-float 29/29, sanitizer 88/88, and no/one/both-backend 29/29
each):

- precise-double verify suite: 55/55 passed;
- sanitizer suite: 85/85 passed, including memcheck, initcheck, racecheck,
  synccheck, ASan, and UBSan;
- no-optional-backend suite: 27/27 passed;
- mixed-float suite: 26/27 passed; only `cli_policy` failed (later fixed by
  the precision-aware fixtures of the first follow-up);
- Solovev trajectory: `251 -> 199 -> 456`, final FSQR `9.583e-17`;
- W7-X trajectory: `1877 -> 1617 -> 2011`, final FSQR `9.778e-13`.

Do not re-freeze either trajectory merely because a modified build converges.
Apply the Class A/B/C rules in `verification.md` and compare controller
decisions plus all six state families.

### 2. Step 1 - Correct safety and corrupted-input boundaries

#### 2.1 Fix the oriented-Jacobian reduction

`src/geometry_impl.cuh` initializes the first finite sample seen by a reduction
lane with:

```cpp
vmin = vmax = a;
```

where `a = abs(g)` and `ov = signJ * g`. A sign-flipped first sample therefore
enters the minimum as positive and can be hidden if later samples assigned to
that lane are valid. Initialize the minimum from `ov` while retaining `a` for
the scale statistic:

```cpp
vmin = ov;
vmax = a;
```

Add a production-path CUDA test that runs `jacobianStatsKernel` followed by
`jacobianFinalizeKernel` on a buffer with exactly one sign reversal in a
lane's first sample. Assert the invalid status, oriented minimum, index, and
that every guarded downstream consumer leaves its output/cache sentinel
unchanged. Run it in ordinary CTest, memcheck, and initcheck. Testing only the
finalizer with a manually populated `ControlRecord` is insufficient.

#### 2.2 Validate v1 restart offsets before indexing

The NetCDF and HDF5 readers cast serialized `restart_stage_offset` values to
`size_t` and use them to index `rst_iter` without proving that the offsets are
nonnegative, monotonic, or bounded by `nrestarts`. Validate all dimensions and
offsets before constructing any `StageReport`:

- first offset is zero when stages exist;
- every offset is nonnegative and monotonic;
- every offset is at most `nrestarts`;
- stage-array dimensions agree with `nstages`;
- restart-array dimensions agree with `nrestarts`.

Malformed files must return a typed failure without an out-of-bounds read.
Add corrupted NetCDF/HDF5 fixtures for negative, descending, and oversized
offsets and run the host readers under ASan/UBSan.

#### 2.3 Close or explicitly revise the refresh-pass terminal contract

On a preconditioner-refresh pass, `invariantPredicateKernel` is invoked with
`classify_converged=0` because the new force normalization is finalized only
at the host fence. Consequently, a pass later classified as converged by the
host still performs in-place preconditioning and its residual reduction.

Prefer making the required normalization available before the device terminal
predicate so every converged/nonfinite pass no-ops preconditioning. If that
would add a control fence or materially change arithmetic, document and test
the narrower accepted invariant: no persistent state or cache mutation, with
deterministic scratch telemetry. In either case, make the implementation and
the completion-plan claim identical.

#### Step 1 exit gate

- The manufactured first-sample sign reversal is detected on device.
- Corrupt restart metadata fails cleanly under ASan/UBSan.
- Refresh-pass terminal behavior has an explicit tested contract.
- Frozen Solovev and W7-X controller decisions and final states remain in the
  accepted equivalence class.

### 3. Step 2 - Repair release and build matrices

#### 3.1 Fix the self-hosted GPU release script

`scripts/ci_gpu.sh` currently caps a Solovev run with `CUMES_MAX_ITER=20` and
expects both `completed 20/20 iterations` and `Done.`. The real executable
returns stage failure and reports `completed 21/1000 iterations`; it exits
before printing `Done.`. Thus the declared GPU release job cannot pass.

Replace the contradictory text checks with a stable, intentional oracle.
Prefer structured telemetry over parsing human console output. At minimum,
assert the documented stage-cap exit code, effective-iteration semantics,
finite positive residuals, and absence of output artifacts when the terminal
status policy forbids them. Run the complete `scripts/ci_gpu.sh` locally on
the TITAN Xp before accepting the fix.

#### 3.2 Make the CLI policy test precision-aware

`tests/cli_policy_test.sh` hard-codes `ftol_array: [1e-14]`. The mixed-float
policy correctly rejects this below its documented `1e-6` floor, so the test
never reaches the compatibility behavior it intends to exercise. Use a
tolerance reachable in every tested precision, or generate a policy-specific
fixture. The full float CTest suite must pass, not merely the CUDA component
subset.

#### 3.3 Implement the optional-backend matrix actually claimed by the docs

Add configure/build presets or equivalent CI entries for:

- both NetCDF and HDF5 enabled;
- NetCDF only;
- HDF5 only;
- neither backend.

Run the host I/O golden, failure, schema-v0/v1 round-trip, CLI suffix, and
float-to-disk tests in each applicable configuration. At review time only the
both-enabled and neither-enabled configurations existed.

#### 3.4 Make precision/device-check flags target-scoped

`CMakeLists.txt` still appends optimization, fast-math, and device-check flags
through global `CMAKE_CUDA_FLAGS` (and host optimization through global
`CMAKE_CXX_FLAGS`). Move policy flags to named project targets with
`target_compile_options` and language/configuration generator expressions.
Do not leak production fast math or `-G` into unrelated tests/adapters. Remove
the sanitizer configuration's optimization-option redefinition warning.
Preserve the existing isolation in which JSON parsing is host-only C++20 and
CUDA operator translation units remain C++17.

#### Step 2 exit gate

- `scripts/ci_gpu.sh` passes end to end on the available GPU.
- precise-double, mixed-float, sanitizer, and all four backend configurations
  configure, build, and pass their applicable tests.
- warnings-as-errors is clean without nvcc option-redefinition warnings.
- compile-command inspection confirms parser/backend dependencies and policy
  flags are confined to their intended targets.

### 4. Step 3 - Finish the durable I/O contract

Binary publication uses the checked `publishAtomic` path, but NetCDF/HDF5
currently call `nc_close`/`H5Fclose` and then `renamePublish`. Library close
flushes library buffers but is not the documented checked OS-level file
`fsync`. In addition, `fsyncDirectoryOf` ignores `fsync` and `close` errors.

Create a common publication helper suitable for library-owned file handles:

1. finish and check every NetCDF/HDF5 write and library flush/close;
2. open the completed same-directory temporary file and check `fsync`;
3. close that descriptor and check the result;
4. atomically rename it over the destination;
5. open and `fsync` the containing directory, propagating failure where the
   platform promises this capability;
6. preserve the old destination and clean the temporary file on every
   pre-rename failure.

Add fault-injection tests around write, flush, file-fsync, close, rename, and
directory-fsync boundaries. Confirm that `main` returns failure and an
existing destination remains valid whenever publication fails.

#### Step 3 exit gate

All binary, NetCDF, and HDF5 backends satisfy the same documented atomic and
durable publication protocol, with checked failures and preservation tests.

### 5. Step 4 - Reconcile tests, docs, and repository hygiene

After Steps 1-3:

- update `verification.md` from its stale 53-test count or, preferably, avoid
  embedding a count that changes whenever a sanitizer variant is added;
- remove claims that NetCDF-only/HDF5-only CI passes until those jobs exist and
  are green;
- remove the stale statement that no CI/event-DAG tests exist;
- describe the GPU release script's actual structured acceptance contract;
- amend `performance.md` and the completion plan so the modern-GPU gate is
  marked postponed, not completed;
- remove the tracked `test_host_config_scratch_*.json` files;
- change `test_host_config` to use a per-test temporary directory with RAII
  cleanup so interrupted or parallel tests do not leave repository-root
  debris;
- ensure every completion banner distinguishes "implementation commits
  landed" from "all acceptance gates passed".

#### Step 4 exit gate

- A clean checkout has no generated fixtures before or after the test matrix.
- Documentation contains no known contradiction with source behavior or CI.
- Full precise Solovev and W7-X trajectories pass with recorded provenance.
- The branch may then be described as code-overhaul complete, with only the
  explicitly postponed hardware-dependent validation remaining.

### 6. Postponed - Modern-GPU performance validation

This work requires a second, modern CUDA GPU and is intentionally not assigned
to the immediate closure agent.

When hardware becomes available:

1. Run the fixed-iteration Solovev and W7-X harness using the same commit,
   precision policy, toolkit provenance, warm-up, and measurement method as
   the TITAN Xp baseline.
2. Record GPU model, compute capability, driver/toolkit, clocks, power/thermal
   state, arena/cuFFT/graph memory, setup/output time, median, p95, measured
   noise floor, and a 95% confidence interval over thermally stable repeats.
3. Check the performance policy: a claimed improvement must exceed
   `max(5%, noise floor)` at the lower confidence bound and must not regress
   the other primary workload by more than 2% at its upper confidence bound,
   unless a separately reviewed correctness/memory justification applies.
4. Re-run the complete numerical trajectory/state gate; architecture-specific
   performance must never substitute for correctness equivalence.
5. Update `performance.md` with measured results and only then make a
   cross-architecture acceptance claim.

Until then, retain the TITAN Xp results as the measured baseline and label all
modern-GPU conclusions as unknown/deferred.

---

<a id="reader-rank-hardening-handoff"></a>

## 2026-08-17 23:46:16+08:00 — V1 container reader rank-hardening handoff

**Former path:** `docs/reader-rank-hardening-handoff.md`
**First tracked:** [`611e8d7`](https://github.com/12ff54e/cuMES/commit/611e8d7929431ab4579249362ba5bef1febaf096) at 2026-08-17T23:46:16+08:00

> **Follow-up closed (2026-08-18).** This exact-rank task remains closed, and
> commit `de265cd` also closes the subsequent resource/schema audit in
> [`v1-reader-resource-hardening-handoff.md`](#v1-reader-resource-hardening-handoff).
> Final acceptance is restored; modern-GPU validation remains separately
> POSTPONED.

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

### 1. Verdict and scope

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

### 2. Finding - NetCDF v1 reader trusts variable shape

File: `src/cumes/io/netcdf_writer.cpp`, `NetcdfV1Reader::read`.

#### Unsafe sites

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

#### Required NetCDF helper contracts

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

### 3. Finding - HDF5 v1 reader trusts dataspace and attribute rank

File: `src/cumes/io/hdf5_writer.cpp`, `Hdf5V1Reader::read` and
`getStrAttr`.

#### Unsafe sites

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

#### Required HDF5 helper contracts

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

### 4. Checked dimensions and allocation limits

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

### 5. Required regression tests

Extend `tests/test_io_restart_offsets.cpp` or add a focused host-only
`test_io_malformed_shapes.cpp`. Each backend must include:

#### NetCDF fixtures

- a state family with rank 1 and rank 3;
- a state family with swapped or mismatched rank-2 dimensions;
- a scalar outcome variable encoded as a rank-1 array with multiple values;
- a stage/restart variable encoded as rank 0 and rank 2;
- correct rank but wrong datatype;
- `ns`/`mnmax` beyond the representable or documented resource bound.

#### HDF5 fixtures

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

### 6. Documentation reconciliation

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

### 7. Exit gates

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

### 8. Review evidence before this handoff

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

---

<a id="v1-reader-resource-hardening-handoff"></a>

## 2026-08-18 00:05:59+08:00 — V1 reader resource-hardening handoff

**Former path:** `docs/v1-reader-resource-hardening-handoff.md`
**First tracked:** [`ac7f94e`](https://github.com/12ff54e/cuMES/commit/ac7f94ea2fb75388bf70f6dea5ad9fe1b7e17a60) at 2026-08-18T00:05:59+08:00

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

### 1. Verified baseline

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

#### Closure results

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

### 2. P1 — reject HDF5 variable-length provenance strings

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

### 3. P1 — impose a practical state-allocation cap

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

### 4. P2 — finish the exact HDF5 datatype contract

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

### 5. P2 — validate all closed-range serialized values

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

### 6. Documentation reconciliation

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

### 7. Exit gates

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
