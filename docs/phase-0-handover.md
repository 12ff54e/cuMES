# cuMES Phase 0 handover — overhaul/phase-0 branch

Status date: 2026-08-13. Branch: `overhaul/phase-0`, based on `631a16a`
(the blueprint's named baseline). This document records what Phase 0 work was
completed, how it was verified, and what is intentionally deferred and why.

## 1. Scope agreed

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

## 2. Commits (631a16a..HEAD, 14 total)

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

## 3. How the verification anchor works

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

## 4. Trajectory verification (what was checked)

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

## 5. Test suite (14 registered)

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

## 6. Intentionally deferred (and why)

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

## 7. Next steps (Phase 1)

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

## 8. Reproducibility notes

- Float W7-X prescribed-current does **not** converge at ftol ≤ 1e-3 (float
  floor ~3e-3); the capture recipe uses per-config float ftol (solovev=1e-6,
  w7x=1e-2).
- `DUMP_CUMES_VERIFY` is hardcoded at `src/solver.cu:9`; the dump machinery is
  compiled in but runtime-gated by `CUMES_DUMP=1`.
- The forensic `cumes_state.bin` stays double on disk regardless of T (the
  binary writer converts T→double).
