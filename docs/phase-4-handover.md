# cuMES Phase 4 handover — pure controller and observers

Status date: 2026-08-15. Branch: `overhaul` (Phase 0 `bd26857` + Phase 1
`12bcc44` + Phase 2 `168170a` + Phase 3 `c21564c` + this Phase 4 work). This
document records what Phase 4 of `docs/cuda-overhaul-blueprint.md` delivered,
how it was verified, and what was deliberately deferred and why.

## 1. Scope

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

## 2. What changed

### New `cumes/solver` headers (pure host, templated on T)

| File | Contents |
| ---- | -------- |
| `control_record.hpp` | `RestartReason` (0/1/2, part of the frozen telemetry contract), `JacobianStatus<T>`, `InvariantVerdict<T>`, `Damping<T>`, `RestartDecision<T>`, and the aggregate `ControlRecord<T>` (the single-fence target; see §5). |
| `iteration_controller.hpp` | `IterationController<T>` + `kPreconInterval = 25`: owns iter2/iter1/log-anchor, the ten-sample 1/tau history, the running-minimum `res0`, `ijacob`, `delt`, and `ftol`. Methods `next_schedule()` (ijacob 25/50 maintenance), `jacobian_invalid()`, `classify_invariant()`, `decide_restart()`, `after_descent()`, plus pass-invariant accessors (`reset_constraint_reference`, `refresh_preconditioner`, `fsqz_prev`, …). |
| `pass_record.hpp` | `PassRecord`: the typed 15-column per-pass telemetry record (standard-layout 15 doubles, `static_assert`ed) whose field order is the frozen on-disk contract. |
| `stage_solver.hpp` | `StageSolver<T>::run`: one stage's `profilesCreate`/`fourierCreate`/`metricCreate` + `solverRun` + free. State stays owned by the caller. |
| `multigrid_solver.hpp` | `MultigridSolver<T>::run` + `MultigridOutcome<T>`: owns the state across stages, prolongs via `interpolateState`, validates the schedule, emits a `RunReport` with per-stage history, and reports a failing stage without calling `exit()`. |

### Migration

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

### Already-complete Phase-4 deliverables (re-verified, not re-done)

- **Delete dump-driven solver branches.** The only dump-mode-dependent control
  branch (the `iter2 == kDumpIter` extra preconditioner refresh) was removed in
  Phase 0 (`4ea7746`). A grep confirms every remaining `dumpEnabled()`/`kDumpIter`
  reference is observability only — none feeds a decision.
- **Unproduced combined-force buffers.** Removed in Phase 3; `grep` confirms no
  combined-force buffer/dump remains.

## 3. Key design decisions

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

## 4. Verification

### Class A bitwise gate (the critical gate)

`scripts/compare_bitwise.py` against the Phase-3 baseline (`.verify-scratch/
baseline-phase3`, tag `overhaul/phase-3`) after each of the four commits:

| Config | state | trajectory | step_0 | dump manifest |
| ------ | ----- | ---------- | ------ | ------------- |
| double/solovev | OK | OK | OK (10) | OK (235 files) |
| double/w7x    | OK | OK | OK (10) | OK (526 files) |

Both `PASS: byte-identical` — the controller-driven solver replays the frozen
trajectory exactly (same restarts, damping, effective-iteration counts, and
final state).

### Test matrix

| Preset | Result |
| ------ | ------ |
| `verify` (double, both backends) | **20/20** (14 unit + 6 compute-sanitizer memcheck) |
| `float` | **14/14** |

### Controller unit test (`test_controller`)

Drives scripted residual sequences through every branch and asserts the exact
historical semantics: convergence, nonfinite recovery (delt ×0.9 + re-anchor),
the anchor-pass 1/tau history (otav/`dtau`/`b1`/`fac` values), the bad-Jacobian
restart (delt ×0.9, `ijacob`++), the bad-progress restart (delt ÷1.03), the
`ijacob == 25` maintenance reset (delt = 0.98·delt0), the refresh predicate
(age > 10), and effective-iteration / restart-anchor bookkeeping.

### Observer isolation (the second exit-gate clause)

A `CUMES_DUMP=0` Solovev run writes a `cumes_state.bin` **byte-identical** to the
`CUMES_DUMP=1` run — enabling the dump observer changes neither the state hash
nor the iteration count. (This was established in Phase 0; Phase 4 re-verified
it and made the telemetry a typed record so the property is structural.)

## 5. Deferred (documented, not hidden)

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

## 6. Next steps (Phase 5)

Phase 5 is **operator/workspace boundaries**: transform-only `SpectralOperator`,
profiles/geometry/B/force/constraint/residual/preconditioner/descent interfaces,
a stage arena with reported liveness/peak memory, and scalar CPU references at
every boundary — plus wiring `main.cu` to `read_and_validate` + versioned writers
+ the checkpoint reader.
