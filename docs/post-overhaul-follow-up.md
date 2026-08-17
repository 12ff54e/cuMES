# Post-overhaul follow-up handoff

> **FIRST FOLLOW-UP IMPLEMENTED; BOTH RE-REVIEWS CLOSED (2026-08-17).** The
> actionable items below (§§2–5) were implemented and their available-hardware
> gates pass. A later adversarial review found one remaining malformed-file
> memory-safety class in the v1 NetCDF/HDF5 readers; that bounded handoff
> ([`reader-rank-hardening-handoff.md`](reader-rank-hardening-handoff.md)) is
> now closed too, restoring final code acceptance. The frozen trajectories
> remain Class A byte-identical to the `dc0d0c4` baseline, and the legacy
> `.bin` final states are byte-identical. Hardware-dependent §6 remains
> separately postponed. The historical sections below describe the state at
> the time of the first review and must not be read as current acceptance
> status.

Status date: 2026-08-17. Reviewed branch: `overhaul` at
`dc0d0c46f9bd00840cc80389092ae0a957d3700e` (four commits ahead of
`origin/overhaul` at review time).

## 1. Acceptance status (archived review record)

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

## 2. Step 1 - Correct safety and corrupted-input boundaries

### 2.1 Fix the oriented-Jacobian reduction

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

### 2.2 Validate v1 restart offsets before indexing

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

### 2.3 Close or explicitly revise the refresh-pass terminal contract

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

### Step 1 exit gate

- The manufactured first-sample sign reversal is detected on device.
- Corrupt restart metadata fails cleanly under ASan/UBSan.
- Refresh-pass terminal behavior has an explicit tested contract.
- Frozen Solovev and W7-X controller decisions and final states remain in the
  accepted equivalence class.

## 3. Step 2 - Repair release and build matrices

### 3.1 Fix the self-hosted GPU release script

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

### 3.2 Make the CLI policy test precision-aware

`tests/cli_policy_test.sh` hard-codes `ftol_array: [1e-14]`. The mixed-float
policy correctly rejects this below its documented `1e-6` floor, so the test
never reaches the compatibility behavior it intends to exercise. Use a
tolerance reachable in every tested precision, or generate a policy-specific
fixture. The full float CTest suite must pass, not merely the CUDA component
subset.

### 3.3 Implement the optional-backend matrix actually claimed by the docs

Add configure/build presets or equivalent CI entries for:

- both NetCDF and HDF5 enabled;
- NetCDF only;
- HDF5 only;
- neither backend.

Run the host I/O golden, failure, schema-v0/v1 round-trip, CLI suffix, and
float-to-disk tests in each applicable configuration. At review time only the
both-enabled and neither-enabled configurations existed.

### 3.4 Make precision/device-check flags target-scoped

`CMakeLists.txt` still appends optimization, fast-math, and device-check flags
through global `CMAKE_CUDA_FLAGS` (and host optimization through global
`CMAKE_CXX_FLAGS`). Move policy flags to named project targets with
`target_compile_options` and language/configuration generator expressions.
Do not leak production fast math or `-G` into unrelated tests/adapters. Remove
the sanitizer configuration's optimization-option redefinition warning.
Preserve the existing isolation in which JSON parsing is host-only C++20 and
CUDA operator translation units remain C++17.

### Step 2 exit gate

- `scripts/ci_gpu.sh` passes end to end on the available GPU.
- precise-double, mixed-float, sanitizer, and all four backend configurations
  configure, build, and pass their applicable tests.
- warnings-as-errors is clean without nvcc option-redefinition warnings.
- compile-command inspection confirms parser/backend dependencies and policy
  flags are confined to their intended targets.

## 4. Step 3 - Finish the durable I/O contract

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

### Step 3 exit gate

All binary, NetCDF, and HDF5 backends satisfy the same documented atomic and
durable publication protocol, with checked failures and preservation tests.

## 5. Step 4 - Reconcile tests, docs, and repository hygiene

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

### Step 4 exit gate

- A clean checkout has no generated fixtures before or after the test matrix.
- Documentation contains no known contradiction with source behavior or CI.
- Full precise Solovev and W7-X trajectories pass with recorded provenance.
- The branch may then be described as code-overhaul complete, with only the
  explicitly postponed hardware-dependent validation remaining.

## 6. Postponed - Modern-GPU performance validation

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
