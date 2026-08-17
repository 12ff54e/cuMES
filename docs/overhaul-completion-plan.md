# cuMES overhaul completion plan

Status date: 2026-08-17. Reviewed branch: `overhaul` at
`b9d3429bb36bf56be0e5498c4cf01b7356514e1e`.

Phase 11 completed the structural migration and the two reference problems
retain their frozen numerical trajectories. The overhaul is nevertheless not
design-complete: the review found unfinished safety predicates, host/device I/O
boundaries, precision/build policy, and acceptance gates. The remaining work is
split into four ordered steps below.

This plan is deliberately a closure plan. It does not add free-boundary physics,
scientific `wout` output, or speculative CUDA Graph integration.

## Step 1 - Close numerical safety and error boundaries

This step comes first because later performance and I/O measurements are not
meaningful while invalid inputs or invalid device states can continue through
the operator DAG.

### Work

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

### Exit gate

- Invalid profile normalizations fail during host validation, before CUDA
  context/stage construction.
- No library code calls `exit()`.
- Invalid/terminal passes perform no forbidden state or cache mutation.
- Solovev and W7-X preserve their accepted controller decisions and final-state
  equivalence class.
- The new cases pass ordinary CTest and Compute Sanitizer memcheck/initcheck.

## Step 2 - Finish configuration and I/O contracts

### Work

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

### Exit gate

- Binary/NetCDF/HDF5 all use the same host snapshot and `Writer` interface.
- Legacy-v0 golden files retain their exact documented layouts.
- Schema-v1 round trips retain the complete `RunReport` and restart metadata.
- The none/NetCDF-only/HDF5-only/both backend build matrix passes, including
  unwritable paths, interrupted publication, unknown suffixes, and float-to-disk
  conversion.
- Strict and compatibility CLI behavior is covered end to end.

## Step 3 - Finish runtime and performance policy

### Work

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

### Exit gate

- A normal pass has one deliberate control transfer/fence and no allocation;
  sampled console output does not add a device-wide or stream-wide fence.
- One stage performs one arena allocation and one construction of each module.
- `verify-double` reports precise math; fast math appears only in opt-in builds.
- Repeated thermally stable runs report setup/output separately from iteration
  time, median, p95, 95% confidence interval, clocks, and noise floor on both a
  Pascal GPU and a modern GPU. The other primary workload's upper regression
  bound remains at or below 2% unless separately justified.

## Step 4 - Establish the release gate and close the documents

### Work

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

### Exit gate

- The warning-clean CI and sanitizer matrix pass from a clean checkout.
- Full Solovev and W7-X trajectory/state gates pass with recorded build, GPU,
  driver, toolkit, input hash, and output schema provenance.
- Documentation contains no known contradiction with source behavior.
- Only after Steps 1–4 pass should the branch be described as satisfying the
  original overhaul definition of done.

## Ordering and ownership

```text
Step 1: numerical safety
    -> Step 2: config and I/O contracts
        -> Step 3: runtime/performance closure
            -> Step 4: release acceptance
```

Steps should land as separately reviewable commits. Within a step, tests should
land with the behavior they guard. Do not re-freeze a trajectory merely because
it converges: classify the change and apply the corresponding numerical gate.
