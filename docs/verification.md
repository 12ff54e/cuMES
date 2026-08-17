# cuMES verification strategy

The verification tiers and gates of `docs/cuda-overhaul-blueprint.md` §10,
extracted verbatim (sections 1–7 correspond to §10.1–§10.7), plus the current
inventory of what is actually wired into the build, and the change-review
checklist (§13 of the blueprint) as the appendix.

## Current state (2026-08-17, post-overhaul follow-up closed)

The overhaul completion plan (`docs/overhaul-completion-plan.md`, steps 1–4)
is landed in four commits (see `docs/phase-11-closeout-handover.md`):
numerical safety predicates (48713b2), config/I-O contracts (4363e71),
runtime/performance policy (d602d2c), and this release gate. The
post-implementation acceptance review then found a bounded set of loose ends
(`docs/post-overhaul-follow-up.md`); those are closed on top of the four
commits:

- the oriented-Jacobian reduction seeds the lane minimum from the ORIENTED
  value (`vmin = signJ·√g`, not `|√g|`), with a production-path
  first-sample sign-reversal regression (`test_safety_predicates`);
- the v1 NetCDF/HDF5 readers validate restart offsets and array extents
  before indexing (`validateRestartOffsets` in `run_report.hpp`; corrupted
  negative/descending/oversized/extent-mismatch fixtures in
  `test_io_restart_offsets`, host-only so the ASan twin runs the same
  source);
- the refresh-pass terminal contract is CLOSED rather than narrowed: the
  force-norm factors are finalized ON DEVICE before the terminal predicate
  (`forceNormFinalizeKernel`), so every converged/nonfinite pass — refresh
  or not — no-ops preconditioning; the host consumes the same record
  fields, keeping device and host classification bit-identical;
- `scripts/ci_gpu.sh` asserts the documented stage-cap contract (exit 1,
  one-based effective-iteration count `completed 21/1000` for 20 capped
  passes, finite positive residuals, no output artifacts);
- `tests/cli_policy_test.sh` fixtures use `ftol 1e-6` (exactly the float
  floor), so the CLI policy gate passes in every precision;
- the optional-backend matrix is complete: `netcdf-only` and `hdf5-only`
  presets join `nobackend` (and the both-backend verify default), each with
  CI matrix entries;
- precision/device-check flags are target-scoped (no global
  `CMAKE_CUDA_FLAGS`/`CMAKE_CXX_FLAGS` appends; the sanitizer
  configuration's optimization redefinition warning is gone). The CUDA TUs'
  host pass deliberately receives NO `-march=native`: the frozen Class A
  baseline never had it there, and adding it enables FMA contraction that
  diverges the trajectory bitwise (measured);
- NetCDF/HDF5 publication now runs the full checked chain
  (`publishLibraryFile`: reopen temp → checked fsync → checked close →
  atomic rename → checked directory fsync), with fault-injection tests
  around the reopen/fsync/write-flush/rename/directory-fsync boundaries
  plus an end-to-end `output_publication` gate on the real binary;
- `test_host_config` writes its scratch into a per-test temp directory
  (RAII), and the tracked `test_host_config_scratch_*.json` debris is
  removed.
- the v1 NetCDF/HDF5 readers prove the EXACT rank, datatype, and extent of
  every object before reading into a fixed or sized host buffer
  (`docs/reader-rank-hardening-handoff.md`): checked scalar/vector/family
  helpers, exact-rank dataspace probes before `H5Sget_simple_extent_dims`,
  scalar-or-one-element attribute checks with bounded string widths, checked
  dimension narrowing/byte counts with documented resource caps, and typed
  allocation-failure conversion. `test_io_malformed_shapes` (host-only, ASan
  twin) exercises rank 1/3 families, swapped extents, scalar-as-array
  outcomes, rank 0/2 stage datasets, multi-element attributes, wrong
  datatypes, and beyond-INT_MAX dimensions on both backends.

The modern-GPU performance gate remains POSTPONED (no second GPU is
available): the TITAN Xp numbers stay the measured baseline and no
cross-architecture claim is made (see `docs/performance.md`).

- **Verify gate (default `build/`, `verify-double` precise math,
  warnings-as-errors)**: the full CTest suite — unit tests incl. the
  manufactured safety-predicate and event-DAG suites, compute-sanitizer
  memcheck AND initcheck variants of the kernel-driving tests, the CLI
  strict-vs-compatibility policy gate, the end-to-end publication gate,
  and the benchmark smoke. (No test-count is embedded here: it changes
  whenever a sanitizer variant or backend fixture is added.) The frozen
  trajectory oracle: Solovev `251→199→456` FSQR 9.583e-17 and W7-X
  `1877→1617→2011` FSQR 9.778e-13 must reproduce **bit-identically**
  (`scripts/compare_bitwise.py` over the full dump manifests). The
  follow-up closure above was re-verified Class A byte-identical against
  the frozen `dc0d0c4` baseline (both configs: trajectory record, final
  state, step-0 snapshots, and full dump manifests). Note: the blueprint's
  pre-overhaul `FSQZ=4.273e-18` is stale — the axisymmetric backend's
  algorithm update moved the frozen value to `4.274e-18`, which is what
  the baseline itself records.
- **Sanitizer preset (`build-sanitize/`)**: the verify gate plus
  racecheck/synccheck variants of the kernel tests (RUN_SERIAL; racecheck
  exhausts the GPU under parallel runs) and ASan+UBSan twins of the
  host-only libraries/tests (`asan_test_*`).
- **Precision presets**: `verify-double` (default, precise), `fast-double`
  (opt-in `--use_fast_math`, dump machinery compiled out), `mixed-float`
  (float state + documented double reductions; state floor ~1e-7,
  `ftol_array` entries must be >= 1e-6), `debug-double` (precise + `-G`).
  The policy and its flags are recorded in every v1 output.
- **Backend matrix**: verify (NetCDF+HDF5), netcdf-only, hdf5-only, and
  nobackend builds each run their suites — all four configurations exist
  as presets and as hosted CI matrix entries; the v0 NetCDF writer is
  byte-identical to the frozen schema dumps, the HDF5 v0 adapter is
  layout-exact (libhdf5 embeds a per-second timestamp), and v1 containers
  round-trip the complete RunReport + restart metadata.
- **CI**: `.github/workflows/ci.yml` — a hosted build/test matrix
  (verify/nobackend/netcdf-only/hdf5-only/float + ASan/UBSan host tests)
  and a self-hosted GPU job (`scripts/ci_gpu.sh`: full CTest incl.
  sanitizer variants, CLI policy gate, benchmark smoke, and the frozen
  short-trajectory stage-cap contract).
- **Event-DAG audit**: `test_event_dag` pins the scheduling contracts
  (pending-event query/elapsed-time semantics, delayed-kernel event
  ordering, stream-fault surfacing); an Nsight Systems/API-trace audit
  remains the documented manual step before any multi-stream or CUDA Graph
  production variant:
  `nsys profile -o trace ./build/cuMES inputs/solovev.json out.bin`.


## 1. Reference hierarchy

Use three layers of truth:

1. **Local scalar reference:** simple CPU implementations of transforms, staggered geometry, force terms, constraint, residuals, and tridiagonal solves. These diagnose the first wrong component.
2. **Frozen legacy trajectory:** per-iteration component snapshots, residuals, damping values, restart events, and final state from the audited cuMES baseline on safe inputs.
3. **Independent VMEC++ reference:** inputs and outputs generated with a pinned VMEC++ 0.7.0 revision, toolchain, and conversion script.

The legacy solver is not an oracle for a path known to contain undefined behavior. Such paths are validated against a corrected scalar formulation and VMEC++ instead.

Every reference artifact records input hash, code revision, build flags, scalar type, GPU/runtime where applicable, and a schema version. Large binary fixtures should have a manifest with checksums and generation commands.

## 2. CPU unit tests

Required host tests include:

- strict/compatibility JSON parsing, normalized configuration, defaults, aliases, duplicate/unknown keys, wrong types, integer overflow, negative modes, empty stages, and checked extent overflow;
- mode folding/unfolding for signed toroidal input and every product-basis family;
- `GridShape`, surface/mode indexing, and quadrature endpoint weights;
- profile polynomial evaluation, flux normalization, fixed-iota/current policy selection, non-unit `tcon0`, and rejection of unsupported gamma;
- controller residual histories reproducing convergence, ten-sample initialization/zero cases, bad-progress/Jacobian restarts, `ijacob=25/50` maintenance resets, checkpoint refresh/restore, and effective-iteration counting;
- multigrid interpolation at axis, interior points, LCFS, odd/even modes, and all six families;
- output capability preflight and run-status-to-exit-code mapping;
- versioned/legacy serialization round trips and deliberate I/O failures.

## 3. CUDA component tests

| Component            | Minimum CUDA test                                                                                                                                                                                                                    |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Transform            | Constant, single theta mode, single zeta mode, analytical derivatives, all six families, parity, axis/LCFS rules, Nyquist and awkward/max angular sizes, direct CPU comparison, float/double.                                        |
| Axisymmetric backend | Cross-check every intermediate against generic `nzeta=1` backend on several `ns/ntheta/mpol` combinations.                                                                                                                           |
| Geometry             | Manufactured circular/elliptic and deliberately inverted/degenerate surfaces; compare half interpolation, `tau1`, parity correction, `sqrt(g)`, covariant metric, oriented Jacobian status, and guarded no-write behavior pointwise. |
| Magnetic field       | Manufactured lambda/iota/current cases with analytical `B^u`, `B^v`, `B_u`, `B_v`, pressure, and `ncurr=0/1`.                                                                                                                        |
| Force                | CPU scalar weak form for every parity family; isolate radial, poloidal, toroidal, hybrid-lambda, and lambda-axis-exception contributions; include a directional finite-difference/energy consistency check where applicable.         |
| Constraint           | Verify `m=0,1` annihilation by `m(m-1)`, band edges, non-unit `tcon0`, reset/reference behavior, and theta sizes above/below 32.                                                                                                     |
| Residual             | Known vectors, deterministic reduction, zero/denormal/nonfinite inputs, and float-state/double-accumulator mode.                                                                                                                     |
| Preconditioner       | Compare lower/diagonal/upper coefficients and batched solutions with CPU tridiagonal solve for awkward row counts, both parities, `m=1` scaling, `copysign` lambda term, zero modes, scaled near-singular and breakdown cases.       |
| Descent/checkpoint   | Exact component/mode scaling, conditional `m=1` gauge, axis/LCFS rules, post-descent capture, descent-then-restore discard, velocity zero, and optional fused state-only checkpoint.                                                 |
| Prolongation         | Device/CPU agreement for coarse/fine pairs including non-power-of-two sizes.                                                                                                                                                         |

Tests must allocate through the same typed slabs as production. No test may spell a magic component count such as five or six.

## 4. Integration and trajectory tests

Maintain the following tiers:

- `smoke`: a few iterations of small axisymmetric and 3D manufactured cases;
- `short-trajectory`: 25–100 iterations of Solovev and W7-X with component checkpoints at selected iterations;
- `full-regression`: complete multigrid Solovev and W7-X, including prescribed current, constraint resets, and restarts;
- `legacy-compatibility`: existing JSON fixtures and old binary reader/writer;
- `I/O-matrix`: binary plus NetCDF/HDF5 with none/one/both optional libraries, recognized/unknown suffixes, unwritable paths, float-to-on-disk conversion, schema inspection, and restart round trip.

Structured telemetry is compared directly. Do not parse human `printf` output to infer solver behavior.

## 5. Sanitizers and static checks

**Status (2026-08-17): the memcheck matrix was extended and CI exists.**
The `sanitizer` preset registers racecheck + synccheck variants of the kernel
tests (`CUMES_ENABLE_EXTRA_SANITIZER_TOOLS`, RUN_SERIAL in CTest — racecheck
instrumentation exhausts the GPU under parallel runs) and builds dedicated
ASan+UBSan twins of the host-only libraries and their tests
(`CUMES_HOST_SANITIZERS`; the ASan runtime must be first in each executable's
library list, so the sanitized libs are `_asan` copies consumed only by
`asan_test_*` executables — never propagated into CUDA targets). CI lives in
`.github/workflows/ci.yml` (hosted build matrix incl. the optional-backend
configurations + ASan/UBSan host tests; self-hosted GPU release gate), and
`test_event_dag` pins the scheduling contracts. The Nsight Systems/API-trace
audit and formatting/static-analysis jobs remain documented manual steps.

CI jobs:

- host AddressSanitizer and UndefinedBehaviorSanitizer for config, controller, and I/O;
- Compute Sanitizer `memcheck` and `initcheck` on all small CUDA tests;
- Compute Sanitizer `racecheck`/`synccheck` for their supported intra-kernel shared-memory/barrier hazards; do not treat them as proof of inter-kernel global-memory ordering;
- explicit event-DAG stress tests for streams/graphs using randomized delay kernels, versioned poison buffers, and assertions that every consumer observes the intended producer version;
- an Nsight Systems or CUDA API-trace audit of each multi-stream graph variant and snapshot path;
- debug launch checking with a named range on failure;
- compiler warnings as errors for project sources, excluding vendored dependencies;
- formatting and a lightweight CUDA-aware static-analysis pass where supported.

Fixtures must be self-contained. The historical force verifier's dependency on an absent `vmecpp_init.bin` is resolved: `test_force_verify` generates its own fixture (checked-in asset-free).

## 6. Equivalence gates

Classify each change before review:

- **Class A — ownership/scheduling, same arithmetic order:** require bitwise equality of component outputs and the full residual/controller trajectory in the precise build.
- **Class B — reduction/kernel reorder:** require per-operator absolute/relative/ULP thresholds derived from the reference scale, identical finite/status classification, and identical controller decisions on the frozen short trajectories.
- **Class C — numerical algorithm change:** require independent CPU/VMEC++ agreement, physical invariant checks, final-equilibrium comparison, convergence robustness across the fixture matrix, and a written ADR.

Never approve a change only because the final residual is small. Compare R/Z/lambda families, axis/boundary invariants, geometry/field intermediates, restart sequence, and iteration count.

## 7. Performance acceptance

For a performance-motivated change:

- require the lower bound of the 95% confidence interval to show an improvement greater than `max(5%, measured noise floor)` on one named target workload, with the upper confidence bound on the other primary workload's regression at or below 2%, unless the change has a separately approved correctness or memory benefit;
- use repeated thermally stable warm runs and report median, p95, confidence interval, clocks, and noise floor—not a single timing;
- include setup and output separately from effective-iteration time;
- compare on at least the legacy Pascal target and one modern architecture;
- reject peak arena/cuFFT/graph memory growth beyond an agreed baseline ceiling unless the measured performance or correctness benefit explicitly justifies it;
- retain the old backend until the new one passes both numerical and performance gates.

These thresholds are review policy, not a claim that every listed optimization will meet them.

## Appendix: review checklist for every CUDA change (blueprint §13)


Before merging a kernel or scheduling change, answer:

1. Which mathematical domain and layout does every view use?
2. Are all extents and products validated before launch?
3. Is the kernel correct for partial tiles/warps and awkward sizes?
4. Which stream owns the operation, and where is the necessary dependency established?
5. Does it allocate, synchronize, or copy in the hot loop?
6. What is the register/shared-memory/occupancy effect on Pascal and a modern GPU?
7. Which CPU/intermediate/trajectory test detects a wrong result?
8. Is arithmetic order preserved? If not, which equivalence class and tolerance apply?
9. What benchmark demonstrates benefit, including regressions on the other primary shape?
10. Can the previous backend remain available until the new one passes all gates?
