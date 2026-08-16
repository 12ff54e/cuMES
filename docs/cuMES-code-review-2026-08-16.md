# cuMES Code Review Log — branch `overhaul` (main...HEAD)

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

## 1. Correctness — silent wrong results / crashes on validated inputs

### 1.1 `src/fourier_impl.cuh:999` — forwardReduceKernel drops theta points when nThetaRed > 16 — CONFIRMED
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

### 1.2 `src/fourier_impl.cuh:188` — mpol=2 makes the de-alias plan batch 0 → startup crash — CONFIRMED
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

### 1.3 `src/prolongation_impl.cuh:99` — exit(EXIT_FAILURE) mid-solve on validated equal-ns stages — CONFIRMED
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

### 1.4 `src/cumes/io/io_common.hpp:72` — checkStateDimensions cast wraps negative → std::terminate on corrupt header — CONFIRMED
`static_cast<long long>(*needed) > *sz` casts a checked-`size_t` byte count; a
corrupt int32 header with `ns*mnmax` in [2^63, 2^64) (e.g. ns=2147483647,
mnmax=95000000) wraps negative, the file-size bound silently passes, and the
readers then execute `fam.resize(~2e17)` → `std::bad_alloc`. main.cu:358 catches
only `cumes::CumesError`, so the process `std::terminate`s instead of producing the
intended "dimensions implausible" error (checkpoint.cpp:91-95,
legacy_binary_v0.cpp:76-82).
**Fix:** compare in `size_t` / `long double`, or bound ns/mnmax against sane
maxima before the multiply.

### 1.5 `src/main.cu:210` — validation warnings are never printed — CONFIRMED
Unknown input keys (json_reader.cpp:359-362) and skipped out-of-range boundary
harmonics (validated_problem.cpp:158-164) are collected into `vp.warnings()`
explicitly "so the caller can report them" (validated_problem.cpp:228-230), but
main.cu never iterates `warnings()`. A typo'd key (`"n_theta": 24`) or an
`m >= mpol` rbc harmonic runs silently with defaults/dropped harmonics — the legacy
parser printed `cuMES: WARNING: unknown input key ... ignored` and
`skipping mode m=... n=...` to stderr.
**Fix:** print `vr.value().warnings()` to stderr after successful validation.

### 1.6 `src/main.cu:310` — `--output-schema v1` silently ignored for .nc/.h5 — CONFIRMED
The flag is parsed and validated for any output (main.cu:129-138) but applied only
in the binary branch (main.cu:305-311); the nc/h5 branch (326-331) calls
`outputSave<Real>()` without the schema, and the NetCDF/HDF5 writers are hard-wired
to the v0 fixed-capacity layout (output_netcdf.cpp:27-33). `--output-schema v1
out.nc` silently writes v0 with no warning or error.
**Fix:** reject the combination, or implement v1 for nc/h5.

### 1.7 `src/main.cu:145` — `--input`/`--output` documented but unhandled — CONFIRMED
The file header documents `--input <path>` (line 4) and `--output <path>` (line 9),
but the parse loop handles only `--output-schema`/`--restart`/`--restart-legacy`/
`--checkpoint`; `cuMES --input x.json out.bin` hits the "unknown option" branch and
exits EXIT_FAILURE. The diff both advertises and fails to implement the flags.
**Fix:** implement the flags or delete them from the header comment.

### 1.8 `include/cumes/io/legacy_provenance.hpp:62` — 9+-stage configs produce self-inconsistent v0 provenance — CONFIRMED
`from_validated` stores `p.n_grids = s.stages.size()` uncapped while the stage
arrays truncate at `kMaxGrids=8` (line 20). The legacy parser hard-failed >8 stages
("entries exceed the 8-entry capacity", main:src/input_json.cu:104-107); the new
validation has no stage-count cap. A 9-stage JSON now solves all 9 stages while the
v0 writers emit scalar `n_grids=9` against an `ngrids=8` dimension
(output_netcdf.cpp:67, output_hdf5.cpp:81/128) — stages 9+ silently dropped from
the provenance.
**Fix:** cap or validate the stage count against `kMaxGrids` again.

---

## 2. Correctness — diagnostics, provenance, CLI/tooling contracts

### 2.1 `src/solver_impl.cuh:609` — dump iter-0 diagnostic reads d_r_e unsynchronized — CONFIRMED
With `CUMES_DUMP=1`, the iter-0 block does a synchronous `cudaMemcpy` of
`rs.d_r_e` on the legacy default stream right after the inverse transform was
enqueued on the `cudaStreamNonBlocking` compute stream (stream.hpp:17) — the
legacy stream does not wait for it. The legacy code synced
(main:src/solver.cu:675-679: `cudaEventRecord(ev1); cudaEventSynchronize(ev1);`),
and the file's own `dumpDeviceArray` documents the hazard and calls
`cudaDeviceSynchronize()` first (solver_impl.cuh:469-474). Result: stale/garbage
`[loop diag]` print and `dump/cuMES/debug_r_e.bin`.
**Fix:** add `cudaDeviceSynchronize()` (or route through `dumpDeviceArray`).

### 2.2 `src/geometry_impl.cuh:535` — jacobianStatsKernel vmax under-reports on sign-flipped √g — CONFIRMED
The new-min branch (`else if (ov < vmin) { vmin = ov; ... }`) never updates
`vmax = fmax(vmax, a)`, so a flipped element (ov = -a) taking that branch excludes
its magnitude from `max|√g|`. Legacy reduced both over `fabs(g)` where
`vmax >= vmin` made the skip harmless. Impact: only the BAD-JACOBIAN diagnostic
printf (solver_impl.cuh:1290-1292) under-reports max|√g| — the validity decision
is unaffected because `min_oriented <= 0` short-circuits the `||` chain first, and
no-flip runs are exactly legacy-equivalent.
**Fix:** update `vmax` unconditionally in both branches.

### 2.3 `include/cumes/io/run_report.hpp:35` — StageReport.restarts is never populated — CONFIRMED
The only construction site (multigrid_solver.hpp:65-72) never touches `restarts`,
yet versioned_binary.cpp:84-85 serializes it into every v1 container — the restart
history (which the frozen regression bar tracks, e.g. W7-X's restart events) is
always recorded as 0. The controller knows the events
(iteration_controller.hpp:165-190, `RestartReason::kBadJacobian/kBadProgress`) but
nothing plumbs them into the report.
**Fix:** carry restart events in `SolverResult<T>` and fill `StageReport.restarts`,
or drop the field and the container slot.

### 2.4 `src/main.cu:83` — fill_provenance re-hashes the input file after the solve — CONFIRMED
`read_and_validate` parses the file at startup (no raw bytes retained), then
post-solve `fill_provenance` re-opens and FNV-1a-hashes the path. A file modified
or replaced mid-solve records a `source_hash` of bytes the solver never consumed
(TOCTOU), plus the unconditional whole-file double read.
**Fix:** hash the bytes once at read time and carry them into the report.

### 2.5 `tests/test_force_verify.cu:154` — converged-force gate is tautological — CONFIRMED
The recompute path (lines 110-129) runs the same production
Geometry/MagneticField/Force/ToroidalFft operators the solver just used, so a
broken force formula that still permits convergence (e.g. zero forces on m≥2)
makes solverRun converge AND the recomputed residuals ~1e-16 — all three
`kFailThresh=1e-4` CHECKs pass. Contradicts the header's claim "a broken force
formula shows up as O(1) residuals" (the file's own comment at 150-152 admits the
shared kernels).
**Fix:** recompute forces via an independent CPU reference (as
test_force_reference.cu does) instead of the production kernels.

### 2.6 `scripts/capture_baseline.sh:157` — float-capture sed hardcodes 3 ftol entries — CONFIRMED (empirical)
The sed rewrite always produces `[ft, ft, ft]`. A 2- or 4-stage `--configs` entry
fails validation (`ftol_array length must match ns_array`, json_reader.cpp:291);
a multi-line `ftol_array` silently escapes the line-based sed and the float build
then hard-errors on the double-tuned ftols below the 1e-6 floor
(precision_policy.hpp:33). Shipped configs are single-line 3-stage, so the default
path works.
**Fix:** JSON-aware substitution, or a solver-side per-stage ftol override env
(like `CUMES_MAX_ITER`/`CUMES_DELT0`).

### 2.7 `scripts/capture_baseline.sh:101` — absolute --build path mangled via $OLDPWD — CONFIRMED (empirical)
`"$OLDPWD/$build/cuMES"` concatenates into `<repo-root>//<abs>/cuMES` for an
absolute `--build`; the preflight `[ -x "$BUILD/cuMES" ]` (line 89) passes but the
launch dies with exit 127 under `set -euo pipefail` (the only diagnostic is buried
in run.log). Same mangle for `--float-build` (line 159) and the `--schema` reruns
(116-119).
**Fix:** normalize `--build` to an absolute path once at startup.

### 2.8 `src/output.cpp:112` — check_cuda throw bypasses the atomic-publish contract — CONFIRMED
`cumes::check_cuda(cudaMemcpy(buf, d, nb, D2H), tag)` inside `writeFam` throws
`CumesError`; the documented failure contract (close temp, remove temp, return
false — the `fail` lambda at output.cpp:94-101) is only wired to fwrite failures.
A GPU fault during the final T→double copy leaves `<out>.tmp.<pid>` on disk and
`fp` unclosed (main catches the error and exits). Same throw point at
output_hdf5.cpp:153 and output_netcdf.cpp:154, which additionally skip
H5Fclose/nc_close + remove(tmp).
**Fix:** wrap the copy in try/catch and run the fail path, or use unchecked
`cudaMemcpy` + manual error handling consistent with the writers' contract.

### 2.9 `src/output_hdf5.cpp:179` — staging-buffer leak on write failure — CONFIRMED
`H5_CHECK` returns false on any per-mode `H5Dwrite` failure before
`delete[] dbuf; delete[] buf;` (185-186). NetCDF twin identical (NC_CHECK at
165-170, deletes at 171-172). One-shot leak of ~2·ns·mnmax·8 bytes (~270 KB for
W7-X) immediately before exit; visible only under LeakSanitizer.
**Fix:** delete before the macro's return-false, or RAII the buffers.

### 2.10 `tests/test_regression_kernels.cu:435` — testPcr leaks a 43-buffer RealSpaceStorage — CONFIRMED
`testPcr` creates `RealSpaceStorage` via `realSpaceCreate(p)` (43 cudaMallocs with
the default `arena=nullptr`) and frees only `d_f`/`mt`; the sibling `testDealias`
calls `realSpaceFree(rs)` (line 417). 7 ns values × double/float = 14 leaked
43-buffer sets, reclaimed only at process exit.
**Fix:** add `realSpaceFree(rs)`.

### 2.11 `tests/test_geometry_iso.cu:116` — coverage check relies on uninitialized memory — PLAUSIBLE
The kernel-coverage gate counts exact `0.0` entries in `bsupu`/`bsubu`, but those
buffers are never zero-initialized (geometry_impl.cuh:64-65 allocates via
cudaMalloc/arena with no memset). On fresh allocations the driver's zeroed pages
would make a skipped point read 0.0 and the check fire; on reused non-zero memory
it would not — the test's ability to catch the launch-shape regression it exists
for depends on allocator state. The codebase itself memsets similar buffers "so a
diagnostic dump … is deterministic" (fourier_impl.cuh:252-266).
**Fix:** memset the buffers (or use a sentinel fill) before the kernel.

### 2.12 `tests/test_geometry_iso.cu:91` — unchecked cudaMemcpy in the m=1 block — PLAUSIBLE (framing)
The copies at 91/96 are indeed raw `cudaMemcpy` (failure → uninitialized `hcc`,
test still exits 0), but they are NOT a regression (identical in
main:tests/test_geometry_iso.cu) and NOT unique (lines 112/115/121 are also raw).
**Fix:** route through `cc()` for consistency.

### 2.13 `cmake/CumesCudaArchitectures.cmake:18` — default arch list breaks CUDA 11.0–11.7 — CONFIRMED
`61;75;80;86;89` is set with no toolkit-version conditional, while CLAUDE.md
documents "CUDA Toolkit >= 11" and CumesDependencies.cmake:14 checks
`find_package(CUDAToolkit 11 REQUIRED)`. On CUDA 11.0–11.7 `nvcc` errors
"Unsupported gpu architecture compute_89" on the default `cmake -B build` (11.8
is the first toolkit supporting sm_89). The comment pointing to
docs/performance.md for the compat policy is stale — that doc contains no such
text.
**Fix:** gate sm_89 on `CUDAToolkit_VERSION >= 11.8` (and sm_86 on >= 11.1),
or document the override.

### 2.14 `src/json_parser.cpp:8` + `src/cumes/config/json_reader.cpp:13` — ZQ_JSON_PARSER_IMPLEMENTATION defined twice — CONFIRMED
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

## 3. Correctness — latent (mechanism real, no current trigger)

### 3.1 `include/cumes/solver/iteration_controller.hpp:97` — jacobian_invalid adds an absolute min_oriented <= 0 gate — PLAUSIBLE
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

### 3.2 Defaulted moves + freeing destructors on 5 operator classes — PLAUSIBLE
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

### 3.3 `src/axisymmetric_impl.cuh:278` — rzconKernel launched on OR but writes both outputs unconditionally — PLAUSIBLE
The interface documents "rCon/zCon may be null views to skip that output"
(spectral_operator.hpp:33-37); the generic backend guards each pointer
(fourier_impl.cuh:455-456) and launches per-slot with `(rCon, nullptr)`/
`(nullptr, zCon)`. The axisymmetric backend launches on
`rCon.data() != nullptr || zCon.data() != nullptr` into a kernel that writes both
`rCon[idx]` and `zCon[idx]` with no null check → device illegal-address fault for
one-null callers. No current caller passes exactly one null (solver passes both,
tests pass both null).

### 3.4 `src/cumes/io/versioned_binary.cpp:57` — short-family guard still writes — PLAUSIBLE
`if (fam.size() != n) ok = false;` then proceeds to `write_f64_array(fp,
fam.data(), n)` — an OOB read of a short host vector (same pattern
checkpoint.cpp:48-49). No in-tree trigger: every producer resizes families to
exactly `family_size()` (snapshot_bridge.cuh:42, checkpoint.cpp:92/127,
versioned_binary.cpp:129), and even when tripped, `ok=false` still removes the
temp and returns an error — the defect is the OOB read itself.
**Fix:** return before the write when the size mismatches.

### 3.5 `src/cumes/io/versioned_binary.cpp:62` — precision tag from string-compare — PLAUSIBLE
`precision = (report.build.scalar_type == "float") ? 1 : 0` is a stringly-typed
convention for a format discriminator. All in-repo producers write exactly
"double"/"float" from the same sizeof ternary (main.cu:81, test_io_golden.cu:149),
and the reader ignores the trailer field entirely, so nothing misfires today — a
foreign producer writing "single"/"fp32" would silently record 0=double.
**Fix:** a typed enum on RunReport.

### 3.6 `src/fourier_impl.cuh:1038` — enqueue_inverse/enqueue_forward ignore the passed view bundles — PLAUSIBLE
`(void)geometry` / `(void)real_force` and the captured `rs_` is used instead,
while `AxisymmetricOperator` honors the passed views (axisymmetric_impl.cuh
266-292). Divergent contracts of the same `SpectralOperator` interface: a caller
passing non-aliasing views gets correct results on the axisymmetric backend and
wrong results on the generic one. Today's only caller wraps views over the same
`rs` the operator was constructed with (solver_impl.cuh:519-535), and the
toroidal header notes the aliasing — no wrong output in the current codebase.

### 3.7 `include/cumes/runtime/stream.hpp:44` — synchronize() on a moved-from Stream syncs the default stream — CONFIRMED (use-after-move only)
Moves null the source (`other.stream_ = nullptr`), and `synchronize()` calls
`cudaStreamSynchronize(stream_)` with no null guard; `cudaStreamSynchronize(0)`
is defined behavior — it waits on the **legacy default stream**, which does not
sync nonblocking-stream work — and returns cudaSuccess. A use-after-move thus
reports success while never waiting for the moved-away stream's work. No current
call site moves a Stream and then synchronizes the source.
**Fix:** assert/throw on `stream_ == nullptr`.

### 3.8 `include/cumes/state/mode_table.cuh:35` — DeviceModeTable implicit copy → double-free — PLAUSIBLE
Raw owning aggregate (d_xm/d_xn + arena_backed) with no deleted copy ops and
`modeTableFree` freeing unconditionally (35-38). Every call site uses
guaranteed-elision copy-init and frees once, so no copy occurs today; a
pass-by-value or explicit copy followed by modeTableFree on both would
double-cudaFree.
**Fix:** delete the copy ops (or RAII-own the pointers).

---

## 4. Cleanup — reuse (duplicated implementations)

### 4.1 `src/output.cpp:23` — tempPathFor/publishAtomic re-implemented at four sites — CONFIRMED
`cumes::io_detail::tempPathFor`/`publishAtomic` exist in
src/cumes/io/io_common.hpp:20/27 (used by checkpoint.cpp and legacy_binary_v0.cpp);
output.cpp:23/33 re-implements the same protocol with bool+stderr instead of the
reason-string convention, and output_hdf5.cpp:53 / output_netcdf.cpp:38 hand-roll
the temp-path string inline. Four sites, two conventions — a fix to the shared
helper will not reach the three writers.

### 4.2 `src/forces_impl.cuh:28` — geometryParityViews byte-identical to geometry_impl.cuh:46 — CONFIRMED
Verified byte-identical (`sed -n 27,39p` vs `45,57p` diff clean); the header
comment says "mirror of geometry_impl.cuh's helpers". Both TUs already include
`cumes/state/real_fields.cuh` — the natural single home for one inline
definition. Any view-bundle change must be applied twice or the force kernel's
views silently diverge.

### 4.3 `src/output_hdf5.cpp:153` — D2H + T→double block copy-pasted into three writers — CONFIRMED
output_hdf5.cpp:153-154, output_netcdf.cpp:154-155, output.cpp:112-113 are
identical modulo the error tag. The on-disk-double contract (the Python compare
scripts parse doubles regardless of T) is re-implemented three times; one shared
helper next to io_common.hpp's serialization primitives is mechanically feasible.

### 4.4 `src/cumes/io/checkpoint.cpp:58` — read_checkpoint duplicates VersionedBinaryReader — CONFIRMED
checkpoint.cpp:67-96 vs versioned_binary.cpp:108-135: same magic/version/ns/mnmax
header shape, same `checkStateDimensions`, same six-family f64 loop — differing
only in magic string ("CUMECKP1" vs "CUMES001") and error labels; writer halves
likewise (checkpoint.cpp:42-49 vs versioned_binary.cpp:52-59, the v1 writer then
appending its provenance trailer). Any layout evolution must be edited and
re-tested in both.

### 4.5 `tests/test_geometry_ncurr.cu:53` — manufactured-state fixture copy-pasted across 4 tests — CONFIRMED
The six-family manufactured state appears at test_geometry_ncurr.cu:52-76,
test_constraint_tcon.cu:53-84 (quadratic envelope — deliberate per its comment),
test_geometry_iso.cu:34-60 (fillState), and a reduced variant at
test_cuda_graph.cu:95-109. tests/support/cumes_test_support.cuh hosts no state
builder, though its comment says helpers move in "when a second consumer
appears" — there are now 3-4.

### 4.6 `benchmarks/graph_realpass.cu:76` — benchmark harness copy-pasted across 3 files — CONFIRMED
`now_us` (fixed_iteration.cu:55-58 vs graph_realpass.cu:76-79) byte-identical;
`median`, the `need` CLI lambda, the read_and_validate block, and the operator
stack + CUMES_FORCE_GENERIC selection structurally verbatim across
fixed_iteration.cu / graph_realpass.cu / graph_overhead.cu (only stderr prefixes
differ). Every harness change must land in three files or silently diverge.

### 4.7 `tests/support/cumes_test_support.cuh:19` — cc() re-implements cumes::check_cuda — CONFIRMED (trivial)
cuda_status.hpp:52-58 throws `CumesError`; the test helper does the identical
check as fprintf+exit. The duplication is literal (cuda_status.hpp's own header
comment says check_cuda replaces "the per-file checkCuda/cc/ccf helpers that used
to exit(1)"), though the exit-based form is a defensible test-UX choice (an
uncaught throw → std::terminate loses the tag message).

---

## 5. Cleanup — simplification (dead code / derivable state)

### 5.1 `include/cumes/solver/equilibrium_operator.hpp:96` — member op_ written, never read — CONFIRMED
Only the declaration and `op_(op)` in the constructor exist; `enqueue` uses the
duplicate `transform_op_` (solver_impl.cuh:515). Delete `op_`.

### 5.2 `include/cumes/solver/control_record.hpp:62` — ControlRecord<T> is dead; InvariantVerdict<T> templates on unused T — CONFIRMED
`ControlRecord` has no non-comment reference in any TU (the header's own comment
admits it was the unlanded Phase 6A single-fence form). `InvariantVerdict` holds
only two bools — the T parameter exists purely to spell
`InvariantVerdict<double>` at the call site. Delete ControlRecord; de-template
InvariantVerdict.

### 5.3 `include/cumes/runtime/device_context.hpp:26` — DeviceContext is dead infrastructure — CONFIRMED
Only tests/test_runtime.cu constructs it; production main.cu:275 creates its own
`cumes::Stream`. The header comment ("the solver keeps the legacy default stream
until the Phase 6A scheduling work") is stale — Phase 6A is landed.

### 5.4 `src/precon_impl.cuh:982` — enqueue_apply retains dead xm/xn parameters — CONFIRMED
`(void)xm; (void)xn; // legacy signature` — the body never touches them, yet the
sole caller (solver_impl.cuh:975) still plumbs `transform.xm()/xn()` and the
header retains them. Remove the parameters and update the call site.

### 5.5 `CMakeLists.txt:257` — CUMES_HOST_SANITIZERS re-declares all host targets — CONFIRMED
The asan block duplicates every source list and link setup of the four host
libraries + four test executables (257-316). The block's own comment explains the
deliberate reason (sanitized runtime FIRST in each asan executable's library
list), so this is a maintainability cost, not a bug: a new TU in `cumes_core`
must be added to two source lists or the sanitized build silently diverges.

### 5.6 `docs/mathematics.md:21` — normative contract corrupted by markdown artifacts — CONFIRMED
83 occurrences of `*{`/`\sum*`-style artifacts: `s*j` for `s_j`, `\sum*{m,n}`,
`R^{cc}*{mn}`, `S*{mn}=m*{\mathrm{scale}}n*{\mathrm{scale}}`, `\mu*0` for
`\mu_0` (line 198). The file declares itself the normative numerical contract
("a change that alters any expression below is at least Class B") yet the
mixed-gauge formulas (e.g. `(f_Rss ± f_Zcs)/√2`, lines 91-92) are semantically
garbled for a contract document.

### 5.7 `src/output_hdf5.cpp:130` — provenance dims hardcoded 32/16/16 — CONFIRMED
`d32[1]={32}`/`d16x16[2]={16,16}` (and output_netcdf.cpp:69-71) hardcode what
`LegacyInputProvenance::kMaxAxis=32, kMaxM=16, kMaxN=16`
(legacy_provenance.hpp:22-24) already defines — the same function uses
kMaxGrids/kMaxCoeff for the other dims. A capacity change would silently truncate
the writes.

### 5.8 `include/cumes/state/seed_state.hpp:178` — six separate uploads instead of the slab — CONFIRMED
restart_state stages six host arrays and issues six H2D memcpys in the exact
`state_slab()` order (Rcc Zsc Lsc Rss Zcs Lcs; spectral_storage.hpp:7-8,52), which
was built precisely so "the six old per-family copies become one". init_state
(lines 108-119) has the same six-upload block. One 6·mnmax·ns staging buffer +
one memcpy replaces both (the double→T conversion is the only reason for host
staging).

---

## 6. Cleanup — efficiency

### 6.1 `src/fourier_impl.cuh:510` — full 12·mpol·ns·nz2 memset every inverse pass — CONFIRMED
`inversePackKernel` writes every bin n<=ntor of all 12 slots each pass, so only
the tail bins ntor+1..nz2-1 are stale; the memset clears the whole buffer
(12·12·99·19·16 ≈ 4.3 MB for W7-X, ~29 MB for nzeta=256) every iteration. The
tail-zero can be folded into inversePackKernel (threads with n==ntor zero the
tail of their slots) — a one-time construction memset does not work because the
forward D2Z overwrites the bins each pass. Correctness is unaffected.

### 6.2 `benchmarks/graph_realpass.cu:203` — pageable staging makes cudaMemcpyAsync synchronous — CONFIRMED
`new Real[16]` is pageable, so the benchmark's `cudaMemcpyAsync` degrades to a
synchronous 2-hop copy, while the production path it measures uses
`cumes::PinnedBuffer<double>(16)` (solver_impl.cuh:1087; pinned_buffer.hpp:3-5
documents exactly why). Both branches use the same h_ctl, so the
graph-vs-direct saving estimate mostly cancels — the biased part is the
absolute per-pass wall. Use PinnedBuffer like production.

### 6.3 `src/output_hdf5.cpp:165` — H5Dget_space inside the per-mode loop — CONFIRMED
`H5Dget_space(ds)`/`H5Sclose(fs)` run once per mode per family (6·mnmax handle
pairs, e.g. ~800-1800 for W7-X) though the file dataspace `ds` is invariant
across modes. Fetch once before the loop.

### 6.4 `src/solver_impl.cuh:509` — d_f_spec_/d_control_/d_psum_ outside the stage arena — CONFIRMED
The constructor receives the stage DeviceArena and carves precon_/constraint_
from it, but allocates the solver's own three buffers as standalone cudaMallocs —
the stage was consolidated so "one cudaMalloc per stage instead of ~110" and the
arena's peak/liveness report understates the real footprint. (Note: d_f_spec_ is
the largest of the operator's own buffers, but the arena's d_zeta_real is ~5.5×
larger for W7-X.)

---

## 7. Cleanup — altitude (bandaids / fragile decoupling)

### 7.1 `src/solver_impl.cuh:607` — ~278 lines of dump machinery interleaved in enqueue — CONFIRMED
8 `#ifdef DUMP_CUMES_VERIFY` blocks (623-678, 702-719, 748-804, 810-866,
875-903, 920-937, 959-964, 977-1021) interleave the per-iteration DAG builder;
`dumpDeviceArray` allocates a fresh `new T[nelem]` and full
`cudaDeviceSynchronize()` per call, and `dumpEnabled()` re-reads the env var at
every one of ~6 entry points per iteration. The blueprint (§6.12) wants
arithmetic-only enqueue with observers as subscribers — extracting the dump
windows into the observer/record mechanism would shrink enqueue by a third.

### 7.2 `include/cumes/solver/stage_solver.hpp:37` — stage_arena_bytes hand-sums every module's sizes — CONFIRMED
The function's own comment admits "keep the two in sync" with the modules'
authoritative `alloc_span` calls (43 arrays, 15 half-grid,
25·nH+9·ns+7·mnmax·ns+…, 64 KiB slack). Any buffer added/removed in a module
silently drifts the plan — an underestimate fails only at runtime, an
overestimate wastes memory silently. (Correction to the original claim: the two
benchmarks reuse `stage_arena_bytes` itself; they duplicate the surrounding
operator-stack setup, not the byte arithmetic.)
**Fix:** per-module size-report functions or a measuring-arena dry run.

### 7.3 `src/solver_impl.cuh:1128` — h_ax[64] magic bound decoupled from the ntor cap — CONFIRMED
`T h_ax[64]; // ntor+1 <= 64 for the hardcoded inputs` followed by a
`cudaMemcpy2D` of `p.ntor + 1` elements. The inputs are no longer hardcoded; the
real guarantee is the validator's `ntor in [0, 15]`
(validated_problem.cpp:92-93). Raise that cap without touching this line and the
stack array overflows in the production printIterRow path.
**Fix:** size from `p.ntor + 1` (or a small std::vector).

---

## 8. Cleanup — conventions (CLAUDE.md exact rules)

### 8.1 `src/solver_impl.cuh:541` — four cudaEventCreate calls unchecked — CONFIRMED
Breaks the CLAUDE.md rule that CUDA calls are "error-checked through the
centralized cumes::check_cuda/check_cufft" (the four `cudaEventDestroy` calls at
553-554 are likewise unchecked). A failed creation leaves an uninitialized handle;
the later `cudaEventRecord` is also unchecked, so the failure surfaces as garbage
timing instead of a CumesError at the true failure point.

### 8.2 `src/fourier_impl.cuh:519` — kTile/nKTiles/kTileA/kTileS/nKTilesA/nKTilesS/blkX/invSmem are camelCase — CONFIRMED
CLAUDE.md: "**Variables:** `snake_case`". Runtime variables, not constants (the
kCamelCase escape), repeated at lines 387/403/633/734; `plainPerEl` at
solver_impl.cuh:1265 likewise.

### 8.3 `include/cumes/solver/stage_solver.hpp:90` — realSpaceCreate/Free + modeTableCreate/Free non-RAII — CONFIRMED
Breaks CLAUDE.md: "xCreate/xFree replaced by RAII classes owning their buffers."
The pairing is manual (90-91, 113-114) and a throw between them skips the frees —
harmless on the always-arena stage path (both frees no-op when arena_backed) but
a full leak of ~30 cudaMallocs for any nullptr-arena caller (e.g.
test_constraint_tcon.cu).

### 8.4 `include/cumes/state/seed_state.hpp:65` — host staging buffers missing h_ prefix — CONFIRMED
`c`, `s`, `zsc`, `zcs`, `lsc`, `lcs` (lines 65-67) and again in restart_state
(138-143). CLAUDE.md: "Host pointers: `h_` prefix (e.g., `h_rmnc`, `h_cos`)."

### 8.5 `tests/test_geometry_ncurr.cu:122` — raw cudaMalloc/cudaFree for the stats probe — CONFIRMED (style-only)
Breaks CLAUDE.md: "All device allocations via RAII (DeviceBuffer/DeviceArena)".
The free is present and checked, so no leak — a `DeviceBuffer<T> stats(4)` is a
drop-in replacement. (test_regression_kernels.cu:475 also raw-mallocs `d_f`.)

### 8.6 `include/cumes/io/snapshot_bridge.cuh:27` — bare size products vs the checked_mul mandate — CONFIRMED
checked_size.hpp:3-6: "Every derived element-count product … must go through
these helpers, not a bare `a * b`." snapshot_bridge.cuh:27-29 computes
`ns() * mnmax()` and `kCount * one` bare. No realistic overflow today (mpol
capped at 16, validated ints) — a mandate-conformance gap, not a hazard.

---

## Appendix — REFUTED candidates (checked, found wrong or guarded)

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
