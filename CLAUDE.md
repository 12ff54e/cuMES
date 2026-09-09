# cuMES agent guide

`AGENTS.md` is a symlink to this file. Edit `CLAUDE.md` and keep the symlink;
there should be one shared set of repository instructions.

## Project and scope

cuMES implements the core VMEC magnetic-equilibrium algorithm. The default
native backend uses CUDA; the browser backend uses WebGPU and WebAssembly.
Both support fixed and free boundaries. Host code handles configuration,
iteration control, I/O, and parts of the vacuum solve. This is a research and
scaffolding project; qualification is specific to the case, precision, and
backend tested.

Convergence means all configured cuMES discrete residuals satisfy their
thresholds after finite/geometry/Jacobian validity gates. VMEC++ 0.7.0 is an
independent diagnostic reference, not the convergence oracle. A fixed-point
checkpoint replay is useful evidence; a small FSQR alone is insufficient.

## Working approach and commits

- Inspect `git status` and the relevant implementation before editing. Preserve
  unrelated user changes and keep the patch focused on the requested result.
- Carry authorized work through implementation, relevant checks, and commits.
  Resolve routine choices from existing code; ask for clarification when a
  missing requirement materially changes the result and cannot be inferred.
- Read the relevant sections of the documents below as needed. Use code,
  presets, and tests to establish current behavior; numerical and format
  contracts still require deliberate, validated changes. Dated measurements
  and overhaul histories are evidence for their recorded configurations, not
  universal defaults or acceptance values for another backend.
- Reuse existing operators and dependency APIs. Prefer a small integration
  layer over a second implementation of the same mathematics. Avoid unrelated
  refactors, new frameworks, and speculative generalization.
- Treat host/device data exchange and synchronization as optimization targets
  alongside kernel time. Keep reusable state and intermediates resident,
  transfer only the slices or compact reductions that host consumers need,
  and batch copies with required completion points. Preserve numerical gates,
  controller decisions, and vacuum update scheduling; see `docs/performance.md`.
- Commit meaningful, coherent steps progressively as they are completed and
  validated. Keep each commit focused and reviewable; include its validation
  summary. A small task may need only one implementation commit.
- Keep routine logs, benchmark captures, build artifacts, and progress notes
  outside the repository, normally in `../tmp/`. Do not make commits solely
  to record them. Retain durable contracts, regression fixtures, and design
  decisions in the repository when the change needs them.
- Reuse a compatible build directory; use a separate directory when changing
  backend/toolchain or comparing revisions. Check the cached source directory
  before rebuilding, since sibling worktrees can share `../tmp/` paths.
- Respect submodule boundaries. If a dependency needs changes, validate and
  commit them in that submodule before committing the parent gitlink update.
  Initialize pinned dependencies without updating them to unrelated revisions.

## Build and run

C++ and CUDA translation units use strict C++20, without GNU extensions.
`CMakeLists.txt`, `CMakePresets.json`, and `cmake/` define the build options.
Choose the backend affected by the task; the commands below are entry points,
not a requirement to run every configuration for every edit.

### Native CUDA

```bash
git submodule update --init --recursive
cmake --preset verify
cmake --build --preset verify -j
./build/cumes inputs/solovev.json --output out.bin
ctest --preset verify
```

The `verify` preset selects precise double arithmetic and warnings as errors,
with optional NetCDF/HDF5 output enabled when the libraries are available.
Other configure/build presets include `float`, `debug`, `fast`, `sanitizer`,
`fixed-only`, `nobackend`, `netcdf-only`, `hdf5-only`, and `profiling`.
Use the sanitizer and optional-dependency matrices when the affected code
requires them; GPU sanitizer runs are costly and racecheck is serialized.

Use a CUDA toolkit/host compiler combination that supports the project's
C++20 code. CMake defaults the CUDA host compiler to `/usr/bin/g++-12`; override
`CMAKE_CUDA_HOST_COMPILER` for another supported toolchain. Architecture
selection lives in `cmake/CumesCudaArchitectures.cmake` and accepts an explicit
`CMAKE_CUDA_ARCHITECTURES` override. These CUDA settings do not apply to Wasm.

### WebGPU / WebAssembly

On this machine, the Emscripten SDK and shared cache can be used as follows:

```bash
source /lustre/qzhong/emsdk/emsdk_env.sh
export EM_CACHE="$PWD/../tmp/cumes-emscripten-cache"
git submodule update --init --recursive deps/webgpu-fft deps/vacuum-field
emcmake cmake --preset webgpu
cmake --build --preset webgpu -j
ctest --preset webgpu
```

The preset builds in `../tmp/cumes-build-webgpu`; an explicit `-B` can select
an isolated directory. It sets `CUMES_BACKEND=WEBGPU`,
`CUMES_PRECISION_POLICY=mixed-float`, and `CUMES_USE_FLOAT=ON`. The WebGPU build
must not enable CUDA or require native NetCDF/HDF5 libraries.

Serve the generated `webgpu/cumes_webgpu.html` over a secure context. The
existing local nginx preview for the preset is:
`http://localhost:6969/magnetic-equilibrium-solver/tmp/cumes-build-webgpu/webgpu/cumes_webgpu.html`.
Use the URL matching the actual build directory and reload the generated HTML
after rebuilding; cached HTML can still select an older versioned runtime.

- Default page: boundary setup/editor; users can switch fixed/free modes.
- `?mode=test`: numerical verification in a worker.
- `?preset=w7x`: W7-X in the fixed-boundary editor; `&grids=3` selects
  multigrid. Historical `?solve=w7x` links open this setup without starting a
  solve. All equilibria use the same Run button.
- `?boundary=free&coils=solovev` (or `w7x`, `cth_like`): free-boundary preset;
  `&run=1` starts it.
- `vacuum=webgpu` selects the opt-in paired-f32 vacuum kernels with Wasm-double
  LU; add `vacuum_lu=webgpu` for resident paired-f32 LU (at most 256 unknowns).
  `vacuum=host` is the default/reference.
- `precision=float|double` selects scalar-f32 or paired-f32 plasma arithmetic.
  Paired words provide higher precision; this is not native WGSL f64. The
  fixed editor defaults to scalar, while free-boundary and W7-X examples
  default to paired precision.

## Code map and architecture contracts

| Area | Implementation |
| --- | --- |
| Shared configuration, layout, I/O, host controller | `include/cumes/`, `src/cumes/` |
| Native entry point / public embedding API | `src/main.cu`, `include/cumes/solver/equilibrium_solver.hpp` |
| CUDA operators | `include/cumes/{transforms,physics,numerics,solver}/`, `src/kernels/`, `src/*_{double,float}.cu` |
| WebGPU operators / shaders | `include/cumes/webgpu/`, `src/webgpu/`, `src/webgpu/shaders/` |
| Browser UI, worker, Emscripten bridge, presets | `webgpu/` |
| Shared vacuum library / browser coupling | `deps/vacuum-field/`, `src/free_boundary_impl.cuh`, `src/webgpu/vacuum.cpp` |
| Native tests / browser harnesses | `tests/`, `scripts/test_webgpu_*.mjs`, `scripts/webgpu_*.mjs` |

Preserve these contracts unless the task intentionally changes them:

- Per-pass mathematics: spectral state → inverse transforms → geometry and
  magnetic field → scheduled vacuum update → MHD and constraint forces →
  forward transforms → residuals, preconditioner, descent, and controller.
- Six spectral families are ordered `Rcc, Zsc, Lsc, Rss, Zcs, Lcs`, with
  `surface + mode * ns` within a family. Real-space arrays use
  `point + surface * nZnT`, with theta contiguous. Full-grid state and
  half-grid metric/field quantities are staggered. Follow
  `docs/data-layout.md` and `docs/mathematics.md` for parity, normalization,
  axis/LCFS, and radius-reference semantics; a raw displaced slab is not a
  complete physical state.
- CUDA operators own buffers through `DeviceBuffer`/`DeviceArena`, expose
  typed views, and use centralized CUDA/cuFFT error checks. Allocate reusable
  device scratch at stage setup, not in the iteration loop. Kernel modules
  normally use `_impl.cuh` bodies and float/double instantiation TUs; follow
  the existing module's placement rather than moving files for conformity.
- WebGPU operators use WebGPU buffers, command encoders, and asynchronous
  completion under `cumes::webgpu`. Preserve fixed-boundary residency and
  batched readbacks. CUDA allocation, stream, cuFFT, and `.cu` conventions
  apply to the native backend, not to the browser operators.
- Free-boundary coupling retains vacuum activation/restart state, `nvacskip`
  scheduling, LCFS pressure forces, preconditioner terms, and multigrid
  persistence. Reuse `deps/vacuum-field` and the existing coupling. The browser
  builds its WebGPU backend and HOST-double reference. `vacuum=webgpu` selects
  paired-f32 vacuum kernels, with Wasm-double LU by default and resident
  paired-f32 LU through `vacuum_lu=webgpu`. GPU LU retains pivoting, full/partial
  factor reuse and deferred error checks; systems above 256 unknowns require
  Wasm LU. `vacuum=host` retains the HOST path. Both run from the solver worker
  and reuse the same coupling.
- Browser free-boundary assets contain coil geometry and small configuration
  files. Generate field grids in memory with the existing MAKEGRID code.
  Preserve Solovev/W7-X/cth_like selection and coil uploads; do not ship field
  grids or add NetCDF solely for the browser path.
- Keep the dependency direction from cuMES to its libraries. Optimizer
  objectives and policy belong to meow/integration code; use the public
  `EquilibriumSolver` API for embedding rather than the CLI or output files.

## Precision, inputs, and output

- Native operators support float/double templates; `Real` selects the CLI's
  type. CUDA float device arithmetic must remain free of FP64 instructions;
  norm sums use float-float accumulation and the host controller uses double
  (ADR-0015 supersedes ADR-0001). This restriction does not prohibit host/Wasm
  double arithmetic, including the browser vacuum solve.
- Native float inputs reject stage tolerances below `1e-6`; this is an input
  floor, not a convergence guarantee. Fixed-boundary 3-D float uses radius
  reference storage; the qualified W7-X `1e-5` case also uses compensated
  geometry, including compensated m=1 toroidal sums for single-grid cold
  starts (`docs/w7x-single-grid-float.md`). Browser scalar/paired behavior has
  separate qualification; scalar free-boundary W7-X can stall above tolerance.
- Parse/validate input through the shared config API. Unknown keys are errors
  by default; native `--compatibility` changes input handling only. Native
  library solves ignore process-global `CUMES_*` controls unless requested.
  See `README.md` for CLI controls and `webgpu/browser_bridge.js` for browser
  options; do not assume a native environment variable configures the page.
- Native `--newton` opts fixed-boundary axisymmetric double solves (`ntor=0`,
  `nzeta=1`) into guarded Newton–Krylov corrections. It is off by default and
  preserves configured multigrid stages, tolerances, and caps. The library
  equivalent is `SolveRequest::enable_newton`; this is not a browser option.
  See `docs/adr/0016-opt-in-newton-corrections.md` for policy and qualification.
- Native `INPUT_FILE` is positional. `--output` defaults to
  `$PWD/cumes-output.bin`; `--boozer-output` is an alternative, mutually
  exclusive output. Known suffixes select compiled output backends; unknown
  suffixes and unavailable backends are errors.
- Configuration schema v1 is distinct from the native binary version (currently
  8) and checkpoint version (currently 6). Spectral state remains double on
  disk. Preserve reader compatibility and full provenance. Consult
  `docs/output-formats.md`, `configs/schema-v1.json`, and the readers/writers
  before changing serialization.

## Coding conventions

- Follow `.clang-format` for C++/CUDA and nearby conventions for JavaScript and
  WGSL. Avoid reformatting unrelated code. The installed pre-commit hook
  formats and re-stages whole staged C++/CUDA files; inspect the resulting
  diff, especially when a file also contains unrelated unstaged changes.
- Types: `PascalCase`; functions/variables: `snake_case`; constants and scoped
  enum values: `CAPITAL_SNAKE_CASE`. Keep established physics abbreviations
  such as `ns`, `mnmax`, `delt`, `fsqr`, `rmnc`, and `nZnT`.
- Scalar-templated types expose `using val_type = T;` as their first public
  member. Use descriptive aliases for secondary type parameters.
- Prefer RAII, `std::vector`, `std::span`, `std::string_view`, and
  `std::optional<std::reference_wrapper<T>>` for nullable borrowed values.
  Raw pointers belong at necessary device/C-library interop boundaries and
  existing device-view escape hatches. Use `d_`/`h_` for device/host pointers.
- Keep numerical expressions and ownership visible. Share setup and scratch
  when they are reused; add caching or abstraction to address a demonstrated
  need, not a hypothetical one.

## Validation matched to the change

- Documentation-only changes: check accuracy, paths, commands, and the diff;
  no solver rebuild is needed. For UI/bridge changes, run the relevant existing
  Node checks and browser smoke test. For shader/operator/controller changes,
  build the affected backend and run its relevant numerical gates. Shared
  physics or vacuum changes need coverage of each affected backend and
  precision.
- Existing CTest and browser harnesses are the starting point. Add regression
  tests for meaningful behavior or numerical defects, using analytic cases,
  invariants, and independent scalar references. Avoid tests that only repeat
  the implementation or pin incidental markup. Native tests are standalone
  executables; shared CUDA helpers live in
  `tests/include/cumes_test_cuda_helper.cuh`.
- Classify numerical changes using `docs/verification.md` §6: Class A preserves
  arithmetic and requires bitwise outputs/controller trajectories; Class B
  uses justified numerical error bounds with unchanged classification and
  controller decisions; Class C changes the algorithm and requires convergence,
  invariants, robustness, independent comparison, and an ADR. Compare the
  recorded baseline with matching inputs, flags, precision, and backend;
  historical CUDA-double iteration counts are not browser-float targets.
- Browser CTest checks do not replace executing WGSL on a real adapter.
  `scripts/webgpu_validate_run.mjs` handles Chrome conformance/solve checks;
  `scripts/webgpu_firefox_validate.mjs` uses local headless Firefox through
  geckodriver. Follow `docs/webgpu-port.md` for setup and numerical gates.
- Run GPU solves/benchmarks serially on each adapter. With the user's forwarded
  Chrome at `localhost:9333`, create tabs in the existing window, not new
  windows or isolated browser contexts. Preserve the user's tabs and settings;
  the browser gates redirect test setup to their tab's session storage.
  Use `CUMES_CLOSE_TEST_TAB=1` with the Chrome validator for cleanup. Apply
  Firefox test preferences only to a disposable profile. An empty adapter
  name does not itself mean WebGPU is unavailable.
- Once the relevant checks pass, broaden testing when shared-code impact,
  failures, unresolved concerns, or qualification requirements justify it.
  Report unavailable checks explicitly rather than implying they passed.
- For speed claims, compare warmed repeated runs on the same workload and
  configuration, retain correctness checks, and separate setup, iteration,
  and output time. State the hardware/browser and measurement variability.
  Use `docs/performance.md` for full performance qualification; a result on
  one adapter supports a claim for that measured setup only.

## Documentation to consult as needed

| Document | Use when working on |
| --- | --- |
| `docs/architecture.md`, `docs/library-api.md` | Operator ownership, dependency direction, embedding, tangents |
| `docs/mathematics.md`, `docs/data-layout.md` | Numerical formulas, Fourier/parity conventions, storage |
| `docs/output-formats.md`, `docs/dump-files.md` | Results, checkpoints, scientific fields, diagnostic dumps |
| `docs/verification.md`, `docs/performance.md` | Numerical equivalence gates, qualification, benchmarks |
| `docs/webgpu-port.md`, `webgpu/presets/README.md` | Browser integration, precision, browser testing, coil provenance |
| `docs/adr/0012-bspline-fixed-boundary-transfer.md` | Native multigrid transfer and its diagnostic alternatives |
| `docs/adr/0015-float-only-device-arithmetic.md`, `docs/w7x-float-float.md`, `docs/w7x-double-compensation.md` | Precision policy and qualified compensated geometry |
| `docs/adr/`, `docs/overhaul-history.md`, `docs/cuda-overhaul-blueprint.md` | Design decisions and historical implementation evidence |
