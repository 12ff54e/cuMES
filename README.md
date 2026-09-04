# cuMES — CUDA Magnetic Equilibrium Solver

Pure-CUDA, ground-up reimplementation of the core VMEC stellarator equilibrium
algorithm. All computation runs on GPU; the CPU host is a thin orchestrator.
This is a pedagogical / scaffolding project — not production-grade, but the
architecture and physics are real.

An experimental browser backend is available through Emscripten and
emdawnwebgpu. It provides a CUDA-free build and the complete fixed-boundary
iteration DAG for both axisymmetric and folded 3-D equilibria: transforms,
half-grid geometry, fixed-iota and prescribed-current magnetic closure,
radial/poloidal/toroidal force, spectral-condensation constraint, full `(m,n)`
preconditioner, accelerated descent, controller recovery, and multigrid
transfer. The default browser gate runs the controller-complete three-stage
Solovev solve with persistent velocity, constraint, preconditioner, and
rollback state. On the NVIDIA TITAN Xp through Dawn's Vulkan backend it
converges in `72 -> 31 -> 247` effective iterations.
The shipped W7-X case executes the same integrated path in the browser,
including its prescribed-current closure; its iteration-3 residual triple
matches native CUDA mixed-float at `(1.141e+01, 7.079e+00, 1.012e-01)`. A
dedicated `?solve=w7x` browser entry point runs all three W7-X stages; the
hardware-qualified run converges in `82 -> 35 -> 25` effective iterations.
The converged spectral state and run provenance are published as a version-8
native binary through a browser download link and verified by an in-Wasm
round trip. The download also contains the complete half/full-grid scientific
field block and is accepted by the standard plotting workflow.
See
[the WebGPU port status](docs/webgpu-port.md).

**Independent comparison implementation:** [`proximafusion/vmecpp`](https://github.com/proximafusion/vmecpp)
(CPU-based C++ VMEC solver) at tag 0.7.0. cuMES convergence is defined by its
own discrete force residuals and validity gates; vmecpp is a diagnostic
cross-check, not the convergence oracle (see [Verification](#verification)).

**Status: working.** The default fixed-boundary path uses shaped cold starts,
an axisymmetric start policy, asynchronously prepared B-spline multigrid
transfer, and qualified one-shot time-step recovery. The
prior audited trajectories remain available with `CUMES_SEED_ENVELOPE=0`,
`CUMES_AXISYM_LAMBDA_SEED=0`, `CUMES_DELT0=0.9`, and
`CUMES_DISABLE_STEP_RECOVERY=1`; the earlier Catmull-Rom and linear multigrid
transfers remain available with `CUMES_FORCE_CATMULL_PROLONGATION=1` and
`CUMES_FORCE_LINEAR_PROLONGATION=1`:

| case | multigrid stages | effective iters | final FSQR |
| ---- | ---------------- | --------------- | ---------- |
| Solovev (`inputs/solovev.json`) | 5 → 11 → 55 | 235 → 193 → 326 (754) | 9.973e-17 |
| W7-X (`inputs/w7x.json`) | 33 → 66 → 99 | 1315 → 1419 → 1372 (4106) | 9.997e-13 |

## Quick start

```bash
git submodule update --init --recursive
cmake --preset verify          # double precision, both backends, -Werror
cmake --build build -j
./build/cumes inputs/solovev.json --output out.bin
./build/cumes inputs/solovev.json --boozer-output boozer.bin
./build/deps/magnetic-coordinate/cumes-boozer out.bin --output boozer.nc
ctest --test-dir build --output-on-failure
```

The current WebGPU milestone builds separately and requires core WebGPU/WGSL
single precision:

```bash
source "/lustre/qzhong/emsdk/emsdk_env.sh"
export EM_CACHE="$PWD/../tmp/cumes-emscripten-cache"
emcmake cmake --preset webgpu
cmake --build --preset webgpu -j
ctest --preset webgpu
python3 -m http.server --directory ../tmp/cumes-build-webgpu/webgpu
# open http://localhost:8000/cumes_webgpu.html in a WebGPU-capable browser
# or run the complete folded W7-X path:
# http://localhost:8000/cumes_webgpu.html?solve=w7x
```

The page runs GPU/CPU conformance cases and prints `cuMES WebGPU self-test:
PASS` when dispatch and readback agree. The artifact test checks the generated
HTML/JavaScript/Wasm bundle; browser execution is the numerical gate.

The default build also links the `magnetic_coordinate` library into cuMES and
produces the standalone `cumes-boozer` converter from
`deps/magnetic-coordinate`. `--output PATH` writes the native PEST-like result;
`--boozer-output PATH` instead transforms cuMES's converged in-memory snapshot
and writes the Boozer result. These options are mutually exclusive, so one run
never publishes both forms. The standalone converter reads an existing
schema-v8 native binary equilibrium. Both Boozer paths run the same transform
and produce the mixed
`(s, theta_b, zeta)` representation; `zeta_b = zeta + nu`. Disable this
component with `-DCUMES_BUILD_MAGNETIC_COORDINATE=OFF`. If its submodule is
absent, configuration warns and continues without Boozer support.
The Boozer output suffix selects `.bin`, `.nc`, `.h5`, or `.hdf5`; optional
NetCDF/HDF5 libraries are detected at configure time. All three containers
store only the six real Fourier parity families, never complex coefficients.

`deps/vacuum-field` is optional. A fixed-boundary-only build neither configures
nor links it:

```bash
cmake --preset fixed-only
cmake --build build-fixed-only -j
```

The equivalent cache switch is `-DCUMES_USE_VACUUM_FIELD=OFF`. When enabled
(the default), a missing submodule is detected and support is disabled with a
CMake warning. A fixed-only executable rejects `lfreeb=true` during input
validation with instructions for enabling the dependency.

**Requirements:** CUDA Toolkit ≥ 11, CMake ≥ 3.20, GPU compute capability ≥ 6.1
(Pascal or newer). If the host gcc is > 12, `CMAKE_CUDA_HOST_COMPILER` must
point to g++-12 (set in `CMakeLists.txt`). Built CUDA architectures: 61
(Pascal), 75 (Turing), 80 (Ampere), 86, 89 (Ada). NetCDF and HDF5 are optional:
a plain `cmake -B build` detects them and continues with unavailable backends
disabled. Without NetCDF, binary output and in-memory MAKEGRID free-boundary
calculations remain available.

`deps/BSplineInterpolation` is a direct header-only submodule used to prepare
the fixed-boundary multigrid transfer matrices. It is deliberately separate
from the copy used internally by `deps/magnetic-coordinate`, so either
dependency can be updated or disabled independently. If the direct submodule
is absent, cuMES falls back to Catmull-Rom prolongation; the explicit build
switch is `-DCUMES_USE_BSPLINE_PROLONGATION=OFF`.

## Architecture

The solver is a single `cumes` operator library. The strangler-fig migration is
complete: the legacy `include/*.cuh` kernel structs are gone, and the production
kernels live directly in operator classes that own their device buffers (RAII
`DeviceBuffer`/`DeviceArena`) and expose typed view bundles.

- **Device operator modules** — `src/kernels/<mod>_impl.cuh` + explicit
  `double`/`float` instantiation TUs (`src/<mod>_{double,float}.cu`): `fourier`
  (`ToroidalFftOperator`), `geometry` (`GeometryOperator`/`MagneticFieldOperator`),
  `forces` (`ForceOperator`), `solver` (`EquilibriumOperator`), `profiles`,
  `precon`, `constraint`, `prolongation`, `axisymmetric`.
- **Host-side `cumes` namespace** — `include/cumes/*` + `src/cumes/*`: config
  validation, RAII CUDA runtime, typed non-owning views, the I/O stack, and the
  host orchestration (`MultigridSolver`, `StageSolver`, `IterationController`).

All GPU allocations happen at startup; there are zero `cudaMalloc` calls in the
hot loop. See [`docs/architecture.md`](docs/architecture.md) for the full shape.

## Data flow per iteration

```
Spectral coefs (rmnc, zmns, lmnc)              ← degrees of freedom
         │
         ▼  [ToroidalFftOperator::inverse_fused — inverse DFT + rCon/zCon]
Real-space geometry R, Z, λ + derivatives      (ns × ntheta × nzeta)
         │
         ▼  [GeometryOperator / MagneticFieldOperator]
Half-grid: √g, g_uu, g_uv, g_vv, B^θ, B^ζ    (ns-1 × ntheta × nzeta)
         │
         ▼  [FreeBoundaryOperator when lfreeb: CUDA NESTOR]
Vacuum LCFS field / pressure jump               (fixed-boundary path bypasses)
         │
         ▼  [ForceOperator]
Real-space forces F_R, F_Z, F_λ                (ns × ntheta × nzeta)
         │
         ▼  [ConstraintOperator (bandpass) + forward DFT]
Spectral forces                                 (3, mnmax, ns)
         │
         ▼  [ResidualOperator + Preconditioner + DescentOperator]
v = fac×(b1·v + delt·f) ,  x += delt·v        (Garabedian accelerated descent)
```

The whole per-iteration DAG is composed in `EquilibriumOperator::enqueue` on one
explicit compute stream, with a single deliberate host fence per iteration for
the convergence decision.

## Physics implemented (mirroring vmecpp)

- **MHD force balance** in flux coordinates with vmecpp's m-parity even/odd
  decomposition; staggered half-grid metric (`√g`, covariant `g_uu/g_uv/g_vv`,
  contravariant `B^θ`/`B^ζ`) between flux surfaces.
- **Hybrid λ-force** blending two `bsubv` interpolations with a radial
  `2·kPDamp·(1-s)` weight (kPDamp = 0.05).
- **Spectral-condensation constraint** (bandpass-filtered de-aliased `rCon −
  rCon0` force with vmecpp's `tcon` multiplier profile).
- **Preconditioning**: radial tridiagonal solve (LCFS row excluded) + λ
  preconditioner from flux-surface metric averages.
- **Iteration control** (`IterationController`): Garabedian accelerated descent
  with VMEC_8_52 damping, BAD_JACOBIAN / BAD_PROGRESS restarts, and `ijacob
  25/50` maintenance resets.
- **Multigrid grid sequencing** via `Prolongation` (linear-in-s on scalxc-scaled
  odd-m coefficients, axis extrapolation, LCFS copied exactly).

## Build & Run

### Precision presets

| preset | policy | notes |
| ------ | ------ | ----- |
| `verify` (default) | verify-double | precise double, NetCDF+HDF5, `-Werror` |
| `float` | mixed-float | float state/FFT + documented double reductions |
| `fast` | fast-double | opt-in `--use_fast_math`, dump machinery compiled out |
| `debug` | debug-double | precise + `-G` |
| `sanitizer` | verify-double | compute-sanitizer memcheck/initcheck/racecheck/synccheck + ASan/UBSan host twins |
| `webgpu` | mixed-float | Emscripten + emdawnwebgpu; experimental fixed-boundary axisymmetric/3-D solver |

The backend matrix (`nobackend`, `netcdf-only`, `hdf5-only`) rounds out the
optional-backend presets. Every computation is `template<typename T>`; `Real`
(`include/vmec_types.h`) is the compile-time switch (`-DCUMES_USE_FLOAT=ON`).

### CLI

```
./build/cumes [OPTION]... INPUT_FILE
```

- `INPUT_FILE` is mandatory and positional.
- `--output <path>` / `-o <path>` selects the result path; without it, cuMES
  writes `$PWD/cumes-output.bin`.
- Every backend writes the schema-v1 container — a versioned state payload with
  full provenance for binary, NetCDF and HDF5.
- `--restart <checkpoint>` / `-r <checkpoint>` and `--checkpoint <path>` /
  `-c <path>` read/write a v6 checkpoint; v1–v5 remain readable.
- Strict schema-v1 behavior is the **default**: unknown input keys are errors,
  and unknown output suffixes are rejected. The `--compatibility` flag restores
  vmecpp-style warn-and-ignore parsing for unknown input keys (input-side only;
  the output suffix policy stays strict).

A `.nc`/`.h5` suffix dispatches to the host-only NetCDF/HDF5 writers when
compiled in (the availability preflight runs before any CUDA work). Non-fatal
validation warnings print as `cuMES: WARNING: ...` on stderr.

### Input

`inputs/*.json` follow the vmecpp indata schema (flat keys: `mpol`, `ntor`,
`nfp`, `ns_array`/`niter_array`/`ftol_array` multigrid stages, `am`/`ac`/`ai`/
`aphi` profile coefficients, `raxis_c`/`zaxis_s`, and `rbc`/`zbs` boundary
harmonics). The JSON is parsed and validated host-side into a `ValidatedProblem`
(`cumes-config-v1` normalized schema, `configs/schema-v1.json`); no solver code
parses input.

Free-boundary input accepts either a precomputed `mgrid_file`, or `coils_file`
together with Makegrid parameters. `coils_file` may use legacy MAKEGRID
coils-dot geometry or strict `cumes-coils-v1` JSON, documented in
`deps/vacuum-field/docs/coil-formats.md`. The parameters may be supplied by
`makegrid_parameters_file`, or embedded as the complete parameter object under
the key `makegrid_parameters`. If both parameter keys are present, cuMES warns
and uses the embedded object. These paths construct the response table once in
memory before iteration; `extcur` remains in Amperes.
See `inputs/free_bdy/solovev_free_bdy_coils.json` and
`inputs/free_bdy/solovev_free_bdy_embedded.json`.

### Precision

- The default build is double (`Real = double`); residuals reach ~1e-14.
- Single precision (`float` preset) stalls at ~1e-7 (the float rounding floor):
  the double-tuned stage ftols can never be met, so float builds hard-error at
  startup unless the `ftol_array` entries are relaxed to ≥ 1e-6.
- On-disk state files stay double regardless of `T`; dump files are `T`-native.
- The per-pass control record (residuals, Jacobian stats, force-norm factors)
  is double in both builds; the device norm reductions accumulate in double.

### Environment variables

| Variable | Effect |
| -------- | ------ |
| `CUMES_FORCE_GENERIC` | `=1` forces the generic cuFFT backend on axisymmetric shapes (default: the axisymmetric direct-poloidal backend) |
| `CUMES_FORCE_CATMULL_PROLONGATION` | `=1` selects four-point Catmull-Rom coarse-to-fine transfer (the previous fixed-boundary default) |
| `CUMES_FORCE_LINEAR_PROLONGATION` | `=1` selects two-point linear coarse-to-fine transfer (default for axisymmetric free-boundary and float runs) |
| `CUMES_MAX_ITER` | iteration cap (overrides every stage's cap in a multigrid run) |
| `CUMES_DELT0` | absolute initial time-step override (bypasses qualified axisymmetric/free-boundary stage scaling) |
| `CUMES_DISABLE_STEP_RECOVERY` | `=1` disables qualified fixed-boundary time-step recovery (diagnostic reference trajectory) |
| `CUMES_SEED_ENVELOPE` | override cold-start shaping (fixed 3-D multigrid `0.12`, fixed 3-D single-grid `0.129`, free 3-D `0.12` through `ns=25` and `0.03` above, coarse fixed-axisymmetric `-0.07`; `0` restores the reference envelope) |
| `CUMES_AXISYM_LAMBDA_SEED` | override the axisymmetric geometric lambda predictor scale (fixed/free defaults `0.65`/`1.0`; `0` restores zero lambda) |
| `CUMES_VACUUM_ACTIVATION_THRESHOLD` | override the free-boundary vacuum handover residual sum (default `3e-2`; `1e-3` restores the reference gate) |
| `CUMES_DUMP` | enables debug/dump output |

## Verification

The frozen reference trajectories are cuMES's own audited baselines and serve as
the internal regression oracle (`build/compare_runs` +
`build/compare_bitwise` + CTest). vmecpp is used only as an independent
diagnostic comparison:

The four comparison tools are host-only C++ executables built by CMake. They
can also be built without configuring cuMES:

```bash
scripts/build_compare_tools.sh build/compare-tools
# A non-HDF5 tool is also a directly compilable translation unit:
g++ -std=c++20 scripts/compare_runs.cpp -o compare_runs
```

`compare_wout` reads vmecpp HDF5 output, so the standalone builder uses
`h5c++` or `pkg-config hdf5`; the other three tools need only the standard
library and the POSIX `sha256sum` subprocess used by `compare_bitwise`.

- **Solovev 5→11→55**: tuned axisymmetric start and cubic B-spline transfer
  give 235 → 193 → 326 effective iters, final FSQR 9.973e-17.
  `CUMES_FORCE_CATMULL_PROLONGATION=1` restores 235 → 190 → 341;
  `CUMES_FORCE_LINEAR_PROLONGATION=1` restores 235 → 193 → 387.
  `CUMES_SEED_ENVELOPE=0`,
  `CUMES_AXISYM_LAMBDA_SEED=0`, and `CUMES_DELT0=0.9` restore
  251 → 199 → 456 and FSQR 9.583e-17.
- **W7-X 33→66→99**: 1877 → 1617 → 2011 effective iters (total 5505), final
  FSQR 9.778e-13 for the diagnostic reference controller. The default
  recovery, shaped cold start, and cubic B-spline transfer converge in
  1315 → 1419 → 1372 iterations (total 4106), FSQR 9.997e-13, and a
  checkpoint restart remains below tolerance and converges at iteration 1.
  `CUMES_FORCE_CATMULL_PROLONGATION=1` restores the 4160-pass transfer
  trajectory;
  `CUMES_FORCE_LINEAR_PROLONGATION=1` restores the 4507-pass transfer
  trajectory. `CUMES_SEED_ENVELOPE=0`
  restores the recovery-only 4944-pass trajectory.
- **Single-grid W7-X**: with `n_grids=1` (`ns_array={99}`), one-shot recovery
  and the single-grid seed converge in 2465 effective iterations (down from
  2711 with the reference seed and 2953 with the reference controller), FSQR
  9.959e-13.
  `CUMES_SEED_ENVELOPE=0` restores the 2711-iteration recovery-only
  trajectory; additionally setting
  `CUMES_DISABLE_STEP_RECOVERY=1` restores the 2953-iteration reference
  trajectory.

See [`docs/verification.md`](docs/verification.md) for the full tier/gate
structure (equivalence classes, sanitizers, equivalence gates, performance
acceptance).

## Status vs VMEC++

| VMEC++ feature | Status |
| -------------- | ------ |
| FFT-accelerated transforms | cuFFT backend (batched 1D ζ-FFT + direct poloidal), mirroring vmecpp's FFTX structure |
| Multigrid grid sequencing | Implemented (`ns_array`/`niter_array`/`ftol_array` stage loop + `Prolongation`) |
| De-aliased constraint force | Implemented (spectral-condensation bandpass + fused rCon/zCon) |
| Hot restart / checkpointing | Implemented (`--checkpoint` / `--restart`) |
| Adaptive time-step (Jacobian resets) | Implemented (restart/maintenance delt control, vmecpp VMEC_8_52) |
| Free boundary / vacuum solver | Implemented: NESTOR with file-backed or in-memory Makegrid coil fields |
| Mercier stability, jxbout, wout | Post-processing; not needed for the core loop |
| Python interface | Not yet; C++/CUDA executable only |

## Tests

Standalone executables, no test framework; CPU reference mirrors GPU kernel
logic. Run them via `ctest --test-dir build` or directly from `build/`:

- **Transforms / geometry**: `test_fourier`, `test_geometry_iso`,
  `test_geometry_ncurr`, `test_axisym_backend`, `test_constraint_tcon`,
  `test_regression_kernels`
- **Physics**: `test_forces`, `test_force_reference`, `test_force_verify`
- **Numerics / solver**: `test_tridiagonal`, `test_controller`,
  `test_accumulation`, `test_safety_predicates`, `test_operator_views`
- **Runtime / graph**: `test_runtime`, `test_arena`, `test_event_dag`,
  `test_cuda_graph`
- **Config / I/O**: `test_input_json`, `test_host_config`, `test_host_io`,
  `test_checkpoint`, `test_io_golden`, `test_io_restart_offsets`,
  `test_io_malformed_shapes`, `test_output_failure`
- **CLI policy shells**: `cli_policy_test.sh`, `output_publication_test.sh`

Both `double` and `float` are instantiated in every build.

## Directory structure

```
cuMES/
├── CMakeLists.txt          Library split + executable/tests
├── include/
│   ├── vmec_types.h        `Real` alias + parity/basis convention
│   ├── solver.cuh / output.cuh   thin app shims
│   └── cumes/              the operator library (config, core, io, numerics,
│                           physics, runtime, solver, state, transforms)
├── src/
│   ├── main.cu             Entry point: validate config → multigrid → output
│   ├── kernels/            Templated kernel bodies (<mod>_impl.cuh), one file per operator
│   ├── <mod>_{double,float}.cu   explicit instantiation TUs
│   └── cumes/              Host-side C++ (config, io, core, runtime)
├── tests/                  Standalone correctness tests + support/
├── benchmarks/             fixed_iteration + graph_overhead harnesses
├── scripts/                four compare_*.cpp tools + standalone build script
├── inputs/                 solovev.json, w7x.json (vmecpp indata schema)
├── configs/                schema-v1.json (cumes-config-v1)
└── docs/                   architecture, mathematics, verification, ADRs, history
```

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — current architecture and build split
- [`docs/mathematics.md`](docs/mathematics.md) — normative numerical contracts
- [`docs/data-layout.md`](docs/data-layout.md) — storage/layout contracts
- [`docs/verification.md`](docs/verification.md) — verification tiers and gates
- [`docs/performance.md`](docs/performance.md) — measured performance + acceptance policy
- [`docs/adr/`](docs/adr/) — architecture decision records

## License

MIT — see [LICENSE](LICENSE).
