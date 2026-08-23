# cuMES — CUDA Magnetic Equilibrium Solver

Pure-CUDA, ground-up reimplementation of the core VMEC stellarator equilibrium
algorithm. All computation runs on GPU; the CPU host is a thin orchestrator.
This is a pedagogical / scaffolding project — not production-grade, but the
architecture and physics are real.

**Reference implementation:** [`proximafusion/vmecpp`](https://github.com/proximafusion/vmecpp)
(CPU-based C++ VMEC solver) at tag 0.7.0. vmecpp is cuMES's correctness
reference (see [Verification](#verification)).

**Status: working.** The solver reproduces its two frozen reference trajectories
(cuMES's own audited baselines, re-derived after the CUDA overhaul):

| case | multigrid stages | effective iters | final FSQR |
| ---- | ---------------- | --------------- | ---------- |
| Solovev (`inputs/solovev.json`) | 5 → 11 → 55 | 251 → 199 → 456 | 9.583e-17 |
| W7-X (`inputs/w7x.json`) | 33 → 66 → 99 | 1877 → 1617 → 2011 (5505) | 9.778e-13 |

## Quick start

```bash
cmake --preset verify          # double precision, both backends, -Werror
cmake --build build -j
./build/cumes inputs/solovev.json out.bin    # positional: <input> <output>
ctest --test-dir build --output-on-failure
```

**Requirements:** CUDA Toolkit ≥ 11, CMake ≥ 3.20, GPU compute capability ≥ 6.1
(Pascal or newer). If the host gcc is > 12, `CMAKE_CUDA_HOST_COMPILER` must
point to g++-12 (set in `CMakeLists.txt`). Built CUDA architectures: 61
(Pascal), 75 (Turing), 80 (Ampere), 86, 89 (Ada).

## Architecture

The solver is a single `cumes` operator library. The strangler-fig migration is
complete: the legacy `include/*.cuh` kernel structs are gone, and the production
kernels live directly in operator classes that own their device buffers (RAII
`DeviceBuffer`/`DeviceArena`) and expose typed view bundles.

- **Device operator modules** — `src/<mod>_impl.cuh` + explicit
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

The backend matrix (`nobackend`, `netcdf-only`, `hdf5-only`) rounds out the
optional-backend presets. Every computation is `template<typename T>`; `Real`
(`include/vmec_types.h`) is the compile-time switch (`-DCUMES_USE_FLOAT=ON`).

### CLI

```
./build/cumes [--input <path>] [--output <path>] [flags]
```

- Positionals fill the two slots `<input> <output>`; the `--input`/`--output`
  flags override them.
- Every backend writes the schema-v1 container — a versioned state payload with
  full provenance for binary, NetCDF and HDF5.
- `--restart <checkpoint>` / `--checkpoint <path>` (read/write a v1 checkpoint).
- Strict schema-v1 behavior is the **default**: unknown input keys are errors,
  an explicit output path is required, and unknown suffixes are rejected. The
  `--compatibility` flag restores vmecpp-style warn-and-ignore parsing for
  unknown input keys (input-side only; the output policy stays strict).

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
| `CUMES_MAX_ITER` | iteration cap (overrides every stage's cap in a multigrid run) |
| `CUMES_DELT0` | initial time step |
| `CUMES_DUMP` | enables debug/dump output |

## Verification

The frozen reference trajectories are cuMES's own audited baselines and serve as
the internal regression oracle (`scripts/compare_runs.py` +
`scripts/compare_bitwise.py` + CTest). vmecpp is used only as an independent
correctness reference:

- **Solovev 5→11→55**: 251 → 199 → 456 effective iters, final FSQR 9.583e-17.
- **W7-X 33→66→99**: 1877 → 1617 → 2011 effective iters (total 5505), final
  FSQR 9.778e-13. The converged final state agrees with the vmecpp/wout
  reference at ~1e-5 in R/Z and ~1e-4 in the weakly-determined near-axis λ.
- **Per-iteration fidelity (W7-X, single-grid)**: invariant residuals track
  vmecpp at ≤1e-8 over the entire run; the converged state matches the wout at
  ≤1.5e-9 in all six families.
- **Single-grid regression**: with `n_grids=1` (`ns_array={99}`) the run
  converges at 2953 effective iters, FSQR 9.924e-13.

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
| Free boundary / vacuum solver | Fixed boundary only |
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
│   ├── <mod>_impl.cuh      Templated kernel bodies (one file per operator)
│   ├── <mod>_{double,float}.cu   explicit instantiation TUs
│   └── cumes/              Host-side C++ (config, io, core, runtime)
├── tests/                  Standalone correctness tests + support/
├── benchmarks/             fixed_iteration + graph_overhead harnesses
├── scripts/                compare_runs.py / compare_states.py / compare_bitwise.py
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
