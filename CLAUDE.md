# CLAUDE.md — cuMES (CUDA Magnetic Equilibrium Solver)

## Project Overview

cuMES is a pure-CUDA, ground-up reimplementation of the core VMEC stellarator
equilibrium algorithm. All computation runs on GPU; the CPU host is a thin
orchestrator. This is a pedagogical / scaffolding project — not production-grade,
but the architecture is real.

Reference implementation: `https://github.com/proximafusion/vmecpp` (CPU-based
C++ VMEC solver) at tag 0.7.0 — cuMES's correctness reference (see Status).
The codebase is post-CUDA-overhaul (blueprint:
`docs/cuda-overhaul-blueprint.md`); details live in `docs/` — see the
[documentation map](#documentation-map).

## Build & Run

```bash
# in folder cuMES
cmake --preset verify          # verify-double: precise math, all backends, -Werror
cmake --build build -j

# an output path is REQUIRED — there is no default output file
./build/cumes inputs/solovev.json out.bin        # positional: <input> <output>

ctest --test-dir build --output-on-failure

# sanitizer preset: compute-sanitizer memcheck/initcheck/racecheck/synccheck
# variants of the kernel tests + ASan/UBSan host twins (racecheck RUN_SERIAL)
cmake --preset sanitizer && cmake --build build-sanitize -j
ctest --test-dir build-sanitize

# single precision (mixed-float: float state + double reductions)
cmake --preset float && cmake --build build-float -j
# other presets: fast (fast-double, opt-in --use_fast_math, dump machinery
# compiled out), debug (debug-double, precise + -G); optional-backend matrix:
# nobackend / netcdf-only / hdf5-only
```

**Requirements:** CUDA Toolkit >= 11.8 (C++20 host/device support), CMake >= 3.20,
GPU compute capability >= 6.1. The project-wide language standard is strict
C++20 — no GNU extensions — for host C++ and CUDA TUs alike (root
`CMakeLists.txt`). If the host gcc is > 12, `CMAKE_CUDA_HOST_COMPILER` must
point to g++-12 (set in `CMakeLists.txt`). CUDA architectures: 61 (Pascal),
75 (Turing), 80 (Ampere), 86, 89 (Ada).

## CLI & Environment

- Positional `<input> <output>`; `--input`/`--output` flags override the slot
  they name. Default input: `inputs/solovev.json`; the output path is REQUIRED.
- `--restart <checkpoint>` / `--checkpoint <path>` — read/write the v2
  checkpoint (`docs/output-formats.md` §4).
- Every backend writes the schema-v1 container (versioned binary/NetCDF/HDF5
  with full provenance; a `.nc`/`.h5` suffix dispatches to the host-only
  NetCDF/HDF5 writers when compiled in). Formats: `docs/output-formats.md`.
- Strict behavior is the DEFAULT: unknown input keys are validation errors and
  unknown suffixes are rejected. `--compatibility` restores vmecpp-style
  warn-and-ignore for unknown input keys (input-side only; the output policy
  stays strict). Warnings print as `cuMES: WARNING: ...` on stderr.

Environment variables:

| Variable | Effect |
| -------- | ------ |
| `CUMES_FORCE_GENERIC` | `=1` forces the generic cuFFT backend on axisymmetric shapes (default: the axisymmetric direct-poloidal backend) |
| `CUMES_MAX_ITER` | iteration cap; overrides every stage's cap in a multigrid run |
| `CUMES_DELT0` | initial time step |
| `CUMES_DTAU_FLOOR` | floor on the damping parameter dtau |
| `CUMES_DUMP` | master switch for dump/debug output |
| `CUMES_DUMP_ITER` / `CUMES_E2_START` | which iterations the windowed dump files fire on |

**Precision:** every computation is `template<typename T>` (double or float);
`Real` (include/vmec_types.h) is the compile-time switch
(`-DCUMES_USE_FLOAT=ON`). Tests instantiate both types in every build. On-disk
state stays double (Python scripts unaffected); dump files are T-native. Float
runs stall at ~1e-7 and hard-error at startup when any `ftol_array` entry is
below 1e-6 — relax the stage ftols for float experiments. The per-pass control
record (residuals, Jacobian stats, force-norm factors) is DOUBLE in both builds
(ADR-0001): device norm reductions accumulate in double and the host controller
(`IterationController<double>`) sees them unrounded.

## Directory Structure

```
cuMES/
├── CMakeLists.txt          Library split + executable/tests (architecture.md §2)
├── include/
│   ├── vmec_types.h        `Real` alias + parity/basis convention comment
│   ├── solver.cuh          SolverResult<T> + solver_run declaration (app shim)
│   ├── output.cuh          output_print declaration (app shim)
│   ├── fft_traits.h        FftTraits<T>: cuFFT type/enum/exec dispatch
│   ├── JsonParser.h        The JSON engine of the input config parser (cumes_config_json)
│   └── cumes/              The operator library (see below)
├── src/
│   ├── main.cu             Entry point: validate config → multigrid stage loop → output
│   ├── kernels/            Templated kernel bodies (<mod>_impl.cuh), one file per operator
│   ├── <mod>_{double,float}.cu   Explicit instantiation TUs (cumes_cuda_{double,float})
│   │                       modules: fourier geometry forces solver profiles precon
│   │                                constraint prolongation axisymmetric
│   ├── cumes/              Host-side C++ (config, io, core, runtime)
│   └── output*.cpp         Binary/NetCDF/HDF5 writers + format dispatcher
├── tests/                  Standalone correctness tests (no framework) + support/
├── benchmarks/             fixed_iteration + graph_overhead harnesses
├── scripts/                compare_runs.py / compare_states.py / compare_bitwise.py
├── inputs/                 solovev.json, w7x.json (vmecpp indata schema)
└── docs/                   See the documentation map below
```

`include/cumes/`: `config` (ProblemSpec → ValidatedProblem, DeviceParams<T>),
`core` (GridShape, checked arithmetic), `io` (output specs, checkpoint,
versioned containers), `runtime` (DeviceBuffer/DeviceArena/Stream), `state`
(spectral/real-space storages, typed views), `transforms` (SpectralOperator +
ToroidalFftOperator + AxisymmetricOperator), `physics`
(Geometry/MagneticField/Force/Constraint/Profiles), `numerics`
(Residual/Descent/Prolongation/Preconditioner + tridiagonal backends), `solver`
(EquilibriumOperator, IterationController, StageSolver, MultigridSolver).

## Architecture: Data Flow per Iteration

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

The per-iteration DAG is composed in `EquilibriumOperator::enqueue`
(`src/kernels/solver_impl.cuh`); `solver_run` is a thin loop over the pure-host
`IterationController` + that DAG on one explicit compute stream, with one
deliberate host fence per iteration. All operators own their device buffers
(directly or via one `DeviceArena` carved per stage) and expose typed view
bundles; no legacy workspace structs remain.

## Key Design Decisions

- **cuFFT transforms** — batched 1D real FFT in the toroidal (ζ) direction +
  direct poloidal synthesis/reduction; the constraint module (rCon/zCon,
  de-aliasing) reuses the same plans/scratch with compact sub-batch plans
  (architecture.md §3).
- **Column-major storage** — `array[point + surface*nZnT]` for real space,
  `array[surface + mode*ns]` for spectral.
- **Staggered half-grid** — dynamic variables on full grid (flux surfaces);
  metric elements on half grid (between surfaces). Prevents checkerboard
  instability. Matches VMEC convention.
- **All GPU allocations at startup** — scratch arrays allocated once, reused
  every iteration. Zero `cudaMalloc` calls in the hot loop.
- **Host checks convergence** — residual reduction runs on GPU; the scalar
  comparison `fsq < ftol` happens on host.
- **Precision via templates** — `T` = double (verified default; residuals to
  ~1e-14) or float (stalls at ~1e-7; relax ftols). cuFFT dispatches through
  `FftTraits<T>`.
- **Explicit instantiation split** — each operator's kernels live in
  `src/kernels/<mod>_impl.cuh`, included only by its `_double.cu`/`_float.cu` TUs (one
  scalar type per TU), so kernels may declare dynamic shared memory directly
  as `extern __shared__ T[]`.
- **Config validation** — the JSON input is parsed and validated host-side into
  a `ValidatedProblem` (`cumes-config-v1` schema, `configs/schema-v1.json`);
  no solver code parses input.
- **Class A / Class B changes** — Class A = bit-identical to cuMES's own frozen
  trajectory (Solovev `251→199→456` FSQR 9.583e-17, W7-X `1877→1617→2011`
  FSQR 9.778e-13) — the internal regression oracle for every refactor,
  independent of any vmecpp bit-exactness target. Class B = ULP-equivalent
  with identical controller decisions (iteration counts + restart sequence),
  a deliberate re-freeze. Verify with `scripts/compare_runs.py` + CTest
  (verification.md §6).

## Coding Conventions

### Naming

- **Types:** `PascalCase` (e.g., `DeviceParams`, `ToroidalFftOperator`)
- **Functions:** `snake_case` (e.g., `solver_run`, `eval_two_power`)
- **Variables:** `snake_case` (e.g., `d_rmnc`, `delta_s`, `nZnT`);
  compact physics/Fortran-derived abbreviations are exempt and stay as-is
  (`ns`, `mnmax`, `delt`, `dtau`, `fsqr`, `rmnc`, `nZnT`, `jF`-style index
  names)
- **Constants:** `CAPITAL_SNAKE_CASE` (e.g., `SIGN_JACOBIAN`, `MU_0`),
  including scoped-enum values (`VERIFY_DOUBLE`, `NONE`, `OK`)
- **Templated types:** every templated struct/class aliases its scalar type
  parameter as `using val_type = T;` (first public member; secondary type
  params get descriptive aliases like `error_type`)
- **Host pointers:** raw pointers in host code only where absolutely
  necessary (CUDA/C-library interop, `main(argc, argv)` plumbing,
  `SpectralStorage::family_ptr()`/`state_slab()` device escape hatches, and
  device-side kernel/operator members). Everything else: `std::vector`/
  `std::span`/`std::string_view`, `std::optional<std::reference_wrapper<T>>`
  for nullable params, `DeviceBuffer` for test-harness device allocations
- **Device pointers:** `d_` prefix; **host pointers:** `h_` prefix
- **Operators:** `cumes` namespace, RAII classes owning their buffers

### GPU Memory

- Column-major throughout: `index(point, surface) = point + surface * nZnT`
- All device allocations via RAII (`DeviceBuffer`/`DeviceArena`), error-checked
  through the centralized `cumes::check_cuda`/`check_cufft` in `cumes/runtime`
- New device code belongs behind an operator class with typed views; kernel
  bodies live in the module's `src/kernels/<mod>_impl.cuh`

### API pattern

- Each operator module has a `.hpp` header (types + declarations) under
  `include/cumes/<area>/` and kernel bodies in `src/kernels/<mod>_impl.cuh`,
  explicitly instantiated by `src/<mod>_{double,float}.cu`
- Host-side modules are plain C++ under `src/cumes/`

### Tests

- Standalone executables, no framework dependency
- CPU reference implementation mirrors GPU kernel logic
- Pattern: create known input → run GPU → run CPU reference → compare
- Shared builders live in `tests/support/cumes_test_support.cuh`

## Status

The CUDA overhaul is design-complete (phases 0–11 + the four closure steps +
post-overhaul follow-up and reader-rank hardening, all re-verified Class A
byte-identical against the frozen `dc0d0c4` baseline — full record in
`docs/overhaul-history.md`). The frozen baselines (cuMES's own audited
reference outputs, independent of any vmecpp bit-exactness target):

- Solovev: 251 → 199 → 456 effective iters, final FSQR 9.583e-17.
- W7-X: 1877 → 1617 → 2011 effective iters (total 5505), final FSQR 9.778e-13.
  The converged final states agree with the vmecpp/wout reference at ~1e-5 in
  R/Z, ~1e-4 in the weakly-determined near-axis λ.
- Single-grid regression (`n_grids=1`, ns_array={99}): 2953 effective iters,
  FSQR 9.924e-13. Per-iteration residuals track vmecpp at ≤1e-8 over the
  ENTIRE run; the converged state matches the wout at ≤1.5e-9 in all six
  families (wout comparisons must read the FULL-grid `lmns_full`, not the
  half-grid `lmns`).
- The multigrid final state is a different member of the (near-degenerate)
  λ-gauge family than the single-grid run (~2.7e-4 in rmncc(0,1)) — intrinsic
  to the continuation, not a cuMES artifact. Restarting from the multigrid
  final state converges at iter 1 (a genuine fixed point).

Known issues:

1. **Axis representation (state-file only).** The dumped axis m>0 coefficients
   are constant-extrapolated from j=1 (`extrapolate_axis_kernel`), so they equal
   the j=1 values (vmecpp keeps them 0). The real-space axis geometry agrees
   (1e-15) and axis coefficients do not enter the forces — this shows up only
   when diffing state files / wout axis rows.
2. **Float builds reject impossible tolerances.** Float stalls at ~1e-7, so
   double-tuned stage ftols (1e-16/1e-12) can never be met — relax
   `ftol_array` entries to >= 1e-6 for float experiments.

## Scope (vs VMEC++)

| Feature | Status |
| ------- | ------ |
| FFT-accelerated transforms | Implemented (cuFFT: batched 1D ζ-FFT + direct poloidal) |
| Multigrid grid sequencing | Implemented (`ns_array`/`niter_array`/`ftol_array` stage loop + `Prolongation`) |
| De-aliased constraint force | Implemented (bandpass inside `ConstraintOperator`, fused rCon/zCon in `inverse_fused`) |
| Hot restart / checkpointing | Implemented (v2 checkpoint: `--checkpoint` / `--restart`) |
| Adaptive time-step (Jacobian resets) | Implemented (restart/maintenance delt control, vmecpp VMEC_8_52) |
| Free boundary / vacuum solver | Not implemented — fixed boundary only |
| Mercier stability, jxbout, wout | Not implemented — post-processing, not needed for the core loop |
| Python interface | Not implemented — C++/CUDA executable only |

## Documentation Map

| Document | Contents |
| -------- | -------- |
| `docs/architecture.md` | operator library, build/library split, per-iteration pipeline, dependency rules |
| `docs/mathematics.md` | normative numerical contracts: coordinates, Fourier representation/quadrature, geometry, fields, force, constraint, preconditioner, damping/descent, prolongation |
| `docs/data-layout.md` | storage/layout contracts (state, real-space, quadrature) |
| `docs/output-formats.md` | on-disk containers: v1 binary/checkpoint/NetCDF/HDF5, Python reader |
| `docs/dump-files.md` | the `CUMES_DUMP` diagnostics: file manifest, formats, naming scheme |
| `docs/verification.md` | verification tiers/gates, equivalence classes (Class A/B/C), review checklist |
| `docs/performance.md` | measured performance + acceptance policy |
| `docs/overhaul-history.md` | phase-by-phase overhaul record and closeout handovers |
| `docs/cuda-overhaul-blueprint.md` | the original overhaul plan |
| `docs/adr/` | architecture decision records |
