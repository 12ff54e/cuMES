# CLAUDE.md — cuMES (CUDA Magnetic Equilibrium Solver)

## Project Overview

cuMES is a pure-CUDA, ground-up reimplementation of the core VMEC stellarator
equilibrium algorithm. All computation runs on GPU; the CPU host is a thin
orchestrator. This is a pedagogical / scaffolding project — not production-grade,
but the architecture is real.

Reference implementation: `https://github.com/proximafusion/vmecpp` (CPU-based C++ VMEC solver) at tag 0.7.0.

The codebase completed a CUDA overhaul (blueprint: `docs/cuda-overhaul-blueprint.md`,
phases 0–11; see `docs/architecture.md` for the current shape and the
`docs/phase-*-handover.md` trail for history). Everything below describes the
post-overhaul code.

## Build & Run

```bash
# in folder cuMES
cmake --preset verify          # verify-double: precise math, all backends, -Werror
cmake --build build -j

# Run main solver (default input inputs/solovev.json; an output path is
# REQUIRED in strict mode — the default cumes_state.bin is compatibility-only)
./build/cuMES inputs/solovev.json out.bin        # positional: <input> <output>

# run tests (the full CTest suite, incl. compute-sanitizer memcheck AND
# initcheck variants; the count varies with the sanitizer/backend
# configuration, so no fixed number is documented)
ctest --test-dir build --output-on-failure

# full sanitizer matrix (dedicated preset): memcheck/initcheck/racecheck/
# synccheck compute-sanitizer variants of the kernel tests + host-only
# targets built with ASan+UBSan. Racecheck is slow; the variants are
# RUN_SERIAL in CTest.
cmake --preset sanitizer && cmake --build build-sanitize -j
ctest --test-dir build-sanitize

# single-precision build (mixed-float policy: float state + double reductions)
cmake --preset float && cmake --build build-float -j
# other named precision presets: fast (fast-double, opt-in --use_fast_math,
# dump machinery compiled out), debug (debug-double, precise + -G)
```

CLI: `--input <path>` / `--output <path>` (flags override the two positional
slots; positionals fill the first free slot), `--output-schema legacy-v0|v1`
(v1 = versioned container with full provenance for binary, NetCDF and HDF5),
`--restart <checkpoint>`, `--restart-legacy <six-family payload>`,
`--checkpoint <path>` (write a v1 checkpoint after solve; the container
records per-stage restart histories). Strict schema-v1 behavior is the
DEFAULT: unknown input keys are validation errors, an explicit output path is
required, and unknown suffixes are rejected — the named `--compatibility`
flag restores the vmecpp-style warn-and-ignore parsing, the cumes_state.bin
default, and the unknown-suffix fallback. A `.nc`/`.h5` suffix dispatches
to the host-only NetCDF/HDF5 writers when compiled in (the availability
preflight runs before any CUDA work). Non-fatal validation warnings (unknown
input keys in compatibility mode, skipped out-of-range boundary harmonics)
print as `cuMES: WARNING: ...` on stderr.
`CUMES_FORCE_GENERIC=1` forces the generic cuFFT transform backend on
axisymmetric shapes (default: the axisymmetric direct-poloidal backend).
`CUMES_MAX_ITER`/`CUMES_DELT0`/`CUMES_DUMP` are the env-gated run knobs.

**Requirements:** CUDA Toolkit >= 11, CMake >= 3.20, GPU compute capability >= 6.1.
If the host gcc is > 12, `CMAKE_CUDA_HOST_COMPILER` must point to g++-12 (set in
`CMakeLists.txt`).

**Precision:** every computation function is `template<typename T>` (double or
float); the executable's `Real` alias (include/vmec_types.h) is the compile-time
switch (`-DCUMES_USE_FLOAT=ON`). The tests instantiate both types in every build.
On-disk state files stay double (Python scripts unaffected); dump files are
T-native. Float runs stall at ~1e-7 residuals and never reach the hardcoded
ftol — float builds hard-error at startup when any `ftol_array` entry is below
1e-6, so relax the stage ftols for float experiments. The per-pass control
record (residuals, Jacobian stats, force-norm factors) is DOUBLE in both
builds (ADR-0001): the device norm reductions accumulate in double and the
host controller (`IterationController<double>`) sees them unrounded, while
the state/descent physics stays `T`. Device code is split into
explicit double/float instantiation TUs (`src/*_double.cu` / `src/*_float.cu`,
one scalar type per TU), so kernels may declare dynamic shared memory directly
as `extern __shared__ T[]` (the old non-templated `dynSharedBase()` indirection
was removed 2026-08-16).

**CUDA architectures:** 61 (Pascal), 75 (Turing), 80 (Ampere), 86, 89 (Ada).

## Directory Structure

```
cuMES/
├── CMakeLists.txt          Library split + executable/tests (see architecture.md §2)
├── include/
│   ├── vmec_types.h        `Real` alias + parity/basis convention comment
│   ├── solver.cuh          SolverResult<T> + solverRun declaration (thin app shim)
│   ├── output.cuh          outputSave/outputPrint declarations (app shim)
│   ├── fft_traits.h        FftTraits<T>: cuFFT type/enum/exec dispatch
│   ├── JsonParser.h        Legacy parser kept only for the netcdf/hdf5 v0 writers
│   └── cumes/              The operator library (see below)
├── src/
│   ├── main.cu             Entry point: validate config → multigrid stage loop → output
│   ├── <mod>_impl.cuh      Templated kernel bodies, one file per operator module
│   ├── <mod>_double.cu     Explicit double instantiation TU (cumes_cuda_double)
│   ├── <mod>_float.cu      Explicit float instantiation TU (cumes_cuda_float)
│   │                       modules: fourier geometry forces solver profiles precon
│   │                                constraint prolongation axisymmetric
│   ├── cumes/              Host-side C++: config, io, core, runtime
│   └── output*.cpp         Binary/NetCDF/HDF5 writers + format dispatcher
├── tests/                  Standalone correctness tests (no framework) + support/
├── benchmarks/             fixed_iteration + graph_overhead harnesses
├── scripts/                compare_runs.py / compare_states.py / compare_bitwise.py
├── inputs/                 solovev.json, w7x.json (vmecpp indata schema)
└── docs/                   Blueprint, mathematics, verification, architecture,
│                           ADRs, phase handovers
```

`include/cumes/`: `config` (ProblemSpec → ValidatedProblem, DeviceParams<T>,
validation), `core` (GridShape, checked arithmetic, Result), `io` (output
specs, checkpoint, versioned containers), `runtime` (DeviceBuffer/DeviceArena/
Stream/DeviceContext), `state` (SpectralStorage, RealSpaceStorage,
typed views), `transforms` (SpectralOperator interface + ToroidalFftOperator +
AxisymmetricOperator), `physics` (Geometry/MagneticField/Force/Constraint/
Profiles operators), `numerics` (Residual/Descent/Prolongation/Preconditioner +
tridiagonal backends), `solver` (EquilibriumOperator, IterationController,
StageSolver, MultigridSolver).

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

The whole per-iteration DAG is composed inside `EquilibriumOperator::enqueue`
(`src/solver_impl.cuh`); `solverRun` is a thin loop over the pure-host
`IterationController` + that DAG on one explicit compute stream, with one
deliberate host fence per iteration. All operators own their device buffers
(directly or via one `DeviceArena` carved per stage) and expose typed view
bundles; no legacy workspace structs remain.

## Key Design Decisions

| Decision                           | Rationale                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **cuFFT transforms**               | The transforms mirror vmecpp's FFTX structure: a batched 1D real FFT in the toroidal (ζ) direction plus direct poloidal synthesis/reduction; the constraint module (rCon/zCon, de-aliasing) reuses the same plans/scratch, with compact sub-batch plans (only the participating slots/modes/surfaces: 2·(mpol−2)·(ns−1) elements for the deAlias bandpass, 4·mpol·ns for rCon/zCon). W7-X on TITAN Xp: inverse 0.43 ms/iter, forward 0.39 ms/iter; full run 7.79 s→4.98 s after the 2026-08-03 pass (PCR tridiagonal solve, coalesced θ-major access, slot-split poloidal accumulation, sync removal). |
| **Column-major storage**           | Standard for cuBLAS; natural for per-surface indexing: `array[point + surface * nZnT]` for real space, `array[surface + mode * ns]` for spectral.                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| **Staggered half-grid**            | Dynamic variables on full grid (flux surfaces); metric elements on half grid (between surfaces). Prevents checkerboard instability. Matches VMEC convention.                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| **All GPU allocations at startup** | Scratch arrays allocated once, reused every iteration. Zero `cudaMalloc` calls in the hot loop.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| **Host checks convergence**        | Residual reduction runs on GPU, but the scalar comparison `fsq < ftol` happens on host.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **Precision via templates**        | All computation is `template<typename T>`; `Real` (vmec_types.h) + `-DCUMES_USE_FLOAT=ON` selects float, otherwise double. Double is the verified default — residuals pushed to 1e-14. Single precision stalls at ~1e-7 (float rounding floor), so float runs need a relaxed ftol. cuFFT dispatches through `FftTraits<T>` (D2Z/Z2D ↔ R2C/C2R).                                                                                                                                                                                                                                                            |
| **Explicit instantiation split**   | Each operator's kernels live in `src/<mod>_impl.cuh` included only by its `_double.cu`/`_float.cu` TUs (one scalar type per TU, linked as `cumes_cuda_double`/`cumes_cuda_float`). This makes direct `extern __shared__ T[]` legal in the templated kernels (see Precision).                                                                                                                                                                                                                                                                                                                            |
| **Config validation**              | The JSON input is parsed and validated host-side into a `ValidatedProblem` (`cumes-config-v1` normalized schema, `configs/schema-v1.json`); the solver consumes `ValidatedProblem` + per-stage `DeviceParams<T>` packs. No solver code parses input.                                                                                                                                                                                                                                                                                                                                                    |
| **Class A / Class B changes**      | Class A = bit-identical to the frozen trajectory (Solovev `251→199→456` FSQR 9.583e-17, W7-X `1877→1617→2011` FSQR 9.778e-13) — the regression oracle for every change. Class B = ULP-equivalent with identical controller decisions (iteration counts + restart sequence), a deliberate re-freeze. Verify with `scripts/compare_runs.py` + CTest.                                                                                                                                                                                                                                                      |

## Fourier Transform Details

### Parity convention (vmecpp m-parity)

Real-space arrays are split by **m parity**, matching vmecpp:

- **Even m** (m=0,2,4,...) → `e` arrays
- **Odd m** (m=1,3,5,...) → `o` arrays

Each parity array receives the FULL contribution from the mode:

- R: `rmncc*cos(mθ)cos(nζ) + rmnss*sin(mθ)sin(nζ)`
- Z: `zmnsc*sin(mθ)cos(nζ) + zmncs*(-cos(mθ)sin(nζ))`
- λ: `lmnsc*sin(mθ)cos(nζ)`

This is NOT a trigonometric-factor split (cos-cos vs sin-sin) — it is an
m-parity split. The Jacobian, metric, and force kernels all assume this convention
(matching vmecpp's `jacobian_kernel.h`, `metric_kernel.h`, `mhdforce_kernel.h`).

### Basis convention

- R: `cos(mθ - nζ)` (rmnc coefficients)
- Z: `sin(mθ - nζ)` (zmns coefficients)
- λ: `cos(mθ - nζ)` (lmnc coefficients)

### Mode numbering

`mode = m * (ntor+1) + n` with `m = 0..mpol-1`, `n = 0..ntor`
(folded n ≥ 0 basis; `mnmax = mpol*(ntor+1)`). Mode 0 is the (0,0) DC mode.
The physical `n` is `n*nfp` (the ζ grid covers one field period).

### Forward DFT normalization

The forward quadrature is vmecpp's reduced-grid trapezoid over θ ∈ [0, π]
(`nThetaRed = ntheta/2+1` points), weight `w = mscale*nscale/(nZeta*(nThetaRed-1))`
with endpoint halving and `mscale = √2 (m>0)`, `nscale = √2 (n>0)` — the
projection onto the orthonormal basis (NOT the `1/nZnT, 2/nZnT` round-trip
identity; the descent step re-applies `mfac*nfac`). The inverse DFT uses the
raw basis (no normalization — the cuMES state is plain physical; the λ state
additionally carries `mscale*nscale` inside the values).

### Transform implementation

12-slot packing per poloidal mode (vmecpp `kBatch`: rmkcc/rmkss + ζ-derivative
slots for R, Z, λ), batched 1D cuFFT D2Z/Z2D of length `nzeta` (batch
`12*mpol*ns`), direct poloidal accumulation/reduction over θ with small
per-mode tables (`d_cos_th` etc.) — structurally identical to vmecpp's
`fft_toroidal.cc`. The same plans and scratch (`d_zeta_spectra`/`d_zeta_real`)
serve the constraint module: rCon/zCon (xmpq-weighted reconstruction) and the
de-aliasing bandpass (full-grid sc/cs analysis → D2Z → normalized coefficients
→ Z2D synthesis). The original direct-sum kernels were removed after A/B
validation (commit 373172f^ has them).

### Derivative computation

Derivatives are computed analytically during the inverse DFT:

- `∂/∂θ cos(mθ-nζ) = -m sin(mθ-nζ)`
- `∂/∂θ sin(mθ-nζ) = +m cos(mθ-nζ)`
- `∂/∂ζ cos(mθ-nζ) = +n sin(mθ-nζ)`
- `∂/∂ζ sin(mθ-nζ) = -n cos(mθ-nζ)`

Derivatives are in "index space" (not physical radians). The Jacobian and metric
are consistent with this convention; only the absolute scaling is affected.

## Geometry

### Jacobian (half-grid)

The Jacobian is the parity-staggered expression from geometry.cu
(`jacobian_kernel.h` convention), NOT a direct `(1 + λ_θ)` formula:

```
tau    = tau1 + dSHalfDsInterp * tau2        (dSHalfDsInterp = 1/4)
tau1   = ru12 * zs - rs * zu12
tau2   = odd-parity correction keeping the Jacobian sign-stable
√g     = tau * r12                           (no signJ factor here)
dVdsH  = signJ * Σ √g * wInt                 (signJ = -1 applied at
                                               computeForceNormPartials)
```

### Metric (half-grid)

The stored metric is the **covariant** metric g_uu, g_uv, g_vv (no /g²
factor — the contravariant notation g^ij in the older docs was wrong):

- `g_uu = R_θ² + Z_θ²` (parity-mixed half-grid average)
- `g_uv = R_θ * R_ζ + Z_θ * Z_ζ` (3D only)
- `g_vv = R_ζ² + Z_ζ²` (plus the 3D toroidal part)

each with the even/odd `sFi²`/`sFo²`/`sH` parity weighting of
`geometryKernel` (mirroring vmecpp `metric_kernel.h`).

### Contravariant B (half-grid)

- `B^θ = (lamscale * λ_ζ + χ') / √g`
- `B^ζ = (lamscale * λ_θ + Φ') / √g`

(with the toroidal λ derivative stored as -∂λ/∂ζ; `χ' = chipH` for the
fixed-iota case, ncurr=0).

### Radial derivatives

`R_s = (R[j+1] - R[j]) / Δs` — normalised by the radial grid spacing.

## Solver: Garabedian Accelerated Descent

The solver uses a pseudo-time descent with momentum (second-order Richardson):

```
v_new = fac * (b1 * v_old + delt * f)
x_new = x_old + delt * v_new
```

**Adaptive damping:** `inv_tau` is the residual **log-ratio**
`min(|ln(fsq / fsq_prev)|, 0.15) / delt` (NOT the sqrt of the max residual —
the older docs were wrong), averaged over a 10-iteration window.
`b1 = 1 - dtau`, `fac = 1/(1 + dtau)`.

**Time-step control (vmecpp VMEC_8_52, in `IterationController`):** delt is
adapted at restarts and maintenance resets — ×0.9 on bad-Jacobian/nonfinite
passes, ÷1.03 on bad-progress, and 0.98/0.96×delt₀ at the ijacob 25/50
maintenance reset — matching vmecpp's Evolve control block. (The old docs'
"adaptive time-step — not yet" row described exactly this; it is implemented.)

**Convergence:** `max(fsqr, fsqz, fsql) < ftol` where `ftol = 1e-14`.

### What's intentionally omitted vs VMEC++

| VMEC++ feature                       | Status                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FFT-accelerated transforms           | cuFFT backend (batched 1D ζ-FFT + direct poloidal), mirroring vmecpp's FFTX structure                                                                                                                                                                                                                                                                                                                                                                      |
| Multigrid grid sequencing            | Implemented: per-config `ns_array`/`niter_array`/`ftol_array` stage loop (Solovev 5→11→55, W7-X 33→66→99), each stage seeded by the previous converged state via `Prolongation` — vmecpp `InterpolateToNextMultigridStep`, linear in s on scalxc-scaled odd-m coefficients, 2·x₁−x₂ axis extrapolation, odd-m zeroed at the axis, LCFS copied exactly. A stage that exhausts its cap without meeting ftol fails the run (vmecpp semantics) |
| De-aliased constraint force          | Implemented (spectral-condensation bandpass inside `ConstraintOperator`; the fused rCon/zCon synthesis lives in `ToroidalFftOperator::inverse_fused`)                                                                                                                                                                                                                                                                                                    |
| Hot restart / checkpointing          | Implemented (v1 checkpoint container: `--checkpoint` / `--restart`; legacy six-family payload: `--restart-legacy`)                                                                                                                                                                                                                                                                                                                                        |
| Free boundary / vacuum solver        | Fixed boundary only                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Mercier stability, jxbout, wout      | Post-processing; not needed for core loop                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Adaptive time-step (Jacobian resets) | Implemented — the restart/maintenance delt control above (vmecpp VMEC_8_52); no per-pass continuous adaptation (matches vmecpp)                                                                                                                                                                                                                                                                                                                            |
| Python interface                     | Not yet; C++/CUDA executable only                                                                                                                                                                                                                                                                                                                                                                                                                           |

## Coding Conventions

### Naming

- **Types:** `CamelCase` (e.g., `DeviceParams`, `ToroidalFftOperator`, `ValidatedProblem`)
- **Functions:** `camelCase` (e.g., `fourierCreate`, `computeGeometry`)
- **Variables:** `snake_case` (e.g., `d_rmnc`, `delta_s`, `nZnT`)
- **Constants:** `kCamelCase` (e.g., `kSignJacobian`, `kNsVal`, `kFtol`)
- **Device pointers:** `d_` prefix (e.g., `d_rmnc`, `d_gsqrt`)
- **Host pointers:** `h_` prefix (e.g., `h_rmnc`, `h_cos`)
- **Operators:** `cumes` namespace, `xCreate`/`xFree` replaced by RAII classes
  owning their buffers (`ToroidalFftOperator`, `GeometryOperator`, …)

### GPU Memory

- Column-major throughout: `index(point, surface) = point + surface * nZnT`
- All device allocations via RAII (`DeviceBuffer`/`DeviceArena`), error-checked
  through the centralized `cumes::check_cuda`/`check_cufft` in `cumes/runtime`
- New device code belongs behind an operator class with typed views; kernel
  bodies live in the module's `src/<mod>_impl.cuh`

### API pattern

- Each operator module has a `.hpp` header (types + declarations) under
  `include/cumes/<area>/` and its kernel bodies in `src/<mod>_impl.cuh`,
  explicitly instantiated by `src/<mod>_{double,float}.cu`
- Host-side modules are plain C++ under `src/cumes/`

### Tests

- Standalone executables, no framework dependency
- CPU reference implementation mirrors GPU kernel logic
- Pattern: create known input → run GPU → run CPU reference → compare
- Shared builders live in `tests/support/cumes_test_support.cuh`

## Status and Known Issues

**Status (2026-08-17): the CUDA overhaul is design-complete. Blueprint phases
0–11 finished the strangler-fig migration (legacy kernel structs and
`dynSharedBase()` removed, measured bit-identical); the four closure steps of
`docs/overhaul-completion-plan.md` then landed as separately reviewable
commits — 48713b2 numerical safety predicates, 4363e71 config/I-O contracts,
d602d2c runtime/performance policy, and the release gate (warnings-as-errors,
initcheck, CI, event-DAG tests, docs). Every step was verified **Class A
byte-identical** against the frozen baselines (including the
`--use_fast_math` removal: precise double math IS the frozen codegen), so no
re-freeze occurred anywhere. The post-implementation acceptance review's
loose ends (`docs/post-overhaul-follow-up.md`) are also closed: the
oriented-Jacobian first-sample fix + production regression, validated v1
restart offsets + corrupted fixtures, the refresh-pass terminal contract
closed via device-side force-norm finalization (`forceNormFinalizeKernel` —
convergence is no longer structurally disabled on refresh passes), the
repaired `scripts/ci_gpu.sh` oracle, precision-aware CLI fixtures, the
complete optional-backend preset/CI matrix, target-scoped precision flags
(deliberately NO `-march=native` on the CUDA TUs' host pass — the frozen
codegen never had it and adding it diverges the trajectory), and the checked
library-publication chain (`publishLibraryFile`) with fault-injection tests.
Both frozen trajectories were re-verified Class A byte-identical against the
`dc0d0c4` baseline after every change. The frozen baselines are:**

- Solovev: 251 → 199 → 456 effective iters, final FSQR 9.583e-17 — the
  final stage matches vmecpp's playground reference exactly (456, 9.99e-17).
- W7-X: 1877 → 1617 → 2011 effective iters (total 5505), final FSQR
  9.778e-13; vmecpp multigrid runs 1877 → 1635 → 2012. The converged final
  states agree at ~1e-5 in R/Z, ~1e-4 in the weakly-determined near-axis λ.
- The multigrid final state is a different member of the (near-degenerate)
  λ-gauge family than the single-grid-99 run: rmncc(0,1) differs by
  2.7e-4 vs the single-grid state/wout — and vmecpp's own single-grid vs
  multigrid states differ by exactly the same 2.7e-4 (intrinsic to the
  continuation, not a cuMES artifact). Restarting the solver from the
  multigrid final state converges at iter 1 (it is a genuine fixed point).
- Single-grid regression: with `n_grids=1` (ns_array={99}) the run is
  bit-identical to the pre-multigrid code (compare_runs.py PASS, 2953
  effective iters, FSQR 9.924e-13, state identical).
- The per-iteration residuals track vmecpp at ≤1e-8 over the ENTIRE run with an
  identical restart sequence, and the single-grid converged state matches the
  wout at ≤1.5e-9 in all six families including the λ gauge modes. Note:
  scripts/compare_converged_state.py must read the FULL-grid wout `lmns_full`,
  not the half-grid `lmns`.

1. **Axis representation (state-file only, real-space-irrelevant).** cuMES
   constant-extrapolates the axis row from j=1 (extrapolateAxisKernel — m=1
   all six families plus the m=0 lmncs "chi-force leftover", matching vmecpp's
   extrapolateTowardsAxis), so the dumped axis m>0 coefficients equal the j=1
   values (e.g. rmncc(1,0)@axis = 0.0497 for W7-X, 0.319 for Solovev), while
   vmecpp keeps them 0 (or s^(m/2)-extrapolated). The real-space axis geometry
   agrees (step_A verified at 1e-15), and the axis coefficients do not enter
   the forces, so this only shows up when diffing state files / wout axis rows.

2. **Float builds reject impossible tolerances.** Float runs stall at ~1e-7
   (float rounding floor), so the double-tuned stage ftols (1e-16/1e-12) can
   never be met. main hard-errors at startup when any `ftol_array` entry is
   below 1e-6 in a float build — for float experiments, relax the
   `ftol_array` entries to >= 1e-6 (and note `CUMES_MAX_ITER`/`CUMES_DELT0`
   override every stage's cap when set).

3. **Issue pass (2026-08-12).** The findings in `cuMES-issues.md` were
   verified and fixed: OOB reads (iotaH LCFS row, 5-vs-6 family test
   buffer), the >128-row PCR limit, the ntheta>32 de-alias drop, stale/
   unproduced combined force buffers, tcon0 propagation, gamma rejection,
   negative-m boundary modes, empty multigrid schedules, writer status
   propagation + backend preflight, degenerate-division guards, shuffle
   masks, ζ-tiled launch blocks, float tolerance rejection, dead-storage/
   cuBLAS removal, CTest registration, and the parser validation gaps
   (wide ints, angular caps, unknown-key warnings, aux/asym type checks).
   Both config regressions re-verified: Solovev 251→199→456 (FSQR 9.58e-17),
   W7-X 1877→1617→2011 (FSQR 9.78e-13).
