# AGENTS.md — cuMES (CUDA Magnetic Equilibrium Solver)

## Project Overview

cuMES is a pure-CUDA, ground-up reimplementation of the core VMEC stellarator
equilibrium algorithm. All computation runs on GPU; the CPU host is a thin
orchestrator. This is a pedagogical / scaffolding project — not production-grade,
but the architecture is real.

Reference implementation: `https://github.com/proximafusion/vmecpp` (CPU-based C++ VMEC solver) at tag 0.7.0.

## Build & Run

```bash
# in folder cuMES
cmake -B build -G Ninja
cmake --build build -j

# Run main solver
./build/cuMES

# run tests
./build/test_fourier

# single-precision build (all computation templated on T; Real = float)
cmake -B build-float -G Ninja -DCUMES_USE_FLOAT=ON
cmake --build build-float -j
```

**Requirements:** CUDA Toolkit >= 11, CMake >= 3.20, GPU compute capability >= 6.1.
If the host gcc is > 12, `CMAKE_CUDA_HOST_COMPILER` must point to g++-12 (set in
`CMakeLists.txt`).

**Precision:** every computation function is `template<typename T>` (double or
float); the executable's `Real` alias (vmec_types.h) is the compile-time switch
(`-DCUMES_USE_FLOAT=ON`). The tests instantiate both types in every build.
On-disk state files stay double (Python scripts unaffected); dump files are
T-native. Float runs stall at ~1e-7 residuals and never reach the hardcoded
ftol — relax ftol for float experiments. Note: nvcc rejects `extern __shared__`
arrays in function templates instantiated with different element types in one
TU — the dynamic shared-memory base is routed through the non-templated
`dynSharedBase()` helper (see the comment at the top of each .cu that uses it).

**CUDA architectures:** 61 (Pascal), 75 (Turing), 80 (Ampere), 86, 89 (Ada).

## Directory Structure

```
cuMES/
├── CMakeLists.txt          Build config (two targets: cuda_vmec + test_fourier)
├── README.md               Architecture overview, data flow, omitted features
├── include/                Public headers
│   ├── vmec_types.h        Shared GPU data structures (GridParams, SpectralState, …)
│   ├── input.h             Hardcoded Solovev boundary + profile functions
│   ├── fourier.cuh         DFT plan, inverseDFT / forwardDFT declarations
│   ├── geometry.cuh        MetricWorkspace, computeGeometry declaration
│   ├── forces.cuh          computeForces declaration
│   ├── profiles.cuh        profilesCreate / profilesFree declarations
│   ├── solver.cuh          SolverResult, solverRun declaration
│   ├── output.cuh          outputPrint declaration
│   └── refine.cuh          interpolateState declaration (grid sequencing)
├── src/                    Implementation files (.cu = CUDA C++)
│   ├── main.cu             Entry point: multigrid stage loop → solve → output
│   ├── fourier.cu          cuFFT inverse/forward transform kernels (12-slot ζ-FFT)
│   ├── geometry.cu         Jacobian, metric g^ij, contravariant B on half-grid
│   ├── forces.cu           MHD force residuals (brmn/bzmn/crmn/czmn/blmn/clmn)
│   ├── profiles.cu         Radial profile evaluation + GPU upload
│   ├── solver.cu           Fixed-point loop with Garabedian accelerated descent
│   ├── refine.cu           interpolateState: coarse→fine grid state interpolation
│   └── output.cu           Copy results from GPU → print
└── tests/
    └── test_fourier.cu     Standalone correctness tests (no framework)
```

## Architecture: Data Flow per Iteration

```
Spectral coefs (rmnc, zmns, lmnc)              ← degrees of freedom
         │
         ▼  [inverse DFT — GPU kernel]
Real-space geometry R, Z, λ + derivatives      (ns × ntheta × nzeta)
         │
         ▼  [geometry kernel]
Half-grid: √g, g_uu, g_uv, g_vv, B^θ, B^ζ    (ns-1 × ntheta × nzeta)
         │
         ▼  [forces kernel]
Real-space forces F_R, F_Z, F_λ                (ns × ntheta × nzeta)
         │
         ▼  [forward DFT — GPU kernel]
Spectral forces                                 (3, mnmax, ns)
         │
         ▼  [descent kernel — Garabedian step]
v = fac×(b1·v + delt·f) ,  x += delt·v
```

## Key Design Decisions

| Decision                           | Rationale                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **cuFFT transforms**               | The transforms mirror vmecpp's FFTX structure: a batched 1D real FFT in the toroidal (ζ) direction plus direct poloidal synthesis/reduction; the constraint module (rCon/zCon, de-aliasing) reuses the same plans/scratch, with compact sub-batch plans (only the participating slots/modes/surfaces: 2·(mpol−2)·(ns−1) elements for the deAlias bandpass, 4·mpol·ns for rCon/zCon). W7-X on TITAN Xp: inverse 0.43 ms/iter, forward 0.39 ms/iter; full run 7.79 s→4.98 s after the 2026-08-03 pass (PCR tridiagonal solve, coalesced θ-major access, slot-split poloidal accumulation, sync removal). |
| **Column-major storage**           | Standard for cuBLAS; natural for per-surface indexing: `array[point + surface * nZnT]` for real space, `array[surface + mode * ns]` for spectral.                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| **Staggered half-grid**            | Dynamic variables on full grid (flux surfaces); metric elements on half grid (between surfaces). Prevents checkerboard instability. Matches VMEC convention.                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| **All GPU allocations at startup** | Scratch arrays allocated once, reused every iteration. Zero `cudaMalloc` calls in the hot loop.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| **Host checks convergence**        | Residual reduction runs on GPU, but the scalar comparison `fsq < ftol` happens on host.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **Precision via templates**        | All computation is `template<typename T>`; `Real` (vmec_types.h) + `-DCUMES_USE_FLOAT=ON` selects float, otherwise double. Double is the verified default — residuals pushed to 1e-14. Single precision stalls at ~1e-7 (float rounding floor), so float runs need a relaxed ftol. cuFFT dispatches through `FftTraits<T>` (D2Z/Z2D ↔ R2C/C2R).                                                                                                                                                                                                                                                        |

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

**Convergence:** `max(fsqr, fsqz, fsql) < ftol` where `ftol = 1e-14`.

### What's intentionally omitted vs VMEC++

| VMEC++ feature                       | Status                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FFT-accelerated transforms           | cuFFT backend (batched 1D ζ-FFT + direct poloidal), mirroring vmecpp's FFTX structure                                                                                                                                                                                                                                                                                                                                                                      |
| Multigrid grid sequencing            | Implemented: per-config `ns_array`/`niter_array`/`ftol_array` stage loop (Solovev 5→11→55, W7-X 33→66→99), each stage seeded by the previous converged state via `interpolateState` (refine.cu) — vmecpp `InterpolateToNextMultigridStep`, linear in s on scalxc-scaled odd-m coefficients, 2·x₁−x₂ axis extrapolation, odd-m zeroed at the axis, LCFS copied exactly. A stage that exhausts its cap without meeting ftol fails the run (vmecpp semantics) |
| Free boundary / vacuum solver        | Fixed boundary only                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Mercier stability, jxbout, wout      | Post-processing; not needed for core loop                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Hot restart / checkpointing          | Not yet                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| Adaptive time-step (Jacobian resets) | Fixed step; add when convergence is poor                                                                                                                                                                                                                                                                                                                                                                                                                   |
| De-aliased constraint force          | Not yet                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| Python interface                     | Not yet; C++/CUDA executable only                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Input file parsing                   | Hardcoded in `input.h`                                                                                                                                                                                                                                                                                                                                                                                                                                     |

## Coding Conventions

### Naming

- **Types:** `CamelCase` (e.g., `GridParams`, `SpectralState`, `FourierPlan`)
- **Functions:** `camelCase` (e.g., `fourierCreate`, `computeGeometry`)
- **Variables:** `snake_case` (e.g., `d_rmnc`, `delta_s`, `nZnT`)
- **Constants:** `kCamelCase` (e.g., `kSignJacobian`, `kNsVal`, `kFtol`)
- **Device pointers:** `d_` prefix (e.g., `d_rmnc`, `d_gsqrt`)
- **Host pointers:** `h_` prefix (e.g., `h_rmnc`, `h_cos`)

### GPU Memory

- Column-major throughout: `index(point, surface) = point + surface * nZnT`
- All device allocations via `cudaMalloc`, freed with `cudaFree`
- Error checking: always check `cudaError_t` return values

### API pattern

- `xCreate(params)` — allocates GPU memory, returns struct
- `xFree(struct)` — frees all GPU allocations
- Each module has a `.cuh` header (types + declarations) and a `.cu` source

### Tests

- Standalone executables, no framework dependency
- CPU reference implementation mirrors GPU kernel logic
- Pattern: create known input → run GPU → run CPU reference → compare

## Known Issues / Next Steps

\*\*Status (2026-08-07): grid sequencing is implemented and verified against
vmecpp. Both configs now run multi-stage by default (Solovev 5→11→55,
W7-X 33→66→99, mirroring the reference JSONs), each stage seeded by the
previous converged state via `interpolateState` (refine.cu). Verification
vs vmecpp's own multigrid runs:

- Solovev: 251 → 199 → 456 effective iters, final FSQR 9.58e-17 — the
  final stage matches vmecpp's playground reference exactly (456, 9.99e-17).
- W7-X: 1877 → 1617 → 2011 effective iters (total 5505), final FSQR
  9.78e-13; vmecpp multigrid runs 1877 → 1635 → 2012. The converged final
  states agree at ~1e-5 in R/Z, ~1e-4 in the weakly-determined near-axis λ.
- The multigrid final state is a different member of the (near-degenerate)
  λ-gauge family than the single-grid-99 run: rmncc(0,1) differs by
  2.7e-4 vs the single-grid state/wout — and vmecpp's own single-grid vs
  multigrid states differ by exactly the same 2.7e-4 (intrinsic to the
  continuation, not a cuMES artifact). Restarting the solver from the
  multigrid final state converges at iter 1 (it is a genuine fixed point).
- Single-grid regression: with `n_grids=1` (ns_array={99}) the run is
  bit-identical to the pre-multigrid code (compare_runs.py PASS, 2953
  effective iters, FSQR 9.924e-13, state identical).\*\*
  (The 2026-08-02 status below remains true for single-grid runs: the
  per-iteration residuals track vmecpp at ≤1e-8 over the ENTIRE run with an
  identical restart sequence — BJs iter2 3,5,8,11,15; BPs 51,64,78,91 — and
  the single-grid converged state matches the wout at ≤1.5e-9 in all six
  families including the λ gauge modes. Note: scripts/compare_converged_state.py
  must read the FULL-grid wout `lmns_full`, not the half-grid `lmns`.)

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
   never be met. main now hard-errors at startup when any `ftol_array` entry
   is below 1e-6 in a float build — for float experiments, relax the
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
