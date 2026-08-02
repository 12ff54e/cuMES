# CLAUDE.md — cuMES (CUDA Magnetic Equilibrium Solver)

## Project Overview

cuMES is a pure-CUDA, ground-up reimplementation of the core VMEC stellarator
equilibrium algorithm. All computation runs on GPU; the CPU host is a thin
orchestrator. This is a pedagogical / scaffolding project — not production-grade,
but the architecture is real.

Reference implementation: `../vmecpp` (CPU-based C++ VMEC++ solver).

## Build & Run

```bash
# in folder cuMES
cmake -B build -G Ninja
cmake --build build -j

# Run main solver
./build/cuMES

# run tests
./build/test_fourier
```

**Requirements:** CUDA Toolkit >= 11, CMake >= 3.20, GPU compute capability >= 6.1.
If the host gcc is > 12, `CMAKE_CUDA_HOST_COMPILER` must point to g++-12 (set in
`CMakeLists.txt`).

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
│   └── output.cuh          outputPrint declaration
├── src/                    Implementation files (.cu = CUDA C++)
│   ├── main.cu             Entry point: init params → create state → solve → output
│   ├── fourier.cu          DFT basis precomputation + inverse/forward transform kernels
│   ├── geometry.cu         Jacobian, metric g^ij, contravariant B on half-grid
│   ├── forces.cu           MHD force residuals (brmn/bzmn/crmn/czmn/blmn/clmn)
│   ├── profiles.cu         Radial profile evaluation + GPU upload
│   ├── solver.cu           Fixed-point loop with Garabedian accelerated descent
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
Half-grid: √g, g^uu, g^uv, g^vv, B^θ, B^ζ    (ns-1 × ntheta × nzeta)
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

| Decision | Rationale |
|----------|-----------|
| **Custom DFT kernels, not cuBLAS** | The DFT is a structured sum (cos/sin per surface), not a generic gemm. Custom kernels fuse derivative computation with the transform, saving memory bandwidth. |
| **Column-major storage** | Standard for cuBLAS; natural for per-surface indexing: `array[point + surface * nZnT]` for real space, `array[surface + mode * ns]` for spectral. |
| **Staggered half-grid** | Dynamic variables on full grid (flux surfaces); metric elements on half grid (between surfaces). Prevents checkerboard instability. Matches VMEC convention. |
| **All GPU allocations at startup** | Scratch arrays allocated once, reused every iteration. Zero `cudaMalloc` calls in the hot loop. |
| **Host checks convergence** | Residual reduction runs on GPU, but the scalar comparison `fsq < ftol` happens on host. |
| **double precision** | Mandatory — residuals pushed to 1e-14. Single precision stalls at ~1e-7. |

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
`mode = m * ntor + n` with `m = 0..mpol-1`, `n = 0..ntor-1`.
Mode 0 is the (0,0) DC mode.

### Forward DFT normalization
Mode-dependent weights:
- Mode 0 (DC): `weight = 1 / nZnT`  (since Σ cos²(0) = N)
- All other modes: `weight = 2 / nZnT`  (since Σ cos² = N/2 for m>0)

This ensures round-trip identity: `forward(inverse(x)) = x`.

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
`√g = R * (R_θ * Z_s - R_s * Z_θ) * (1 + λ_θ) * signJ`

The `(1 + λ_θ)` factor accounts for the poloidal angle transformation.
`signJ = -1` for right-handed flux coordinates.

### Metric (half-grid)
- `g^uu = (R_θ² + Z_θ²) / g²`
- `g^uv = (R_θ * R_ζ + Z_θ * Z_ζ) / g²`
- `g^vv = (R_ζ² + Z_ζ²) / g²`

### Contravariant B (half-grid)
- `B^θ = ι * Φ' / √g`
- `B^ζ = Φ' / √g`

### Radial derivatives
`R_s = (R[j+1] - R[j]) / Δs` — normalised by the radial grid spacing.

## Solver: Garabedian Accelerated Descent

The solver uses a pseudo-time descent with momentum (second-order Richardson):

```
v_new = fac * (b1 * v_old + delt * f)
x_new = x_old + delt * v_new
```

**Adaptive damping:** `inv_tau = sqrt(max(fsqr, fsqz, fsql))`, averaged over a
10-iteration window. `b1 = 1 - dtau`, `fac = 1/(1 + dtau)`.

**Convergence:** `max(fsqr, fsqz, fsql) < ftol` where `ftol = 1e-14`.

### What's intentionally omitted vs VMEC++

| VMEC++ feature | Status |
|----------------|--------|
| FFT-accelerated transforms | Custom DFT kernels instead (simpler, ~30 lines) |
| Multigrid grid sequencing | Single fixed radial grid |
| Free boundary / vacuum solver | Fixed boundary only |
| Mercier stability, jxbout, wout | Post-processing; not needed for core loop |
| Hot restart / checkpointing | Not yet |
| Adaptive time-step (Jacobian resets) | Fixed step; add when convergence is poor |
| De-aliased constraint force | Not yet |
| Python interface | Not yet; C++/CUDA executable only |
| Input file parsing | Hardcoded in `input.h` |

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

**Status (2026-08-02 night 2): the θ-parameterization difference is RESOLVED.
The rollback-backup refresh now captures the POST-descent state (vmecpp's
RestartIteration(NO_RESTART) semantics — the descent runs before the
time-step control), the per-iteration residuals track vmecpp at ≤1e-8
relative over the ENTIRE run (restart sequence identical at all 9 restarts:
BJs iter2 3,5,8,11,15; BPs 51,64,78,91), and the W7-X run converges at 2962
iters, FSQR 9.91e-13 (vmecpp: 2953, 9.92e-13). The converged state matches
the wout at ≤1.5e-9 in ALL SIX FAMILIES — rmncc 3.0e-10, zmnsc 2.3e-10,
lmnsc 1.5e-9, rmnss 1.9e-10, zmncs 1.4e-10, lmncs 1.5e-9 — including the λ
gauge modes (lmnsc(1,0), lmncs(0,1)) that previously sat at 1.4e-2/2.5e-3.
(The latter was partly a comparison bug: scripts/compare_converged_state.py
read the half-grid-interpolated wout `lmns`; it must read `lmns_full`.)**

1. **Axis representation (state-file only, real-space-irrelevant).** cuMES
   constant-extrapolates the axis row from j=1 (extrapolateAxisKernel — m=1
   all six families plus the m=0 lmncs "chi-force leftover", matching vmecpp's
   extrapolateTowardsAxis), so the dumped axis m>0 coefficients equal the j=1
   values (e.g. rmncc(1,0)@axis = 0.0497 for W7-X, 0.319 for Solovev), while
   vmecpp keeps them 0 (or s^(m/2)-extrapolated). The real-space axis geometry
   agrees (step_A verified at 1e-15), and the axis coefficients do not enter
   the forces, so this only shows up when diffing state files / wout axis rows.

2. **No multigrid.** Single fixed radial grid; adding grid sequencing would
   improve robustness for difficult equilibria.
