# cuMES — CUDA Magnetic Equilibrium Solver

Pure-CUDA, ground-up reimplementation of the core VMEC stellarator equilibrium
algorithm. All computation runs on GPU; the CPU host is a thin orchestrator.
This is a pedagogical / scaffolding project — not production-grade, but the
architecture and physics are real.

**Status: working.** For the Solovev test case (`solovev.json`), the solver
converges to the correct equilibrium (invariant residual < 1e-16) from the
default initial state in **209 iterations** (vmecpp reference: 252), with no
restarts and a constant time step. The converged equilibrium matches the
vmecpp reference (`solovev.out.h5`) to 6–8 digits (axis R_00 = 3.990923 vs
3.99092308).

## Architecture

```
main.cu                  Entry point — init → solve → output
   │
   ├── input.h           Hardcoded Solovev input (boundary + profiles)
   ├── vmec_types.h      Shared GPU data structures (GridParams, SpectralState, …)
   │
   ├── profiles.cu/cuh   1D radial profiles (iota, pressure, mass) on device
   ├── fourier.cu/cuh    cuFFT transforms (12-slot packing, batched 1D ζ-FFT)
   ├── geometry.cu/cuh   Jacobian, metric g^ij, contravariant/covariant B
   ├── forces.cu/cuh     MHD force residuals (R, Z, λ) in real space
   ├── constraint.cu/cuh Spectral-condensation constraint force (de-aliased)
   ├── precon.cu/cuh     Radial tridiagonal + λ preconditioners
   ├── solver.cu/cuh     VMEC_8_52 iteration control (damping, restarts)
   └── output.cu/cuh     Pull results from GPU → print
```

## Data Flow per Iteration

```
Spectral coefs (rmnc, zmns, lmnc)              ← degrees of freedom
         │
         ▼  [inverse DFT — custom kernel]
Real-space geometry R, Z, λ + derivatives      (ns × ntheta × nzeta)
         │
         ▼  [geometry kernel]
Half-grid: √g, g^uu, g^uv, g^vv, B^θ, B^ζ    (ns-1 × ntheta × nzeta)
         │
         ▼  [forces kernel]
Real-space forces F_R, F_Z, F_λ                (ns × ntheta × nzeta)
         │
         ▼  [constraint force (de-aliased) + forward DFT]
Spectral forces                                 (5, mnmax, ns)
         │
         ▼  [precondition + descent kernel — Garabedian step]
v = fac×(b1·v + delt·f) ,  x += delt·v
```

## Physics implemented (all matching vmecpp)

- **MHD force balance** in flux coordinates with the vmecpp m-parity
  even/odd decomposition convention (odd real-space carries
  `physical/max(√s, √(1/(ns-1)))`; λ carries an extra √2).
- **Hybrid λ-force** (`forces.cu`): the two bsubv interpolations (half-grid
  average vs `gvv/gsqrt·(lamscale·λ_θ + Φ')` with √s_H-weighted odd part)
  blended with `radialBlending = 2·kPDamp·(1-s)`, kPDamp = 0.05 — verified to
  machine precision against vmecpp's binary at the same state.
- **Spectral-condensation constraint** (`constraint.cu`): bandpass-filtered
  `(rCon − rCon0)` force with vmecpp's tcon multiplier profile.
- **Preconditioning** (`precon.cu`): radial tridiagonal solve (LCFS row
  excluded) + λ preconditioner from flux-surface metric averages.
- **Iteration control** (`solver.cu`): VMEC_8_52 damping
  (`dtau = delt·otav/2`, 10-iteration window), BAD_JACOBIAN / BAD_PROGRESS
  restarts with state backup, convergence on the invariant residuals.

## Build & Run

```bash
# in folder cuMES
cmake -B build -G Ninja
cmake --build build -j
./build/cuMES
```

Requirements: CUDA Toolkit ≥ 11, CMake ≥ 3.20, GPU with compute capability ≥ 6.1
(Pascal or newer). If the host gcc is > 12, `CMAKE_CUDA_HOST_COMPILER` must
point to g++-12 (set in `CMakeLists.txt`).

The reference implementation lives in `../vmecpp` (CPU-based C++ VMEC++
solver); `../scripts/compare_step.py` compares binary dumps of the two codes.

### Precision

All computation is templated on a scalar type `T`; the executable's precision
is a compile-time choice:

```bash
cmake -B build-float -G Ninja -DCUMES_USE_FLOAT=ON   # single precision
cmake --build build-float -j
```

- The default build is double precision (`Real = double` in `vmec_types.h`).
- Single precision is ~1/32-rate-fp64-free on consumer GPUs, but the invariant
  residuals stall at ~1e-7 (the float rounding floor): the hardcoded
  `ftol` (1e-16) is never reached, so float runs report NOT CONVERGED unless
  the tolerance is relaxed for float experiments.
- The on-disk state file (`cumes_state.bin`) and `vmecpp_init.bin` stay double
  regardless of `T` — the Python comparison scripts are unaffected.
- The debug dumps (`dump/cuMES/*.bin`) are `T`-native: only readable by
  same-build tooling (e.g. `test_geometry_iso` is double-build-only).
- The unit tests instantiate both double and float in every build
  (`./build/test_fourier` runs both legs).

### Environment variables

All knobs are off by default; plain runs write nothing extra and print no
debug output.

| Variable | Effect |
|----------|--------|
| `CUMES_MAX_ITER` | iteration cap (default 1000); overrides EVERY stage's cap in a multigrid run |
| `CUMES_DELT0` | initial time step (default 0.9) |
| `CUMES_DTAU_FLOOR` | floor on the damping parameter dtau (experiments) |
| `CUMES_DUMP_ITER` | which iterations the windowed dump files fire on (default 150) |
| `CUMES_E2_START` | first iteration of the per-iteration force dumps (default 560) |
| `CUMES_DUMP` | master switch for all dump/debug output (`=1` enables) |
| `CUMES_LOAD_INIT` | load an initial state from `vmecpp_init.bin` (handoff protocol) |

## Verification

- **Multi-radial-grid (default, both configs)**: each stage runs with its
  own iteration cap and ftol, seeded by the previous stage's converged state
  via `interpolateState` (`src/refine.cu`, vmecpp's
  `InterpolateToNextMultigridStep` semantics). Verified against vmecpp's own
  multigrid runs:
  - **Solovev 5→11→55**: 251 → 199 → 456 effective iters, final FSQR
    9.58e-17; the final stage matches the vmecpp playground reference
    exactly (456 iters, 9.99e-17).
  - **W7-X 33→66→99**: 1878 → 1617 → 2011 effective iters (total 5506),
    final FSQR 9.78e-13 (vmecpp multigrid: 1877 → 1635 → 2012); converged
    states agree at ~1e-5 in R/Z and ~1e-4 in the weakly-determined
    near-axis λ. The final state is a different member of the (near-
    degenerate) λ-gauge family than the single-grid-99 equilibrium
    (rmncc(0,1) differs by 2.7e-4) — vmecpp's own single-grid vs multigrid
    states differ by the same 2.7e-4, so this is intrinsic to the
    continuation, and the multigrid final state is a genuine fixed point
    (restarting from it converges at iter 1).
- **Single-grid regression**: with `n_grids=1` the run is bit-identical to
  the pre-multigrid code (compare_runs.py PASS; W7-X converges at iter 2953
  effective, 2962 raw passes incl. 9 restarts, FSQR 9.924e-13).
- **Per-iteration fidelity (W7-X, single-grid)**: invariant residuals track
  vmecpp at ≤1e-8 relative over the entire run; the cuFFT transform backend
  reproduces the direct-sum trajectory byte-identically in the per-iteration
  logs; the converged state matches the wout at ≤1.5e-9 in all six families.
- **Solovev (axisymmetric, single-grid ns=11)**: converges at iter 252
  (default start) or iter 237 (started from vmecpp's iter-150 state via
  `CUMES_LOAD_INIT=1`), no restarts, delt constant 0.9, final invariant
  residual ~8e-17.

Note: `scripts/compare_runs.py` (two-log trajectory comparison) is only
valid for single-grid logs — per-stage iteration counters restart at 1, so
rows of later stages overwrite earlier ones. Use `compare_states.py` for
final-state checks.

## Status vs VMEC++

| VMEC++ feature | Status |
|----------------|--------|
| FFT-accelerated transforms | Implemented: cuFFT (batched 1D real FFT in the ζ direction + direct poloidal synthesis, mirroring vmecpp's FFTX structure); the constraint module's rCon/zCon and de-aliasing bandpass reuse the same plans/scratch (compact sub-batches: 2·(mpol−2)·(ns−1) and 4·mpol·ns elements instead of the full 12·mpol·ns). See `src/fourier.cu`. |
| Performance (W7-X, TITAN Xp) | Full run 7.79 s → **4.98 s** after the 2026-08-03 pass (PCR tridiagonal solve, coalesced θ-major access, slot-split poloidal accumulation, compact constraint sub-batch FFTs, sync removal): transforms 0.43/0.39 ms/iter (inverse/forward), converged at effective iter 2953 (2962 passes) with FSQR 9.924e-13 — per-iteration residuals bit-identical to the pre-pass run, state ≤2e-10 in all six families. |
| 3D (lthreed) equilibria | Implemented and verified against vmecpp (W7-X, mpol=ntor=12): per-iteration residuals track vmecpp at ≤1e-8 over the whole run; converges at iter 2953 effective (vmecpp 2953) with FSQR 9.924e-13 (vmecpp 9.92e-13); the converged state matches the wout at ≤1.5e-9 in all six families. |
| Multigrid grid sequencing | Implemented: per-config `ns_array`/`niter_array`/`ftol_array` stage loop (Solovev 5→11→55, W7-X 33→66→99), each stage seeded by the previous converged state via `interpolateState` (refine.cu) — linear-in-s interpolation on scalxc-scaled odd-m coefficients, exact at old grid points, LCFS pinned. A stage that exhausts its cap without meeting ftol fails the run (vmecpp semantics). See `src/refine.cu` + the Verification section. |
| Free boundary / vacuum solver | Fixed boundary only |
| Mercier stability, jxbout, full wout | Post-processing; not needed for the core loop |
| Hot restart / checkpointing | Add later |

## Tests

```bash
./build/test_fourier        # DFT round-trip + derivative checks
./build/test_forces         # force-kernel diagnostic solve
./build/test_force_verify   # forces near zero on a converged vmecpp state
```

## License

MIT — see [LICENSE](LICENSE).
