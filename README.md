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
   ├── fourier.cu/cuh    Precomputed DFT basis + forward/inverse transforms
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

### Environment variables

All knobs are off by default; plain runs write nothing extra and print no
debug output.

| Variable | Effect |
|----------|--------|
| `CUMES_MAX_ITER` | iteration cap (default 1000) |
| `CUMES_DELT0` | initial time step (default 0.9) |
| `CUMES_DTAU_FLOOR` | floor on the damping parameter dtau (experiments) |
| `CUMES_DUMP_ITER` | which iterations the windowed dump files fire on (default 150) |
| `CUMES_E2_START` | first iteration of the per-iteration force dumps (default 560) |
| `CUMES_DUMP` | master switch for all dump/debug output (`=1` enables) |
| `CUMES_LOAD_INIT` | load an initial state from `vmecpp_init.bin` (handoff protocol) |

## Verification

- **Iter-1 baseline**: spectral forces match vmecpp at the same initial state
  to ~6e-14 (frcc/fzsc) / 7e-15 (flsc).
- **Handoff (same-state)**: given vmecpp's iter-150 state, real-space λ-force
  matches at 3e-14 and preconditioned spectral forces at ≤2e-6.
- **Full run**: converges at iter 209 (default start) or iter 76 (started
  from vmecpp's iter-150 state), no restarts, delt constant 0.9, final
  invariant residual ~7–9e-17.

## What's Intentionally Omitted (for clarity)

| VMEC++ feature | Why omitted |
|----------------|-------------|
| FFT-accelerated transforms | Custom DFT kernels are simpler and easier to read |
| Multigrid grid sequencing | Single fixed radial grid |
| Free boundary / vacuum solver | Fixed boundary only |
| Mercier stability, jxbout, full wout | Post-processing; not needed for the core loop |
| Hot restart / checkpointing | Add later |
| 3D (lthreed) equilibria | Implemented and verified against vmecpp (W7-X, mpol=ntor=12): the full iter-1 chain matches at 1e-9..1e-13, cuMES converges on its own (iter ~2794 vs vmecpp 2953), and the converged states match the vmecpp wout at ~1e-4 (residual-normalization control-path residual — see CLAUDE.md) |

## Tests

```bash
./build/test_fourier        # DFT round-trip + derivative checks
./build/test_forces         # force-kernel diagnostic solve
./build/test_force_verify   # forces near zero on a converged vmecpp state
```

## License

MIT — see [LICENSE](LICENSE).
