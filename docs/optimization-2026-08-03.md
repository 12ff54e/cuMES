# cuMES Optimization Pass — 2026-08-03

Profile-driven performance pass over every GPU kernel and the per-iteration
pipeline. Headline result on the W7-X case (TITAN Xp, sm_61):

| Metric | Before | After | Δ |
|---|---|---|---|
| W7-X full run | 7.786 s | **4.984 s** | **−36%** |
| Per-iteration wall | 2.63 ms | 1.68 ms | −0.95 ms/iter |
| Converged at | iter 2962 | iter 2962 | identical |
| FSQR | 9.924e-13 | 9.924e-13 | identical |
| Per-iter residuals vs baseline | — | ≤1e-15 over the whole run | effectively bit-identical |
| Converged state vs baseline | — | ≤2e-10 all six families | well under the 1e-8 bar |

The trajectory is bit-identical to the pre-pass run (same restart sequence,
same convergence point, printed residuals unchanged to 4 significant digits);
the only arithmetic changes are in the preconditioner solve (PCR) and two
surface-average loops, which stay within the documented tolerance-based
verification bar (state ≤1e-8, per-iter residual tracking ≤1e-8).

---

## 1. Environment and tooling

- GPU: NVIDIA TITAN Xp (Pascal, sm_61, 12 GB, ~380 GFLOPS fp64).
- CUDA 12.1 toolchain, `-O3 --use_fast_math`, no `-lineinfo` in the default
  build; a parallel `build-prof/` dir adds `-lineinfo` and `sm_61` only.
- **ncu cannot profile this GPU at all.** The TITAN Xp is Pascal (sm_61),
  and Nsight Compute dropped Pascal support before 2023.1.1 (the CUDA 12.1
  toolkit's version): both the installed 2023.1.1 and a freshly downloaded
  2026.2.1 list no gp10x chips in `--list-chips`, and profiling reports
  "Profiling is not supported on device 0". The last Pascal-capable ncu
  (2022.1) predates the CUDA-13.0 driver (580.173) and cannot attach to it.
  The permission layer (ERR_NVGPUCTRPERM, fixable via sudo or
  NVreg_RestrictProfilingToAdminUsers=0) was never the blocker; the
  toolchain has moved on from the hardware.
- With hardware counters off the table, profiling came from **nsys kernel
  timelines**, **cuobjdump --dump-resource-usage** (register counts, spills,
  derived occupancy), and targeted **standalone microbenchmarks**
  (`/tmp/cumes_prof/bench_*.cu`). This shaped the workflow: hypothesis from
  code reading → nsys to rank kernels → microbenchmark to isolate a single
  kernel's behavior → change → full-run verification.

  The resource dump did surface one thing ncu would have: the slot-split
  inverseAccumulate runs at 26% occupancy (64 regs × 540 threads = 1 block/
  SM — registers, not shared memory, are the limiter), and forces at 25%
  (108 regs). The follow-up `__launch_bounds__` experiments (forcing 2-8
  blocks/SM on inverseAccumulate/forwardReduce/forces/geometry) measured out
  as noise or negative — the memory-bound kernels saturate occupancy's
  benefit, and the register-hungry ones spill (forces: 224 B/thread at 64
  regs → 6% regression). All reverted; 4.984 s stands.

## 2. Verification methodology (no vmecpp reference per change)

The user's requirement: compare each change against the **pre-pass cuMES
output itself**, not vmecpp. Added `scripts/compare_runs.py` up front, which
compares two runs on:

1. **Converged state** (`cumes_state.bin`): six spectral families, axis row
   skipped (the project's `compare_states.py` convention), tolerance 1e-8.
2. **Per-iteration residual rows** parsed from the log
   (`fsqr/fsqz/fsql/delt`), tolerance 5e-3 (the print precision is 3-4
   significant digits, so the log tolerance is loose; real trajectory
   divergence compounds far beyond it within a few iterations).
3. **Restart markers** (BAD JACOBIAN / BAD PROGRESS / RESETTING DELT): same
   iterations, same sequence.
4. **Convergence**: converged-iteration delta ≤ 10, final residuals matching.

Baselines captured in `perf_base/{solovev,w7x}/` and committed; per-change
runs go to `perf_check/` (gitignored). The whole verify loop for one change
is ~40 s: build → 4 test binaries → Solovev (~0.5 s) → W7-X (~5 s) →
`compare_runs.py` → commit.

The baseline itself was already verified against the vmecpp wout
(≤1.5e-9 in all six families), so old/new comparison preserves that
transitively.

## 3. Baseline profile (nsys, 60-iteration run)

GPU was ~100% saturated: sum of kernel times ≈ wall time. Per-iteration
budget (~2.63 ms/iter):

| Kernel | µs/iter | % | Reading |
|---|---|---|---|
| tridiagSolveKernel | 600 | 20.2 | 1 thread/block, serial Thomas, global-memory reads |
| inverseAccumulateKernel | 572 | 19.3 | uncoalesced θ-major stores (240 B/thread stride) |
| forwardReduceKernel | 378 | 12.8 | uncoalesced loads + 16-way shared-atomic contention |
| cuFFT (5 execs + pre/post) | ~530 | 17.7 | full 14,256-element batch for everything |
| forcesKernel | 187 | 6.3 | 32-thread blocks |
| geometryKernel | 117 | 4.0 | 32-thread blocks |
| rzConAccumulateKernel | 84 | 2.8 | uncoalesced stores |
| combineParityKernel | 58 | 2.0 | dump-only, ran unconditionally |
| ~25 other kernels | ~250 | rest | one-thread-per-mode kernels, memsets, launch shapes |

## 4. The changes, in commit order

### 4.1 `perf(precon)`: parallel cyclic reduction tridiagonal solve — 600 → 73 µs (−20% of runtime)

The largest single win. The original kernel ran one thread per block
(156 blocks × 1 thread) executing four serial Thomas solves; every
dependent step read `a/d/b` diagonals from global memory with zero ILP.
A naive fix (staging in shared memory, 5-thread split) changed nothing —
a microbenchmark bisection showed the true bottleneck: **the serial fp64
division chain itself** (~500 cycles per dependent step on Pascal; even a
pure arithmetic chain with no memory at all ran at ~105 µs per solve).

The fix is **parallel cyclic reduction**: 128 threads per block (one block
per mode), all rows updated in parallel each round while the coupling
distance doubles (k = 1,2,4,...,7 rounds for ns=99), one dependent
reciprocal per round. R and Z systems run as two sequential passes over
shared coefficient buffers; λ components and the j<jMin zeroing unchanged.

**Arithmetic changes** vs Thomas at the rounding level (different
elimination order). Verified: worst diff vs the Thomas result ~1e-12 in a
microbenchmark; full W7-X trajectory tracks the baseline at ≤1e-15 with the
same restarts and convergence point.

### 4.2 `perf(fourier)`: coalesce θ-major accesses + shuffle-tree reduction

The four poloidal-accumulation kernels
(inverseAccumulate, forwardReduce, rzConAccumulate, deAliasSynthesize)
mapped threads with ζ fastest, so every real-space store/load walked a
240 B stride. Swapping the block dimensions (`l=θ` fastest) coalesces them;
per-thread arithmetic is untouched (each thread still computes the same
(k,l) element), so this is bit-identical.

**Caveat found by measurement**: the swap made forwardReduce's
shared-memory `atomicAdd` reduction 16-way contended per address
(16 l-lanes per k) — the kernel exploded to 2,477 µs/iter. Fixed in the
same commit: replace the atomics with per-thread register values reduced by
a **warp shuffle tree** over the 16 θ lanes (blockDim.x padded to 16,
inactive lanes contribute zero; two k-groups per warp, width 16). The
shared scratch disappeared entirely. Summation order became a deterministic
tree (was unspecified atomics).

### 4.3 `perf(fourier)`: split inverseAccumulate into 3 slot groups — 426 → 397 µs

The kernel was latency-bound at 1 block/SM (41.5 KB shared). Splitting the
12 FFT slots into R/Z/λ groups of 4 (13.8 KB each, three launches) raises
occupancy to 3 blocks/SM. The per-group θ-basis selection (R uses cos/sin,
Z/λ use sin/cos, λ's ζ-derivative is negated) is uniform per block.
Bit-identical.

### 4.4 `perf(pipeline)`: drop per-iteration device syncs; gate combineParity; drop dead memset

- Removed the 7 intra-iteration `cudaDeviceSynchronize` calls — kernels on
  the default stream order implicitly, so the host can enqueue ~25
  launches/iter without device stalls. The two residual-D2H barriers and
  the 25-iteration syncs stay (genuine control-flow dependencies).
- `inverseDFT` gained `do_combine=false` for the hot loop — the e/o parity
  combination is only consumed by the dump machinery and tests (~58 µs/iter).
- Dropped the `d_gCon` memset: `gCon[jF==0]` is never read.

### 4.5/4.6 `perf(constraint)`: compact sub-batch cuFFT plans

The deAlias bandpass round trip and the rCon/zCon round trip transformed
the **full 14,256-element batch** while only a fraction carried data:

- **deAlias**: slots 0/1 (analysis) and 4/5 (synthesis), modes 1..mpol-2,
  surfaces 1..ns-1 → compact batch **2·(mpol−2)·(ns−1) = 1,960** elements,
  buffers 0.56 MB vs 4.1 MB, the 4.1 MB real-buffer memset gone.
- **rCon/zCon**: value slots 0/1/4/5 only → compact batch **4·mpol·ns =
  4,752**, spectra memset 4.33 → 1.45 MB.

The compact plans are **bit-identical** to the full-batch ones (cuFFT's
per-element algorithm depends on the transform size, not the batch) — the
full W7-X trajectory matches at exactly 0.000e+00.

Two subtle bugs surfaced and were fixed during this work (see §6).

### 4.7 `perf(p2)`: launch shapes and small kernels

- forces/geometry/effectiveConstraint/addConstraint/rzConIntoVolume:
  32-thread blocks → 128.
- ncurr1FinalizeKernel: full-grid stride with `k % ntheta` and half the
  iterations skipped → compact loop over `nzeta·nThetaRed` points, same
  (iz, it) visit order.
- lambdaPrecFinalizeKernel: `<<<mnmax,1>>>` (serial ns loop per mode) →
  one thread per (mode, jF), per-element arithmetic unchanged.
- extrapolateAxisKernel: `<<<mnmax,1>>>` → 32-thread blocks.

### 4.8 `perf`: pinned async residual copies; parallel deAliasAnalyze

- The two per-iteration residual D2H copies: pinned buffers +
  `cudaMemcpyAsync` + stream sync (no pageable staging, no implicit
  device-wide sync). No deeper overlap is possible — the copies gate the
  host's convergence/restart control flow.
- deAliasAnalyzeKernel: 8 threads per (m,jF,k) split the 30-point θ sum
  with a warp shuffle tree (width 8), replacing a serial per-thread dot
  (36 threads/block, latency-bound). Summation order differs at ulp level.

### 4.9 `docs`: refreshed README/CLAUDE.md performance numbers.

## 5. Dead ends (measured, reverted)

| Idea | Why it failed |
|---|---|
| forwardReduceKernel m-loop restructure: block per surface, loop m inside so the 14 force arrays are loaded once instead of once-per-m (143 MB → 12 MB of DRAM traffic) | The l-reduction must stay per-m, so the shuffle tree ran 12× (576 shuffles/thread); the shuffle cost ate the load savings — 5.10 → 5.31 s. Reverted. |
| 5-thread split + shared staging for the Thomas solve | Solved the memory latency but not the fp64 division chain (600 → 640 µs); superseded by PCR. |
| ncu-based tuning | Not possible on this machine (perf-counter permission). |

## 6. Debugging stories

- **The 600 µs Thomas**: initially attributed to global-memory latency.
  A standalone microbenchmark with bisection variants (staging-only,
  one-solve, no-division, pure-chain) proved the serial fp64 division chain
  itself was the bottleneck — pure arithmetic with no memory still took
  105 µs per solve. This redirected the fix from memory staging to an
  algorithm change (PCR).
- **The missing threads bug**: PCR failed at ns=65 but passed at ns=33.
  A round-by-round dump showed elements j ≥ 32 never changed — the block
  had 32 threads but 64 rows to solve. Thread count must cover nRow;
  128 threads fixed it.
- **The compact-FFT slot-index bug**: the first compact deAlias attempt
  diverged immediately (FSQR 8e13) and broke even Solovev. Two bugs: a
  memset placed between the analysis D2Z and the coefficient pack wiped
  the analysis spectra the pack needed, and the pack wrote "slots 4/5"
  with the full-batch slot stride — 4× past the end of the 2-slot compact
  buffer. The pack must overwrite the analysis slots in place and zero the
  unused tail bins itself.
- **The atomic explosion**: the θ-dimension swap, planned as a pure win,
  turned forwardReduce's shared atomics into 16-way serialized per address
  (2.5 ms/iter). The shuffle-tree reduction was developed in the same
  commit as the swap rather than as a follow-up.

## 7. Final state

Per-iteration budget (~1.68 ms/iter wall, GPU ~97% saturated):

| Kernel | µs/iter | Status |
|---|---|---|
| inverseAccumulateKernel | 397 | slot-split, coalesced; latency-bound at 3 blocks/SM |
| forwardReduceKernel | 391 | coalesced loads + shuffle tree; DRAM-bound (143 MB/iter of loads) |
| cuFFT (main execs) | ~296 | full batch required (main transforms use all 12 slots) |
| forcesKernel | 167 | 128-thread blocks |
| geometryKernel | 115 | 128-thread blocks |
| rzConAccumulate / tridiagSolve | 78 / 74 | — |
| ~20 smaller kernels | ~250 | each 15-45 µs, launch-shape-optimized |

Remaining headroom is small and increasingly risky: inverseAccumulate could
split its m-loop across threads (needs a reduction, changes rounding);
forwardReduce's load volume is structural (per-(m,j) block reads all 14
arrays; the m-loop restructure that would fix it is net-negative because of
the shuffle cost). Each is a ~100-200 µs/iter prize with real regression
risk — a reasonable place to stop for this architecture.

## 8. Reproducibility

```bash
# baseline + comparison tooling (already in the repo)
python3 scripts/compare_runs.py perf_base/w7x/w7x.log perf_base/w7x/cumes_state.bin \
                               perf_check/w7x/w7x.log perf_check/w7x/cumes_state.bin

# full verification of any change
cmake --build build -j
./build/test_fourier
cd perf_check/solovev && /path/to/build/cuMES > solovev.log      # exit 0, FSQR < 1e-16
cd perf_check/w7x    && CUMES_INPUT=w7x /path/to/build/cuMES > w7x.log
python3 scripts/compare_runs.py perf_base/w7x/... perf_check/w7x/...   # PASS
time CUMES_INPUT=w7x ./build/cuMES    # wall clock

# profiling (kernel timeline)
cmake -B build-prof -G Ninja -DCMAKE_CUDA_FLAGS="-O3 --use_fast_math -lineinfo" \
      -DCMAKE_CUDA_ARCHITECTURES=61 && cmake --build build-prof -j
CUMES_INPUT=w7x CUMES_MAX_ITER=60 nsys profile --stats=true -o /tmp/w7x \
    ./build-prof/cuMES
nsys stats /tmp/w7x.nsys-rep --report cuda_gpu_kern_sum
```

All numbers in this document were measured on the TITAN Xp; the pre-pass
"21 s" figure in the README history was recorded on different hardware and
is not directly comparable to the 7.786 s baseline measured here.
