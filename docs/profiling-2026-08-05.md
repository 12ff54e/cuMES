# cuMES Profiling Session — 2026-08-05 (RTX 4090 @ sbt)

Session goal: continue the optimization work with real hardware counters (ncu),
which the Pascal TITAN Xp could not provide. Outcome: ncu is **blocked on sbt
by a driver-level permission** (`RmProfilingAdminOnly=1`, needs root), so the
session fell back to the nsys + microbenchmark workflow. The 4090's per-
iteration loop is **GPU-bound at 0.53 ms/iter** (3.1× faster than TITAN Xp);
further kernel tuning measured out as noise or negative and was reverted. The
real discovery is environmental: **every CUDA process on sbt pays ~4.9 s of
driver context creation**, which dominates the full W7-X run (6.6 s = 4.9 s
context + 1.6 s loop + 0.3 s exit).

---

## 1. Environment: how to build and run on sbt

- Host `sbt`, user `shibotong` (via `ssh sbt`). 7× RTX 4090 (sm_89), driver
  555.52.04, kernel 5.15.0-107. GPUs 0-5 run other users' jobs; use
  `CUDA_VISIBLE_DEVICES=6`.
- **CUDA toolkit**: the HPC SDK 23.7 install (`/opt/nvidia/hpc_sdk/...`)
  contains CUDA 12.2. Its `cuda/bin/nvcc` is NOT usable directly with CMake —
  the implicit link dirs it reports (`cuda/targets/x86_64-linux/lib`) do not
  exist in this install. The HPC SDK **wrapper** `compilers/bin/nvcc` resolves
  the real toolkit at `cuda/12.2/` (with the proper `targets/` layout) and
  works with `find_package(CUDAToolkit)`. Use it as `CMAKE_CUDA_COMPILER`.
- `load_env` sources `./.proxy` **relative to the CWD** — source it from
  `~/.qzhong` (`cd ~/.qzhong && source ./load_env`), then add
  `/opt/nvidia/hpc_sdk/Linux_x86_64/23.7/cuda/bin` and the profiler dirs to
  PATH (ncu is directly in `Nsight_Compute/`, nsys in `Nsight_Systems/bin`).
  A ready-made `~/.qzhong/cumes_env.sh` does all of this.
- Build: `cmake -B build-prof -G Ninja -DCMAKE_CUDA_COMPILER=.../compilers/bin/nvcc
  -DCMAKE_CUDA_FLAGS="-O3 --use_fast_math -lineinfo" -DCMAKE_CUDA_ARCHITECTURES=89`.
  Note: the CMakeLists `set(CMAKE_CUDA_ARCHITECTURES ...)` overrides the `-D`
  flag, so build-prof actually contains all five archs — harmless, just bigger.
- Baseline verification on the 4090: `compare_runs.py` **PASS** — residuals
  exactly 0.000e+00 vs the committed TITAN Xp baseline, identical restarts,
  converged at 2962, state ≤5.6e-10 (the small state drift is cuFFT's
  per-arch kernel selection; well under the 1e-8 bar).

## 2. The 4.9 s context creation (sbt machine property)

Phase timers in main.cu showed the entire cuMES startup (state init, profile
evaluation, cuFFT plan creation — the latter only ~4 ms) is ~0.2 s. The
remaining ~4.7 s of the 5.0 s startup is **the CUDA driver's context creation**
at the first CUDA call (`cudaSetDevice`/`cublasCreate`): a trivial binary with
no kernels pays the same 4.85 s. Deterministic per process (not a JIT-cache
artifact: `CUDA_CACHE_DISABLE` and repeated runs are identical), independent of
which GPU (busy or idle), not in libcuda's dlopen (1 ms), not in cuFFT/cuBLAS
lazy init (isolated: 19 ms). The nsys API trace places it between
`cuModuleGetLoadingMode` and the first allocation — inside
`cuDevicePrimaryCtxRetain`, in two ~2.3 s chunks. No userland mitigation;
suspects are driver/box-level (no nvidia-persistenced; 7-GPU peer probing at
context creation). Needs the box owner to investigate.

**Consequence**: full W7-X on sbt = 6.6 s, of which only 1.6 s is the solver
loop. Per-run loop improvements are invisible here but still matter on
startup-fast machines (e.g. the TITAN Xp box).

## 3. Per-iteration budget on the RTX 4090 (nsys, 60-iter run)

Loop wall 0.534 ms/iter (differenced t2962−t60), GPU ~95% busy. All kernels
scaled 3-4× vs Pascal (fp64: 1.29 TFLOPS vs 0.38):

| Kernel | µs/iter | vs TITAN Xp | Reading |
|---|---|---|---|
| forwardReduceKernel | 110 | 391 | DRAM-bound (~100 MB/iter of loads, 12× re-read across m) |
| inverseAccumulateKernel ×3 | 106 | 397 | latency-bound; 540-thr blocks, 58 regs → 34 warps/SM |
| cuFFT (5-8 execs incl. compact) | ~55 | ~296 | exec-count overhead ~5 µs each |
| forcesKernel | 41 | 167 | latency-bound; 104 regs → 16 warps/SM |
| geometryKernel | 33 | 115 | — |
| tridiagSolveKernel | 27 | 74 | — |
| computeResidualsKernel ×2 | 27 | — | — |
| rzConAccumulateKernel | 20 | 78 | — |
| ~20 smaller kernels | ~100 | ~250 | — |

## 4. Kernel-level experiments (measured, reverted)

| Idea | Result |
|---|---|
| inverseAccumulate m-split across threads (2×6) | Dead on arrival: the m-groups land in different warps (z-dim), so the cross-group shuffle can't reduce; also register math caps occupancy at ~34 warps/SM regardless of shape. |
| inverseAccumulate single-pass 1080-thr blocks | Invalid launch (>1024 threads/block max) — silent no-op, kernel never ran. |
| inverseAccumulate k-split grid (99,2), 270-thr blocks | 42 warps/SM (via `__launch_bounds__(270,5)`, 40 regs) — 38.6 µs vs 35.7 µs baseline: *slower* (each k-half block re-stages the full spectra; occupancy is not the lever). |
| inverseAccumulate basis tables via `__constant__` | 35.5 µs — no change (same broadcast-cached path as L1). |
| inverseAccumulate basis tables staged in shared | 32.4 µs isolated (−9%), integrated and verified bit-identical (compare_runs PASS, trajectory ≤1e-15) — but **no in-situ gain** (35.6 vs 35.3 µs in the real pipeline; tables already L1-resident). Reverted per the no-noise-changes rule. |
| forces `__launch_bounds__(128,6)/(128,8)` (80/64 regs) | 51.1/52.1 µs vs 46.4 µs baseline: −10/−12% — spills hurt on Ada too. Confirms the Pascal-session dead end. |
| forwardReduce m-loop restructure (block per surface) | Not retried: the per-(m,k,v) l-reduction forces 576 shuffles/thread either way; Pascal measured net-negative, and Ada's shuffle rate doesn't change the economics. |

## 5. ncu status on sbt

`ncu` (2023.2.0, supports ad102) fails with `ERR_NVGPUCTRPERM`:
`/proc/driver/nvidia/params` has `RmProfilingAdminOnly: 1` — hardware counters
are admin-only (device files are world-writable; CUDA itself works). Fix needs
root, one of: `nvidia-modprobe -c 0 -m 0` (per-user grant), or reload the
module with `NVreg_RestrictProfilingToAdminUsers=0`. Until then the workflow
stays nsys timelines + `cuobjdump --dump-resource-usage` (registers/spills) +
standalone microbenchmarks (`/tmp/cumes_prof/bench_*.cu` on sbt).

## 6. Takeaways

1. The 4090 loop (0.53 ms/iter) is at/near its structural floor: the top
   kernels are DRAM- or latency-bound with the block shapes register pressure
   dictates, and every occupancy experiment measured negative or flat.
2. The 4.9 s/process context creation on sbt dominates wall time there;
   worth a box-owner investigation (persistenced, driver state, 7-GPU peer
   init) — it dwarfs any possible solver tuning on this machine.
3. Unblocking ncu (one root command) would let a future session see the
   stall breakdown the microbenchmarks could only guess at.
