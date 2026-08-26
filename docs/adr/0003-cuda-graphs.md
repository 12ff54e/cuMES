# ADR-0003 — CUDA Graph capture for the pre-control DAG

Status: accepted; integrated for fixed-boundary passes (2026-08-26 update)

## Context

The regular iteration enqueues ~24 submissions per evaluated pass (kernels,
cuFFT transforms, and one control-record D2H copy) on one stream, then fences
once. §8.11 asks whether capturing the pre-control portion as CUDA Graphs
reduces submission overhead enough to matter, and to "select by measured shape".

## What was built

- `cumes::CudaGraph` (`include/cumes/runtime/cuda_graph.hpp`): RAII over
  `cudaStreamBeginCapture`/`EndCapture` → `cudaGraphInstantiate` → `launch`,
  with partial-capture teardown on a throwing enqueue.
- `tests/test_cuda_graph.cu`: proves a three-kernel DAG and a **cuFFT inverse
  transform** captured into a graph replay bit-identically to direct stream
  execution — i.e. cuFFT-in-graph works on CUDA 12.1 / sm_61 with the Phase-6B
  explicit work area.
- `benchmarks/graph_overhead.cu`: empty-kernel microbenchmark of per-pass
  submission cost (an upper bound, since real kernels overlap submission with
  GPU execution).

## Measured result (TITAN Xp, sm_61)

| quantity | value |
| -------- | ----- |
| enqueue cost | 1.944 µs / kernel |
| enqueue 24 kernels (one pass) | 46.65 µs |
| graph launch (24-kernel graph) | 6.43 µs |
| **per-pass submission savings (upper bound)** | **40.23 µs** |
| capture + instantiate (one variant) | 436.5 µs |

Against the fixed-iteration benchmark (same GPU):

| shape | µs / iter | max graph savings |
| ----- | ---------- | ----------------- |
| Solovev ns=55 | 213 µs | ~19% (submission-bound) |
| W7-X ns=99 | 2108 µs | ~1.9% (GPU-bound) |

## Original decision (TITAN Xp)

**Do not integrate CUDA Graphs into the production solver.** The
`AxisymmetricOperator` is active and the real production DAG has been measured,
so this is no longer a pending measurement decision. The real-pass saving is
about 8–10% on the submission-bound axisymmetric shape and about 0.5% on W7-X,
which does not clear the complexity and portability bar for this codebase.

An integration would need several graph variants, a per-pass update for the
`m=1` gauge `zeroZ` scalar, and the first-pass `update_iota_chi` special case
for `ncurr=0`. The tested CUDA stack also exposed fragile graph-node parameter
introspection, pushing a robust implementation toward manual graph construction
or variant re-instantiation.

## Original consequences

- `CudaGraph` + `test_cuda_graph` + `cumes_benchmark_graph_overhead` are kept as
  the measurement primitive and the cuFFT-in-graph correctness gate.
- The solver's production path is unchanged (stream execution; Solovev
  bit-identical to Phase 8). No graph backend is wired in, so there is no
  maintenance obligation on the hot loop.

## Real-kernel re-measurement

ADR-0004 landed the
`AxisymmetricOperator` production wiring (28.6% median win, Solovev
214.6 → 153.2 µs/iter), and a new harness `cumes_benchmark_graph_realpass`
(`benchmarks/graph_realpass.cu`) captures the **real** per-iteration DAG —
`EquilibriumOperator::enqueue`, the exact production pipeline, built from the
same operator stack as `cumes_benchmark_fixed_iteration` — and gates the
replay **bit-identical** to direct stream execution before timing anything
(control record, max |diff| = 0 on both shapes; the W7-X capture proves
cuFFT-in-graph across the full DAG, not just one transform).

TITAN Xp, sm_61, CUDA 12.1, `--passes 200 --warmup 5`:

| quantity | Solovev (axisym, ns=55) | W7-X (generic, ns=99) |
| -------- | ----------------------- | --------------------- |
| submission: direct enqueue / pass | 53.8–60.4 µs | 98.8 µs |
| submission: graph launch / pass | 7.3–7.7 µs | 10.9 µs |
| **submission saving / pass** | **46.6–52.7 µs** | **87.9 µs** |
| production wall (DAG + control D2H + fence): direct | 145.6–146.6 µs | 1751.6 µs |
| production wall: graph | 133.6–135.5 µs | 1742.9 µs |
| **end-to-end saving / pass** | **~11 µs (≈7.6%)** | **8.8 µs (≈0.5%)** |
| GPU time / pass: direct → graph | 146.4–147.8 → 134.7–136.0 µs | 1753.2 → 1744.0 µs |
| capture+instantiate (base / refresh / zeroZ variants) | 74–77 / 555–560 / 142–155 µs | 132 / 588 / 297 µs |
| `cudaGraphExecKernelNodeSetParams` (zeroZ update) | 1.5–1.6 µs | 1.5 µs |

Interpretation:

- The empty-kernel upper bound (40.2 µs/pass) slightly **under**estimated the
  real submission cost (46.6–52.7 µs on axisym — the real pass has more
  submissions than the K=24 assumption), but most of it is **hidden by GPU
  execution**, exactly as the deferral suspected: the end-to-end saving is
  ~11 µs/pass (7.6%) on the submission-bound axisym shape and 8.8 µs/pass
  (0.5%) on the GPU-bound W7-X shape. The graph also trims GPU idle gaps
  between passes (GPU time drops ~11 µs on axisym).
- The per-pass `zeroZ` scalar update the integration needed is **mechanically
  solvable**: `cudaGraphExecKernelNodeSetParams` costs 1.5 µs and provably
  changes the launch output (functional gate in the harness). Re-instantiation
  per variant is also cheap: ≤ 0.6 ms per variant per stage.
- **New blocker discovered:** on this CUDA 12.1 stack `cudaGraphKernelNodeGetParams`
  is unusable on the real captured DAG (returns success but neither copies
  parameter values nor leaves the caller's pointer array intact — the harness
  crashed on garbage entries the driver wrote; even a fresh
  `cudaGraphAddKernelNode` segfaulted inside libcuda after replays of the real
  DAG, while minimal repros pass). An integration therefore cannot find the
  zeroZ node via GetParams: it must either construct the graph manually
  (keeping its own node/param handles) or re-instantiate per variant. A
  struct-by-value kernel parameter also triggered an illegal memory access in
  the manual path; plain-int parameters work.

That decision was correct for the measured Pascal stack, but is superseded for
fixed-boundary execution by the modern-GPU result below.

## 2026-08-26 decision update (RTX 4090, CUDA 12.9)

The production solver now lazily captures the complete fixed-boundary
`EquilibriumOperator::enqueue` DAG and caches one executable per schedule
shape. Refresh and non-refresh variants are separate; non-refresh variants are
discarded after new host normalization factors are published. This avoids
fragile graph-node introspection and parameter mutation entirely.

Six order-alternated fixed-iteration pairs on GPU 2, after a 1000-pass preheat,
gave these medians of per-run medians:

| shape | direct stream | production graph | wall reduction |
| ----- | ------------: | ---------------: | -------------: |
| Solovev `ns=55` | 117.14 µs/iter | 82.33 µs/iter | 29.7% |
| W7-X `ns=99` | 562.10 µs/iter | 534.53 µs/iter | 4.9% |

Both graph and direct paths produced identical final-state hashes (Solovev
`3c09fd3260b003a8`, W7-X `ce46a7cbe693601a`). Free-boundary and verification-
dump execution remain on direct streams because host work splits or observes
the DAG. Set `CUMES_DISABLE_CUDA_GRAPHS=1` to select the direct fixed-boundary
path for diagnosis or comparison. The graph change clears the adoption gate on
Solovev and is a 4.9% non-regressing improvement on W7-X; the combined retained
W7-X optimization clears the gate independently.

CUDA event timestamps recorded inside a captured graph cannot be queried
reliably on the tested CUDA 12.9 stack. The solver therefore reports transform
timing as unavailable during graph replay; fixed-iteration wall timing and
Nsight remain the supported graph-path measurements.
