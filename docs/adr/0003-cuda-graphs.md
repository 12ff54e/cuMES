# ADR-0003 — CUDA Graph capture for the pre-control DAG (measured, integration deferred)

Status: accepted (Phase 9, blueprint §8.11)

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

## Decision

**Defer the four-variant solver integration.** The measured savings are an
empty-kernel upper bound and are meaningful only for the submission-bound
axisymmetric shape. Two higher-leverage steps precede it:

1. **Wire the Phase-7 `AxisymmetricOperator`** into the Solovev production path.
   Solovev's 213 µs/iter is dominated by the generic backend's length-1 cuFFT
   transforms; removing them changes the submission profile far more than a
   graph would, and the graph question should be re-measured on that new shape.
2. **Re-measure with real kernels** (not empty) once (1) lands, to confirm the
   ~40 µs upper bound survives overlap with GPU execution.

The full integration also needs a per-pass kernel-parameter update for the
`m=1` gauge `zeroZ` scalar (a `cudaGraphExecKernelNodeSetParams` or a
re-instantiate), plus the first-pass `update_iota_chi` special case for
`ncurr=0`, so it is not a plain four-variant capture.

## Consequences

- `CudaGraph` + `test_cuda_graph` + `cumes_benchmark_graph_overhead` are kept as
  the measurement primitive and the cuFFT-in-graph correctness gate.
- The solver's production path is unchanged (stream execution; Solovev
  bit-identical to Phase 8). No graph backend is wired in, so there is no
  maintenance obligation on the hot loop.

## Status update (2026-08-16): re-measured with REAL kernels — still deferred

The ADR's stated next step — "re-measure with real kernels once the
axisymmetric wiring lands" — is now done. ADR-0004 landed the
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

Updated decision: **remain deferred.** The measured end-to-end gain (~8% on
the one submission-bound shape, 0.5% on the GPU-bound shape) is real but does
not clear the complexity bar for a pedagogical codebase — especially with the
CUDA 12.1 graph-node API fragility pushing an integration toward manual graph
construction of the whole DAG. `CudaGraph`, `test_cuda_graph`,
`cumes_benchmark_graph_overhead`, and now `cumes_benchmark_graph_realpass`
remain as the measurement/correctness primitives; the question should be
re-opened only if the shape mix changes (e.g. a much smaller submission-bound
production shape) or a newer CUDA stack fixes the node APIs.
