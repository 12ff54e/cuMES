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
