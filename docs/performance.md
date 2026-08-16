# cuMES performance

Status date: 2026-08-16 (Phase 10). This documents what has been *measured*
(never claimed), on which hardware, and the acceptance policy that governs
future optimization. It supersedes the historical profiling notes
(`optimization-2026-08-03.md`, `profiling-2026-08-05.md`) for decision-making;
those remain as raw inputs.

## 1. Measurement harness (§8.1)

`cumes_benchmark_fixed_iteration` runs one radial stage at a config's final
shape (Solovev `ns=55`, W7-X `ns=99`) with `ftol=0` (never converges), discards
a warm-up, and emits JSON: GPU identity, shape, arena/cuFFT bytes, setup vs
solve time, median/p95 wall µs/iter, and an FNV final-state hash. The seed
construction is shared with the CLI via `cumes/state/seed_state.hpp`.

A `cumes::SolverBench` observer samples the single control fence; it is pure
host observability (`bench == nullptr` in production leaves the hot loop
byte-identical). A 2-pass smoke gate (`cumes_benchmark_smoke`) is registered in
CTest.

## 2. Steady-state wall time (TITAN Xp, sm_61)

Measured by the fixed-iteration harness (Phase 9, updated post-Phase-10 with
the axisymmetric production wiring — see ADR-0004):

| shape | µs / effective iteration |
| ----- | ------------------------ |
| Solovev `ns=55` (axisymmetric) | ~153 (was ~213 on the generic backend) |
| W7-X `ns=99` (3D) | ~2110 |

Historical hotspot profiles (different sessions; not cross-GPU ratios — see the
blueprint §2) put the W7-X iteration work in the transform accumulation/reduction
(~0.4 ms each on TITAN Xp) and the forward reduction (~0.11 ms on RTX 4090),
with force, geometry, tridiagonal, and residual work smaller — i.e. structural
transform work dominates over generic occupancy tuning, and host synchronization
matters more as kernels shorten.

## 3. Phase 9 experiments and their outcomes

The exit gate for §8.10–§8.12 was *"measure, then adopt or remove."* Three
experiments were run on the TITAN Xp; one was adopted.

### 3.1 Mixed-float double accumulation — adopted (ADR-0001)

`cumes::NormAccum<T>` (`float → double`, `double → double`) widened the
accumulator of the three control-feeding reductions. **Class A** for the double
build (identity); **Class B** for float, where `test_accumulation.cu` shows a
~20× lower summation error (4.6e1 → 2.3e0 on a 1M-term dynamic-range case). It
does not unlock a lower float `ftol` — the float-state rounding floor dominates.

### 3.2 R/Z vs λ force-kernel split — not adopted (ADR-0002, retired Phase 10)

The split reduced registers 108 → 82/54 but was **1.20–1.45× slower**: the
force kernel is input-traffic-bound, so the two-kernel split doubled the
geometry/field loads. The prototype was removed in Phase 10; the conclusion
(§8.10's remaining radial-tile / force+projection fusion ideas trade global
traffic for registers/shared memory and are unlikely to pay) is the durable
result.

### 3.3 CUDA Graph capture — integration deferred (ADR-0003)

Empty-kernel microbenchmark: enqueue 1.94 µs/kernel, graph launch 6.43 µs/pass
for 24 kernels → **~40 µs/pass upper-bound saving**. That is ~19% of Solovev
(submission-bound) but ~2% of W7-X (GPU-bound). Integration is deferred until
the Phase-7 `AxisymmetricOperator` is wired into Solovev and a real-kernel
re-measure confirms the bound. `CudaGraph` + `test_cuda_graph` (which proves
cuFFT-in-graph replays bit-identically) are retained as the measurement
primitive.

## 4. Acceptance policy (blueprint §10.7)

A performance-motivated change is accepted only when, on one named target
workload, the lower bound of the 95% confidence interval shows an improvement
`> max(5%, measured noise floor)`, while the upper bound on the other primary
workload's regression is `<= 2%` — unless a separately approved correctness or
memory benefit justifies it. Requirements:

- repeated thermally stable warm runs; report median, p95, CI, clocks, and the
  noise floor (never a single timing);
- setup and output reported separately from effective-iteration time;
- compared on the legacy Pascal target *and* one modern architecture;
- peak arena/cuFFT/graph memory growth beyond an agreed baseline is rejected
  unless the measured benefit justifies it;
- the old backend is retained until the new one passes both numerical and
  performance gates.

Equivalence class precedes any timing claim: Class A requires bitwise equality;
Class B requires per-operator ULP bounds and identical controller decisions;
Class C requires independent CPU/VMEC++ agreement and a written ADR.

## 5. Retained historical inputs

- `optimization-2026-08-03.md`, `profiling-2026-08-05.md` — raw session notes;
  the TITAN Xp and RTX 4090 measurement scripts remain historical comparison
  inputs, not acceptance goldens.
- `docs/adr/0001..0003` — the decisions behind §3.
