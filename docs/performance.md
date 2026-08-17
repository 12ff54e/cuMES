# cuMES performance

Status date: 2026-08-17 (overhaul completion plan step 3.4 re-measurement).
This documents what has been *measured* (never claimed), on which hardware,
and the acceptance policy that governs future optimization. It supersedes the
historical profiling notes (`optimization-2026-08-03.md`,
`profiling-2026-08-05.md`) for decision-making; those remain as raw inputs.

> **2026-08-17 re-measurement note (completion plan step 3.4).** The numbers
> in §2 were re-measured with the overhaul's final build configuration:
> precise double math (`CUMES_PRECISION_POLICY=verify-double`, no
> `--use_fast_math` — its removal proved **trajectory byte-identical**, so
> the frozen baselines stand under precise math), the single-construction
> stage arena (one allocation + one module construction per stage, no
> measuring pass), and the fence-delivered axis/boundary telemetry (the
> production path has exactly one deliberate control fence per pass and no
> per-print synchronization or allocation). Setup and iteration time are
> reported separately. Only Pascal numbers are reported here: no modern GPU
> was attached to this session, so the two-architecture comparison remains
> an open measurement on the RTX-class target.

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

Measured by the fixed-iteration harness with 300 timed passes after 50 warmup
passes, three thermally stable repeats each (completion plan step 3.4,
2026-08-17, `verify-double`):

| shape | median µs/iter | p95 µs/iter | repeats |
| ----- | -------------: | ----------: | ------: |
| Solovev `ns=55` (axisymmetric) | 167.0 / 167.1 / 166.5 | 175.7 / 177.2 / 175.2 | 3 |
| Solovev `ns=55` (generic backend, `CUMES_FORCE_GENERIC=1`) | 179.1 | 194.9 | 1 |
| W7-X `ns=99` (3D, generic toroidal-FFT) | 1757.8 / 1757.7 (first repeat 1874.4: cold-clock ramp, discarded) | 1772.4 / 1772.5 | 3 |

Setup (arena + module construction + profile/table upload, one pass per
stage): ~3.5 ms Solovev / ~3.9 ms W7-X, reported separately from the
iteration time. The axisymmetric backend retains a measured ~7% advantage
over the generic backend on the same shape (167 vs 179 µs/iter).

CUDA Graph re-measurement on the REAL pass DAG (ADR-0003 follow-up,
`cumes_benchmark_graph_realpass`, Solovev `ns=55`): enqueue submission
63.9 µs/pass vs graph launch 9.8 µs/pass (54.2 µs saved), production-pattern
wall 124.0 vs 111.3 µs/pass (12.8 µs ≈ 10% saved), GPU time 125.3 vs
112.9 µs. The real-pass bound is therefore confirmed beneficial on the
submission-bound Solovev workload; production graph integration remains a
separate decision (see §3.3).

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
for 24 kernels → **~40 µs/pass upper-bound saving**. The 2026-08-17 REAL-pass
re-measure (see §2) confirmed a ~12.8 µs/pass (≈10%) wall saving on Solovev,
with bitwise replay fidelity (`test_cuda_graph`). Production graph
integration is still deferred: it remains a separately reviewable scheduling
change outside the completion plan's scope, and the W7-X workload is
GPU-bound where the submission saving is proportionally small. `CudaGraph`
+ `test_cuda_graph` are retained as the measurement primitive.

## 4. Acceptance policy (verification.md §7)

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

**Modern-GPU validation status: POSTPONED (2026-08-17).** No second, modern
CUDA GPU is attached to this machine, so the "one modern architecture" half of
this policy has not been run — it is neither passed nor failed. Until suitable
hardware is available, the TITAN Xp measurements above remain the measured
baseline and every modern-GPU conclusion is explicitly unknown/deferred
(`docs/post-overhaul-follow-up.md` §6). The deferred procedure is fixed: same
commit, precision policy, toolkit provenance, warm-up, and measurement method
as the TITAN Xp baseline; record GPU model, compute capability, driver/toolkit,
clocks, power/thermal state, arena/cuFFT/graph memory, setup/output time,
median, p95, measured noise floor, and a 95% confidence interval; re-run the
complete numerical trajectory/state gate afterwards.

Equivalence class precedes any timing claim: Class A requires bitwise equality;
Class B requires per-operator ULP bounds and identical controller decisions;
Class C requires independent CPU/VMEC++ agreement and a written ADR.

## 5. Retained historical inputs

- `optimization-2026-08-03.md`, `profiling-2026-08-05.md` — raw session notes;
  the TITAN Xp and RTX 4090 measurement scripts remain historical comparison
  inputs, not acceptance goldens.
- `docs/adr/0001..0003` — the decisions behind §3.
