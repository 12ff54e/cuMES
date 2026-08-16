# cuMES Phase 9 handover — graphs and high-risk fusion

Status date: 2026-08-16. Branch: `overhaul` (Phase 0 `bd26857` + Phase 1
`12bcc44` + Phase 2 `168170a` + Phase 3 `c21564c` + Phase 4 `1b0d099` + Phase 5
`2b9aaf8` + Phase 6 `759f933` + Phase 7 `a59fff1` + Phase 8 `1069939` + this
Phase 9 work). This document records what Phase 9 of
`docs/cuda-overhaul-blueprint.md` delivered, how it was verified, and what was
deliberately deferred and why.

## 1. Scope

Phase 9 is **graphs and high-risk fusion** (§8.10–§8.12). Its exit gate is
unusual among the phases: *"each backend has an ADR, differential tests, and
measured benefit on named hardware. Unsuccessful experiments are removed rather
than becoming maintenance paths."* The four deliverables and their outcomes:

| Blueprint § | Deliverable | Status |
| ----------- | ----------- | ------ |
| §8.1 | Fixed-iteration benchmark harness | **Done** (`a01fd37`, `67625a1`) |
| §8.12 / §8.8 | Mixed-float double reductions | **Done, adopted** (`4769fda`, `85fda96`) |
| §8.10 | Force split prototype (R/Z vs lambda) | **Done, measured, not adopted** (`2d05071`) |
| §8.11 | CUDA Graph variants | **Measured, integration deferred** (`e6602eb`) |
| §8.10 / §8.3 | Fused descent/checkpoint + force/projection | **Deferred** (§5) |

Three of the four were measured on the TITAN Xp (sm_61, 12 GB) — the named
hardware the acceptance policy requires — and one (double reductions) is adopted
into the float build.

## 2. What changed

### 2.1 Benchmark harness (`a01fd37`, `67625a1`, §8.1)

`cumes_benchmark_fixed_iteration` runs one radial stage at a config's final
shape (Solovev ns=55, W7-X ns=99) with `ftol=0` (never converges), discards a
warm-up, and emits JSON: GPU identity, shape, arena/cuFFT bytes, setup vs solve
time, median/p95 wall µs/iter, and an FNV final-state hash.

- The seed construction (`init_params`/`init_state`/`restart_state`) was moved
  verbatim out of `main.cu` into `include/cumes/state/seed_state.hpp` so the
  harness and CLI share one copy (a pure relocation; Solovev bit-identical).
- A new opt-in `cumes::SolverBench` observer is threaded through `solverRun`
  and sampled at the single control fence. It is pure host observability —
  `bench == nullptr` in production leaves the hot loop byte-identical.
- Steady-state TITAN Xp: **Solovev ~213 µs/iter, W7-X ~2.11 ms/iter**.
  Registered as a 2-pass CTest smoke gate (`performance;smoke`).

### 2.2 Mixed-float double accumulation (`4769fda`, `85fda96`, §8.8/§8.12)

`cumes::NormAccum<T>` (`float → double`, `double → double`) now drives the
accumulator of the three control-feeding reductions — `computeResidualsKernel`,
`rzNormKernel`, `forceNormReduceKernel`. The *terms* keep their `T` arithmetic;
only the summation width changes. ADR-0001.

- **Class A** for the double build (`NormAccum<double> == double`). After the
  refinement commit (`85fda96`) the term/result expressions are byte-for-byte
  the originals, so bit-identity is by construction, not by compiler identity-
  cast elision.
- **Class B** for the float build. `test_accumulation.cu` shows a ~20× lower
  summation error on a 1M-term dynamic-range case (4.6e1 → 2.3e0). It does
  **not** unlock a lower float `ftol` (the float-state floor dominates); a full
  double `ControlRecord` is the larger follow-up (ADR-0001).

### 2.3 Force split prototype (`2d05071`, §8.10)

`computeForcesSplit` splits the monolith into `rzForcesKernel` (12 families) +
`lambdaForcesKernel` (4 families), copied verbatim, behind the retained
monolith. ADR-0002.

- ptxas (sm_61): monolith **108 registers, 0 spills**; split **82 / 54**.
- `test_force_split.cu`: **bit-identical** (max |diff| = 0 across all sixteen
  families, double + float), but **1.20–1.45× slower** (the kernel is
  input-traffic-bound, so doubling the geometry/field loads dominates the
  occupancy win). Not adopted; retained only as the differential gate.

### 2.4 CUDA Graph primitive + measurement (`e6602eb`, §8.11)

- `cumes::CudaGraph` (capture/instantiate/launch RAII) in
  `include/cumes/runtime/cuda_graph.hpp`.
- `test_cuda_graph.cu` proves a three-kernel DAG **and a cuFFT inverse
  transform** replay bit-identically to stream execution — cuFFT-in-graph works
  on CUDA 12.1 / sm_61 with the Phase-6B explicit work area.
- `cumes_benchmark_graph_overhead` (empty-kernel microbenchmark): enqueue
  1.94 µs/kernel, graph launch 6.43 µs/pass for 24 kernels, **~40 µs/pass
  savings upper bound**; capture+instantiate 436 µs/variant. That is ~19% of
  Solovev (submission-bound) vs ~2% of W7-X (GPU-bound). ADR-0003 defers the
  four-variant solver integration until the Phase-7 `AxisymmetricOperator` is
  wired and a real-kernel re-measure confirms the bound.

## 3. Key design decisions

1. **"Measured, then adopted or removed."** Each §8.10/§8.11 backend was built
   behind the production path (never wired into `solverRun`), given a
   differential test, measured on the TITAN Xp, and then either adopted (double
   reductions) or documented as not-adopted (force split, graphs) — matching the
   exit gate. The one *adopted* change is the cheapest and only Class-A-for-
   double one.
2. **Bit-identity by construction, not by assumption.** The 9.5 refinement
   (`85fda96`) removed the identity `A(...)`/`T(...)` casts so the double build's
   term expressions are verbatim — an explicit `double(x)` cast on an already-
   double expression *should* be elided, but removing it removes any doubt that
   `-use_fast_math` FMA contraction is preserved. This was driven by the W7-X
   stage-1 check (below).
3. **The force split is a negative result kept as a gate, not a backend.** Its
   value is that it empirically classifies the force path as input-traffic-bound
   (not register-bound), which redirects §8.10's remaining experiments (radial
   tile, force+projection fusion) away from register-reduction strategies.
4. **Graphs are measured but not integrated.** The empty-kernel savings are an
   upper bound and only meaningful for the submission-bound axisymmetric shape;
   the higher-leverage change for that shape is the already-built (Phase 7)
   `AxisymmetricOperator`, so wiring graphs before it would optimize the wrong
   thing.

## 4. Verification

### Class A bitwise gate (the critical gate)

Both configs reproduce the frozen trajectory byte-for-byte against the Phase-8
baseline `f83a3ca`:

| Config | Effective iterations | Final residual | Final state |
| ------ | -------------------- | -------------- | ----------- |
| Solovev | 251 → 199 → 456 | FSQR 9.583e-17 | bit-identical |
| W7-X | 1877 → 1617 → 2011 (5505) | FSQR 9.778e-13 | bit-identical (`compare_states.py` 0.0) |

**Note on the recorded W7-X count.** `CLAUDE.md` and `README.md` still say
"1878 → 1617 → 2011 (5506)", but the *actual* frozen baseline — recorded in the
Phase 5 and Phase 6 handovers ("byte-identical … W7-X 1877 → 1617 → 2011") and
reproduced here by building `f83a3ca` directly — is **1877 → 1617 → 2011
(5505)**. The stage-1 count moved 1878 → 1877 with the Phase-6B device-only
force-norm reduction (a Class B reorder); `CLAUDE.md`/`README` were not updated.
This Phase-9 handover is the corrected record.

### Test matrix

| Preset | Result |
| ------ | ------ |
| `verify` (double, sanitizers ON) | **39/39** (24 unit + 14 compute-sanitizer memcheck + 1 smoke) |
| `float` | compiles; `test_accumulation`, `test_tridiagonal`, `test_force_split` run both legs |

New tests: `test_accumulation` (double-accum policy), `test_force_split`
(monolith ≡ split + timing), `test_cuda_graph` (graph + cuFFT-in-graph), and the
`cumes_benchmark_smoke` CTest gate.

## 5. Deferred (documented, not hidden)

- **Four-variant solver graph integration** (ADR-0003). Deferred until (a) the
  Phase-7 `AxisymmetricOperator` is wired into the Solovev production path, and
  (b) a real-kernel (non-empty) re-measure confirms the ~40 µs upper bound. It
  also needs a per-pass kernel-parameter update for the `m=1` gauge `zeroZ`
  scalar and the `ncurr=0` first-pass `update_iota_chi` case, so it is not a
  plain four-variant capture.
- **Fused descent/checkpoint** (§8.3 option 2, Phase 9 "optional"). Writes the
  post-descent state to the checkpoint from registers; a Class B change to the
  delicate post-descent checkpoint ordering. The benchmark harness now exists to
  measure it.
- **Force-plus-projection fusion** (§8.10, high risk). The force-split result
  (input-traffic-bound) makes this unlikely to pay; needs its own measured gate.
- **Full double `ControlRecord`** (ADR-0001). Thread double invariants through
  the host controller, not just the device accumulation.
- **Retirement of the force-split prototype and the Phase-7 rzCon reference
  path** — Phase 10 scope, once the differential tests are permanent regressions.

## 6. Next steps

1. Wire `AxisymmetricOperator` into the Solovev production path (Phase 7
   follow-up) — the highest-leverage change for the submission-bound shape, and
   the prerequisite for re-measuring graph benefit.
2. With that in place, re-measure graph submission savings with real kernels and
   decide on the four-variant integration (ADR-0003).
3. Land the §8.3 fused descent/checkpoint behind the benchmark harness.
4. Phase 10: retire the now-superseded prototypes (force split, rzCon reference
   path, `computeForcesSplit`), correct the stale W7-X count in
   `CLAUDE.md`/`README.md`, and publish the ADR-backed performance/architecture
   docs from these tested contracts.
