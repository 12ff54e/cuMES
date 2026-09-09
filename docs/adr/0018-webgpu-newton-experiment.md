# ADR-0018: Opt-in WebGPU Newton experiment

Date: 2026-09-09

Status: Experimental; disabled by default. This is a Class C algorithm change.

## Context

Main's native Newton policy uses CUDA operators and a double preconditioned
residual. The browser uses paired-f32 state and physics with a scalar-f32
radial preconditioner. Native double qualification does not establish that
finite differences of this browser residual are useful or accurate.

## Decision

Expose `newton=1` for paired (`precision=double`), fixed-boundary,
axisymmetric (`ntor=0`, `nzeta=1`) resident solves. Unsupported combinations
fail with an explicit error. Ordinary solves retain the existing policy and
arithmetic. Browser paired precision is not native WGSL f64.

Use the existing iteration operators through `enqueue_iteration_probe`.
Freeze the accepted spectral state, preconditioner elements and matrix,
constraint references and `tcon`, normalization factors, and the established
Z-m1 force gauge. Each probe recomputes geometry, magnetic fields, current
closure, and constrained forces. A disabled `ReadbackBatch` drops observation
copies and prohibits mapping during the probe. Geometry and norm status
records are copied into dedicated device scratch before operator reuse.

Krylov vectors, orthogonalization, and the preconditioned residual use f32.
Trial states use paired arithmetic, preserving displacements below one f32
ULP. Independent coordinates exclude the fixed R/Z boundary, dependent axis,
lambda m=0, and inactive parities. Trial construction restores the m=1 axis
from its first off-axis row. The existing normalized Fourier basis maps
independent coordinates into the stored state.

For the frozen residual `F`, coordinate map `B`, and direction `q`, apply
`A(q) = (F(x) - F(x + h B q)) / h`, where
`h = epsilon / max(abs(B q))`. Solve `A(delta) = F(x)` using restarted GMRES
with two orthogonalization sweeps and a true residual evaluation after each
restart. There is one control readback after the eagerly enqueued inner
solve. Invalid/nonfinite probes and singular projected systems reject the
correction.

The outer policy follows ADR-0016: 32 Krylov steps and basis vectors,
relative inner target `1e-3`, forward difference `epsilon=1e-6`, attempts
every 100 effective iterations beginning at 100, epoch age greater than 20,
and no attempt on reference/preconditioner refresh or a terminal/invalid
base. Try scales `1`, `0.5`, `0.25`, and `0.125`; accept only a valid state
whose sum of all three constrained, normalized invariant residuals improves
by more than 5%. Inner convergence alone is not an acceptance criterion.

Acceptance clears velocity and controller momentum. Rejection reevaluates
the exact saved base and requires its original residual triple exactly
before normal iteration continues. Trial evaluations do not advance the
outer controller or alter its checkpoint. The frozen caches are retained
until the normal schedule refreshes them.

`newton_step` changes epsilon only for explicit experiments. `newton_probe=1`
records a seven-step forward-difference sweep against a central `1e-4`
reference at the first eligible attempt of each stage. The sweep checks the
saved base, preconditioner buffers, and paired constraint references for
bitwise preservation. The central reference is a consistency diagnostic,
not an independent derivative oracle.

## Validation and limits

The WebGPU conformance suite exercises independent coordinate/mask
expectations, signed-zero base preservation, sub-ULP paired displacement,
identity and restarted nonsymmetric GMRES, tiny/zero right-hand sides,
singular systems, and nonfinite maps. Full Chrome conformance passed on an
NVIDIA GeForce RTX 3060 Ti; the affected Wasm build and all 14 browser CTest
checks passed.

Focused live runs used the declared axisymmetric matrix inputs below, keeping
their grids, iteration caps, and physics and applying the page's displayed
paired tolerance `1e-12` to every stage. This explicitly differs from the
native matrix tolerance `1e-16`. Each baseline/candidate pair used identical
browser input bytes, default operator flags, and the same adapter.

| Input | Baseline iterations | Newton iterations | Newton final residual triple |
| --- | ---: | ---: | --- |
| `00_solovev_reference` | 507 | 333 | `(9.903e-13, 5.272e-14, 2.357e-15)` |
| `15_vmecpp_analytical_ncurr1` | 596 | 397 | `(9.696e-13, 2.650e-14, 2.901e-18)` |

All three stages converged in both variants. Each Newton run accepted one
full-scale correction per stage, with more than 5% merit improvement. All
frozen-cache snapshots were unchanged. Forward/central JVP discrepancies at
epsilon `1e-6` stayed below `3.0e-5` in these six measured epochs. The complete
step sweep also exhibits the expected competition between truncation and
f32 rounding; the step was not tuned separately for either case.

A deliberate `newton_step=1e-30` Solovev run produced four singular-inner-solve
rejections. It recovered the exact 507-record baseline controller trace and
identical spectral and scientific-field SHA-256 digests. This exercises
rollback after actual production-oracle probes, including derived-field
reconstruction, rather than only testing a coordinate copy.

Both variants passed exported-field finiteness, configured-stage/residual,
axis/parity, oriented-Jacobian, and nonnegative B² checks. Fixed boundary
coefficients were bitwise identical between variants. Their existing paired
basis conversion differs from the binary64 input boundary by at most
`2.22e-15` in the reference case, so native exact-input-bit tests are not a
browser precision contract. Browser output also continues to record scalar
f32 plus paired-state provenance; it does not claim native double storage
arithmetic.

At equal native poloidal angles, Newton/baseline maximum physical R/Z surface
displacements were `5.72e-6 m` and `7.74e-8 m`, respectively. Relative L2 B²
differences were `6.43e-7` and `1.27e-8`. Independent stored VMEC++ 0.7.0 CPU
references have the same physical inputs and converge at their own `1e-16`
tolerance. Newton/reference maximum surface displacements were `6.59e-6 m`
and `3.03e-7 m`; corresponding baseline/reference values were `1.22e-5 m`
and `3.09e-7 m`. These are coordinate-dependent diagnostics, not substituted
convergence criteria or universal error bounds. The existing matrix validator
checks the reference physics and lambda conversion.

This supports an opt-in experiment for these measured cases, with useful
JVPs despite the scalar-f32 preconditioner. It is not general qualification
or a speed claim: warmed repeated timing was not performed. Native 19-case
double measurements do not transfer to the browser. No scalar, 3-D,
free-boundary, cross-browser, or checkpoint-replay Newton qualification is
claimed.

The existing Chrome validator accepts `CUMES_INPUT_JSON` for a fixed input
loaded into the tab-local advanced editor (`?preset=w7x`); the page retains
its displayed precision tolerance. `CUMES_CAPTURE_OUTPUT=1` records the
binary and separate spectral/field digests. Numerical logs and experimental
captures belong in `../tmp/`, outside the repository.
