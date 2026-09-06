# ADR-0014: Fixed radius reference for float state

- Status: Default for fixed-boundary 3-D float; other solve classes unchanged
- Updated: 2026-09-07
- Date: 2026-09-06

## Context

W7-X fails to converge in mixed float at `ftol=1e-5`. Quantizing only the
large R_00 coefficient of a converged ns=99 double state raises its
double-evaluated FSQR from about `1e-12` to `2.1083e-4`. Quantizing R_00
relative to its edge value instead gives `4.8304e-9`. Norm-factor errors
are below one part per million and do not explain the stall.

## Decision

Use radius-reference storage by default for fixed-boundary 3-D float solves.
`CUMES_RADIUS_REFERENCE=0` and `SolveRequest::use_radius_reference=false`
restore the previous absolute-coefficient representation. Retain double boundary coefficients as an immutable m=0 Rcc
reference. Store, update, transform and radially differentiate the small
float displacement. Restore the reference in absolute-radius terms and
retain its toroidal derivatives. Carry it through grid refinement, and
export physical coefficients through the existing double snapshot and
checkpoint formats. Double, axisymmetric and free-boundary solves ignore
the option. No convergence thresholds or validity gates change.

## Default change and cost (2026-09-07)

The representation is now the default in the public solve request, parameter
builder and float fixed-iteration benchmark. `CUMES_RADIUS_REFERENCE=0`,
`SolveRequest::use_radius_reference=false`, or the benchmark's
`--radius-reference 0` retains the previous representation.

This is an accuracy choice, not an improvement in every respect. On TITAN Xp,
three alternating W7-X ns=99 benchmark runs (300 warmup + 500 measured passes,
same tight checkpoint, native odd reconstruction) give median per-pass times
of 525.07 microseconds with absolute coefficients and 543.09 microseconds
with reference storage: approximately 3.4% extra cost. The default is justified
by retaining radial detail and the improved convergence behavior, rather than
by a per-pass speedup. These timings are separate from the incremental cost of
float-float reconstruction. The run records are in
`/lustre/qzhong/cumes-diagnostics/w7x-float-investigation/default-reference/`.

Validation of the default change: 64/64 float tests and the four selected
verify API/reference tests pass, including reference memcheck and initcheck.
The new default with poloidal compensation reproduces the previously explicit
reference run's checkpoint and final-stage telemetry byte-for-byte; its
iteration counts remain 149 → 277 → 322. Tests cover the explicit opt-out and
unchanged double, axisymmetric and 3-D free-boundary defaults.

## Evidence and limits

The first two W7-X grids now converge at `1e-5` in 149 and 278 effective
iterations. The final grid still stalls: its best maximum residual is
`1.4070e-5` over 5000 passes. A tight double checkpoint restarted in the
modified float solver converges at ns=99, and its exported float checkpoint
converges again on the first replay pass.

Reference storage alone is a partial improvement. It originally defaulted to
off while the remaining inverse-transform error was investigated. The
subsequent selective float-float reconstruction completes W7-X cold-start
convergence at 1e-5. On 2026-09-07 the reference representation became the
default for fixed-boundary 3-D float because it preserves small radial
structure and improves the tested W7-X trajectory. This does not establish
convergence for every 3-D configuration or tolerance; broader qualification
remains open, and the float-float reconstruction is still a separate opt-in. Extra double
geometry arithmetic and compensated descent updates did not complete
convergence and are not retained.

Double W7-X and Solovev retain byte-identical checkpoints and final-stage
telemetry; `compare_runs` reports identical state families and restart
sequences. The float suite passes 63/63 and verify passes 99/99, including memcheck
and initcheck variants. The new reference test also passes a separate float
Compute Sanitizer memcheck run. Measurements, reproduction instructions,
and the remaining limitations are in the
[investigation](../w7x-float-convergence.md).

## Diagnostic follow-up

A subsequent [stage-substitution investigation](../w7x-float-inverse-diagnostic.md)
identifies odd R/Z inverse reconstruction as the remaining float bottleneck.
A double-arithmetic GPU oracle correcting only `r_o` and `z_o`, with float
state and float outputs, completes cold-start convergence at `1e-5`. This
was followed by the retained [selective float-float implementation](../w7x-float-float.md).
The full double oracle remains diagnostic; the minimal poloidal correction
converges with approximately 3.2% measured extra per-pass cost on TITAN Xp.
