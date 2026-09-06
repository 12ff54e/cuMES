# ADR-0014: Experiment with a fixed radius reference in float

- Status: Experimental, opt-in; not qualified as a default numerical policy
- Date: 2026-09-06

## Context

W7-X fails to converge in mixed float at `ftol=1e-5`. Quantizing only the
large R_00 coefficient of a converged ns=99 double state raises its
double-evaluated FSQR from about `1e-12` to `2.1083e-4`. Quantizing R_00
relative to its edge value instead gives `4.8304e-9`. Norm-factor errors
are below one part per million and do not explain the stall.

## Decision

Provide `CUMES_RADIUS_REFERENCE=1` and the equivalent explicit
`SolveRequest::use_radius_reference` option for fixed-boundary 3-D float
solves. Retain double boundary coefficients as an immutable m=0 Rcc
reference. Store, update, transform and radially differentiate the small
float displacement. Restore the reference in absolute-radius terms and
retain its toroidal derivatives. Carry it through grid refinement, and
export physical coefficients through the existing double snapshot and
checkpoint formats. Double, axisymmetric and free-boundary solves ignore
the option. No convergence thresholds or validity gates change.

## Evidence and limits

The first two W7-X grids now converge at `1e-5` in 149 and 278 effective
iterations. The final grid still stalls: its best maximum residual is
`1.4070e-5` over 5000 passes. A tight double checkpoint restarted in the
modified float solver converges at ns=99, and its exported float checkpoint
converges again on the first replay pass.

This is a partial improvement. Full Class C acceptance still requires
cold-start convergence on the final grid and broader numerical
qualification. Consequently the option defaults to off. Extra double
geometry arithmetic and compensated descent updates did not complete
convergence and are not retained.

Double W7-X and Solovev retain byte-identical checkpoints and final-stage
telemetry; `compare_runs` reports identical state families and restart
sequences. The float suite passes 63/63 and verify passes 99/99, including memcheck
and initcheck variants. The new reference test also passes a separate float
Compute Sanitizer memcheck run. Measurements, reproduction instructions,
and the remaining limitations are in the
[investigation](../w7x-float-convergence.md).
