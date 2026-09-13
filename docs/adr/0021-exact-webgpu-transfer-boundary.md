# ADR-0021: Preserve the stored LCFS during WebGPU radial transfer

- Status: Accepted
- Date: 2026-09-13
- Numerical classification: Class C for affected multigrid trajectories

## Problem and decision

The asymmetric heliotron benchmark exposed a fixed-boundary displacement of
`5.96e-8` m after WebGPU radial refinement to 256 surfaces. Evaluating the
endpoint through WGSL division and square-root normalization need not produce
exactly one. The resulting high-word error survives paired-f32 low-word
recombination and changes the prescribed physical boundary.

Copy each stored spectral profile's final value directly to the new final
surface, with zero velocity. Apply this to both linear and Catmull–Rom transfer
and to six- and twelve-family states. Interior interpolation, odd-m axis
regularity, paired low-word transfer, and the solver controller remain as
specified previously. For a free-boundary stage transition, this preserves the
current boundary until subsequent vacuum-coupled descent updates it.

The correction restores an existing invariant, but the corrected boundary can
change subsequent residuals and controller decisions. It therefore receives
Class C qualification rather than a claim of identical affected trajectories.

## Validation and limits

Real-adapter conformance checks compare transfer against the existing scalar
reference, require zero velocity and exact endpoint equality, and exercise
both interpolations with six-family 5-to-8 and twelve-family 128-to-256 grids.
The complete Chrome conformance suite also checks symmetric Solovev multigrid
convergence and the scalar/paired asymmetric operator references.

The three fixed-boundary cases in
[the Fortran VMEC benchmark](../../benchmarks/asymmetric_vmec/README.md)
provide full-solve residual, finite-state, oriented-Jacobian and LCFS checks.
That document records the tested browser, precision, resolutions, independent
VMEC differences, and the native fixed-point checkpoint replays. Paired output
LCFS errors are checked against `16*epsilon_f32^2*max(1, max(abs(Rcc_edge)))`;
the transfer operator itself requires exact stored endpoint equality.

This change does not establish agreement with Fortran VMEC at residual-sized
error, automatic axis repair, scalar convergence for every high-aspect-ratio
case, or additional free-boundary qualification.
