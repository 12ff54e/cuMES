# ADR-0016: Expose the qualified Newton policy as an explicit option

- Status: Accepted, opt-in only
- Date: 2026-09-08

## Context

The [19-case axisymmetric study](../axisymmetric-newton-qualification.md)
finds substantial speedups for some inputs on TITAN Xp and RTX 4090, including
19.21% / 14.51% for prescribed-current Solovev. The same policy slows several
Ada cases. Axisymmetry alone does not justify changing the default, but users
can explicitly select the policy for their workloads.

## Decision

Expose `--newton` in the ordinary CLI and
`cumes::SolveRequest::enable_newton = true` in the embedding API. The default
is false. Require double precision, fixed boundary, `ntor=0` and `nzeta=1`;
unsupported requests fail before GPU setup. All configured multigrid stages,
tolerances and iteration caps remain in force. Passing the flag again selects
corrections for a checkpoint restart; a converged checkpoint still exits on
its first residual evaluation.

Use the tested policy without per-case parameter selection:

- forward differences with maximum physical coefficient perturbation `1e-6`;
- 32 Krylov steps and a 32-vector basis, inner relative tolerance `1e-3`;
- corrections every 100 effective iterations, starting at 100 on each stage;
- epoch age greater than 20, stable m=1 gauge, no preconditioner/reference
  refresh, and a valid, finite, nonterminal preconditioned base evaluation;
- trial scales `1`, `1/2`, `1/4`, `1/8`, requiring valid finite geometry and a
  reduction of at least 5% in the sum of the three invariant residuals.

The exact descent-coordinate map, GPU GMRES and frozen-epoch Newton operator
live in `include/cumes/numerics/`, with kernels in
`src/kernels/newton_impl.cuh` and explicit double/float instantiation TUs.
Experimental probes use compatibility aliases for this same implementation.
Float component instantiations remain testable; the full-solver flag rejects
float until separately qualified. The supported speed evidence uses precise
`verify-double` arithmetic.

Correction scratch and the pinned GMRES control mirror are allocated before
iteration, only when enabled. Probes use the live equilibrium operator with
its preconditioner, constraint reference/multiplier and normalization frozen.
Every probe recomputes geometry, fields, forces and prescribed-current closure.
Probe and line-search evaluations temporarily suppress dump output so they
cannot overwrite real outer-iteration observation files.
Outer dump windows describe the base evaluation; when a correction is
accepted, that iteration's scalar controller record describes the accepted
state. The two observations need not describe the same coefficients.

An accepted correction zeroes velocity and resets damping samples and the
running residual minimum. It preserves timestep, iteration/restart counters,
gauge history, and refresh cadence, then returns to normal controller
classification. A rejected correction restores the state and re-evaluates its
fields and current closure with the same caches; its residual triple must
match before ordinary descent continues. Inner breakdown is a rejection,
not full-solver convergence or failure. All three original component
thresholds still define convergence.

## Verification

The promotion changes code ownership and adds explicit option plumbing. Its
kernel arithmetic and launch ordering are preserved from the experiment.
One baseline and one enabled solve for every case on each GPU (76 complete
solves) reproduce the archived corresponding state, every derived field,
all stage residuals/counts/restarts, and Newton correction/evaluation counts
exactly. This includes the `mpol=4` rejected correction and prescribed-current
closure. A full default W7-X solve also reproduces its archived state, derived
fields and stage records exactly. Default-off execution allocates no Newton
workspace.

Durable tests cover CLI help and unsupported requests, default-off/API
validation, controller momentum reset without epoch changes, and complete
Solovev, `mpol=4` and prescribed-current solves with final-grid replay. The
promoted GMRES and coordinate tests exercise both scalar types. Exact stage
counts are asserted for the qualified precise-double/B-spline build; other
builds retain physical, convergence and replay checks.

The complete verify suite passes 69/69 tests, including installed-package
consumption and the float-kernel FP64 audit. The float build passes its CLI,
API, controller and kernel-type checks. On RTX 4090, the full prescribed-current
flag run passes CUDA 12.9 memcheck and initcheck with graphs enabled and
produces bit-identical results to its unsanitized run. A dump-enabled Solovev
run preserves the qualified state and stage records without writing fake
iteration-1000 probe files.

Full promotion evidence and commands are retained in
`../tmp/cumes-newton-flag-20260908/`. The original timing and independent VMEC++
comparisons remain in `../tmp/cumes-axisymmetric-newton-20260908/`. Promotion
checks establish numerical equivalence to that experiment; they are not a new
cross-GPU performance measurement. The original CUDA 12.1 graph-initcheck
caveat remains documented in the qualification report.

## Consequences

Enabling the flag deliberately selects a Class C convergence trajectory.
For the qualified Solovev input, stage counts become `144 / 113 / 227`
instead of `235 / 193 / 326`; every configured tolerance is satisfied.
Users can compare the enabled policy on their own inputs without rebuilding
or applying an experimental patch. The default policy and its performance
qualification remain unchanged.

The existing native `RunReport` schema does not serialize solver-request
options. Preserve the CLI command or embedding request alongside the output
for reproduction; verbose stage logs identify Newton policy and correction
counts. This addition does not change the native/checkpoint format.
