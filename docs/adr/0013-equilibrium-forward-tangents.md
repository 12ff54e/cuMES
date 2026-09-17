# ADR-0013: Fixed-boundary equilibrium forward tangents

Status: Implemented for the qualified fixed-boundary precise-double scope

## Context

An optimizer varies fixed-boundary Fourier coefficients `x`, cuMES solves for
an equilibrium state `u`, and the optimizer evaluates a residual vector from
the resulting fields.  Re-solving the nonlinear equilibrium once per boundary
coefficient is robust but makes a dense least-squares Jacobian prohibitively
expensive.  The final Landreman QH construction stage has 120 boundary
variables and 44,353 optimizer residuals.

The Jacobian discussed here is the derivative of the equilibrium map.  It is
unrelated to the coordinate Jacobian `sqrt(g)` and its validity gate in the
existing solver.

## Decision

cuMES will provide forward sensitivities of a converged fixed-boundary
equilibrium.  It will not know about QS, aspect-ratio, iota, or optimizer
weights.  Those target definitions and their chain rule remain in meow.

For the converged discrete equilibrium equations

```text
F(u, x) = 0,
```

the tangent for a boundary direction `dx` is defined by

```text
F_u du = -F_x dx.
```

The production tangent operator must differentiate the discrete CUDA
operators directly.  It must not perturb the boundary or re-run the nonlinear
solver.  Boundary finite differences are retained only as an independent test
oracle.

The first qualified scope is precise-double, stellarator-symmetric,
fixed-boundary equilibrium.  Free-boundary vacuum response and mixed-float
tangents are later qualifications, not silent fallbacks.

## Ownership and API boundary

cuMES owns:

- a typed boundary-direction representation in the validated folded Fourier
  basis;
- a retained final-grid linearization context;
- matrix-free applications of `F_u` and `F_x`;
- a preconditioned tangent linear solve;
- tangent spectral state, derived fields, and radial equilibrium profiles.

meow owns:

- the mapping from optimizer variables to cuMES boundary directions;
- directional derivatives of QS, aspect ratio, and mean iota;
- assembly of the dense residual Jacobian required by TRF;
- decisions about dense, block, or matrix-free optimization.

The public cuMES interface will expose a solve-and-linearize session rather
than adding target-specific data to `SolveOutcome`.  Multiple tangent
directions reuse one converged primal state and one final-grid operator
context.  A block interface may execute directions together on the GPU, but
its mathematical result is identical to applying one direction at a time.

## Discrete linearization contract

The differentiated residual is the unpreconditioned, gauge-fixed spectral
equilibrium residual before Garabedian descent.  Adaptive time-step control,
restart decisions, convergence tests, and multigrid prolongation are not part
of `F`.

For fixed boundary:

- prescribed LCFS `R` and `Z` tangent rows equal the supplied boundary
  direction;
- interior `R`, `Z`, and lambda are solved variables;
- axis regularity and the existing lambda/m=1 gauge convention are imposed in
  both the primal residual and tangent operator;
- the tangent derived-field snapshot uses exactly the primal output grid and
  layout.

The existing nonlinear preconditioner may precondition the tangent Krylov
solve, but it is not treated as `F_u^{-1}` and cannot define the derivative.

## Qualification gates

1. **Contracts and host algebra.** Boundary/tangent snapshot types use checked
   layout operations. Synthetic equilibrium tangents validate meow's target
   JVPs.
2. **Reusable residual evaluation.** Final-grid residual evaluation is
   separate from descent and retains its workspaces after convergence. The
   published converged state produces the same invariant residual without
   changing the frozen nonlinear trajectory.
3. **Analytic CUDA JVP.** CPU references and directional finite differences
   validate the tangent transform, geometry, magnetic-field, profile, force,
   constraint, and derived-field operators.
4. **Linear tangent solve.** Restarted GMRES uses explicit boundary and gauge
   rows, reports its linear residual, and fails explicitly on
   stagnation or unsupported problem classes.
5. **meow integration.** A solve-and-linearize session is cached at each
   optimizer point. Every boundary degree of freedom maps to a direction; each
   equilibrium tangent propagates through the target JVP into the matrix passed
   to `meow::JacobianFunction`.
6. **Qualification.** Small Solovev and three-dimensional fixtures compare
   tangent columns against centered nonlinear finite differences, and compare
   the analytic and finite-difference QH Jacobians by column norm, directional
   prediction, `J^T r`, accepted TRF steps, and final objective while preserving
   the existing Class-A nonlinear trajectories.

The fixed-boundary QH directional and gradient gates are required in addition
to a successful build.

## Implementation outcome

The public implementation consists of `BoundaryTangent`,
`EquilibriumTangent`, and `EquilibriumLinearization`. The retained session
evaluates analytic CUDA JVPs, solves the gauge-fixed tangent system with
right-preconditioned restarted GMRES, and materializes the spectral,
geometry/magnetic-field, and `Phi'`/`chi'`/`iota`/`I`/`G` derivatives consumed
by optimizer targets. It never evaluates an optimizer target.

The cuMES sensitivity suite checks the dual algebra, nonlinear operator JVP,
residual JVP, boundary solve, and target-facing materialization. The meow
integration independently checks folded optimizer directions and compares a
strict QH target tangent against centered nonlinear equilibrium solves. At the
mode-1 analytic QH start, the complete residual-vector derivative differs by
5.2% while the coordinate-invariant objective derivative differs by 0.14%.
The residual-vector difference reflects the equilibrium lambda/radial gauge
branch; both tangent states satisfy the linearized equilibrium residual.

The production meow driver uses the default `1e-4` relative linear tolerance
for trust-region work. One accepted-step scaling runs completed every QA mode
through 80 variables and every QH mode through 120 variables. A future block
interface can reduce the still-linear cost in the number of boundary
directions without changing this contract.

## Resident tangent GMRES

The CUDA tangent solve defaults to a resident backend through
`TangentLinearOptions::backend`. It reuses `DeviceGmres`, including its device
reductions, two-pass classical Gram-Schmidt, Givens rotations, and true
residual checks. The original host modified Gram-Schmidt backend remains
selectable as `TangentLinearBackend::HOST`. Reordering the reductions and
using reorthogonalization is a numerical algorithm change (Class C), rather
than a bitwise scheduling change.

The retained workspace owns a private ordered stream, uploads the primal and
active-degree maps once, and packs/gathers the existing analytic JVP and
preconditioner on the device. cuFFT plans follow the supplied compute stream;
public residual and field materialization calls can subsequently reuse the
same evaluator on the default stream. Boundary/gauge conventions and physical
operators are unchanged.

`DeviceGmres` accepts incremental submission and separate absolute and relative
tolerances. The tangent caller reads one compact control record after at most
16 Arnoldi steps, avoiding a full restart's unnecessary JVPs after projected
convergence. Each completed cycle recomputes the true residual before reporting
convergence. The existing fixed-submission Newton interface delegates to the
same primitives and retains its fixed operator budget. Restart 300 is supported;
the basis limit protects integer Hessenberg indexing instead of imposing the
former limit of 256.

Qualification compares Solovev, QA, and QH states, fields, target columns,
true residuals, and iteration counts against the retained host backend on
Pascal and Ada. Manufactured float/double GMRES systems cover incremental
restarts, capped budgets, singular/nonfinite maps, and basis 300. The boundary
regression also covers zero RHS, absolute stopping, workspace reuse, backend
switching, unpreconditioned restarts, and prescribed LCFS rows. The existing
Solovev nonlinear finite-difference oracle and meow target-derivative checks
remain required. Measurements and their scope are in
[performance.md](../performance.md).

This backend accelerates the same implicit tangent map. It does not resolve
the documented differences between that map and the cold nonlinear optimizer
derivative, and does not qualify analytic Jacobians as a replacement for the
fast rundown's cold finite differences.

## Consequences

A dense Jacobian still contains one column per boundary variable, so forward
tangents do not make its arithmetic independent of parameter count.  They do
replace many full nonlinear solves with linear solves that reuse the same
operator and preconditioner, and they admit block GPU execution.  If the
parameter count later dominates, the same JVP foundation can support a
matrix-free Gauss-Newton method; a scalar-objective adjoint is a separate
future decision.
