# ADR-0011: Use cubic coarse-to-fine multigrid transfer

- Status: Accepted
- Date: 2026-08-31

## Context

Every refined radial stage previously started from a two-point linear
interpolation of the converged coarse spectral coefficients. The equilibrium
profiles are radially smooth, so linear transfer discards coarse-grid
curvature and leaves the fine stage to reconstruct it through full force
evaluations. This is an iteration-algorithm limitation, independent of initial
time-step or cold-seed parameter tuning.

Odd poloidal modes require interpolation in the existing `scalxc`-decomposed
coordinate to preserve their near-axis regularity. Any replacement must also
copy the LCFS exactly, zero the new odd-mode axis, reset velocity, and leave
the convergence residual unchanged.

## Decision

Use a four-point Catmull-Rom interpolant on the uniform normalized-flux grid.
Apply it independently to all six spectral families in the same decomposed
coordinate used by the linear transfer. At either radial endpoint, obtain the
missing neighbor by linear extrapolation. Exact coarse nodes, the LCFS, the
odd-mode axis extrapolation, and velocity reset remain explicit contracts.

Enable cubic transfer for precise-double fixed-boundary continuation and 3-D
free-boundary continuation. Retain linear transfer for axisymmetric
free-boundary runs, where the moving-boundary response regressed, and for
mixed-float until separately qualified. `CUMES_FORCE_LINEAR_PROLONGATION=1`
is the diagnostic opt-out.

## Evidence

All retained cases satisfy every configured residual component:

| workload | linear transfer | cubic transfer | reduction | final FSQR |
| -------- | --------------: | -------------: | --------: | ---------: |
| W7-X fixed `33 → 66 → 99` | 4507 | 4160 | 7.70% | `9.986e-13` |
| Solovev fixed `5 → 11 → 55` | 815 | 766 | 6.01% | `9.695e-17` |
| CTH-like free `15 → 25` | 431 | 424 | 1.62% | `9.920e-11` |

W7-X stage counts change from `1315 → 1559 → 1633` to
`1315 → 1443 → 1402`; Solovev changes from `235 → 193 → 387`
to `235 → 190 → 341`. Restarting the cubic W7-X final-grid checkpoint
converges in one iteration with the identical residual triple.

The rejected axisymmetric free-boundary case changed from 1025 to 1045 passes,
so it retains linear transfer. A dedicated double/float CUDA test compares
both schemes against a CPU scalar implementation on a non-integer refinement
ratio and checks all six families, exact LCFS preservation, odd-axis zeroing,
and velocity reset under memcheck and initcheck.

## Consequences

- Refined stages begin closer to their fine-grid fixed points without adding
  force evaluations, persistent memory, or a new convergence parameter.
- The prolongation kernel reads two additional neighboring coarse rows; this
  one-time stage-boundary cost is negligible relative to saved iterations.
- Default multigrid trajectories are Class C numerical changes. The explicit
  linear opt-out preserves direct A/B access to the previous trajectories.
