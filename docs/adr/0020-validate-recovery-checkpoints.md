# ADR-0020: Retain the preceding checkpoint until a descent is validated

- Status: Accepted for native CUDA recovery
- Date: 2026-09-13

## Context

meow's QH construction with the newer inverse Fourier implementation reaches
a cold finite-difference input that fails on both the old and new cuMES
revisions. The first radial stage saves its post-descent state at effective
iteration 25. The next pass rejects that geometry, restores the same invalid
checkpoint, and repeats iteration 26 until the 10,000-pass budget is exhausted.
Reducing the time step cannot repair a state that is rejected before descent.

Capturing the post-descent state is intentional and preserves the established
VMEC trajectory. Moving every checkpoint to the pre-descent state would alter
successful restarts, including weakly determined lambda modes. The defect is
discarding the preceding checkpoint before checking the replacement.

## Decision

Keep two state-only checkpoint slabs in the native CUDA solver. On the existing
refresh decision, rotate their ownership and copy the post-descent state into
the new checkpoint. Mark it pending until the next evaluated state passes both
the oriented-Jacobian gate and the finite invariant-residual gate.

If either gate rejects that pass, discard the pending checkpoint and restore
the preceding one. Existing velocity reset, time-step reduction, restart
anchoring, cache refresh, iteration caps and convergence thresholds still
apply. A successful validation keeps the new checkpoint. An invalid initial
state has no preceding checkpoint and retains the existing failure behavior.

This changes recovery policy (Class C). It preserves operator arithmetic and
the checkpoint selected on trajectories whose refreshed checkpoints pass
validation. There are no additional iteration kernels, transfers, or fences:
rotation only swaps host buffer ownership and each refresh still performs one
device-to-device copy. The extra allocation is made at stage setup and costs
`6 * ns * mnmax * sizeof(T)` bytes: 100,800 bytes on the QH final double grid
(`ns=50`, `mnmax=42`). The WebGPU implementation is outside this change.

## Verification

`tests/fixtures/qh_checkpoint_recovery.json` is the exact positive column-30
input from meow's QH mode-3 construction on RTX 4090. It retains the accepted
axis predictor, all three radial stages, `tcon0=2`, and `1e-12` tolerances.
Changing it to a single radial stage would change the cold initialization
envelope and remove the failure being tested.

The fixture now converges on TITAN Xp / CUDA 12.1 and RTX 4090 / CUDA 12.9.
The pending checkpoint is discarded at iteration 26. With B-spline transfer,
the three stages on TITAN Xp take 1,139, 1,039 and 874 effective iterations.
`test_checkpoint_recovery` also exercises the Catmull-Rom transfer used by
meow, checks every original stage tolerance, exact prescribed boundary,
finite output fields, positive oriented Jacobian and magnetic energy, and
one-pass final-grid replay with preserved coefficients. The same test fails
against the pre-fix library for both transfer policies.

All 63 tests in the local double integration build pass, including the
float-kernel type audit. The recovery regression passes memcheck and initcheck
with zero errors. Solovev, W7-X, and fixed QA/QH inputs at mode-3 resolution
retain byte-identical spectral/half-grid arrays and stage residual, iteration,
and restart records compared with the pre-fix library on TITAN Xp.

VMEC++ 0.7.0 rejects `tcon0=2`. An independent diagnostic therefore uses
`tcon0=1` in both solvers while retaining the other input values. Both converge
at `1e-12`; the maximum R/Z surface displacement at equal native coordinates
is 2.20 mm and the RMS displacement is 0.668 mm on a 64-by-64 angular grid,
excluding the dependent axis row. This comparison has no gauge alignment and
is a diagnostic, not a replacement convergence criterion. Production meow
inputs retain `tcon0=2`.

Raw before/after runs, comparison scripts, sanitizer logs, and the downstream
construction qualification are retained under
`../tmp/meow-performance-20260912/qh-recovery/`, with the RTX 4090 artifacts in
the corresponding `/tmp/meow-performance-20260912/qh-recovery/` on `gervais`.
