# Boozer field-period coordinate correction plan

## Problem

`magnetic-coordinate` solves the Boozer toroidal shift `nu` in physical
toroidal radians. Its Fourier solve uses `m*iota + n*nfp`, where `n` is the
field-period Fourier integer. The version-2 output contract nevertheless
describes the stored field-period angle as if

```text
zeta_b = zeta + nu.
```

Writing the source and Boozer field-period angles as `alpha = nfp*zeta` and
`alpha_b = nfp*zeta_b`, the consistent relation is

```text
alpha_b = alpha + nfp*nu.
```

The missing factor has little visible effect on a QA diagnostic, whose desired
modes have toroidal integer zero, but it corrupts the helical phase used by a
QH `B_mn` diagnostic. The plotting code also labels the stored mixed grid
`(theta_b, alpha)` as a grid uniform in both Boozer angles.

## Ownership

- `deps/magnetic-coordinate` owns the transform, its public output contract,
  and manufactured coordinate tests.
- cuMES owns readers and equilibrium plotting for magnetic-coordinate output.
- meow owns the optimizer-side symmetry diagnostic and comparison figures.
- Generated equilibrium, Boozer, and plot artifacts remain outside either
  repository, in the requested optimization result directories.

The equilibrium solver and the Landreman target evaluator are unchanged. The
QS objective is evaluated directly from equilibrium quantities and does not
depend on a Boozer transform.

## Versioned correction

1. Publish output schema version 3 rather than silently changing version 2.
   Version 3 keeps `nu` in physical toroidal radians and names the stored
   toroidal grid `alpha`, with `alpha_b = alpha + nfp*nu`.
2. Add an analytic `nfp > 1`, toroidally varying manufactured test that checks
   both the physical shift and the field-period map/Jacobian.
3. Make cuMES construct a genuinely uniform `(theta_b, alpha_b)` plot grid by
   periodically inverting the toroidal map at fixed `theta_b`. Geometry and
   `B` are interpolated from the mixed source grid only after that inversion.
4. Make meow's `B_mn` quadrature retain `nfp` from the header and use
   `alpha_b = alpha + nfp*nu` and
   `d alpha_b/d alpha = 1 + nfp*dnu/dalpha`.
5. Convert QH outputs with at least `48 x 32` output angles. A `96 x 64` grid
   is preferred for regenerated presentation artifacts; the result must be
   checked against `48 x 32` before it is accepted.

## Verification and commits

Changes are committed by owner and only after the associated checks pass:

1. plan and contract decision in cuMES;
2. schema-v3 writer/reader and manufactured tests in `magnetic-coordinate`;
3. cuMES reader/remap/plot tests plus the updated dependency revision;
4. meow symmetry-consumer tests;
5. regenerated untracked QH artifacts and a numerical comparison with the
   supplemental `booz_xform` result.

The acceptance diagnostic is the edge symmetry-breaking spectrum. The broken
version-2 consumer produced approximately `1.81e-2`; a corrected `48 x 32`
calculation produced `5.49e-4`, compared with `4.78e-4` from the supplemental
terminal QH `booz_xform` file. The residual difference is evaluated separately
from the coordinate bug because the local optimization stopped much earlier
than the supplemental refinement.
