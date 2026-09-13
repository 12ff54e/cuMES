# Comparing asymmetric equilibria with Fortran VMEC

Matching the boundary, profiles, grid and nominal `FTOL` does not fully match
the finite-dimensional equilibrium problem. Fortran VMEC2000 and cuMES use
different effective spectral-condensation strengths for `lasym=true`, and
their iteration histories can freeze different m=1 coordinate coefficients.
The QH audit below isolates these effects from the prescribed-current closure.

The Fortran reference in this document is
[`c965d31`](https://github.com/hiddenSymmetries/VMEC2000/tree/c965d31faf732ca77d280ef509a7bdefe7797292).
These statements describe that implementation, not every VMEC distribution.
The measured cuMES implementation is `c60fc7b`, CUDA double on TITAN Xp.

## Constraint strength and residual normalization

cuMES uses a normalized Fourier bandpass on the full periodic theta grid.
Its radial constraint multiplier is linear in the supplied `tcon0`; see
[`constraint_impl.cuh`](../src/kernels/constraint_impl.cuh),
[`fourier_impl.cuh`](../src/kernels/fourier_impl.cuh), and
[the mathematical contract](mathematics.md#8-spectral-condensation-constraint).
Fortran has two additional asymmetric factors:

1. `bcovar.f` replaces the supplied value with `min(abs(tcon0), 1)` and halves
   the computed `tcon` profile when `lasym` is true.
2. `fixaray.f` uses the full-theta integration normalization for `lasym`,
   while `alias.f` integrates paired half-theta contributions with a further
   half factor. Its bandpass is half the normalized full-period projection.

For the same geometry, preconditioner and constraint reference, the combined
effective strength, expressed in cuMES's normalized bandpass convention, is

```text
cuMES:                  supplied tcon0
Fortran, lasym=false:    min(abs(supplied tcon0), 1)
Fortran, lasym=true:     min(abs(supplied tcon0), 1) / 4
```

Thus the QH namelist's `TCON0=2` corresponds to cuMES `tcon0=0.25` for an
explicitly matched-constraint comparison. Passing 2 to both codes makes the
cuMES constraint eight times stronger. This factor multiplies the artificial
spectral-condensation contribution, not the MHD force or prescribed current.

The source locations are
[`bcovar.f`](https://github.com/hiddenSymmetries/VMEC2000/blob/c965d31faf732ca77d280ef509a7bdefe7797292/Sources/General/bcovar.f),
[`fixaray.f`](https://github.com/hiddenSymmetries/VMEC2000/blob/c965d31faf732ca77d280ef509a7bdefe7797292/Sources/Initialization_Cleanup/fixaray.f),
and [`alias.f`](https://github.com/hiddenSymmetries/VMEC2000/blob/c965d31faf732ca77d280ef509a7bdefe7797292/Sources/General/alias.f).

There is also a force-residual normalization difference. On the same QH state,
with the effective constraint matched, cuMES's unpreconditioned Fourier forces
are twice Fortran's, and its three squared residuals are four times Fortran's.
This global factor preserves the force zero, but a common numeric threshold
does not impose the same stopping criterion. In particular, the shared nominal
`FSQZ < 1e-6` m=1-freezing threshold can switch at different states.

The normalized cuMES projection deliberately preserves the toroidal-rotation
equivalence of a symmetric state and its twelve-family representation. That
invariant is covered by `test_asymmetric_solver` and
[ADR-0020](adr/0020-non-stellarator-symmetry.md). Copying Fortran's quarter
factor into every asymmetric solve would change that contract. Reference
comparisons must explicitly record effective constraint strength; the audit
does not change the production force projector or input interpretation.

## The m=1 coordinate constraint

Both solvers mix the m=1 `Rss/Zcs` and `Rsc/Zcc` forces, then suppress their
difference components once the residual is small. With decayed velocity, this
freezes the corresponding state differences

```text
Rss(m=1,n,s) - Zcs(m=1,n,s)
Rsc(m=1,n,s) - Zcc(m=1,n,s).
```

The values depend on the initial interior state, multigrid transfer and
iteration history. The QH fixture gives cuMES a converged reference axis, but
cuMES constructs its own interior with zero lambda. It does not start from
Fortran's complete interior state. Matching only the axis and LCFS therefore
does not match the frozen m=1 coefficients.

See [`residue.f90`](https://github.com/hiddenSymmetries/VMEC2000/blob/c965d31faf732ca77d280ef509a7bdefe7797292/Sources/General/residue.f90),
cuMES's `m1_constraint_kernel` in
[`solver_impl.cuh`](../src/kernels/solver_impl.cuh), and
[`control_policy.hpp`](../include/cumes/solver/control_policy.hpp).

At finite radial and angular resolution, different frozen coordinate choices
can produce different iota profiles even when both projected force systems
converge. Tightening `FTOL` alone cannot align those frozen coefficients.

## QH isolation experiment, 2026-09-13

This is the regenerated finite-pressure, prescribed-current
Landreman–Sengupta JPP 2019 section 5.5 case in
[`benchmarks/asymmetric_vmec`](../benchmarks/asymmetric_vmec/README.md):
`nfp=5`, `mpol=8`, `ntor=12`, `ntheta=22`, `nzeta=28`, final `ns=51`.
The first iota sample is at half-grid `s=0.01`, not extrapolated `s=0`.

The unchanged original Fortran `add_fluxes` routine, evaluated on reconstructed
cuMES geometry and lambda, reproduces native iota within `1.56e-10` over the
whole profile. The prescribed enclosed current is `5000*s` A in both codes;
the maximum current difference in this audit is `5.46e-11` A.

For the force comparison, VMEC's `wout` state was converted back to its internal
full-mesh representation. This requires undoing the half-mesh lambda average
and flux normalization and restoring m=1 axis storage. The reconstruction
follows [`load_xc_from_wout.f`](https://github.com/hiddenSymmetries/VMEC2000/blob/c965d31faf732ca77d280ef509a7bdefe7797292/Sources/Initialization_Cleanup/load_xc_from_wout.f)
and [`wrout.f`](https://github.com/hiddenSymmetries/VMEC2000/blob/c965d31faf732ca77d280ef509a7bdefe7797292/Sources/Input_Output/wrout.f).
Its reconstructed fields agree with the `wout` fields to approximately
`1e-11`. A separately linked Fortran executable injected this same state
before the inverse transform and dumped the original force evaluation.

Measured on that common state:

| Quantity | Result |
| --- | ---: |
| Original cuMES / Fortran filtered constraint amplitude | 8.00000000037 |
| Ratio after setting cuMES `tcon0=0.25` | 1.00000000005 |
| Maximum Fourier-force difference before matching, comparing cuMES with twice Fortran | 1.26414e-4 |
| Maximum Fourier-force difference after matching, same normalization | 1.39927e-12 |

The original cuMES equilibrium gives Fortran residuals
`(1.3743e-7, 3.4185e-7, 1.1281e-13)`: lambda/current closure agrees, while the
R/Z equations disagree because of constraint strength. Conversely, the cuMES
equilibrium with matched strength and strict tolerance gives Fortran residuals
`(2.4976e-17, 1.2961e-17, 5.8520e-18)`. Fortran computes the same iota for that
state, including its difference from Fortran's own cold-start solution.

The final isolation replaced only the two m=1 difference families with the
strict Fortran values, retained cuMES's other coefficients as the starting
state, and relaxed all remaining degrees of freedom with `tcon0=0.25`.
An isolated CUDA build held those difference forces at zero throughout this
restart; it used `delt=0.1`. This controlled intervention prevents the changed
initial state from temporarily releasing the m=1 constraint. It is a diagnostic
state alignment, not an independently initialized production benchmark.

| Comparison | Final FTOL | cuMES iota at s=0.01 | Fortran iota at s=0.01 | Relative difference |
| --- | ---: | ---: | ---: | ---: |
| Original inputs and independent interior states | 1e-12 | 0.70009338 | 0.69050827 | 1.3881% |
| Original constraint, stricter solves | 1e-16 | 0.70189941 | 0.69457715 | 1.0542% |
| Matched constraint, independent cold start | 1e-16 | 0.69684769 | 0.69457715 | 0.3269% |
| Matched constraint and frozen m=1 state | 1e-16 | 0.69458182 | 0.69457715 | 0.000672% |

In the last row, the maximum absolute difference over the entire iota profile
is `1.68e-5`. All three cuMES residuals are below `1e-16`; the fields are finite,
the oriented Jacobian is valid, and the LCFS is unchanged. The final checkpoint
also converges in one evaluation using the **unmodified production binary**
with `tcon0=0.25`. The isolated build is needed to select the reference's frozen
coordinate state, not to make that final state pass cuMES's normal gates.

The evidence identifies constraint normalization, iteration-dependent m=1
coordinates, and near-axis stopping sensitivity as the causes of the original
percent-level discrepancy. It does not identify a different prescribed-current
iota formula. Residual convergence, a fixed-point replay, and agreement of
volume alone do not establish a mesh-independent near-axis iota.

Audit scripts, executable hashes, state conversions, original Fortran dumps,
JSON data and PNG/PDF figures are retained outside the repository under
`../tmp/asym-vmec-benchmark/qh-current-audit/` and
`../tmp/asym-vmec-benchmark/qh-state-audit/`. No production numerical code was
changed for this investigation.
