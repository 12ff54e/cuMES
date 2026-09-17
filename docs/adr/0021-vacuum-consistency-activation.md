# ADR-0021: check vacuum consistency when pressure coupling activates

Status: accepted, 2026-09-14.

## Problem

The NCSX c09r00 free-boundary benchmark aborted on its second evaluation,
while vacuum pressure was still OFF. Its cold-start magnetic axis gave
`ctor=-0.237234`, `bsubu_vac=-0.151611`, and `rbtor=2.432417`.
The loop-current check treated this diagnostic vacuum evaluation as an
accepted pressure-coupled state, preventing the fixed-boundary startup
relaxation from improving the axis.

## Decision

Apply the existing toroidal-field sign and 1% loop-current consistency checks
from the first pressure-coupled vacuum update onward. OFF-state diagnostic
updates still run, but do not enforce those two coupling conditions. Finite
checks, activation/restart logic, update scheduling, and pressure arithmetic
are unchanged. An inconsistent active update remains a hard error.

This is a Class C controller change: it changes whether some cold starts can
proceed. It does not lower the convergence tolerances or qualify every
resulting equilibrium against VMEC.

## Validation

`test_vacuum_bridge` uses a deliberately under-resolved axis filament near a
circular LCFS. It verifies that the OFF update survives a measured current
mismatch above 1%, then rejects the same mismatch at activation.

CUDA-double `test_vacuum_bridge`, `test_free_boundary_solver`, and
`test_asymmetric_free_solver` pass. The symmetric Solovev regression retains
its recorded stage counts and bitwise repeatability. NCSX now passes all three
configured stage tolerances; its independent vacuum/equilibrium comparison is
recorded separately in the free-boundary benchmark study.
The CUDA-float asymmetric free-boundary regression also passes at its existing
`1e-5` axisymmetric and `1e-6` three-dimensional tolerances.

Chrome 153 on Windows 10 / RTX 3060 Ti also converged the paired-precision
Solovev and asymmetric tokamak free-boundary cases using the HOST vacuum
path. Their final residuals were respectively
`(9.943e-13, 4.744e-14, 2.696e-14)` and
`(9.507e-13, 4.814e-13, 9.936e-14)` at the configured `1e-12` tolerance.
