# Predeclared axisymmetric Newton qualification matrix

These 16 inputs were fixed on September 8, 2026, before running the expanded
qualification. Run every case, including baseline failures, iteration caps and
regressions. The matrix is not a collection selected for Newton successes.

The fixed candidate policy uses forward differences, 32 Krylov vectors and a
32-vector basis, every 100 effective iterations starting at 100, epsilon
`1e-6`, and an acceptance merit ratio of `0.95`. The runner owns the existing
validity, epoch and residual guards. This directory contains inputs and
provenance; it does not change production policy.

All cases are fixed-boundary, stellarator symmetric, axisymmetric and double
precision. Every case retains three configured stages with iteration caps
`[1000, 2000, 2000]` and tolerances `[1e-16, 1e-16, 1e-16]`. Radial grids are
`[5, 11, 55]` except the explicitly named radial-resolution case, which uses
`[5, 11, 99]`. No configured stage is skipped.

| ID | Input variation | Resolved mpol / ntheta | Provenance |
|---|---|---|---|
| 00_solovev_reference | Existing reference | 6 / 18 | cuMES input |
| 01_solovev_circular | R=4+cos(theta), Z=sin(theta) | 6 / 18 | Generated Solovev variation |
| 02_solovev_elongation_low | Z m=1 coefficient 1.2 | 6 / 18 | Generated Solovev variation |
| 03_solovev_elongation_high | Z m=1 coefficient 2.0 | 6 / 18 | Generated Solovev variation |
| 04_solovev_triangularity_stronger | R m=2 coefficient -0.18 | 6 / 18 | Generated Solovev variation |
| 05_solovev_triangularity_reversed | R m=2 coefficient +0.068 | 6 / 18 | Generated Solovev variation |
| 06_solovev_pressure_half | am = [0.0625, -0.0625] | 6 / 18 | Generated Solovev variation |
| 07_solovev_pressure_double | am = [0.25, -0.25] | 6 / 18 | Generated Solovev variation |
| 08_solovev_pressure_quadratic | am = [0.125, -0.25, 0.125] | 6 / 18 | Generated Solovev variation |
| 09_solovev_pressure_flatcore | am = [0.125, 0, -0.125] | 6 / 18 | Generated Solovev variation |
| 10_solovev_mpol4 | Lower spectral resolution | 4 / 14 | Generated Solovev variation |
| 11_solovev_mpol10 | Higher spectral resolution | 10 / 26 | Generated Solovev variation |
| 12_solovev_radial99 | Final radial grid 99 | 6 / 18 | Generated Solovev variation |
| 13_solovev_ntheta36 | Denser quadrature, same modes | 6 / 36 | Generated Solovev variation |
| 14_vmecpp_circular | R=6+2cos(theta), Z=2sin(theta), zero pressure, ai=[0.9,-0.65] | 8 / 22 | Adapted VMEC++ fixture |
| 15_vmecpp_analytical_ncurr1 | Analytical Solovev, prescribed current and substantial pressure | 13 / 32 | Adapted VMEC++ fixture |

The elongation and triangularity labels identify coefficient variations, not
exact geometric elongation or triangularity values. Cases 00–13 share the
cuMES Solovev base and must not be presented as independent physical devices.
In particular, pressure factors of two are local variations of the base;
the upstream analytical case supplies a distinct pressure/current regime.

## Independent fixture provenance and adaptations

The pinned files in `sources/` make generation independent of a local VMEC++
checkout. The upstream inputs were read from VMEC++ tag `v0.7.0`, commit
`335ef66441d82980331d0062ab8d0398eff50818`, in
`src/vmecpp/cpp/vmecpp/test_data/`. Their MIT license is included as
`sources/VMECpp-LICENSE.txt`. `manifest.json` records the original path and
SHA-256 of every source and generated input, and every changed top-level field
with its before/after values.

The circular-tokamak fixture retains its physical inputs. Its original
single-stage controls (`ns=17`, tolerance `1e-20`, cap `3000`) are deliberately
replaced by the common schedule above. The resulting input is an adapted
benchmark, not an untouched upstream regression fixture.

The analytical Solovev fixture retains its full boundary point set, current
profile, `curtor=2823753.28289`, pressure profile/scale, flux, `mpol=13` and
`delt=1.0`. Three explicit adaptations are made:

1. Replace its original `ns=31`, cap `2000`, tolerance `1e-16` schedule by the
   common three-stage controls.
2. Reflect the poloidal coordinate as `theta_old = pi - theta_new`:
   `R_m_new = (-1)^m R_m_old` and
   `Z_m_new = (-1)^(m+1) Z_m_old`. This preserves the geometric boundary point
   set and supplies the orientation used by cuMES's negative-Jacobian contract.
   Independent CPU comparisons should use this same adapted input.
3. Replace its upstream zero R-axis seed by `[4.0]`, a central initial seed.
   This is not an independently converged axis estimate.

That last case is a separate upstream prescribed-current fixture within the
Solovev family, not an unrelated equilibrium family. Its convergence under
the standardized controls is an outcome to record, not assumed in advance.

## Regeneration and runner contract

```sh
python3 benchmarks/axisymmetric_newton/generate.py
python3 benchmarks/axisymmetric_newton/generate.py --check
```

Generation and byte checks perform no equilibrium solves or GPU work.
`manifest.json` contains `cases`, each with an `id` and an `input` path relative
to this directory. `expected` contains the exact configured stage controls and
resolved angular resolution; it is not an expected success result. Runners
must check input hashes, keep baseline and candidate inputs identical, and
retain outcomes for all 16 IDs. Results and timing reports belong outside the
input manifest.

## Finite-pressure supplement: IDs 16–18

After the original 16-case screen, code inspection confirmed that `am` is a
pressure polynomial in SI Pa: `eval_mass_profile` in
`include/cumes/config/profile_functions.hpp` multiplies it by
`DeviceParams<T>::MU_0 * pres_scale`, and
`src/kernels/profiles_impl.cuh` uploads the result to `pres_H`. The original
Solovev central pressure is only 0.125 Pa, so its half/double and profile
variations cover a physically small pressure range.

The following three supplemental cases were declared before their own runs
to address that units-based coverage gap. They were added after observing the
original screen, not presented as part of its initial predeclaration, and
were not selected from timing or convergence results for these new inputs.
Keep all three outcomes, including failures and regressions.

| ID | Pressure polynomial in Pa | Changed field |
|---|---|---|
| 16_solovev_pressure1000_linear | 1000(1-s) | am = [1000, -1000] |
| 17_solovev_pressure5000_linear | 5000(1-s) | am = [5000, -5000] |
| 18_solovev_pressure1000_quadratic | 1000(1-s)^2 | am = [1000, -2000, 1000] |

Only `am` changes from the pinned cuMES Solovev source. Geometry, flux, fixed
iota, axis seed, step, `mpol=6`, angular resolution, grids `[5,11,55]`, caps
`[1000,2000,2000]`, and all three `1e-16` tolerances are retained. The same
fixed Newton policy applies to all three cases.

The separate `pressure_manifest.json` follows the same runner schema and
records the relationship to the original manifest. The original
`manifest.json` and its 16 input files are unchanged. Its SHA-256 remains
`90f28299f35d97d577873620e030f2b88facf9eaa08114ba0aca981f9f90f776`.
Regenerate or check the supplement independently:

```sh
python3 benchmarks/axisymmetric_newton/generate_pressure.py
python3 benchmarks/axisymmetric_newton/generate_pressure.py --check
```

The supplemental generator refuses to proceed if the original manifest or
pinned Solovev source has changed, and writes only IDs 16–18 and the separate
pressure manifest. It performs no solver or GPU work.
