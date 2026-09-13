# Free-boundary benchmarks against original Fortran VMEC

These benchmarks found two defects: premature consistency checks during an
uncoupled cold start, and the wrong analytic toroidal branch in the NESTOR
singular integrals. They are fixed in cuMES `acee660` and `d423c4a`
(vacuum-field `20f4a57`). See [ADR-0021](../../docs/adr/0021-vacuum-consistency-activation.md)
and [vacuum-field ADR-0004](../../deps/vacuum-field/docs/adr/0004-singular-fourier-branches.md).

## Sources and scope

[`manifest.json`](manifest.json) pins every input and field-grid checksum.
Large mgrid files are downloaded to a selected directory outside the repo.
The reference executable is original Fortran VMEC2000
[`c965d31`](https://github.com/hiddenSymmetries/VMEC2000/tree/c965d31faf732ca77d280ef509a7bdefe7797292),
built with mpifort, Release, OpenBLAS and MPICH ScaLAPACK, and run with one
process/BLAS thread. VMEC++ is not the equilibrium reference.

| Case | Source | Symmetry | Resolution |
| --- | --- | --- | --- |
| DIII-D | [STELLOPT VMEC benchmark](https://github.com/PrincetonUniversity/STELLOPT/tree/05e013b622291ab57b98bafd5b22de8405139bd6/BENCHMARKS/VMEC_TEST) | LASYM, up/down asymmetric tokamak | nfp=1, mpol=12, ntor=0, theta=48, ns=16/32/64/128 |
| NCSX c09r00 | [pinned low-resolution deck](https://github.com/uwplasma/vmex/blob/f09288b37bac7d7122e98220192795e76770cd1f/examples/data/input.ncsx_c09r00_free_lowres), original MAKEGRID field asset | stellarator-symmetric 3-D control | nfp=3, mpol=7, ntor=6, 20×24 angles, ns=9/15/25 |
| CTH-like | [pinned LASYM fixture](https://github.com/uwplasma/vmex/blob/f09288b37bac7d7122e98220192795e76770cd1f/examples/data/input.cth_like_free_bdy_lasym_small) | LASYM, **synthetic** field grid | nfp=5, mpol=5, ntor=4, 16×20 angles, ns=15 |

CTH-like is a negative benchmark. The supplied synthetic case is not a
converged original-VMEC reference and does not qualify physical asymmetric
stellarator free-boundary accuracy. DIII-D supplies asymmetric axisymmetric
coverage; NCSX supplies symmetric 3-D coverage.

## Matching the inputs

Preparation retains physical profiles, currents, flux and the starting
boundary. Signed-n Fourier coefficients and legacy axis families are mapped
to the existing shared cuMES configuration format. Modes beyond the selected
mpol/ntor are truncated in both codes. A legacy ZAXIS n=0 sine entry has no
physical contribution.

Native constraint strength is `min(abs(Fortran TCON0),1)` for symmetric
inputs and one quarter of that for LASYM; see
[the normalization audit](../../docs/fortran-vmec-comparisons.md).
Thus native TCON0 is .25 for DIII-D/CTH and 1 for NCSX. Nominal FTOL values
remain squared residual thresholds in each solver's own normalization.

The following explicit changes are applied to **both** input decks:

- DIII-D's intentionally cap-limited coarse stages become a converging
  schedule: FTOL=1e-8/1e-10/1e-11/1e-12, caps=10000/10000/10000/20000.
- NCSX retains FTOL=1e-6/1e-8/1e-10 and caps=4000 per stage. The optional
  `--ncsx-strict` study changes only the final FTOL to 1e-12.
- CTH retains FTOL=1e-10 and extends its cap from 1000 to 20000. Its grid has
  one circuit, so the unused second EXTCUR entry is removed. Original VMEC
  ignores that surplus entry; cuMES deliberately validates circuit counts.

## Measured results, 2026-09-14

CUDA double on TITAN Xp, corrected vacuum operator:

| Case | Final nominal FTOL | Native stage iterations | Relative volume difference | Maximum absolute iota difference |
| --- | ---: | --- | ---: | ---: |
| DIII-D | 1e-12 | 395 / 678 / 898 / 1764 | -2.17e-7 | 5.45e-6 |
| NCSX | 1e-10 | 163 / 267 / 489 | 3.95e-5 | 5.16e-4 |
| NCSX, stricter final stage | 1e-12 | 163 / 267 / 804 | 3.80e-5 | 1.68e-4 |
| CTH-like synthetic | 1e-10 | fails active current check | not qualified | not qualified |

DIII-D and both NCSX runs pass all configured stages, all three residuals,
finite-field and oriented-Jacobian checks. Their reference VMEC runs also
pass all three final residual thresholds. Relative magnetic-energy
differences are respectively -1.63e-8, 2.62e-6 and 2.61e-6. VMEC's `wb`
is multiplied by `4*pi^2` for this energy comparison; its `volume_p` is
already the physical volume.

Cold NCSX previously failed the OFF-state current check. After correcting
activation alone, it converged but differed by -.992% in volume and .04636
in iota. Evaluating the same LCFS and axis in the two vacuum solvers isolated
the second defect: geometry, coil interpolation and axis-current fields
agreed to roundoff, but singular source/kernel projections used the wrong
`T+/-` and `S+/-` branches for the `mu-nv` basis. The correction reduced
common-state vacuum-pressure RMS disagreement from .795% to `3.69e-11`.
It affects both symmetric and asymmetric nonzero-toroidal-mode vacuum solves.

Checkpoint restart replays also converge, but are **not one-pass fixed-point
replays**: vacuum state/factors are reconstructed during startup. DIII-D
takes 51 replay iterations, changing iota by at most `1.87e-7` and volume
relatively by `4.73e-9`. NCSX takes 86, with maximum iota changes `1.29e-5`
at FTOL=1e-10 and `4.69e-6` at 1e-12. These changes are smaller than the
reported code-to-code differences.

CTH's original Fortran process exits successfully and writes `ier_flag=0`,
but after 20,000 iterations its actual residuals are approximately
`(.01161, .005188, .004236)`, far above `1e-10`. cuMES rejects the active
loop-current mismatch instead of presenting a converged equilibrium. Its
1% current guard is retained; this Fortran revision uses a 5% guard.
An additional physical-coil experiment shifting the helical winding by 2 mm
also failed the requested tolerance in both codes. It does not replace the
failed synthetic reference with a qualified case.

## Backend checks

The correction passes 10 native CUDA/operator/integration gates, seven HOST
operator gates, and the float asymmetric free-boundary regression at its
existing tolerances. Independent Gauss-Legendre tests detect the branch
defect before the fix. Historical educational_VMEC golden data contained the
same error: dependent reference entries are now explicitly marked as derived
and regenerated independently, as described in vacuum-field ADR-0004.

On the user's Chrome 153 / Windows 10 / RTX 3060 Ti, 30 WGSL integral
fixtures and their fused fallbacks pass, together with 1,770 arithmetic
comparisons. A three-dimensional asymmetric Solovev consumer case converges
in 1,323 iterations with both HOST and WebGPU vacuum (Wasm-double LU), paired
plasma arithmetic and all residuals below 1e-12. The two outputs differ by
at most `4.23e-7` in stored spectral coefficients, `1.76e-7` in LCFS R/Z
coefficients, and `2.10e-8` relatively in reconstructed volume. This is
backend agreement on that case, not bitwise equivalence or a speed claim.

The DIII-D/NCSX benchmarks above are native runs using their published
mgrid assets. Browser regression coverage uses coil-generated fields; these
mgrid-only decks were not run in Chrome. No native WGSL f64 is implied.

## Reproduction

Python requirements: NumPy, SciPy, h5py, f90nml and Matplotlib. Choose fresh
result directories outside the repository; runners refuse to overwrite them.

```bash
python3 benchmarks/free_boundary_vmec/prepare.py --out "$FREE_INPUTS"
python3 benchmarks/free_boundary_vmec/run.py \
  --inputs "$FREE_INPUTS" --out "$FREE_CUMES" --cumes /path/to/cumes
python3 benchmarks/free_boundary_vmec/run.py \
  --inputs "$FREE_INPUTS" --out "$FREE_VMEC" --vmec /path/to/original/xvmec

python3 benchmarks/free_boundary_vmec/compare.py \
  --cumes "$FREE_CUMES/ncsx/output.h5" \
  --replay "$FREE_CUMES/ncsx/replay.h5" \
  --vmec "$FREE_VMEC/ncsx/wout_ncsx.nc" \
  --out "$FREE_CUMES/ncsx/comparison" --title 'NCSX c09r00'
```

Both full runners return a nonzero status for the failed CTH benchmark while
retaining every selected case's logs/results. They inspect actual residuals;
process exit status and `ier_flag` alone are insufficient. Use `--case` to
select cases, and prepare a separate `--ncsx-strict` input directory for the
strict comparison. `compare.py` writes JSON and PNG/PDF iota/cross-section
figures. Its pass status verifies convergence and validity; reported physical
differences are measurements, not universal acceptance limits.
