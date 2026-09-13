# QH m=1 gauge and convergence study

Measured 2026-09-14 with CUDA double on TITAN Xp. This is the regenerated
Landreman–Sengupta JPP 2019 section 5.5 QH case, **not** the later PRL QH
configuration. Fixture provenance is in
[`benchmarks/asymmetric_vmec`](../benchmarks/asymmetric_vmec/README.md).
The original Fortran reference is VMEC2000
[`c965d31`](https://github.com/hiddenSymmetries/VMEC2000/tree/c965d31faf732ca77d280ef509a7bdefe7797292).

## Decision and setup

Retain the production `FSQZ_previous < 1e-6` m=1 gate and its bootstrap.
These experiments do not establish a more robust replacement. Keep effective
constraint strength explicit in comparisons, and separate residual convergence
from radial/angular resolution of iota.

The [common-state audit](fortran-vmec-comparisons.md) established agreement of
the prescribed-current closure and identified different frozen m=1 coordinate
states. This study supports that finding, but shows that the original ns=51,
mpol=8, ntor=12 run cannot establish a mesh-independent near-axis profile.

All native runs retain the same physical LCFS, profiles, flux and reference
axis. Native `tcon0=.25` matches the effective asymmetric constraint strength
of Fortran `TCON0=2`. Both codes retain their residual normalizations: native
squared residuals on a matched state are four times Fortran's. Equal nominal
FTOL remains a different stopping rule.

Baseline: ns=25/51, FTOL=1e-14/1e-16, DELT=.9, mpol=8, ntor=12, actual
quadrature 22×28, and caps of 80,000 per native stage. Successful native runs
pass every configured residual, finite-field and oriented-Jacobian gate.
The first half-grid sample at ns=51 is **s=.01**, not extrapolated s=0.
Cross-grid comparisons below interpolate to common normalized toroidal flux.

## Gauge policy and initial step

These are isolated compile-time experiments, not new production input keys.
The threshold affects both m=1 difference-force families.

| Policy | DELT | iota(.01) | iota(.05) | ns=25 / 51 iterations |
| --- | ---: | ---: | ---: | --- |
| Production gate, 1e-6 | .9 | .69684769 | .69963133 | 8071 / 13888 |
| Production gate, 1e-6 | .5 | .69324596 | .69794376 | 7915 / 16212 |
| Residual-normalized gate, 4e-6 | .9 | .69697928 | .69968103 | 8103 / 13963 |
| Later freeze, 1e-10 | .9 | .69749047 | .70095559 | 6965 / 13205 |
| Later freeze, 1e-10 | .5 | .69413995 | .69903805 | 6884 / 13506 |
| Freeze throughout cold start | .9 and .5 | **failed** | **failed** | final stage exhausts cap |

Changing DELT changes iota(.01) by about .52% with the production gate and
.48% with the later gate. The 4e-6 normalization adjustment alone changes it
by only .019%. Always freezing a cold seed is insufficiently robust: the .9
run ends at `(3.475e-12, 4.883e-12, 6.776e-15)`, above its `1e-16` target.
The previous successful frozen-gauge experiment was a warm, aligned-state
diagnostic, not a successful always-frozen cold-start policy.

## Stopping tolerance

These runs retain the identical coarse stage at FTOL=1e-14 and change only
the ns=51 stopping tolerance under the production gate.

| Final FTOL | iota(.01) | iota(.05) | Final-stage iterations |
| --- | ---: | ---: | ---: |
| 1e-10 | .69185995 | .69314551 | 215 |
| 1e-12 | .69364005 | .69645486 | 1020 |
| 1e-14 | .69676280 | .69954643 | 6527 |
| 1e-16 | .69684769 | .69963133 | 13888 |

The maximum whole-profile change from 1e-14 to 1e-16 is `9.10e-5` in iota.
The 1e-12 stopping level leaves substantially more profile drift in this
case. Tightening FTOL still cannot align independently frozen coordinates.

## Radial and angular resolution

| Grid, all final FTOL=1e-16 | cuMES iota(.025) | cuMES iota(.05) | Fortran iota(.05) |
| --- | ---: | ---: | ---: |
| ns=51, mpol=8, ntor=12, 22×28 | .69461384 | .69963133 | .69826575 |
| ns=101, same modes/quadrature | .69447506 | .70316768 | .70206952 |
| ns=201, same modes/quadrature | .69812821 | .70579481 | timed out |
| ns=51, same modes, 40×80 | .69449552 | .69951885 | .69820187 |
| ns=51, mpol=10, ntor=15, 40×80 | .68227997 | .68671060 | .68830458 |
| ns=51, mpol=12, ntor=15, 48×96 | .68207451 | .68610256 | not run |
| ns=101, mpol=10, ntor=15, 40×80 | .68055315 | .68885888 | not run |

The ns=101 cold schedule uses FTOL=1e-12/1e-14/1e-16; ns=201 uses
1e-12/1e-13/1e-14/1e-16. Changing the coarse schedule can also change the
frozen gauge, so these initially combine startup and grid sensitivity.
A control starts from the converged ns=51 baseline and transfers to 101 then
201 with m=1 difference forces held at zero throughout. It converges in
1/24029/37683 evaluations and gives iota(.05)=.70579409. Its whole-profile
difference from cold ns=201 is at most `7.25e-5` (`1.74e-5` for s>=.025).
The radial shift at s=.05 is therefore not explained by changing cold-start
gauge history alone.

Increasing quadrature alone changes iota(.05) by .016%, whereas increasing
the retained modes from 8/12 to 10/15 changes it by about 1.85%. The mpol=12
result changes it by another .089%. Radial refinement from 101 to 201 still
changes iota(.05) by .374%. A joint radial/angular plateau is not established.

The original Fortran ns=101, angular-refined and quadrature-refined runs all
converged to 1e-16. The ns=201 process reached its 1,800-second limit, with
last printed residuals approximately `(6.00e-15, 4.66e-15, 4.63e-15)`;
it supplies no converged ns=201 reference.

## Reproduction

Use a compatible Ninja CUDA-double build and the Python dependencies in the
asymmetric benchmark README. Choose `STUDY_DIR` outside the repository.

```bash
python3 benchmarks/asymmetric_vmec/gauge_study.py \
  --cumes /path/to/cumes --out "$STUDY_DIR" \
  --case default-cold --case default-cold-step05 \
  --case default-tol10 --case default-tol12 --case default-tol14 \
  --case default-radial101 --case default-radial201 \
  --case default-quadrature --case default-angular --case default-angular12 \
  --case default-combined101

python3 benchmarks/asymmetric_vmec/build_gauge_variant.py \
  --build /path/to/cuda-double-build --threshold 1e-10 \
  --out "$STUDY_DIR/later-build"
python3 benchmarks/asymmetric_vmec/gauge_study.py \
  --cumes "$STUDY_DIR/later-build/cumes" --out "$STUDY_DIR" \
  --case later-cold --case later-cold-step05

python3 benchmarks/asymmetric_vmec/gauge_study.py \
  --vmec /path/to/original/xvmec --out "$STUDY_DIR" \
  --case default-cold --case default-radial101 \
  --case default-angular --case default-quadrature
python3 benchmarks/asymmetric_vmec/gauge_report.py \
  --root "$STUDY_DIR" --out "$STUDY_DIR/report"
```

Build 4e-6 and 1e100 variants similarly for `normalized-cold` and `fixed-*`.
For `fixed-warm-radial201`, pass the always-frozen binary and
`--restart "$STUDY_DIR/default-cold/output.ckpt"`.
The builder overrides one header and links one replacement object in an
isolated directory; production sources and objects are unchanged. The runner
records input/executable hashes and retains failures/timeouts. The report
writes JSON plus PNG/PDF figures, excluding failed runs from converged curves.
A prepared input without a completed run is not a result.
