# Asymmetric fixed-boundary benchmarks against Fortran VMEC

The [QH gauge and convergence study](../../docs/qh-gauge-convergence.md)
adds isolated m=1 policies, stopping-tolerance sweeps, radial/angular
refinement, and a warm transfer with frozen gauge. Its scripts record failed
runs as well as successful ones and generate profile comparison figures.

These three cases exercise `lasym=true` with original Fortran VMEC2000
references. The reference executable was built from
[`c965d31`](https://github.com/hiddenSymmetries/VMEC2000/tree/c965d31faf732ca77d280ef509a7bdefe7797292),
using `mpifort`, Release, OpenBLAS and MPICH ScaLAPACK, with one process and one
BLAS thread. All three reproduced `wout` files report `ier_flag=0` and all three
VMEC residuals below their configured `ftolv`.

| Case | NFP | MPOL / NTOR | Radial stages | Native double FTOL |
| --- | ---: | --- | --- | --- |
| Asymmetric QA, Landreman–Sengupta–Plunk §5.3 | 3 | 4 / 4 | 25 | 1e-12 |
| Asymmetric heliotron | 4 | 7 / 3 | 25, 51, 128, 256 | 1e-12, 1e-12, 1e-12, 1e-14 |
| Finite-pressure asymmetric QH, Landreman–Sengupta §5.5 | 5 | 8 / 12 | 25, 51 | 1e-12 |

`MPOL` retains the VMEC convention, `0 <= m < MPOL`. `manifest.json` records
input hashes, pinned sources and the hashes of the reproduced reference files.
Reference hashes identify these particular captures; rebuilding VMEC on another
platform can change rounding and metadata. Large outputs, logs and generated
figures belong outside the repository.

## Source and initialization contracts

**QA:** the namelist is the
[SIMSOPT test input](https://github.com/hiddenSymmetries/simsopt/blob/c648630cfc5625863b291709c17015bcdcba13af/tests/test_files/input.LandremanSenguptaPlunk_section5p3)
for [Landreman, Sengupta and Plunk (2019), §5.3](https://doi.org/10.1017/S0022377818001344).
Its header documents the reduced resolution and canonical poloidal angle. The
cuMES JSON carries the same boundary, axis, vacuum profiles and stage controls.
The new Fortran run reproduces the distributed reference to rounding precision.

**Heliotron:** the physical data were recovered from DESC's
[external VMEC reference](https://github.com/PlasmaControl/DESC/blob/ad105c5e525fbf26824d6cf9dde48775db0f8a2c/tests/inputs/wout_HELIOTRON_asym_NTHETA50_NZETA100.nc).
The original namelist was unavailable. `sources/input.heliotron` is an explicit
reconstruction with the original final resolution, flux and vacuum profiles;
the multigrid schedule was chosen for this benchmark. Its boundary is

```text
R = 10 + cos(theta) + 0.3 cos(theta - 4 phi)
Z =      sin(theta) - 0.3 sin(theta - 4 phi) + 0.2 cos(4 phi)
```

The toroidally varying vertical displacement breaks stellarator symmetry.
`NTHETA=50` and `NZETA=100` are retained, including for the browser test.

**QH:** the published construction is
[Landreman and Sengupta (2019), §5.5](https://doi.org/10.1017/S0022377819000783).
The checked-in namelist was generated with pyQSC 0.1.3:

```python
from qsc import Qsc
q = Qsc.from_paper("r2 section 5.5", nphi=101)
q.to_vmec("input.qh", r=0.025, ntheta=40, ntorMax=12,
          params=dict(mpol=8, ntor=12, ns_array=[25, 51],
                      ftol_array=[1e-12, 1e-12],
                      niter_array=[10000, 20000]))
```

Here `ntheta=40` samples the near-axis boundary construction; the exported
namelist uses VMEC's default solve grid, 22 theta points and 28 toroidal planes.
The pressure is `3125*(1-s)` Pa, total toroidal current is 5000 A, and the edge
toroidal flux is `pi*0.025**2` Wb. This is a regenerated, truncated verification
configuration. No archived matching paper `wout` was found.

Fortran VMEC normalizes the poloidal angle and repairs the original QH magnetic
axis guess. The cuMES JSON uses the canonical LCFS and converged axis from that
reproduced `wout`. cuMES constructs its own interior state with zero initial
lambda. This qualifies iteration from a supplied axis. The original near-axis
guess gives an invalid initial cuMES Jacobian even after boundary-angle
normalization; automatic axis repair remains unimplemented. cuMES now reports
this immediately instead of repeatedly restoring the same invalid seed.

## Repeat the runs

Use a CUDA double build with HDF5 enabled, and an original Fortran VMEC2000
executable. The runner launches cases serially, records executable/input hashes,
and performs a final-grid checkpoint replay for every successful cuMES solve.
Output directories must be new. `--case qa`, `--case heliotron` and `--case qh`
select individual cases.

```bash
python3 benchmarks/asymmetric_vmec/run.py \
  --vmec /absolute/path/to/xvmec --out ../tmp/asymmetric-fortran
python3 benchmarks/asymmetric_vmec/run.py \
  --cumes /absolute/path/to/cumes --out ../tmp/asymmetric-cuda

python3 benchmarks/asymmetric_vmec/compare.py \
  --cumes ../tmp/asymmetric-cuda/heliotron/cumes.h5 \
  --replay ../tmp/asymmetric-cuda/heliotron/replay.h5 \
  --vmec ../tmp/asymmetric-fortran/heliotron/wout_heliotron.nc \
  --out ../tmp/asymmetric-comparison/heliotron --title 'Asymmetric heliotron'
```

The scripts need Python 3.11 or newer; the comparison additionally needs NumPy,
SciPy, h5py and Matplotlib. It reuses the native
cuMES state reader and plot reconstruction, checks convergence, finite fields,
oriented Jacobians and the fixed LCFS, and optionally checks a one/two-evaluation
checkpoint replay. It writes a JSON diagnostic report and an iota/R–Z overlay.
VMEC comparisons remain diagnostic; they do not replace cuMES's own gates.

The same numeric `tcon0` does **not** give the same asymmetric constraint
strength: pinned Fortran VMEC caps its magnitude at 1 and has two additional
half factors. In cuMES's full-period projection convention, an explicit
matched-constraint run uses `min(abs(Fortran tcon0), 1)/4`. The original fixtures
retain their recorded settings. Also align the frozen m=1 coordinate state
when diagnosing remaining differences; matching the boundary, grid and nominal
`FTOL` alone is insufficient. The
[Fortran comparison audit](../../docs/fortran-vmec-comparisons.md) derives these
rules and isolates their effect on QH.

`iotas[1:]` is compared on the half radial mesh. Axis positions are compared at
the same geometric toroidal angles. The VMEC `wb` angular normalization is
restored with `4*pi**2` before comparing magnetic energies. Lambda is deliberately
excluded: VMEC `wout` stores a smoothed half-mesh lambda with a flux-dependent
normalization, while cuMES stores full-mesh spectral state. Raw lambda or
interior coordinate coefficients require additional gauge/grid alignment.

For native float, provide `--float-tolerance 1e-5`. This explicitly changes the
requested precision test. A successful double solve is not a float guarantee.

## Browser runs

Build a dedicated WebGPU directory following `docs/webgpu-port.md`. With the
forwarded Chrome at port 9333 and the build served by the local preview:

```bash
cumes_test_url=http://localhost:6969/magnetic-equilibrium-solver/tmp/cumes-build-asym-vmec-webgpu/webgpu/cumes_webgpu.html
CUMES_CLOSE_TEST_TAB=1 node scripts/webgpu_validate_run.mjs \
  "$cumes_test_url?mode=test" ../tmp/asymmetric-browser-conformance

CUMES_CLOSE_TEST_TAB=1 CUMES_CAPTURE_OUTPUT=1 \
CUMES_INPUT_JSON="$PWD/benchmarks/asymmetric_vmec/inputs/heliotron.json" \
  node scripts/webgpu_validate_run.mjs \
  "$cumes_test_url?preset=w7x&precision=double&trace=1" \
  ../tmp/asymmetric-browser-heliotron

python3 benchmarks/asymmetric_vmec/compare.py \
  --cumes ../tmp/asymmetric-browser-heliotron-output.bin \
  --browser-report ../tmp/asymmetric-browser-heliotron-result.json \
  --vmec ../tmp/asymmetric-fortran/heliotron/wout_heliotron.nc \
  --out ../tmp/asymmetric-comparison/browser-heliotron
```

The harness creates a tab in the existing window, uses tab-local settings and
closes that tab. The browser's paired precision sets all stage tolerances to
1e-12, including the heliotron's final stage. The captured `*-input.json`
records the actual input. The binary comparison reconstructs physical
diagnostics from the exported state and uses the browser report as convergence
evidence. Paired f32 arithmetic is distinct from native IEEE double.

On 2026-09-13, Chrome 153.0.8010.36 on Windows / NVIDIA GeForce RTX 3060 Ti
passes all three paired-precision cases after the fixes below. Every stage
converges at `1e-12`; the table lists effective stage iterations and the
largest of the three final residuals.

| Case | Stage iterations | Maximum final residual | Maximum LCFS coefficient error (m) |
| --- | --- | ---: | ---: |
| QA | 962 | 9.949e-13 | 8.33e-17 |
| Heliotron | 877, 548, 1040, 1457 | 9.816e-13 | 1.67e-16 |
| QH | 7371, 3924 | 9.893e-13 | 3.33e-16 |

Captured state and scientific-field hashes agree with the browser payload;
all saved scientific fields are finite and all half-grid Jacobians are
negative. QA retains its exact 963-record controller trace and bitwise
scientific payload from before the large-grid scan and endpoint changes.
The original QH axis guess is rejected before any descent samples, with the
new invalid-initial-geometry diagnostic. All 17 WebGPU CTest checks and the
real-adapter conformance suite pass, including the new transfer and large
finite-scan cases. The maximum iota differences from Fortran are `1.14e-3`,
`4.32e-5` and `9.90e-3`, respectively; these are physical diagnostic differences,
separate from the residual thresholds.

## Defects exposed and regression evidence

- The heliotron's native 128-to-256 B-spline transfer uploaded its matrix on
  the default CUDA stream and consumed it on a nonblocking stream. Corrupted
  geometry produced an axis radius around 5097 m. Ordering the upload with its
  consumer fixes the solve. `test_prolongation` covers analytic profiles at
  that size in float/double and six/twelve families. The original QA result's
  53 numeric datasets remain identical.
- Invalid initial geometry had no usable checkpoint, yet both backends retried
  it while shrinking the time step. The new first-pass diagnostic covers cold
  starts, restarts and newly prolonged grids. Existing recovery after descent
  remains in place. Native regression tests use an axis outside the LCFS.
- The plotter used reduced-theta current quadrature for asymmetric states and
  mixed different radial faces in one metric cross term. Analytic metric and
  full-surface current-conservation tests cover both corrections. At sampled
  points from all three CUDA outputs, reconstructed iota agrees with the saved
  scientific fields within `1.6e-10`.
- The 256-surface browser heliotron needs 204.8 MB storage bindings and a
  414 MB batched readback. Device creation now requests the adapter's supported
  buffer limits. Finite scans use two-dimensional dispatch beyond 65,535
  workgroups; browser conformance tests include a NaN beyond that former limit.
  The Chrome harness captures the solver log and downloads its 133 MB output
  in bounded chunks.
- Browser multigrid interpolation changed the heliotron's fixed LCFS by
  `5.96e-8` m through endpoint division/square-root rounding. Copying the stored
  endpoint exactly prevents that high-word error from surviving paired
  recombination. Real-adapter tests require exact endpoints for both linear
  and Catmull–Rom transfer, including twelve-family 128-to-256 refinement.
  [ADR-0021](../../docs/adr/0021-exact-webgpu-transfer-boundary.md) records the
  invariant and numerical qualification.

Recorded CUDA-double stages converge at every requested tolerance; each final
checkpoint replay converges in one evaluation. The initial comparison's
maximum absolute iota differences from Fortran are approximately `1.05e-3`
(QA), `4.24e-5` (heliotron), and `9.59e-3` (QH), concentrated near the axis.
Volume differences are below `3.3e-8` relative, and magnetic-energy differences
are below `5.0e-8` relative. QA's iota difference drops to `1.04e-4` at 101 radial
surfaces. QH still differs by `7.32e-3` after tightening both solvers' final
residual tolerance to `1e-16`. Keeping its physical boundary and refining to
`MPOL=10, NTOR=15, NTHETA=40, NZETA=80`, at 51 surfaces and `1e-13` residual
tolerance, reduces the maximum iota difference to `2.45e-3`. This demonstrates
resolution sensitivity. The subsequent
[same-state force audit](../../docs/fortran-vmec-comparisons.md#qh-isolation-experiment-2026-09-13)
finds an eightfold constraint-strength mismatch in the original QH comparison.
Matching that strength and the frozen m=1 coordinates reduces its first-half-grid
iota difference to `4.67e-6` (`0.000672%`) at `1e-16` final tolerance, and the
maximum profile difference to `1.68e-5`. That result uses controlled state
alignment; the original independent cold-start discrepancy remains recorded
above.

Native float at `1e-5` converges for QA and heliotron. QH's final float stage
stalls around `(1.6e-5, 3.8e-5, 7.2e-9)` and is unqualified at that tolerance.
The double/paired cases, supplied-axis limitation, and remaining iota discrepancy
must be kept distinct from that precision limit. These fixed-boundary results
do not qualify asymmetric free-boundary coupling.
