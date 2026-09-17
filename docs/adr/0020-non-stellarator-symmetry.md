# ADR-0020: Non-stellarator-symmetric equilibria

- Status: Accepted; qualification applies to the configurations below
- Date: 2026-09-10
- Numerical classification: Class C for asymmetric solves

## Decision

Use `lasym=true` with signed-n VMEC `rbs` and `zbc` boundary harmonics and
optional `raxis_s`/`zaxis_c` axis coefficients. The default remains false;
complementary inputs require the explicit flag. Append six complementary
product-basis families to the existing six-family layout. Preserve physical
amplitudes, staggering, radius references, fixed-edge and axis rules, all
residual/validity gates, and the existing controller.

Reuse the existing inverse/forward FFT operators, real-space MHD equations,
preconditioner factors, descent, and multigrid transfer. Project and average
on the full periodic theta grid for asymmetric problems. Retain both Fourier
parities in constraint bandpass filtering and apply the m=1 coordinate gauge
to the complementary Rsc/Zcc pair. This introduces no second equilibrium
algorithm and no float-device FP64 arithmetic.

Symmetric binary v8 and checkpoint v6 writes are preserved. Asymmetric binary
v9 and checkpoint v7 explicitly count the twelve families and append complete
input provenance. NetCDF/HDF5 expose all twelve named datasets; plotting and
restart readers preserve them. See `output-formats.md` for the contracts.

## Qualification

The implementation covers CUDA and WebGPU fixed and free boundaries. CUDA
supports float and double; WebGPU supports scalar-f32 and paired-f32.
`test_asymmetric` checks inverse geometry, analytic angular derivatives,
weak-form forward forces, and bandpass filtering against independent scalar
product-basis quadrature, for ntor=0 and ntor=2 in both scalar types.
`test_asymmetric_solver` solves a tilted, vertically displaced tokamak and a
small 3-D signed-n perturbation using three radial stages. It requires every
configured residual, finite fields, an oriented Jacobian, unchanged LCFS,
complete output round trips, and converged checkpoint replay in at most two
passes. A converged symmetric 3-D state, rotated analytically in toroidal
angle into all twelve families, also passes immediate asymmetric fixed-point
replay. Double uses 1e-12 thresholds; float uses 1e-6. These are qualification
cases, not guarantees for arbitrary asymmetric geometries.

On NVIDIA TITAN Xp / CUDA 12.1, the double tokamak converges in 183/127/152
stage iterations. Original symmetric Solovev and a small 3-D symmetric case
retain bitwise-identical final coefficients and identical stage counts against
the pre-change executable (235/193/326 and 199/138/150 respectively). Operator,
configuration, malformed-I/O, checkpoint, and existing Fourier/geometry/
preconditioner/multigrid regressions also pass.

VMEC++ 0.7.0 rejects asymmetric transforms, so it cannot serve as an independent
asymmetric equilibrium reference. Analytic transform and toroidal-rotation
comparisons establish the new projection mathematics and equilibrium
covariance; broader independent equilibrium qualification is still required
for additional cases.

The browser conformance suite executes all twelve families on a real WebGPU
adapter: inverse geometry and derivatives, direct/FFT weak-form projections,
the complementary m=1 gauge, compact residual norms and nonfinite guards,
preconditioning, descent, and full-period bandpass filtering. Each operator is
compared with its scalar reference for axisymmetric and 3-D shapes in scalar
and paired precision. Boundary preview tests also compare signed-n RBS/ZBC
against the analytic input harmonics. The boundary editor, 2-D cuts, orbit
renderer, and binary output retain all twelve families.

Firefox 155.0.1 with WebGPU enabled (nonfallback adapter; device name hidden by
the browser) converges the three-grid asymmetric tokamak at 1e-12 in paired
precision, with final residuals (9.360e-13, 4.725e-13, 4.871e-15). A small 3-D
signed-n perturbation converges to (9.441e-13, 3.740e-13, 2.289e-14).
The same 3-D boundary in scalar precision converges at the editor's 1e-5
threshold to (9.674e-06, 4.216e-06, 5.598e-08).
The complete browser conformance/Solovev gate and the WebGPU CTest suite pass.
Chrome qualification is recorded below.

## Free-boundary coupling

The existing vacuum library now solves coupled sine/cosine potentials with
four matrix blocks, a fixed constant-potential gauge, and both singular RHS
projections. Full updates rebuild factors; partial updates retain them. cuMES
passes all eight R/Z edge families, averages the full theta grid, and applies
full-grid vacuum pressure to the moving LCFS. Activation, current-consistency
checks, soft restarts, `nvacskip`, and multigrid persistence are retained.
The same coupling drives native CUDA, browser HOST-double vacuum, and browser
paired-f32 vacuum kernels with Wasm-double LU.

MAKEGRID's `number_of_phi_grid_points` must match the equilibrium's effective
`nzeta`, and the field periods must agree. cuMES rejects mismatched grids
before dispatching the vacuum update. The external-field operator interpolates
R/Z on corresponding toroidal planes; it does not interpolate between phi
planes.

The vacuum-library HOST and CUDA suites pass all 20 tests. A physical 3-D
toroidal rotation produces nonzero cosine potential and preserves all eight
vacuum outputs under full and partial updates. Its complete Firefox gate
passes with Wasm-double LU, including nonzero asymmetric geometry and
independent matrix factorization. See the dependency's
[ADR-0003](../../deps/vacuum-field/docs/adr/0003-asymmetric-potential.md).

The native double free tokamak fixture changes one Solovev coil current by 2%
to break up/down symmetry. It converges in 795/445 passes to residuals
(9.605e-13, 1.895e-13, 6.572e-14). Its checkpoint replay reconverges after the
vacuum initialization/restart sequence. A 3-D perturbation with 16 matching
toroidal planes converges in 1348 passes. Both checkpoint replays converge
in 51 passes with every residual below 1e-12. The integration gate checks every
residual, finite fields, Jacobian orientation, moving asymmetric LCFS, complete
checkpoint/provenance round trips, and restart convergence. The original
symmetric free-boundary gate retains its frozen 389/636 trajectory.

Native float initially exposed an overflowing tangent-pole sentinel in the
vacuum library. Reusing the existing WebGPU analytic pole limit fixes those
NaNs; the shared test comparison now rejects nonfinite outputs. See
[vacuum ADR-0004](../../deps/vacuum-field/docs/adr/0004-float-tangent-poles.md).
The asymmetric free integration gate passes at 1e-5 for ntor=0 and at 1e-6
for ntor=1, including two-pass checkpoint replays and every residual/geometry
check. The axisymmetric fine grid stalls above 1e-6, so the input tolerance
floor is not a convergence guarantee. Device-code inspection confirms no
float FP64 instructions.

Firefox 155.0.1 paired plasma with HOST-double vacuum converges this free
fixture in 895/431 passes, with final residuals
(9.513e-13, 4.808e-13, 9.932e-14), and publishes a complete twelve-family
binary v9 result. WebGPU pressure-force checks cover both parities, scalar
and paired inputs, axisymmetric/3-D grids, interior preservation, and finite
gates. The WebGPU CTest suite passes all 17 checks.

The same browser input with paired-f32 vacuum kernels and Wasm-double LU
converges in the same 895/431 passes, to
(9.526e-13, 4.818e-13, 9.953e-14). Relative to HOST vacuum, the largest absolute
final R/Z coefficient difference is 1.7e-7, and lambda differs by at most
3.0e-7. Native double and browser paired/HOST-vacuum solves differ by at most
3.2e-4 in R/Z coefficients and 7.4e-5 in lambda at their accepted tolerances.
The browser state also reconverges through a native double free-boundary
checkpoint replay in 51 passes, with residuals below 1e-12.

## Chrome qualification (2026-09-10)

The user's forwarded Chrome 153.0.8010.36 on Windows 10 / NVIDIA RTX 3060 Ti
passes the full cuMES WebGPU conformance suite, including all twelve families
in scalar and paired precision, axisymmetric/3-D operator references, vacuum
pressure forces, validity gates, and the symmetric Solovev regression. All
17 WebGPU CTest checks also pass. Runs use separate temporary tabs in the
existing browser window, redirect editor settings to session storage, and
execute serially on the adapter.

The following complete asymmetric solves pass every configured residual and
the controller/plot consistency checks. Paired means paired-f32 plasma
arithmetic, not native f64. Its threshold is 1e-12; the scalar editor uses
1e-5. Iterations below sum the effective counts over all radial stages.

| Case | Plasma precision / vacuum | Iterations | Maximum final residual |
| --- | --- | ---: | ---: |
| Fixed tokamak | Paired | 498 | 9.500e-13 |
| Fixed 3-D, ntor=1, nfp=3 | Paired | 608 | 9.503e-13 |
| Fixed 3-D, ntor=1, nfp=3 | Scalar | 94 | 9.620e-6 |
| Free tokamak | Paired / HOST vacuum | 1326 | 9.507e-13 |
| Free tokamak | Paired / WebGPU vacuum, Wasm LU | 1326 | 9.520e-13 |
| Free tokamak | Paired / WebGPU vacuum, resident LU | 1326 | 9.511e-13 |
| Free 3-D, ntor=1, nzeta=16 | Paired / HOST vacuum | 1323 | 9.485e-13 |
| Free 3-D, ntor=1, nzeta=16 | Paired / WebGPU vacuum, Wasm LU | 1323 | 9.474e-13 |

The fixed cases use 5/11/33 radial surfaces; the free cases use 16/32. The
free 3-D case adds the same signed-n perturbations as
`test_asymmetric_free_solver`, with 16 matching MAKEGRID toroidal planes.
Its coil geometry remains axisymmetric; the initial LCFS has toroidal modes.
Downloaded binary v9 outputs retain all twelve families, finite scientific
fields, and negative Jacobians. The fixed LCFS agrees with the embedded
boundary within 4.4e-15 m in paired precision and 4.6e-8 m in scalar precision;
free-boundary outputs show the expected moving LCFS.

The separate vacuum-library browser gate also passes, including resident LU
fixtures through 256 unknowns, singular/nonfinite rejection, independent
factorization checks, and full/partial asymmetric updates with both LU
backends. This qualifies resident LU on this Chrome adapter; the Firefox
limitation below still applies. For the free tokamak, HOST versus WebGPU vacuum
changes final R/Z coefficients by at most 1.1e-7 and lambda by at most 2.2e-7
over the two tested GPU LU choices. These comparisons are numerical
consistency checks, not an independent asymmetric equilibrium benchmark.
For the free ntor=1 case, HOST versus WebGPU vacuum with Wasm LU differs by
at most 8.0e-8 in R/Z and 1.5e-7 in lambda.

The Chrome harness accepts
`CUMES_INPUT_JSON=inputs/free_bdy/asymmetric_tokamak.json`
with `?boundary=free&precision=double&trace=1` and the desired `vacuum` /
`vacuum_lu` query options. Omit `coils=` when injecting a custom free input,
so a named preset does not replace it. Set `CUMES_CAPTURE_OUTPUT=1` and
`CUMES_CLOSE_TEST_TAB=1` to save the binary and close the test tab.

## Remaining qualification limits

Newton corrections, the retained tangent operator, and Boozer export reject
asymmetric states because their operators support only the original six
families. The opt-in resident WebGPU LU limit is 256 total sine/cosine unknowns.
Its existing algebra fixture fails on the Firefox adapter above on both the
original and modified vacuum library, so resident LU remains unqualified on
that adapter; use the default Wasm-double LU there.
