# ADR-0017: Non-stellarator-symmetric equilibria

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

The implementation covers CUDA and WebGPU fixed boundaries. CUDA supports float
and double; WebGPU supports scalar-f32 and paired-f32. `test_asymmetric` checks inverse geometry, analytic angular derivatives,
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
preconditioner/multigrid regressions also pass. Local captures are kept outside
the repository in `../tmp/cumes-asym-validation/`.

VMEC++ 0.7.0 rejects asymmetric transforms, so it cannot serve as an independent
asymmetric equilibrium reference. Analytic transform and toroidal-rotation comparisons establish the
new projection mathematics and equilibrium covariance; broader independent equilibrium qualification is
still required for additional cases.

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
The forwarded Chrome endpoint was unavailable for this qualification.

## Scope of the first implementation step

Free-boundary requests currently reject `lasym=true` explicitly while their
coupling is extended. Newton corrections, the retained tangent
operator, and Boozer export also reject asymmetric states because their
existing operators support only the original six families.
