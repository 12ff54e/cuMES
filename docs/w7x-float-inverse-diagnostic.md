# W7-X: the remaining float error is in odd R/Z inverse reconstruction

Measured 2026-09-06 on the TITAN Xp, CUDA 12.1. This follows the
[reference-plus-displacement investigation](w7x-float-convergence.md).
All convergence experiments use `ftol_array=[1e-5,1e-5,1e-5]`, the original
iteration caps and float linear prolongation. No tolerance, residual
normalization, Jacobian gate or controller policy was relaxed.

## Finding

After centering m=0 Rcc, the remaining bottleneck is the accuracy of the
inverse transform's **odd-parity position fields, `r_o` and `z_o`**.
Float toroidal FFT results, poloidal products/summation and the odd-mode
`1/sqrt(s)` scaling introduce radial rounding noise. Differentiating these
fields, then differentiating pressure-weighted geometry in the force
operator, amplifies that noise.

This is supported by controlled stage substitutions and a full cold-start
intervention: reconstructing only `r_o` and `z_o` with double arithmetic on
the GPU, then storing them as float, gives convergence on all three grids.
The state, every other transformed field, geometry/field/force kernels,
forward transform, preconditioner and descent remain on the existing float
path. The direct double synthesis is a diagnostic oracle, not a newly
qualified production backend.

## Identical-state experiment

Start with the tight ns=99 double checkpoint from the original
investigation. Load it into the centered float representation. Capture its
state immediately after the normal axis extrapolation, export the reference
plus displacement as double, and inject those exact physical coefficients
into the double solver at the same point. The two canonical state dumps
are byte-identical, including the float-rounded LCFS. This avoids the
usual difference between double and float checkpoint boundary patching.

Capture or replace arrays after inverse synthesis, base geometry, magnetic
field, MHD force, constrained force, and forward transformation. Complete
all injected copies before the nonblocking compute stream resumes. Use
one force evaluation per run, with `CUMES_DUMP=1`; there is no descent
history in these comparisons. All reported values below are FSQR.

| Evaluation of the identical state | FSQR |
| --- | ---: |
| Existing centered float path | 1.100809679e-5 |
| Double path | 8.783689754e-7 |
| Double path, replacing only inverse outputs with float outputs | 1.100809434e-5 |
| Float path, replacing only inverse outputs with double outputs rounded to float | 1.351990852e-6 |
| Float path, replacing only base geometry with double outputs rounded to float | 8.782427996e-7 |
| Float path, replacing only magnetic field with double outputs rounded to float | 8.783910059e-7 |
| Float path, replacing only MHD forces with double outputs rounded to float | 8.783971606e-7 |

The float inverse outputs reproduce almost the entire excess residual even
when downstream arithmetic is double. Conversely, accurate inverse outputs
let the remaining float pipeline evaluate below tolerance. The large errors
in pressure/force arrays are primarily propagated inverse-transform errors,
rather than errors originating in those downstream kernels.

### Which inverse outputs?

Replace just one output in the float path with its double counterpart,
rounded into the existing float buffer:

| Replaced inverse output | FSQR |
| --- | ---: |
| None | 1.10081e-5 |
| `r_e` | 1.06506e-5 |
| `r_o` | 3.44149e-6 |
| `z_e` | 1.07021e-5 |
| `z_o` | 9.16374e-6 |
| Any one R/Z angular derivative | approximately 1.1008e-5 |
| Any one lambda derivative or constraint reconstruction | approximately 1.1008e-5 |

Replacing `z_o` also lowers FSQZ from `4.67646e-6` to `2.02383e-6`.
In the converse direction, injecting only float `r_o` into the exact-state
double evaluation raises FSQR from `8.78369e-7` to `8.30843e-6`.
These squared residuals include correlated errors; the rows are
interventions, not additive percentages of an error budget.

The RMS float/double discrepancies in `r_o` and `z_o` are respectively
`6.8944e-8` and `6.9924e-8`. Differencing their errors between radial rows
and dividing by `delta_s=1/98` gives RMS errors of `5.3832e-6` and
`6.0184e-6`. Their relative field errors look small, about `1.5–1.7e-7`,
but the residual is sensitive to this radial variation.

## Inside the inverse transform

The relevant existing expression in `inverse_accumulate_kernel` is

```cpp
T facO = T(1.0) / maxsc;
T fac = (m % 2 == 1) ? facO : T(1.0);
T v0 = fac * (c0 * t0 + c1 * t1);
// ...
v0o += v0;
```

Here `c0/c1` are the float cuFFT results, `t0/t1` are angular basis values,
and `maxsc=max(sqrt(s),sqrt(delta_s))`. R/Z reconstruction therefore rounds
in the FFT, in the poloidal products and sums, and while computing/applying
the odd scaling. The reference representation removes the large m=0 radius
offset, but leaves these odd fields in their original representation.

Further interventions distinguish these contributions:

| Modified part; all downstream operations float | FSQR |
| --- | ---: |
| Existing centered float inverse | 1.10081e-5 |
| Double toroidal transform, rounded to float before existing poloidal synthesis | 6.91996e-6 |
| Existing float toroidal transform, double poloidal synthesis/scaling, rounded float outputs | 6.02865e-6 |
| Same double poloidal synthesis, but retain the float-computed radial scale | 8.16998e-6 |
| Same double poloidal synthesis, but retain float angular basis values | 6.04278e-6 |
| Double complete inverse, rounded float outputs | 1.35199e-6 |

The poloidal substage interventions use an independent CPU double oracle
on captured GPU FFT outputs. On the double FFT input, its reconstructed
fields agree with the existing double GPU inverse within `2.7e-15`
(maximum absolute discrepancy across the checked R/Z position fields).
A direct double GPU synthesis reproduces the complete-inverse intervention
with FSQR `1.351990921e-6`.

Both FFT and subsequent synthesis matter. Accurate poloidal summation
cannot recover digits already lost in the float FFT. Correcting the radial
scale also matters, while float angular-table accuracy contributes much
less in this experiment. There is no evidence of a missing mathematical
normalization factor; applying the correct factor in float adds rounding
noise to an already sensitive calculation.

## Cold-start intervention and checkpoint checks

A diagnostic GPU kernel sums the same folded Fourier series using double
basis tables, double products/accumulators and double odd scaling. It
consumes the original float spectral state and writes float outputs.
Selection controls which reconstructed fields replace the existing inverse
outputs. Every other part of the solver is unchanged.

| Accurate inverse outputs | Effective iterations, ns=33 → 66 → 99 | Final FSQR | Final FSQZ | Final FSQL |
| --- | --- | ---: | ---: | ---: |
| None: reference representation only | 149 → 278 → not converged | 4.3663e-5 | 1.7544e-5 | 7.6399e-9 |
| All inverse outputs | 148 → 184 → 315 | 8.2544e-6 | 5.0905e-6 | 3.5279e-9 |
| Only `r_e`, `r_o`, `z_e`, `z_o` | 148 → 183 → 309 | 8.2841e-6 | 5.4209e-6 | 3.6181e-9 |
| Only `r_o`, `z_o` | 148 → 202 → 313 | 4.6679e-6 | 2.6166e-6 | 2.2212e-9 |

The last checkpoint converges on the first replay pass with the same
odd-field correction (`FSQR=4.667891623e-6`). Evaluating its exact imported
float state through the double path also passes (`FSQR=3.599481099e-6`).
Evaluating it with the original centered float inverse instead reports
`FSQR=1.143938418e-5` and fails the requested tolerance.

Thus the correction enables a cold-start trajectory to reach a state that
also passes an exact-state double evaluation. This is not a lowered
convergence threshold or a norm-reporting workaround.

## Reproduction and implementation status

Artifacts are preserved under
`/lustre/qzhong/cumes-diagnostics/w7x-float-investigation/stage-probe/`;
`/tmp/cumes-w7x-float-investigation` links to the parent investigation.
The artifact directory contains:

- `run_probe.py`, `isolate_inverse.py`, and `poloidal_oracle.py`;
- all case commands, logs, summaries, canonical double array captures and
  checkpoint outputs;
- `solver_impl.before.cuh` / `fourier_impl.before.cuh`, the instrumented
  `.probe.cuh` snapshots, `direct_inverse.cuh`, and source hashes;
- saved `bin-float/cumes` and `bin-double/cumes` diagnostic executables;
- a README with commands for rerunning the stage sweep and cold-start
  correction, and the matching Compute Sanitizer memcheck result.

The temporary substitutions and GPU oracle were removed from the normal
solver source after the experiment. `CUMES_RADIUS_REFERENCE=1` still enables
only the earlier reference representation in normal builds. The normal
source files were restored byte-for-byte to their pre-diagnostic versions.

The oracle is intentionally unoptimized and requires the dump-driven,
non-graph diagnostic execution path. A replay with Compute Sanitizer
memcheck on that path reports zero errors. An initial attempt without
`CUMES_DUMP=1` correctly failed because the prototype's lazy basis allocation
is unsupported inside graph capture; that failed attempt is preserved too.
No graph/performance qualification is claimed for this diagnostic backend.

The subsequent [float-float experiment](w7x-float-float.md) tests how much of
this reconstruction actually needs extra precision. Contrary to the initial
conservative scope proposed here, compensating just the poloidal products,
sums and final multiplication is sufficient for the tested W7-X 1e-5 run.
The toroidal cuFFT can remain float. That opt-in implementation is now
retained separately from the temporary oracle described above.
