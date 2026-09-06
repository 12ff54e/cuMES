# W7-X float convergence investigation

Measured 2026-09-06 at source revision `750d6a4`, NVIDIA TITAN Xp (sm_61),
driver 580.173.02, CUDA Toolkit 12.1.105. Both `verify` and `float` executables
were rebuilt from that revision for the baseline. The opt-in implementation
described below was then developed and tested on the same GPU.

## Outcome

The reported failure at `ftol_array=[1e-5,1e-5,1e-5]` is reproduced.
The baseline float run fails on the **first grid, ns=33**, before any prolongation.
The dominant precision problems are the absolute representation of the large
`R_00(s)` coefficient and roundoff in the transform/geometry/force path.
Residual-normalization errors are much smaller.

These measurements do not prove that no float state can ever satisfy the
tolerance. They show that the baseline representation loses enough precision
to spoil a converged double equilibrium, and that the tested float iterations
stall above the requested tolerance.

## Follow-up: remaining cause isolated

The subsequent [inverse-transform diagnostic](w7x-float-inverse-diagnostic.md)
locates the remaining bottleneck in the odd R/Z inverse reconstruction.
Correcting only `r_o` and `z_o` with double GPU arithmetic while retaining
float state and float output fields gives cold-start convergence at `1e-5`
on all three grids (`148 → 202 → 313`, final FSQR `4.6679e-6`). The exported
state also passes an exact-state double evaluation. This corrective oracle
is diagnostic; the normal opt-in reference implementation below is unchanged.

## Reference-plus-displacement implementation

`CUMES_RADIUS_REFERENCE=1` enables the experiment in the CLI. Embedding
callers use `SolveRequest::use_radius_reference = true`; the process variable
is read only when `use_process_environment` is enabled. The option defaults
to off and applies only to fixed-boundary, nonaxisymmetric float solves.
Double, axisymmetric, and free-boundary policies remain unchanged.

For each m=0 Rcc mode, the state now stores

```text
reference[n] = boundary_Rcc[m=0,n]              # fixed double metadata
displacement[n,s] = float(Rcc[0,n,s] - reference[n])
```

Subtraction occurs before conversion to float, including cold seeds and
checkpoint imports. Inverse R synthesis excludes these reference modes;
radial differences therefore operate on the displacement field directly.
The radially constant reference cancels analytically. Absolute-radius
geometry and force terms add it back, and toroidal derivatives retain the
n>0 reference contribution. The angular reference is synthesized on the GPU;
all evolving state, FFT buffers and real-space arrays remain float.

Refinement carries the same immutable reference, and descent updates only
the displacement. Snapshot/checkpoint output adds the reference back in
double, preserving the existing physical-coefficient format. Derived-field
capture restores the physical even R field. Raw diagnostic dumps carry
separate reference metadata; see [dump formats](dump-files.md).

### Measured results at 1e-5

All three residual components must pass the unchanged tolerance and validity
gates. The reference implementation uses the original iteration caps and
float linear prolongation, with no timestep override.

| Run | Result | FSQR | FSQZ | FSQL |
| --- | --- | ---: | ---: | ---: |
| Baseline float multigrid | Stalls on ns=33 | 8.2574e-5 | 1.5188e-5 | 1.0506e-7 |
| Reference float multigrid | ns=33 and 66 converge in 149 and 278 effective iterations; ns=99 exhausts 5000 passes | 4.3663e-5 | 1.7544e-5 | 7.6399e-9 |
| Reference float, ns=99, restart from tight double checkpoint | Converges in 21 effective iterations / 25 passes | 9.3226e-6 | 3.7377e-6 | 8.0418e-10 |
| Replay of that float checkpoint | Converges on pass 1 | 9.3226e-6 | 3.6528e-6 | 8.0418e-10 |

The best maximum component on valid ns=99 multigrid passes is `1.4070e-5`,
so this is **a useful partial improvement, not a complete W7-X float fix**.
The hot-start replay passes tolerance but is not bit-identical in all three
residuals. A cold-start result at ns=99 remains unqualified.

Several additional experiments did not complete cold-start convergence and
were removed from the implementation:

- Double local poloidal accumulation with the scalar R_00 reference.
- Double local base-geometry arithmetic with all m=0 Rcc references: finest
  grid best maximum residual `1.3335e-5`, final FSQR `3.8116e-5`.
- Persistent compensation of float descent additions, with double local
  sums and float rounding-error storage: best maximum `1.4151e-5`, final
  FSQR `2.8646e-5`. This tests lost small updates; it is not a float-float
  accumulator in the FFT or force summation path.
- A smaller step (`0.1`, recovery disabled) from the prolonged ns=99 seed:
  best maximum `1.2973e-5`, final FSQR `2.5477e-5` after 5000 passes.
  Evaluating that exported state in double gives FSQR `2.0565e-5`, showing
  that the remaining error includes a real discrete force imbalance.

These initial experiments supported preserving small geometry variations,
but did not isolate every remaining source of roundoff. Wider final norm accumulation
alone cannot fix this: those reductions already use double. The
follow-up linked above performs this isolation and identifies the odd R/Z
inverse-transform accuracy as the remaining bottleneck.

### Verification

- Both complete configured CTest suites pass: 63/63 in the float build and
  99/99 in the verify build (including 18 memcheck and 18 initcheck variants).
- `test_radius_reference` checks sub-ULP radial differentiation, toroidal
  reference synthesis and differentiation, tiny descent increments,
  fixed-boundary refinement, seed/restart recentering, and opt-in/type gates.
  Compute Sanitizer memcheck reports zero errors.
- Double W7-X (`1315 → 1419 → 1372`, final FSQR `9.9972988876e-13`) and
  Solovev (`235 → 193 → 326`, final FSQR `9.9729630543e-17`) retain identical
  restart sequences and state families in `compare_runs`. Their checkpoint
  files and final-stage per-pass telemetry are byte-identical to the
  pre-change baselines, even with the experimental environment flag set.
- This does not qualify a new default float numerical policy. The experiment
  remains opt-in; see [ADR-0014](adr/0014-float-radius-reference.md).

## Baseline end-to-end runs

All cases use `inputs/w7x.json`, with only the stated tolerances, grid
sequence, or timestep changed. Process `CUMES_*` overrides were cleared,
then `CUMES_DUMP=1` was enabled. Single-grid ns=33 comparisons use
`CUMES_SEED_ENVELOPE=0.12` to retain the multigrid coarse-start envelope.

| Run | Result | Final FSQR | Final FSQZ | Final FSQL |
| --- | --- | ---: | ---: | ---: |
| Double, three grids, all tolerances 1e-5 | Converged through ns=99 | 9.8155e-6 | 5.0624e-6 | 3.8673e-7 |
| Float, same input | Stage 1 exhausted 3000 passes | 8.2574e-5 | 1.5188e-5 | 1.0506e-7 |
| Float, ns=33, 5000 passes | Not converged | 6.0217e-5 | 1.2677e-5 | 9.2443e-8 |
| Float, three grids, initial step 0.3 | Stage 1 exhausted 3000 passes | 1.9609e-5 | 3.8423e-6 | 2.3094e-8 |
| Float, three grids, initial step 0.1 | Stage 1 exhausted 3000 passes | 3.8561e-5 | 7.4917e-6 | 4.0319e-8 |

The default float timestep settles at `0.7338520966352987`, with three
Jacobian and three progress restarts. It does not collapse toward zero.
Extending the coarse solve from 3000 to 5000 passes does not resolve the
stall. The best maximum component on valid passes of that 5000-pass run is
`4.2830e-5`. Invalid-geometry telemetry rows contain zero sentinels and must
be excluded from any best-residual calculation.

Double reference checkpoints were separately converged to `1e-12` at ns=33
and ns=99. The latter came from the full multigrid run.

## Separate state rounding from evaluation precision

Each row below evaluates a reference checkpoint for one pass, with no prior
descent. Quantization means converting stored coefficients to float and back
to double before loading them into the double solver. Restart applies the
normal axis and fixed-boundary rules; the double cases retain the prescribed
double LCFS, while the float cases use its float representation.

| State / evaluation | FSQR, ns=33 | FSQR, ns=99 |
| --- | ---: | ---: |
| Original checkpoint / double | 9.9729e-13 | 9.9976e-13 |
| Float-quantized checkpoint / double | 1.8371e-6 | 2.1374e-4 |
| Original checkpoint loaded into float / float | 1.1532e-5 | 1.2129e-3 |

Thus, on the fine grid, state quantization alone exceeds `1e-5`, without
float FFTs, float forces, or accumulated iteration history. The float
evaluation introduces additional error. The final state of the stalled
5000-pass float ns=33 solve also fails when evaluated in double
(`FSQR=4.8126e-5`): the trajectory has acquired a real discrete force error,
not merely an inaccurate reported norm.

### Which state coefficients matter?

Selective quantization of the **same ns=99 reference checkpoint**, always
evaluated by the double solver:

| Quantized coefficients | FSQR | FSQZ |
| --- | ---: | ---: |
| Only R_00 | 2.1083e-4 | 3.9761e-5 |
| Every coefficient except R_00 | 1.3108e-6 | 4.3114e-7 |
| Rcc family | 2.1328e-4 | 3.9934e-5 |
| Zsc family | 1.2963e-7 | 2.2559e-7 |
| Both lambda families | 1.2959e-12 | 3.6730e-13 |
| Only R_00, stored relative to its edge value | 4.8304e-9 | 8.8146e-10 |
| All coefficients, with relative storage for R_00 | 1.3288e-6 | 4.3232e-7 |

The relative-storage experiment represents

```text
R_ref = R_00(edge)                              # retained in double
delta_R_00 = float(R_00 - R_ref)                 # small stored value
R_00_for_double_evaluation = R_ref + double(delta_R_00)
```

Here `R_00` ranges from `5.5586` to `5.644874542927441`; its float spacing
is `4.7683716e-7`. The largest displacement from the edge value is only
`0.08627454`. Keeping the reference separate preserves substantially more
radial information. These are **representation experiments evaluated in
double**; the later GPU implementation is evaluated separately above.

## Normalization and the spatial error path

`compute_residuals_kernel` and `rz_norm_kernel`
(`src/kernels/solver_impl.cuh`) accumulate in double for float inputs.
Individual square terms are still evaluated in float. The reduction's
division by `mnmax*ns` is canceled by the invariant evaluation's
multiplication by the same count; the remaining factors are `fNormRZ/4`
for R/Z and `fNormL` for lambda. No missing count factor was found.
Both device and host compare all three normalized squared residuals directly
to `ftol`; they do not square the tolerance again.

The per-surface normalization partials in `compute_norm_partials_kernel`
still use float. Comparing the float checkpoint replay against the
quantized-state double replay gives these relative factor differences:

| Grid | fNormRZ | fNormL |
| --- | ---: | ---: |
| ns=33 | +3.1213e-7 | +5.3459e-7 |
| ns=99 | +2.7688e-7 | -6.0876e-7 |

Substituting the double factors into these float evaluations would change
their reported residuals by less than one part per million. It would not
remove the orders-of-magnitude force error.

On the ns=33 checkpoint comparison, the relative L2 discrepancy grows from
`7.08e-8` in the inverse transform's even-parity R field to `4.64e-5` in the
Jacobian and `1.19e-3` in the radial even-parity force term `armn_e`.
The relevant operations are:

- `inverse_pack_kernel` / `inverse_accumulate_kernel` in
  `src/kernels/fourier_impl.cuh`: float spectra, cuFFT results, poloidal
  accumulation, and real-space storage retain the large R offset.
- `base_geometry_kernel` in `src/kernels/geometry_impl.cuh`: subtracts
  neighboring full-grid fields and divides by `delta_s`.
- `forces_kernel` in `src/kernels/forces_impl.cuh`: differences of half-grid
  pressure/geometry products introduce another factor of `1/delta_s`.
- `descent_step_kernel` in `src/kernels/solver_impl.cuh`: adds increments
  directly to float absolute coefficients, allowing sub-ULP updates to vanish.
  The contribution of this update loss was not separately measured.

At ns=99, `1/delta_s=98`. The two radial differences can amplify absolute
coordinate roundoff approximately as `delta_s^-2` before the residual is
squared. A small relative error in a positive normalization sum does not
capture this sensitivity. Widening a subtraction after its operands have
already been stored in float cannot recover the lost information.

## Interpretation of the baseline

A targeted implementation should preserve the small radial variation of R
through **both** state updates and geometry evaluation: for example,
reference-plus-displacement state storage together with a transform and
radial-derivative path that keeps the large reference separate. Selective
double storage/evaluation is another option. Merely multiplying all lengths
by a constant does not remove the offset-to-variation ratio.

Preserving R_00 only in the state is insufficient if inverse synthesis adds
the reference back into a float real-space array before radial differences.
Likewise, widening only the final residual reductions cannot repair earlier
force errors. A proposed fix must demonstrate full cold-start convergence
through all three grids at `1e-5`, checkpoint replay, and the appropriate
double regression gates. The opt-in implementation above passes the double
gates but does not yet satisfy the full cold-start float requirement.

The rebuilt `test_fourier`, `test_geometry_ncurr`, `test_forces`,
`test_accumulation`, `test_regression_kernels`, and `test_safety_predicates`
all pass on this GPU (6/6). Their operator-level checks do not establish
end-to-end W7-X convergence at the requested tolerance.

## Reproduction and artifacts

Build with `cmake --preset verify` / `cmake --preset float`, then build the
`cumes` target in each directory. Generate a temporary input by replacing
every `ftol_array` entry of `inputs/w7x.json` with `1e-5`; leave the grid sizes,
iteration caps, and other input values unchanged. Run each executable from
its own working directory with `CUMES_DUMP=1`, clearing other `CUMES_*`
overrides. For checkpoint tests, set `ns_array` to the checkpoint grid size,
use one corresponding tolerance/cap, and set `CUMES_MAX_ITER=1` with
`--restart` pointing to the checkpoint.

The complete local record is in `/tmp/cumes-w7x-float-investigation/`:

- Each case directory contains `command.json`, `run.log`, `summary.json`,
  and `dump/cuMES/` with binary telemetry and force-normalization factors.
- `double-ns33/state.ckpt` and `double-qualified/state.ckpt` are the reference
  checkpoints. `quantize99-*` contains the selective-quantization evaluations.
- `reference-final-*` contains the retained implementation runs;
  `m0-ref-compensated-mg`, `m0-reference-widegeom3-mg`, and
  `m0-ref-fine-smallstep` contain the rejected additions.
- `run_case.py` captures isolated runs; `roundoff_probe.py` and `roundoff.json`
  are the earlier CPU sensitivity experiment, superseded by the GPU results
  above for conclusions about solver residuals.

The historical ADR-0001 float smoke was Solovev. Its observed residual scale
does not establish a universal `~1e-7` float floor, and the input requirement
`ftol >= 1e-6` is not a guarantee of convergence for W7-X.
