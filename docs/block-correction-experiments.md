# Block correction experiments, 2026-09-08

Baseline: `a93db11`, with the retained transform optimizations. Every full
solve in this study uses the original input stages, iteration caps and
tolerances. Solovev takes 235/193/326 effective passes and W7-X takes
1315/1419/1372. The new benchmark targets are diagnostics; they introduce no
production controller option or default numerical change.

## Frozen-state protocol

An isolated instrumented build captured coefficients immediately after the
production control fence, before host gates and descent. The capture points
were Solovev ns=5 at passes 100/200, ns=11 at 80/160, ns=55 at 100/250, and
W7-X ns=33 at 400/1000, ns=66 at 300/1000, ns=99 at 100/500/1000. Instrumented
complete solves retain exact baseline checkpoints, stage residuals and restart
sequences on RTX 4090 GPU 2 (`gervais`). Metadata records the residual triple,
Jacobian, force norms, gauge, refresh flags, step and restart age. All 13
captured Jacobians are valid.

These checkpoints are **frozen states, not resumable controller checkpoints**.
The public restart format omits velocity and controller/operator history.
Each probe initializes a fresh preconditioner and constraint reference once,
then freezes their refresh flags and force norms for all perturbations and
trial comparisons. This explains small R/Z residual differences from the
capture metadata. Lambda's geometry-only diagonal is rebuilt once. cuFFT
plans are explicitly bound to the probe stream. A final base replay checks
that the experiment has not contaminated the frozen caches.

Build and run the retained probes:

```sh
cmake --preset verify
cmake --build build --target cumes_benchmark_rz_coupling \
    cumes_benchmark_lambda_correction cumes_benchmark_test_device_bicgstab -j
./build/cumes_benchmark_test_device_bicgstab
./build/cumes_benchmark_rz_coupling --input inputs/w7x.json \
    --restart /path/to/w7x-state.ckpt
./build/cumes_benchmark_lambda_correction --input inputs/w7x.json \
    --restart /path/to/w7x-state.ckpt --iterations 16
```

The same probes can use ordinary solver checkpoints. They do not declare
equilibrium convergence; their diagnostic tolerance is zero to expose the
preconditioned residual even at a converged state.

## Lambda inner solve

With R/Z fixed, the discrete lambda residual is affine, including W7-X's
prescribed-current (`ncurr=1`) closure. That closure must be recomputed on
every map application. Radial half-grid averaging couples adjacent rows;
metric multiplication couples Fourier modes. The physical preconditioned
operator is measurably nonsymmetric, so the prototype uses BiCGStab rather
than assuming that conjugate gradients apply.

`FrozenLambdaOperator` retains production inverse/forward transforms and
magnetic-field/current-closure kernels. Its force kernel computes only the
four lambda force arrays. The homogeneous map zeros the flux/current offsets
using private closure scratch, avoiding finite-difference cancellation.
Odd-mode scaling, physical basis factors, dependent axis entries and inactive
coefficients follow the production conventions. BiCGStab vectors, reductions,
recurrence scalars and updates remain on the GPU; allocations occur at setup.
The fixed iteration budget includes both map applications per inner step.

Initial precise-double TITAN Xp results, 16 inner steps:

| Frozen state | Full-map relative error | Affinity error | Lambda residual before / after full correction | Total residual ratio |
| --- | ---: | ---: | --- | ---: |
| Solovev ns=55, pass 250 | 0 | 3.29e-11 | 7.22e-18 / 3.03e-25 | 1.0109 |
| W7-X ns=99, pass 500 | 0 | 7.96e-12 | 9.69e-13 / 6.69e-17 | 2.9909 |

The total residual is FSQR+FSQZ+FSQL with frozen normalization factors.
Solovev's best tested fractional correction, scale 0.25, reduces it only
0.297%. All W7-X scales 1, 0.5, 0.25 and 0.125 increase it. Both base replays
are exact. The lambda-only force specialization preserves every diagnostic
value exactly; unpaired inner-solve times decrease from 1.772 to 1.631 ms for
Solovev and 45.047 to 41.653 ms for W7-X. These timings establish prototype
cost, not a solver speedup.

An isolated full-solver prototype also tested periodic corrections, including
their setup, two maps per inner step, full residual reevaluation and lambda
velocity reset. With the correction disabled, both original checkpoints are
byte-identical to the preserved baseline executable. Eight inner steps every
250 passes, at scales 0.25 or 1, change Solovev to 235/193/384. Four steps
every 100 passes at scale 0.25 give 241/195/565. The corresponding W7-X trials
exhaust the original coarse-stage cap. Restricting corrections to its final
stage also exhausts the original 5000-pass cap, including the gentler
four-step, scale-0.1 variant. This implementation is rejected: lowering
lambda's residual alone does not improve the coupled descent.

The manufactured BiCGStab test uses a nonsymmetric 4099-row tridiagonal
system, exact diagonal systems at 513 and 1 rows, and zero-RHS reuse in both
precisions. Double's true relative residual is 6.60e-13; float's is 1.03e-6.
Exact systems finish in one step and zero RHS in zero steps. The same test
passes on TITAN Xp and RTX 4090; Ada memcheck and synccheck report zero errors.
`active=0` is a stopped recurrence, not a convergence certificate: callers
must check the true residual and actual equilibrium merit.

## R/Z coupling probe

The first model uses Rcc/Zsc modes m>=1 and two smooth radial shapes,
preserving the LCFS and mixed m=1 gauge. Central differences of the frozen
production preconditioned residual form a small least-squares system; basis
construction, differences, reductions, the dense solve and trial updates run
on the GPU. Acceptance uses all three unpreconditioned normalized residuals.

| Frozen state | Basis size | R-to-Z / R response norm | Best total residual reduction | Ordinary passes at equal evaluation budget |
| --- | ---: | --- | ---: | --- |
| Solovev ns=55, pass 250 | 12 | 0.530–1.565 | 0.0264% | 28 passes: 43.46% |
| W7-X ns=99, pass 500 | 24 | 0.265–0.955 | 0% | 52 passes: 16.42% |

Halving the finite-difference step from 1e-5 to 5e-6 changes response norms
by at most 1.75e-10 relative for Solovev and 2.79e-9 for W7-X. Base replays
are exact, and Ada reproduces the conclusion. Ordinary-pass comparators
start from the same state with a fresh controller; they are not continuations
of the captured production trajectory. This rejects the small smooth basis
as a correction space. It does not rule out coupling the radial preconditioner.

## Artifacts

The session archive is
`../tmp/cumes-block-20260908/` relative to the repository. `states/manifest.json`
records capture provenance and SHA256 checksums; `ada-diagnostics/README.md`
records the remote toolchain, commands, exact tested source copies and logs.
`rz-results/`, the `lambda-*-force-only.log` files and `live-results/` contain
the measurements above. `live-source/` preserves the rejected full-solver
lambda hook. These large local artifacts are not repository fixtures.
