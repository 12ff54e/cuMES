# cuMES library API and optimizer integration

## Overview

The in-process interface provides:

- `cumes::EquilibriumSolver` accepts a `ValidatedProblem` and returns a
  complete host `EquilibriumSnapshot`, converged flux profiles, and
  `RunReport`;
- cold starts and in-memory snapshot restarts are supported;
- embedding calls are quiet and ignore process-global `CUMES_*` controls by
  default, while the CLI explicitly preserves them;
- `parse_problem_spec` maps an in-memory JSON document;
- `cumes::solver`, `cumes::config_json`, `cumes::core`, and `cumes::io` are
  supported build-tree targets, and the static-library closure is installable
  through `find_package(cuMES)`;
- meow owns the integration examples and target implementations; cuMES can
  request their build when `CUMES_MEOW_SOURCE_DIR` names a meow checkout;
- ordinary-C++ direct API and installed-package consumer tests cover the
  embedding boundary.

cuMES is usable as an in-process equilibrium engine inside a nonlinear
least-squares optimizer. The optimizer owns the design vector, the mapping from
that vector to boundary Fourier harmonics, and the objective residuals. cuMES
owns validation and the conversion of the resulting physical problem into a
converged equilibrium.

A normal C++20 program can read or construct a `ProblemSpec`, modify its
boundary harmonics in memory, validate it, call the host-facing cuMES solver,
and receive a complete host equilibrium without using files for solver output,
handling CUDA objects, or duplicating CLI orchestration.

The optimizer qualification is fixed-boundary, precise-double, one GPU,
and deterministic cold starts. The facade continues to expose the behavior of
the configured cuMES build, including existing float and free-boundary paths,
but those paths require their existing independent qualification gates.

## Ownership and dependency direction

The dependency graph remains acyclic:

```text
JSON adapter ──> ProblemSpec ──> validation ──> ValidatedProblem
                                                │
                                                v
                                      cuMES solver facade
                                                │
                                                v
                                  Equilibrium + RunReport
                                                │
                                                v
boundary parameterization + objective ──> meow residual callback
```

cuMES does not depend on Eigen or meow. The generic meow TRF library does not
depend on CUDA or cuMES. An integration executable or optional adapter target
links both libraries.

The optimizer layer owns:

- the ordered design-variable-to-`(family, m, n)` boundary map;
- parameter bounds and scaling;
- the target definition and residual weighting;
- the policy for an invalid boundary or failed equilibrium evaluation.

cuMES owns:

- schema mapping and physical/shape validation;
- boundary folding and mode metadata;
- all CUDA resources and the multigrid solve;
- convergence and numerical-validity classification;
- construction of a complete host equilibrium and structured run report.

`SolveOutcome::timings` separates host-wall setup, multigrid execution, final
spectral-state transfer, and their total. These values are diagnostic metadata
for repeated-solve profiling; `total_device_time_ms` remains the distinct sum
of timed CUDA work reported by the multigrid stages. The multigrid timing is
further partitioned into aggregate stage resource setup, iterative solve,
derived-field output capture, resource teardown, and other orchestration such
as radial prolongation. The subphases sum to `multigrid_wall_ms`.

## Public library surfaces

### Configuration API

The existing four-stage configuration contract remains normative:

```text
JSON -> ProblemSpec -> ValidationReport -> ValidatedProblem -> DeviceParams<T>
```

`ProblemSpec` is the optimizer-editable input. `ValidatedProblem` remains
immutable and must be rebuilt after boundary changes because it caches folded
boundary coefficients, grid shapes, and mode metadata.

The supported configuration target exposes:

- `read_problem_spec(path, options)` for mapping a JSON file;
- `read_and_validate(path, options)` as a convenience path;
- `validate(spec, options)` for optimizer-generated in-memory problems.

`parse_problem_spec(json_text, options)` provides the corresponding in-memory
JSON mapping path for applications that already own configuration text.

### Solver API

The host-facing `cumes::EquilibriumSolver` facade follows the configured
cuMES precision (`Real`, selected at build time as `float` or `double`) while
keeping that scalar choice out of the public interface. Its public header must
not expose CUDA stream types, device pointers, cuFFT handles, or internal
operator classes.

The facade accepts a `ValidatedProblem` and returns a structured `SolveOutcome`:

- a complete `EquilibriumSnapshot`, including all six spectral families and
  final derived fields;
- the `RunReport` and final residual/damping information;
- total device time and the failed multigrid stage, if any;
- an explicit converged/not-converged classification.

Configuration errors remain `ValidationResult` values. CUDA/setup failures use
the existing `CumesError` exception boundary; the CLI converts them to exit
codes while an embedding application can add its own evaluation-failure policy.
The facade never writes result files and never exits the process.

### Output API

Binary, NetCDF, HDF5, checkpoint, and Boozer publication stay separate from the
solver facade. They consume the same complete host snapshot and report returned
by the facade. Optimizer users therefore do not link optional output backends
unless they request them.

## Boundary parameterization and objective contract

The optimizer must use an explicit stable map rather than the incidental order
of sparse JSON harmonic arrays:

```cpp
enum class BoundaryFamily { RBC, ZBS };

struct BoundaryVariable {
    BoundaryFamily family;
    int m;
    int n;
    double scale;
};
```

For each evaluation, the adapter copies an immutable baseline `ProblemSpec`,
replaces the selected harmonics, validates the new spec, and invokes cuMES.
This makes the residual a mathematical function of the optimizer vector and
prevents stale folded-boundary data.

meow minimizes a residual vector, so objectives use scaled residuals such as

```text
r_i(x) = weight_i * (equilibrium_quantity_i - target_i).
```

The residual dimension must remain fixed and every successful residual must be
finite. By default, validation failure, nonconvergence, or numerical failure
terminates the optimization evaluation with context. A synthetic penalty is an
explicit optimizer policy because an arbitrary penalty can corrupt numerical
Jacobians.

Objectives should prefer gauge-invariant physical fields over raw lambda
coefficients. In particular, the documented near-degenerate lambda gauge makes
raw lambda targets unsuitable unless the gauge is deliberately fixed.

The meow `cumes_meow_optimize` example demonstrates this ownership explicitly.
Its optimizer-side plasma-size target reconstructs the final LCFS from
the returned spectral state and computes VMEC-compatible cross-sectional area,
volume, `Rmajor_p`, and `Aminor_p`. The residual callback returns the major- and
minor-radius differences to meow. Neither this target definition nor its
weighting is part of the cuMES solver or meow's generic TRF implementation.

The sibling meow magnetic-gradient target computes the
pointwise half-grid observables

```text
B = sqrt(B^i B_i)
B dot grad(B) = B^s d_s B + B^u d_u B + B^v d_v B
(B cross grad(psi_p)) dot grad(B)
  = psi_p'(s)/sqrt(g) * (B_v d_u B - B_u d_v B).
```

Angular derivatives use periodic Fourier-collocation differentiation, with
the physical `nfp` multiplier in the toroidal direction. The solver facade
exports the converged physical poloidal-flux derivative because prescribed-
current equilibria cannot reconstruct it from the input. The helper returns
fields, not a chosen scalar: surface selection, aggregation, normalization,
and residual weights remain part of the optimizer's target definition.

### QS, QH, and QA optimizer targets

The optimizer-side quasisymmetry target in meow builds a least-squares residual
vector from the equilibrium primitives. For helicity integers `(M,N)`, define

```text
q_QS = ((N - iota*M) (B cross grad(psi)) dot grad(B)
        + (M*G + N*I) B dot grad(B)) / B^3.
```

Here `I(s)=<B_theta>` and `G(s)=<B_zeta>` are the VMEC-compatible covariant
flux functions (`buco` and `bvco`), not the pointwise native covariant field
components. cuMES publishes them as equilibrium profiles. The optimizer
helper makes the flux choice explicit: the requested poloidal flux is
supported, while the conventional Landreman--Paul QS metric can select the
normalized toroidal flux. The sign above follows the requested ordering
`(B cross grad(psi)) dot grad(B)`; reversing the scalar triple product reverses
both terms together and leaves the squared metric unchanged.

For each requested half-grid surface, uniform angular quadrature represents
the flux-surface average using `abs(sqrt(g))`:

```text
f_QS = sum_j w_j <q_QS^2>_j.
```

The API returns pointwise residuals whose squared Euclidean norm is exactly
`f_QS`, along with the per-surface and scalar totals. This is preferable to
returning `f_QS` as one residual, which would make a least-squares optimizer
minimize `f_QS^2`.

The composite residual builders append optimizer-owned scalar residuals:

```text
f_QH = f_QS + (A - A_target)^2
f_QA = f_QS + (A - A_target)^2
             + (integral_0^1 iota(s) ds - iota_target)^2.
```

`A` uses the existing VMEC-compatible plasma-size helper. The iota integral
uses midpoint quadrature on the solver's native uniform half grid. QA checks
that `N=0`; surface selection, nonnegative surface weights, `(M,N)`, and target
values are all explicit optimizer configuration.

In current language, `I=mu0/(2*pi)` times the enclosed toroidal current and
`G=mu0/(2*pi)` times the poloidal current outside the surface, subject to the
equilibrium's orientation convention.

The optional meow `cumes_meow_qs_optimize` integration executable demonstrates the
full composition: boundary vector to validated problem, cuMES solve, QS/QH/QA
residual construction, and meow TRF evaluation. Its command line makes the
flux-gradient convention and physical `(M,N)` explicit. It selects all native
half-grid surfaces only as an example policy.

## Repeated-solve and concurrency policy

`EquilibriumSolver` is a correctness-preserving facade over the multigrid run.
The current solver object intentionally keeps cold-start evaluations
mathematically independent; topology-keyed CUDA resource reuse and controlled
accepted-state continuation are not part of this interface.

Rules for optimizer use are:

- cold start is the deterministic default;
- exact-repeat caching in the adapter is allowed;
- one solver object is not called concurrently;
- parallel evaluations use one solver object per worker/GPU;
- resource reuse must not change the initial state or arithmetic trajectory;
- warm starts are opt-in and must not make residuals evaluation-order
  dependent.

For finite-difference Jacobians, every perturbation starts from the same
accepted central equilibrium rather than from the previous perturbed solve.
Cold starts remain the correctness baseline; any accepted-state continuation
is an explicit optimizer policy.

## Build and packaging

The installed package provides these supported namespaced CMake targets:

```text
cumes::core
cumes::config_json
cumes::solver
cumes::io
```

`cumes::solver` is the configured-precision facade and links the selected CUDA
operator library internally. The lower-level float and double operator targets
remain available to cuMES tests but are not the primary embedding interface.

Independent `CUMES_BUILD_CLI`, `CUMES_BUILD_TESTS`, and
`CUMES_BUILD_BENCHMARKS` options control the optional products. The package
installs its include interfaces, exported target set, and `cuMESConfig.cmake`
using the static-library model.

The CLI links the facade and is limited to:

```text
arguments -> config/output preflight -> solve -> diagnostics/publication
```
