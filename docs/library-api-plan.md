# cuMES library API and optimizer integration plan

## Implementation status

The first in-process milestone is implemented:

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
- `examples/cumes_meow_optimize.cpp` demonstrates the independent integration
  layer and builds when `CUMES_MEOW_SOURCE_DIR` names a meow checkout;
- ordinary-C++ direct API and installed-package consumer tests cover the
  embedding boundary.

Topology-keyed CUDA resource reuse and controlled accepted-state continuation
remain follow-up performance work. The current solver object intentionally
keeps cold-start evaluations mathematically independent.

## 1. Goal and first milestone

cuMES must be usable as an in-process equilibrium engine inside a nonlinear
least-squares optimizer. The optimizer owns the design vector, the mapping from
that vector to boundary Fourier harmonics, and the objective residuals. cuMES
owns validation and the conversion of the resulting physical problem into a
converged equilibrium.

The first milestone is:

> A normal C++20 program can read or construct a `ProblemSpec`, modify its
> boundary harmonics in memory, validate it, call a host-facing cuMES solver,
> and receive a complete host equilibrium without using files for solver
> output, handling CUDA objects, or duplicating CLI orchestration.

The initial optimizer qualification is fixed-boundary, precise-double, one GPU,
and deterministic cold starts. The facade continues to expose the behavior of
the configured cuMES build, including existing float and free-boundary paths,
but those paths require their existing independent qualification gates.

## 2. Ownership and dependency direction

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

## 3. Public library surfaces

### 3.1 Configuration API

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

### 3.2 Solver API

Add a host-facing `cumes::EquilibriumSolver` facade. It follows the configured
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

### 3.3 Output API

Binary, NetCDF, HDF5, checkpoint, and Boozer publication stay separate from the
solver facade. They consume the same complete host snapshot and report returned
by the facade. Optimizer users therefore do not link optional output backends
unless they request them.

## 4. Boundary parameterization and objective contract

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

The `cumes_meow_optimize` example demonstrates this ownership explicitly.
Its optimizer-side `plasma_size_target.hpp` reconstructs the final LCFS from
the returned spectral state and computes VMEC-compatible cross-sectional area,
volume, `Rmajor_p`, and `Aminor_p`. The residual callback returns the major- and
minor-radius differences to meow. Neither this target definition nor its
weighting is part of the cuMES solver or meow's generic TRF implementation.

The sibling optimizer helper `magnetic_gradient_target.hpp` computes the
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

### 4.1 QS, QH, and QA optimizer targets

The optimizer-side `quasisymmetry_target.hpp` builds a least-squares residual
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

## 5. Repeated-solve and concurrency policy

The first implementation is a correctness-preserving facade over the existing
multigrid run. Subsequent performance work introduces a reusable session whose
lifetime owns topology-dependent CUDA resources.

Rules for optimizer use are:

- cold start is the deterministic default;
- exact-repeat caching in the adapter is allowed;
- one solver object is not called concurrently;
- parallel evaluations use one solver object per worker/GPU;
- resource reuse must not change the initial state or arithmetic trajectory;
- warm starts are opt-in and must not make residuals evaluation-order
  dependent.

For finite-difference Jacobians, every perturbation must eventually start from
the same accepted central equilibrium, rather than from the previous perturbed
solve. Until that policy is represented explicitly by the optimizer/evaluator
interface, cold starts remain the correctness baseline.

Direct `printf` calls and process-global environment reads in the current
solver stack are library-hostile. The staged migration is:

1. put the complete solve and snapshot assembly behind the facade;
2. introduce structured logging/run options and make the facade quiet by
   default;
3. translate legacy environment variables into those options in the CLI;
4. reuse streams, plans, arenas, and invariant tables without changing the
   mathematical seed.

## 6. Build and packaging plan

Provide supported namespaced CMake targets:

```text
cumes::core
cumes::config_json
cumes::solver
cumes::io
```

`cumes::solver` is the configured-precision facade and links the selected CUDA
operator library internally. The lower-level float and double operator targets
remain available to cuMES tests but are not the primary embedding interface.

Add independent `CUMES_BUILD_CLI`, `CUMES_BUILD_TESTS`, and
`CUMES_BUILD_BENCHMARKS` options, build/install include interfaces, an exported
target set, and a `cuMESConfig.cmake`. Start with the existing static-library
model; shared-library CUDA ABI work is separate.

The CLI links the facade and becomes only:

```text
arguments -> config/output preflight -> solve -> diagnostics/publication
```

## 7. Implementation sequence

1. Add this plan and an ADR/API contract if later interface decisions require
   compatibility guarantees.
2. Add `EquilibriumSolver` over the configured float/double implementation.
3. Move seed construction, stream creation, multigrid invocation, final state
   transfer, and report input metadata into the facade.
4. Refactor `main.cu` to use the facade without changing output or exit policy.
5. Add public target aliases and install/export packaging.
6. Add a direct-library solve test and a downstream CMake consumer test.
7. Add an optimizer-style example that captures a solver in a residual
   callback while keeping meow and cuMES independently reusable.
8. Replace library printing/environment access with structured options.
9. Add topology-keyed resource reuse and controlled continuation after a
   benchmark identifies the dominant repeated-solve setup costs.

## 8. Acceptance gates

The facade refactor is accepted only when:

- a direct library solve returns a complete, shape-consistent snapshot;
- the CLI and direct API produce identical state and run metadata;
- existing Solovev/W7-X trajectory and fixed-point gates remain unchanged;
- strict/compatibility JSON behavior remains unchanged;
- invalid problems, nonconvergence, and CUDA exceptions remain distinguishable;
- an external C++20 consumer can link without compiling its source as CUDA;
- repeated calls are leak-free under the existing sanitizer matrix;
- the same design vector produces repeatable objective residuals;
- performance measurements separate solve time, setup time, and host transfer
  time before session reuse is introduced.
