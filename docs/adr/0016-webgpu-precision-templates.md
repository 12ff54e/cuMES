# ADR-0016: Shared WebGPU precision templates and evaluation order

## Decision

Maintain each numerical shader family in one WGSL template and expand its
arithmetic, types, and constants with JavaScript during the CMake build. Scalar
operations become WGSL expressions; pair-single operations call a shared
library of explicitly rounded f32 error-free transforms. No preprocessing or
precision dispatch is added to the browser iteration loop.

Use the same evaluation tree for both precisions where the mathematics is the
same. In particular, geometry contractions, force terms, and descent use the
existing paired operation order. Current integration uses parallel angular
integrands, an ordered surface reduction, and parallel field finalization in
both precisions. Precision-dependent input/output layouts and the scalar
radius-reference reconstruction remain explicit template branches.

Compensated arithmetic is also a property of an operation, not only of a
solver mode. Explicit `Pair` types and `pair_*` intrinsics preserve paired
islands inside scalar shaders. Scalar Fourier sums retain their compensated
accumulation policy. The shared library preserves the atomic rounding fences
that prevent browser backends from reassociating error-free transforms.

## Numerical consequences and validation

This is not a Class A refactor of the scalar trajectory. A common evaluation
order changes f32 rounding and can change adaptive controller decisions, so
the scalar migration is qualified as Class C. It does not change the force
model, grids, convergence thresholds, validity gates, or comparison tolerances.
The paired evaluation order is retained wherever the operations were already
shared mathematically.

Qualification uses the existing independent host references for transforms,
geometry, magnetic field, force, constraints, preconditioning, and descent;
the exact rounding tests; selective-compensation tests; and full convergence
and validity checks in both precisions. Execute these in real Chrome and
headless Firefox. Exercise fixed and free boundaries and non-axisymmetric
geometry. Baseline traces distinguish intended scalar rounding changes from
unexpected paired regressions. Browser-specific floating-point trajectories
must be compared on the same adapter and configuration.

The first core migration passed the unmodified Chrome numerical verification
and all 14 CTests. On the RTX 3060 Ti, its scalar verification converged in
302 effective iterations versus 327 before the order change. The paired
free-boundary Solovev trajectory retained all 1,056 controller records exactly.
These counts describe that qualification run, not universal acceptance values.

The template syntax and build entry points are documented in
[`src/webgpu/shaders/README.md`](../../src/webgpu/shaders/README.md).
