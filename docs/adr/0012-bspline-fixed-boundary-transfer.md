# ADR-0012: Apply a reusable B-spline transfer matrix on the GPU

- Status: Accepted
- Date: 2026-08-31

## Context

ADR-0011 established that a smoother coarse-to-fine state reduces the number
of force evaluations needed on refined grids. Its local four-point
Catmull-Rom transfer reduced the main fixed-boundary workloads substantially,
but it does not use all of the converged coarse profile when estimating a new
radial row.

`deps/magnetic-coordinate/BSplineInterpolation` provides
`InterpolationFunctionTemplate1D<3>`, which factorizes the interpolation
system once and reuses it for many profiles. A direct host batch over all 936
W7-X spectral profiles was measured with `INTP_CELL_LAYOUT`: 1.90 ms for
`33 → 66` and 2.94 ms for `66 → 99`. The existing complete GPU Catmull-Rom
boundaries cost 0.084 ms and 0.232 ms. Moving the six state families to and
from the host would therefore make transfer roughly 15 times slower before
counting PCIe staging, even though the better initial state could still reduce
total solver time.

## Decision

Use the interpolation template only to construct the linear
`[ns_new][ns_old]` cubic B-spline maps. At multigrid entry, launch one
background task that builds all scheduled fixed-boundary maps sequentially
while the first stage iterates on the GPU. Consume the completed batch at the
grid transitions, upload each small map, and apply it to every `family × mode`
radial profile in one CUDA kernel. The spectral state remains device-resident.
Direct `Prolongation` callers retain a synchronous construction fallback.

As with the earlier transfers, odd poloidal modes are interpolated after the
`scalxc` decomposition and the extrapolated old-axis value is used as the
first spline sample. The new odd-mode axis is zeroed, velocity is reset, and
the LCFS is copied explicitly rather than relying on an endpoint dot product.

Select the B-spline for precise-double fixed-boundary continuation. Retain
Catmull-Rom for precise-double 3-D free-boundary continuation and linear
transfer for axisymmetric free-boundary and mixed-float runs. If the optional
header dependency is absent, fixed-boundary builds fall back to Catmull-Rom.
`CUMES_USE_BSPLINE_PROLONGATION=OFF` provides the same build-time fallback;
`CUMES_FORCE_CATMULL_PROLONGATION=1` and
`CUMES_FORCE_LINEAR_PROLONGATION=1` provide runtime diagnostic alternatives.

## Evidence

Every retained result satisfies all configured force-residual tolerances:

| workload | previous transfer | B-spline | change | final FSQR |
| -------- | ----------------: | -------: | -----: | ---------: |
| W7-X fixed `33 → 66 → 99` | 4160 | 4106 | -1.30% | `9.997e-13` |
| Solovev fixed `5 → 11 → 55` | 766 | 754 | -1.57% | `9.973e-17` |

W7-X changes from `1315 → 1443 → 1402` to
`1315 → 1419 → 1372`; Solovev changes from `235 → 190 → 341` to
`235 → 193 → 326`. The GPU matrix path reproduces the direct host B-spline
iteration counts and residuals exactly.

Over 10,000 warmed-up optimized repetitions, gervais constructs the W7-X
`33 → 66` and `66 → 99` matrices in median times of 36.7 us and 136.5 us,
respectively. Their 173.3 us total is less than one average W7-X device
iteration (about 462 us), and stage 1 takes 1315 iterations. Solovev's two
matrices total 9.2 us. Consequently the background batch is ready well before
the first qualified transition.

The rejected free-boundary measurements were CTH-like 3-D, 424 → 425, and
axisymmetric Solovev, 1025 → 1074. Their selected policies are unchanged.

## Consequences

- Fixed-boundary refined stages begin closer to the fine-grid fixed point.
- Host matrix construction is overlapped with the coarse GPU solve rather
  than blocking the stage transition.
- The one-time transfer performs a dense radial dot product, but its cost is
  negligible compared with the saved solver iterations and no state PCIe copy
  is introduced.
- The default fixed-boundary trajectory is a Class C numerical change.
- Submodule-free builds remain functional and deterministic through the
  Catmull-Rom fallback.
