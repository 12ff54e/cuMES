# ADR-0002 — R/Z vs lambda force-kernel split (measured, not adopted)

Status: accepted negative result; prototype retired (blueprint §8.10)

## Context

`forcesKernel` (src/forces_impl.cuh) computes all sixteen force families
(armn/azmn/brmn/bzmn/crmn/czmn + blmn/clmn) in one kernel. On sm_61 it compiles
to **108 registers, 0 bytes spill** — high register pressure that limits
occupancy to ~512 threads/SM. §8.10 asks whether splitting the R/Z and lambda
families into two kernels (each with a smaller live working set) wins wall time.

## Decision

Keep the monolithic `forcesKernel` as the sole production path. A differential
prototype split it into `rzForcesKernel` (12 families) and
`lambdaForcesKernel` (4 families), copying the per-family arithmetic verbatim,
to measure the occupancy hypothesis. The split was bit-identical but slower,
so the prototype and its dedicated test were retired after the measurement.

## Measured result (TITAN Xp, sm_61)

| Case | monolith regs | split regs (RZ/λ) | median (monolith / split) | outcome |
| ---- | ------------- | ----------------- | ------------------------- | ------- |
| double axisymmetric ns=5 | 108 | 82 / 54 | 9.22 / 12.29 µs | **1.33× slower** |
| double 3D ns=11 | 108 | 82 / 54 | 11.07 / 13.31 µs | **1.20× slower** |
| float axisymmetric ns=5 | — | — | 6.14 / 8.90 µs | **1.45× slower** |

`test_force_split.cu` proves the split is **bit-identical** to the monolith
(max |diff| = 0 across all sixteen families, double and float) and reproduces the
timing. The register reduction (108 → 82/54) is real, but the kernel is
memory-bound: the split launches two kernels and each re-loads the shared
geometry/field/radial inputs (r/ru/rv, z/zu/zv, gsqrt/guv/gvv, bsupu/bsupv/
totalP, sqrtS) from global memory, roughly doubling input traffic and adding a
second launch. That dominates the occupancy gain.

## Consequences

- The monolith is retained as the production force path; **no production wiring
  was changed** (the Solovev trajectory is bit-identical to Phase 8).
- The split implementation and `test_force_split` are no longer in the tree.
  `test_force_reference.cu` plus the frozen Solovev/W7-X trajectories now pin
  the production monolith.
- The split should not be re-attempted without new evidence, such as a layout
  change that avoids reloading the shared geometry/field inputs.

## Alternatives considered

- **Radial-tile geometry/force fusion** (§8.10, higher risk): trades global
  traffic for registers/shared memory; not attempted — the split result shows
  the force path is input-traffic-bound, so the fusion's extra register pressure
  is unlikely to pay off and would need its own measured gate.
- **Force-plus-projection fusion** (§8.10, high risk): avoid materializing the
  real force arrays; deferred for the same reason.
