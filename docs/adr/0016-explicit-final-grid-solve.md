# ADR-0016: Expose final-grid-only solves explicitly

- Status: Accepted
- Date: 2026-09-08

## Context

Converging every coarse grid to the final tolerance can cost more than solving
directly on the final grid. cuMES already supports single-grid inputs and has
qualified cold-start shaping and timestep policies for them (ADR-0007 through
ADR-0009). An explicit schedule change can exploit those policies while
preserving the requested final equilibrium resolution and convergence tests.
It can also select a different state within the weakly constrained coordinate
family. It therefore requires numerical qualification even though the
single-grid algorithm itself is unchanged.

## Decision

Add `--single-grid` to the CLI. Validate the entire input first, then select its
last `StageRequest` without changing the radial resolution, tolerance or
iteration cap. Revalidate the selected model so its shape and recorded input
parameters describe the executed schedule. Retain input warnings and the
original input source path/hash. The existing solver sees an ordinary
single-grid problem and applies its existing seed, step, validity and stopping
policies. A restart checkpoint must match that final grid.

The option has no effect on an already single-grid input. Without the option,
all explicitly configured stages and tolerances remain in force. Embedding
callers already select the same behavior by supplying one `StageRequest`;
there is no new solver policy or process environment variable.

## Evidence

The implementation is checked against otherwise identical temporary JSON
inputs whose three stage arrays contain only their original final entry. The
CLI regression checks final-stage selection, retained controls, checkpoint
grid matching, compatibility warnings, and validation of discarded stages.

With precise double arithmetic on gervais's RTX 4090, twelve alternating
pairs against `9316169`, following a warmup pair, yielded:

| Case | Multigrid iterations | Single-grid iterations | Median solve ms, multi/single | Paired saving, 95% bootstrap CI |
| --- | --- | ---: | --- | --- |
| W7-X | 1315/1419/1372 | 2465 | 1732.144 / 1277.595 | 25.59% [23.84%, 28.10%] |
| Solovev | 235/193/326 | 354 | 67.925 / 33.805 | 50.29% [49.91%, 50.57%] |

Times are CUDA-event stage solve intervals, including host submission gaps,
excluding startup/output. Process startup is noisy even in identical-binary
A/A trials, so these percentages must not be presented as process wall savings.
Every repeated run reached all configured final tolerances and reproduced its
state exactly. Both single-grid checkpoints converged on replay at iteration
1; recomputed residuals remained below tolerance. All derived field arrays
were finite and the oriented Jacobian was strictly positive. Fixed R/Z
boundary coefficients were unchanged.

W7-X single-grid residuals were (9.958592281908132e-13,
2.3324643946759757e-13, 2.3482288257552583e-13). Compared with the multigrid
state, maximum coefficient differences include Rcc 1.86e-4, Zcs 9.37e-4 and
lambda_cs 3.30e-3. These coordinate-sensitive differences are reported, not
interpreted as physical error bounds. See the [experiment record](../aggressive-optimization-study.md)
for schedule comparisons and independent-solver diagnostics.

## Alternatives

Automatically weakening intermediate tolerances would reinterpret explicit
input requirements. It is not adopted. A safeguarded GPU Pulay prototype
failed to establish reliable solve-time gains and is excluded. Global
initial-step and damping-floor sweeps also failed to improve both shipped
cases. Stage-specific W7-X steps gave smaller, trajectory-dependent gains;
no controller constants are changed by this decision.
