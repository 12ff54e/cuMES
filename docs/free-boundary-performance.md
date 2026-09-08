# Free-boundary performance qualification — 2026-09-09

The retained changes reduce vacuum-kernel and transfer overhead while preserving
the configured equilibrium problem and iteration controls. They comprise
parallel evaluation of axisymmetric regularized source terms, smaller blocks
for the singular right-hand-side kernel, and two device-to-host copies moved
before existing stream fences. Free-boundary execution retains the direct
kernel launch path. Separate prefix/suffix CUDA graphs were tested, then removed
because their incremental benefit was inconsistent across the two GPUs.

The reference is cuMES `bcdd3dab5149dc0c5ec8608574f59f2d6932b398` with
vacuum-field `edbbb2803203d9590af9d5444b1527078a0dc30e`. The retained vacuum
changes are commits `4acd589`, `2eb53c9` and `4d19939`. The final implementation
is cuMES `cc2d91d`; binary and source hashes accompany the timing records. Newton corrections
remain disabled for these free-boundary runs.

The [fixture manifest](../benchmarks/free_boundary/manifest.json) pins source
inputs, every adaptation, and field-data hashes. The [fixture README and
materializer](../benchmarks/free_boundary/README.md) describe reproducible paths
for each machine. Solovev's two field sources exercise setup paths for the same
physical case; they are not independent geometries.

| Case | Radial stages | Tolerances | Iteration caps | Input `nvacskip` |
| --- | --- | --- | --- | ---: |
| Solovev, precomputed mgrid | 16 → 32 | 1e-10, 1e-14 | 10000, 20000 | 6 |
| Solovev, embedded MAKEGRID | 16 → 32 | 1e-10, 1e-14 | 10000, 20000 | 6 |
| CTH-like, finite pressure/current | 15 → 25 | 1e-8, 1e-10 | 2500, 2500 | 9 |
| W7-X, original negative flux | 51 | 1e-12 | 50000 | 6 |
| W7-X, separate positive-flux adaptation | 51 | 1e-12 | 50000 | 6 |

W7-X's single `ns=51` stage is native to its upstream source. No stage is removed,
and no tolerance or iteration cap is relaxed. The configured vacuum spacing,
existing adaptive spacing rule, full/partial update decisions, activation gate,
soft restart and convergence predicates remain unchanged.

The upstream W7-X input has `phiedge=-1.74`. With the pinned external field,
both reference and candidate reject it at the existing toroidal-field sign
check, before a completed native result is available. Its failures remain in
the artifact set and are excluded from speedup claims. The separately declared
`w7x_positive_flux` case changes only `phiedge` to `+1.74` to reproduce the
historical cuMES convention. It is labeled separately throughout; W7-X's ignored
`free_boundary_method=only_coils` selector is removed from both portable inputs.

Baseline Nsight Systems profiles identified different dominant kernels. In the
Solovev profile, serial axisymmetric `gstore` occupied 43.3% of measured GPU
kernel time; in the positive-flux W7-X profile, singular `bvec` occupied 52.0%.
These are diagnostic kernel-time shares from individual profiled runs, not
end-to-end speedups. Raw reports and CSV summaries are retained under
`../tmp/cumes-free-boundary-20260909/pascal/profile-*`.

For axisymmetric `gstore`, each source/image/target term is independent until
the final sum. The implementation evaluates these terms in parallel and stores
only the unweighted `htemp-ga1` value. A second kernel accumulates in the
original ascending source/image order, retaining the original weighted
multiply-add expression. The target-fast scratch holds
`64 * nZnT * nThetaEven` values, allocated once per vacuum operator; the Solovev
double case needs 240 KiB. Full updates rewrite the scratch, while partial
updates retain the existing reuse of regularized contributions.

The singular RHS keeps one thread per Fourier mode and the original serial
surface accumulation. Systems with at most 256 modes use eight threads per
block to distribute work that previously occupied a single block. Larger
systems retain their original 256-thread blocks. The grid uses the selected
block size, explicitly `(mnfull+7)/8` on the optimized path:
the existing generic launch helper computes its grid using 256, so merely
passing it a smaller block size would leave output modes unwritten. The kernel
body and its floating-point operations are unchanged.

The cuMES bridge copies `buco/bvco` into pinned storage before the existing
prefix fence and copies `delbsq` before the existing control fence. Both values
are consumed after the same preceding device work completes. Vacuum operations
retain their default-stream execution and host update order; the final vacuum
scalar download completes before the cuMES suffix consumes its fields.
The pressure-mismatch diagnostic starts at zero and is sampled only after an
edge-force evaluation. This also fixes an inherited pre-activation read of
uninitialized device memory: CUDA 12.9 initcheck reproduced one such read in
Solovev and 47 in CTH on both the untouched baseline and the earlier candidate.
No equilibrium or controller calculation consumes that diagnostic.

Final measurements use [run.py](../benchmarks/free_boundary/run.py) and
[analyze.py](../benchmarks/free_boundary/analyze.py): two warmups per variant,
then seven measured pairs per case, alternating AB/BA from AB. Each GPU runs
one benchmark at a time with CPU affinity 8. Inherited `CUMES_*` variables are
removed. Every run records binary/input/dependency hashes, commands, GPU
telemetry, raw logs, native output and checkpoint where available. Failures and
timing spikes are retained without filtering.

| Machine | GPU | CUDA compiler / target | Host compiler | Driver |
| --- | --- | --- | --- | --- |
| Local | TITAN Xp, GPU 0 | CUDA 12.1.105, native sm_61 | GCC 12.4 | 580.173.02 |
| gervais | RTX 4090, GPU 2 | CUDA 12.9.41, native sm_89 | GCC 12.4 | 570.169 |

Builds use precise double arithmetic with vacuum support and the existing
field-output backends. Configure/build scripts and dependency revisions are
archived alongside each machine's results. Both Ada builds use the same
archived CUDA header compatibility include (six host exception-specification
fixes) and user-local NetCDF installation. Clocks are not fixed; per-run
telemetry records their variation. The local display process remains resident.

The solver metric is the CLI's summed stage CUDA-event interval, printed to
0.001 ms. It includes vacuum work and host gaps within each solver interval,
but excludes outer stage construction, interstage transfer, field setup,
publication and output. Process wall time separately includes startup, field
setup and output; telemetry and numerical validation occur outside that timer.
The final runner uses blocking process completion with a separate timeout
watchdog. Earlier graph/direct experiments used timeout polling that could
quantize process wall measurements by up to 50 ms; their process wall timings
do not support performance claims here.

The reported reduction is the median of paired percentages
`100 * (baseline-candidate) / baseline`. Confidence intervals bootstrap that
median with 20,000 resamples and seed 20260909. Per-variant median, p95, MAD,
minimum and maximum remain in each summary. A positive reduction means faster.

| Case | TITAN Xp baseline → candidate solver ms | Reduction, 95% CI | RTX 4090 baseline → candidate solver ms | Reduction, 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Solovev, mgrid | 921.648 → 580.503 | 37.01% [36.68, 37.07] | 622.545 → 437.262 | 29.69% [18.06, 30.59] |
| Solovev, embedded MAKEGRID | 923.122 → 582.330 | 36.90% [36.34, 37.04] | 622.900 → 432.296 | 30.73% [30.60, 44.30] |
| CTH-like | 1103.252 → 1060.136 | 3.84% [3.73, 3.92] | 975.467 → 948.854 | 2.73% [-1.31, 6.65] |
| W7-X, positive flux | 7337.295 → 6838.747 | 6.77% [6.73, 6.82] | 6916.617 → 5277.761 | 23.70% [22.97, 23.75] |

| Case | TITAN Xp baseline → candidate process ms | Reduction, 95% CI | RTX 4090 baseline → candidate process ms | Reduction, 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Solovev, mgrid | 1313.354 → 971.216 | 26.05% [25.31, 26.81] | 876.645 → 1018.879 | -1.78% [-46.07, 32.68] |
| Solovev, embedded MAKEGRID | 1493.383 → 1144.836 | 23.08% [22.37, 23.66] | 1202.857 → 1102.665 | 18.55% [9.36, 29.96] |
| CTH-like | 1552.942 → 1494.203 | 3.16% [2.56, 4.22] | 1497.331 → 1517.021 | -0.51% [-2.30, 0.30] |
| W7-X, positive flux | 7938.959 → 7432.082 | 6.15% [6.03, 6.59] | 7342.169 → 6052.602 | 22.14% [15.53, 24.42] |

The Solovev and W7-X solver-interval gains clear the 5% lower-confidence-bound
acceptance gate on both GPUs. CTH improves on Pascal; its Ada interval is
inconclusive and excludes a solver regression above 2%. Whole-process results
are less stable: no process-wall gain is established for Ada CTH or the mgrid
Solovev case. Variation outside the solver interval obscures those gains, so
the solver percentages are not presented as process speedups.

| GPU / case | Solver p95 baseline → candidate ms | Solver MAD baseline / candidate ms | Process p95 baseline → candidate ms |
| --- | ---: | ---: | ---: |
| TITAN Xp / Solovev, mgrid | 921.837 → 583.045 | 0.223 / 0.474 | 1320.970 → 983.821 |
| TITAN Xp / Solovev, embedded MAKEGRID | 925.514 → 590.843 | 0.775 / 0.798 | 1511.088 → 1173.713 |
| TITAN Xp / CTH-like | 1117.506 → 1074.092 | 2.576 / 0.482 | 1562.198 → 1507.297 |
| TITAN Xp / W7-X, positive flux | 7344.780 → 6847.348 | 4.763 / 3.075 | 7943.711 → 7454.099 |
| RTX 4090 / Solovev, mgrid | 625.015 → 573.396 | 0.273 / 5.262 | 1499.838 → 1276.190 |
| RTX 4090 / Solovev, embedded MAKEGRID | 851.547 → 442.269 | 0.345 / 0.175 | 1670.274 → 1389.977 |
| RTX 4090 / CTH-like | 1007.249 → 994.772 | 19.255 / 20.092 | 1525.642 → 1531.173 |
| RTX 4090 / W7-X, positive flux | 6920.345 → 5331.611 | 0.133 / 4.328 | 7890.522 → 6269.515 |

The relative solver MAD is at most 2.12% across these runs; raw per-run
telemetry and all outliers remain in the results.

Numerical validation compares every native coefficient and published half/full
field byte, together with numerical run reports: stage counts, residual bits,
restart indices, total iterations and normalized input. Each result must also
retain the requested stage schedule, satisfy all three tolerance components,
have finite fields, positive oriented Jacobian and nonnegative magnetic energy,
and match its checkpoint state exactly. The separate full-trajectory comparison
below also checks every recorded
iteration, including the controller scalars and restart reasons.

The `gstore` unit test additionally compares the original serial GPU loop with
the split implementation bit for bit for two scalar types, three poloidal sizes
and three source fields: 18 fixture checks. The singular RHS test checks 60
output arrays against the frozen original loop, covering both scalar types,
partial blocks, the original-block fallback above 256 modes, toroidal loop
tails and full/partial updates. Its asymmetric cases retain the existing zero
cosine RHS behavior and do not qualify new asymmetric physics. Existing CPU
mirrors and trusted vacuum references remain separate numerical checks.

The final timing matrix contains 180 full-process attempts: 144 successful
solves (including warmups) and 36 retained original-W7-X sign rejections. Every
successful baseline/candidate pair has identical state, fields and numerical
reports within its architecture. Both Solovev forms retain `389 → 636`, CTH
retains `198 → 226`, and positive-flux W7-X retains 1733 effective iterations.

All 71 verify CTests pass. After the diagnostic guard and large-system fallback,
the five affected integration/kernel tests pass again. Three selected float
checks pass, including the FP64-instruction audit. CUDA 12.9 on RTX 4090 passes
memcheck/initcheck for the changed vacuum kernels and complete Solovev/CTH
solves; the original and earlier-candidate initialization failures are retained
alongside the clean guarded runs. The refreshed bvec test exercises the fallback.

Float kernel equivalence does not establish free-boundary float convergence.
Additional Solovev and CTH checks with every stage tolerance raised to `1e-6`
failed in both baseline and candidate before leaving the coarse grid (reporting
5002 and 1274 iterations respectively). Neither produced a completed native
result. Those attempts remain in `pascal/float-check/`; no float solve speedup
or new free-boundary float qualification is claimed.

Full `CUMES_DUMP=1` runs pass on both architectures. A temporary `fopen`
interposer redirects only the per-iteration record filename so every stage is
retained instead of overwritten. It changes no solver data or stage controls;
its source, compiled helpers, commands and checksums remain in `trace/` and
`ada/stage-traces/`. Both ordinary and dump-enabled free-boundary paths use
direct launches. The native output of every dump run also matches both ordinary
baseline and candidate outputs, checking that observation changes no result.

For each architecture, all 1,281 dump files per variant match byte for byte,
including all 15 residual/controller columns at every one of seven configured
stages. Both Solovev inputs record 389/636 passes, CTH records 198/226, and W7-X
records 1736 actual passes for 1733 effective iterations. W7-X's three
bad-Jacobian events occur at actual passes 3/7/14 (effective iterations 3/6/12)
in both versions. These full traces, component dumps, native states and unit
references qualify the retained numerical work as Class A on this matrix.

The [compact raw results](../benchmarks/free_boundary/results/20260909.json)
retain per-run timings, numerical fingerprints, stage controls and telemetry
with their protocol provenance. The full artifact root is
`../tmp/cumes-free-boundary-20260909/`. Final Pascal
measurements are in `pascal/qualified-pairs/`; final Ada measurements are in
`ada/qualified-paired/`. The earlier `final-pairs`/`final-paired` directories
predate the diagnostic guard and are retained separately. Earlier graph, direct,
gstore and block-size
experiments remain in separate directories. Their cumulative speedups are not
presented as isolated gains from an individual kernel change.
