# WebGPU port

## Status

The WebGPU backend is an additive browser backend. CUDA remains the default
and the only backend with the optional free-boundary coupling. The WebGPU port
of cuMES's fixed-boundary solver is complete: it implements and hardware-
qualifies the entire iteration DAG for axisymmetric and folded 3-D equilibria,
multigrid control, native binary result publication, and an interactive
axisymmetric boundary editor.

The current WebGPU correctness milestones are implemented and
browser-validated:

- a top-level `CUMES_BACKEND=WEBGPU` path that never enables CUDA or probes
  CUDA-only dependencies;
- Emscripten's `--use-port=emdawnwebgpu` integration for compilation and link;
- asynchronous adapter/device acquisition, error reporting, storage-buffer
  upload, compute dispatch, copy, and mapped readback;
- the six-family radial prolongation contract in WGSL for linear and
  Catmull-Rom interpolation;
- correct poloidal-m parity for folded `(m,n)` modes, odd-m axis
  regularization, exact LCFS transfer, and velocity reset;
- the axisymmetric inverse transform in WGSL, including parity-separated
  geometry and poloidal derivatives, zero toroidal derivatives, and fused
  `rCon`/`zCon` synthesis;
- the axisymmetric reduced-θ forward projection with axis and fixed/free LCFS
  gates, plus the constraint de-alias bandpass;
- direct 3-D inverse and forward transforms over the full reduced-θ/full-ζ
  grid, covering all six folded parity families, physical `nfp` toroidal
  derivatives, VMEC lambda sign conventions, odd-m scaling, fused constraint
  synthesis, and axis/LCFS force gates; browser conformance agrees value for
  value with the independent C++ float references, and the native CUDA Fourier
  suite provides an external convention check;
- a two-pass direct 3-D spectral-condensation bandpass, with separate `sc` and
  `cs` folded families, CUDA-compatible `n=0`/`n>0` normalization, and
  per-surface `tcon*faccon` scaling; the browser result agrees with its C++
  reference to `1.164e-10` on the conformance case;
- host-generated `f32` Fourier tables, matching the CUDA operator contract and
  avoiding adapter-dependent WGSL transcendental approximations; the strict
  W7-X path carries precision-critical values as normalized `(hi, lo)` `f32`
  pairs through state and velocity, inverse/forward transforms, geometry,
  magnetic closure, MHD force, constraint cancellation, residual
  decomposition, and descent. Explicit storage round trips provide the WGSL
  rounding barriers required by error-free transforms;
- a two-dispatch separable 3-D inverse transform: a short compensated toroidal
  synthesis feeds the poloidal synthesis through persistent storage. For the
  W7-X shape this replaces a 156-term mode loop per real-space point with
  13-term toroidal and 12-term poloidal loops, while improving the hardware
  conformance error from `2.861e-06` to `1.431e-06`. The scalar and paired-f32
  variants keep their large accumulator sets as named locals: Chrome/Dawn's
  D3D12/DXC path returned zero from the otherwise valid dynamically indexed
  function-local arrays, while the named form passes identically on D3D12 and
  Vulkan;
- matching separable forward projection and constraint bandpass pipelines;
  the latter factors both analysis and synthesis into toroidal/poloidal passes,
  reducing the W7-X transform work from about 28 million accumulated terms to
  about 5.2 million without changing its spectral contract;
- the existing JSON mapper, validator, boundary folder, and resolution logic
  compiled into Wasm, with production-shaped axisymmetric cold-start and
  radial-profile initialization;
- generic folded-mode cold-start initialization matching the CUDA boundary and
  magnetic-axis interpolation; the shipped prescribed-current W7-X input now
  parses and initializes its first `ns=33`, `mnmax=156`, 1080-point angular
  stage in-browser with the production `0.12` seed envelope;
- the prescribed-current surface solve: reduced-grid current integrals produce
  `chipH`/`iotaH`, reconstruct `B^theta`, covariant field, and total pressure,
  and update full-grid current/iota profiles using the CUDA extrapolation
  rules;
- the shipped Solovev input parsed in-browser, relaxed to the documented f32
  tolerance floor, initialized, and passed through the WGSL inverse transform;
- the staggered half-grid base-geometry operator in WGSL: parity recombination,
  radial interpolation and derivatives, Jacobian, and covariant metric, now
  over the complete reduced-θ/full-ζ grid with nonzero `g_uv`/`g_vv` 3-D
  coverage;
- fixed-iota contravariant/covariant magnetic fields and total pressure, with
  the CUDA path's finite-Jacobian division guard and full 3-D lambda/toroidal
  geometry contributions;
- radial, poloidal, and toroidal MHD weak-form forces on the full 3-D grid,
  including the `g_uv B^theta B^zeta` couplings, `crmn`/`czmn`, hybrid poloidal
  lambda force, and toroidal `clmn`; the resulting 16 fields feed the direct
  3-D forward projection without a host-side physics approximation;
- first-pass Solovev force projection into the six-family spectral residual
  slab (the initial constraint multiplier is zero, matching CUDA/vmecpp);
- odd-m residual decomposition, the m=1 force gauge, fixed-boundary LCFS norm
  exclusion, and CUDA-compatible `f32` products accumulated into `double` host
  residual sums;
- the later-pass axisymmetric spectral-condensation constraint: LCFS-volume
  reference reset, current-preconditioner `tcon` refresh, `gConEff`, direct
  poloidal bandpass, MHD-force injection, `frcon`/`fzcon`, and constrained
  residual projection;
- the same complete constraint chain on the 3-D angular grid: per-zeta LCFS
  reference reset, reduced-theta/full-zeta `tcon` averages, direct bandpass,
  force injection, and 20-field `frcon`/`fzcon` output ready for toroidal
  projection;
- fixed-boundary radial preconditioner element assembly from half-grid force
  Hessian surface integrals, including parity factors, full-grid diagonal
  averaging, and the LCFS pedestal; these live Solovev `ard`/`azd` values feed
  the constraint multiplier test;
- mode-major R/Z tridiagonal matrix assembly, m=1 correction, magnetic-axis
  `jMin`, per-mode pivot scale, and the axisymmetric lambda diagonal assembled
  from reduced-poloidal surface averages;
- in-place preconditioner application using m=1 force scaling, scale-aware
  pivot guards, CUDA-equivalent paired R/Z parallel cyclic reduction (PCR),
  axis boundary zeroing, and lambda diagonal scaling; PCR scratch lives in a
  persistent storage buffer so the implementation supports the validated
  `ns <= 512` range without exceeding portable workgroup-memory limits;
- the complete 3-D preconditioner generalization: full angular element
  averages, `(m,n)` radial systems with physical `n*nfp` stiffness, mixed
  `mn*g_uv` lambda coupling, per-mode pivot scales, m=1 force scaling for every
  toroidal family, and guarded paired solves across all `mnmax` modes;
- Garabedian accelerated descent for all six spectral families, including
  physical/decomposed basis conversion, the m=1 undone-gauge update, rigid R/Z
  LCFS behavior, free lambda LCFS behavior, and velocity persistence;
- folded `(m,n)` residual decomposition, force normalization, and descent with
  independent poloidal/toroidal basis factors and the m=1 gauge applied to all
  toroidal families;
- the production pre-inverse m=1 axis extrapolation, host force-normalization
  reductions, invariant/preconditioned residual scaling, and the shared
  `IterationController<double>` damping/restart decision; a one-pass native
  CUDA cross-check and the browser path both report first-pass Solovev
  `FSQR=2.565e-02`;
- a repeating two-pass axisymmetric stage slice with persistent velocity,
  constraint reference/`tcon`, cached preconditioner, rollback checkpoint, and
  controller state; WebGPU and native CUDA mixed-float both report the
  effective-iteration-3 residual triple `(1.820e-03, 2.723e-04, 4.029e-04)`;
- a controller-complete coarse-grid loop with top-of-pass maintenance restore,
  oriented-Jacobian recovery, nonfinite recovery, post-descent checkpoint
  refresh/restart, iteration-limit failure, and convergence termination;
  WebGPU and native CUDA mixed-float both converge the `ns=5` Solovev stage at
  effective iteration 72, with terminal residual triples
  `(9.947e-07, 4.857e-07, 3.318e-07)` and
  `(9.912e-07, 4.861e-07, 3.320e-07)` respectively;
- all three fixed-boundary Solovev multigrid stages, with the production float
  linear/scalxc transfer dispatched through WebGPU and every transition checked
  against the independent C++ reference; on the TITAN Xp Vulkan adapter the
  PCR path converges in `72 -> 31 -> 247` effective iterations (350 total) with
  final residual `(8.158e-07, 2.653e-07, 2.915e-10)`; SwiftShader independently
  converges in `72 -> 32 -> 155` (259 total) with
  `(8.666e-07, 3.637e-07, 3.181e-10)`;
- the complete controller-driven 3-D W7-X stage path: inverse transform,
  geometry, prescribed-current closure, MHD force, constraint bandpass,
  toroidal projection, residual decomposition, `(m,n)` preconditioner, and
  descent. The browser and native CUDA mixed-float paths agree at effective
  iteration 3 on `(FSQR,FSQZ,FSQL) = (1.141e+01, 7.079e+00, 1.012e-01)`;
- a selectable `?solve=w7x` browser mode that defaults to the final `ns=99`
  radial grid and publishes the same schema-v8 result form. At the input's
  unmodified `1e-12` tolerance, Chrome 152 on Windows 10 using Dawn's
  D3D12/DXC backend on an RTX 3060 Ti converges in 2812 effective iterations,
  with final residual `(9.985e-13, 2.130e-13, 1.955e-13)` and an
  11,809,091-byte result in 627.2 seconds. `&grids=3` retains the complete
  multigrid integration route: the same adapter converges in
  `1416 -> 1621 -> 1670` (4707 total), with residual
  `(9.996e-13, 2.160e-13, 1.339e-13)` and an 11,809,203-byte result in 1024.2
  seconds. The paired-`f32` TITAN Xp Vulkan route independently converges all
  three grids in `1421 -> 3220 -> 2964` (7605 total), with residual
  `(1.000e-12, 2.115e-13, 1.528e-13)` in 4929.3 seconds;
- cached immutable toroidal basis buffers plus persistent, grow-only operator
  scratch/readback buffers and compute pipelines; this removes hot-loop shader
  recompilation/allocation and keeps the complete high-resolution W7-X run
  stable on Chromium's software adapter;
- production solves skip the independent CPU oracle and per-operator diagnostic
  logging while retaining finite-value, Jacobian, fixed-boundary, and
  preconditioner-breakdown checks. On the TITAN Xp Vulkan adapter, the default
  73-iteration interactive solve completes in 2.33 seconds; a stronger
  `RBC(0,2)=-0.15` boundary converges in 123 iterations and 3.77 seconds at the
  same `1e-5` tolerance. The `?mode=test` route continues to run every GPU/CPU
  comparison;
- schema-v8 native binary publication into Emscripten MEMFS, an in-Wasm
  writer/reader round-trip check, and a browser Blob download link; the
  downloaded `CUMES001` file is accepted by the native reader as `ns=55`,
  `mnmax=6`, and its interior spectral state is within `1.818e-05` relative of
  the native CUDA mixed-float state (the maximum is in lambda);
- complete version-8 scientific fields: final WebGPU geometry and magnetic
  buffers feed the shared host curl/current-density derivation, producing all
  seven half-grid and six full-grid arrays; the 118,736-byte downloaded file
  passes the in-Wasm field round trip and the project plotting workflow;
- an independent C++ float reference evaluated by the browser self-test.
- a responsive browser application with distinct Fourier and contour modes.
  Fourier mode directly edits the stellarator-symmetric `n=0` R-cosine/Z-sine
  harmonics. Contour mode freely moves 16 periodic cubic control points,
  mirrors their partners to retain the solver symmetry, and continuously
  projects 512 contour samples into the same basis capped at `m=5`; it draws
  both the free contour and truncated fit plus their RMS error. The resulting
  input passes through the production validator and three-grid solver, which
  Fourier-synthesizes smooth converged surfaces and exposes the schema-v8
  download without a server-side compute service;
- a narrow C-linkage browser bridge, implemented in
  `webgpu/browser_bridge.js`, keeping DOM, URL, local-storage, and Blob policy
  out of the C++ translation units.

The default self-test parses both embedded inputs, runs a controller-complete
two-pass W7-X slice, then converges all three Solovev stages. The separate W7-X
solve entry point is convergence-qualified end to end at `1e-12` on both an
NVIDIA TITAN Xp through Chrome/Dawn's Vulkan backend and an RTX 3060 Ti through
Chrome/Dawn's Windows D3D12/DXC backend. The page requests the high-performance
adapter and publishes its device/type/backend metadata for automation. On the
Linux qualification host, Chrome's privacy-reduced adapter name is the PCI
device id `0x1b02`; `chrome://gpu` and Vulkan enumerate that id as the TITAN
Xp, and `nvidia-smi` observes the browser GPU process.

## Build and run

The emdawnwebgpu remote port downloads a Dawn package into Emscripten's cache
on first use. Keep both cache and build artifacts outside `/tmp` on this host:

```bash
source "/lustre/qzhong/emsdk/emsdk_env.sh"
export EM_CACHE="$PWD/../tmp/cumes-emscripten-cache"

emcmake cmake --preset webgpu
cmake --build --preset webgpu -j
ctest --preset webgpu
```

The parent workspace is already exposed by nginx with the cross-origin headers
needed by the Wasm application. Open the generated files through that server;
browsers do not reliably initialize WebGPU from `file://` URLs:

```text
http://localhost:6969/magnetic-equilibrium-solver/tmp/cumes-build-webgpu/webgpu/cumes_webgpu.html
```

The generated HTML gives the JavaScript and Wasm files a shared content hash
in their query strings. This prevents a normal browser cache from combining a
runtime from one build with the embedded shaders and C++ module from another.
After upgrading from a build that predates this scheme, use one hard refresh
or add any one-time query parameter to the HTML URL; subsequent rebuilds are
cache-coherent automatically.

In **Fourier** mode the editor exposes `RBC(0,m)` for `m=0..5` and `ZBS(0,m)`
for `m=1..5` as sliders beside a live boundary preview. In **Contour** mode,
16 points define a periodic Catmull-Rom contour; dragging one point mirrors its
partner and a 512-point discrete Fourier transform updates those same
coefficients through `m=5`. The orange target and cyan truncated reconstruction
make the approximation explicit. Select **Run equilibrium** to solve the
fitted boundary. The generated input, editing mode, contour, and coefficients
stay in browser local storage; compute and output generation remain local to
the page. The interactive profile uses stellarator-symmetric axisymmetric
harmonics (`ntor=0`), three grids (`ns=5,11,55`), and a responsive mixed-float
tolerance of `1e-5`.

After convergence the result panel defaults to an interactive 3-D equilibrium
view, with a **2D cut** toggle for the poloidal cross-section. The solver sends
the selected nested surfaces as all six physical Fourier parity families plus
`mpol`, `ntor`, and `nfp`; JavaScript reconstructs the full torus locally. Drag
the canvas to orbit and use the wheel to zoom. The W7-X route presents the same
3-D viewer below its solver log.

Append `?mode=test` for the full GPU/CPU operator conformance suite and
stricter Solovev convergence gate. A successful run finishes with:

```text
cuMES WebGPU self-test: PASS
```

Append `?solve=w7x` to run the fixed-boundary W7-X example directly on its
final `ns=99` radial grid instead of the conformance suite. This avoids the two
coarser browser stages while retaining the input's `1e-12` tolerance. Append
`&grids=3` to retain the complete `33 -> 66 -> 99` integration route. Either
path can be slow on software WebGPU adapters because the 3-D transform uses
separable direct DFT stages.

The page also publishes `data-cumes-webgpu="pass|fail"`, a diagnostic
`data-cumes-detail`, and `data-cumes-adapter`, `data-cumes-adapter-type`, and
`data-cumes-adapter-backend` on `<body>` so browser automation can inspect both
the result and the selected device.
Long W7-X runs additionally publish the current stage, iteration, FSQR, and
last-progress timestamp as `data-cumes-stage`, `data-cumes-iteration`,
`data-cumes-fsqr`, and `data-cumes-last-progress-at`.

## Precision policy

Core WGSL exposes `f32` but not `f64`. The WebGPU preset therefore keeps
`CUMES_USE_FLOAT=ON`, while the selected W7-X solver uses a double-single
representation for its precision-critical path. Each logical value is the
unevaluated sum of two binary32 words. Error-free sum/product kernels retain
the low word, and workgroup-local atomic round trips prevent WGSL compilers
from reassociating away the required rounding points. Every invocation owns a
workgroup slot, so this keeps the strict rounding boundary without paying a
device-memory transaction for each error-free-transform operation.

CUDA Class A byte identity is not a WebGPU acceptance criterion. Ordinary
operator cases retain their float CPU-reference tolerances; dedicated W7-X
cases compare reconstructed `hi + lo` values against double references. The
host reconstructs paired spectral residuals in `double` before accumulating
the invariant norms. This supports the input's original `1e-12` tolerance,
where scalar-f32 W7-X previously stalled near its float floor. Interactive
axisymmetric solves intentionally remain scalar-f32 at their responsive
`1e-5` tolerance.

## Browser performance

Profiling the `ns=99` W7-X hot loop on Chrome/Dawn D3D12 and an RTX 3060 Ti
identified global atomic rounding barriers and the prescribed-current field
finalization as the dominant shader costs. The field finalization now reduces
current once per radial surface and finalizes its 1080 angular points in
parallel. Strict double-single rounding slots now use workgroup memory in the
inverse, geometry, magnetic, force, constraint, projection, decomposition, and
descent shaders; the corresponding per-dispatch storage buffers were removed.

In the same live `mapAsync` timing probe, throughput increased from about 3.78
to 4.47 W7-X iterations/s (roughly 18%). The magnetic-field stage fell from
54.6 ms to 18.3 ms and the toroidal inverse from 18.7 ms to 12.1 ms. Timings
include each operator's queued copies, dispatch, and result mapping, so they
also expose the next architectural bottleneck: intermediate fields still cross
the Wasm/host boundary between adjacent operators.

### Mixed-radix FFT and resident field edges

The strict 3-D forward projection can use the standalone
[webgpu-fft](https://github.com/12ff54e/webgpu-fft) submodule: batched complex
mixed-radix FFTs in zeta, followed by the existing direct poloidal projection.
W7-X keeps its exact 36-point toroidal grid (radices 2, 2, 3, 3); there is no
zero padding or change to angular quadrature. Packing preserves both words,
and unpacking converts the FFT's negative imaginary part to the solver's
positive sine projection. Scalar-f32 and unsupported shapes retain the DFT.
Inverse synthesis and constraint filtering currently retain their separable
direct transforms. FFT summation changes the iterative trajectory, so this
path is qualified by residual convergence, not bit-identical iteration counts.

The standalone dependency's v0.2 API additionally supports packed R2C/C2R
and every integer length from 2 to 1,048,576 (subject to GPU buffer limits),
using multipass radix-2 or Bluestein for larger lengths. Both C++ and JS/TS
interfaces keep execution device-resident. cuMES currently uses its compact
N≤256 C2C shader path; the new general-purpose plans are not automatically
substituted into the solver's existing larger-grid DFT fallback.

`DeviceFields` retains an owning WebGPU buffer handle plus high/low plane
offsets. Geometry, half-grid metrics, magnetic fields, forces, and constrained
forces can feed downstream kernels through device copies instead of Wasm
uploads. The two large force outputs are no longer mapped in production 3-D
solves: at ns=99 this removes 30,792,960 readback bytes per iteration. Device
copies accommodate low-word offsets that do not meet storage-binding alignment.
Producer buffers remain live until consumers are submitted on the same queue.

This is partial residency: geometry/Jacobian validation, host norm calculation,
constraint reference maintenance, and spectral updates still use readbacks.
Production 3-D iterations now collect these reads in one shared mapping,
including the preceding descent update (see the batching measurements below).
The explicit switches are `&resident=0` to restore host transfers and `&fft=1`
to select FFT (`&fft=0` selects direct projection). Residency is enabled by
default; the W7-X example defaults to direct projection based on the complete
solve comparison below. Conformance paths retain full host arrays for comparison.

For a running Chrome session exposed through the user's DevTools tunnel:

```sh
node scripts/webgpu_cdp.mjs eval-file scripts/webgpu_profile.js
node scripts/webgpu_cdp.mjs eval 'cumesGpuProfile.report()'
# Four-way warmed-up comparison in a separate temporary Chrome tab:
node scripts/webgpu_benchmark.mjs 'http://localhost:6969/magnetic-equilibrium-solver/tmp/cumes-build-webgpu-ds/webgpu/cumes_webgpu.html'
```

The probe reports upload, device-copy, and readback byte counts by buffer label,
plus CPU upload-call and `mapAsync` wait duration. Mapping waits include queued
GPU work and synchronization; they must not be described as isolated kernel
or transfer timings. Use full iteration throughput for end-to-end comparisons.

The first combined full `ns=99` Chrome/D3D12 RTX 3060 Ti run converged in
3450 effective iterations with residual `(9.980e-13, 2.009e-13, 1.633e-13)`
and published the 11,809,091-byte output. Navigation-to-result wall time was
357.5 seconds versus the previous 627.2-second direct-transform/host-transfer
run (43% less time, despite a different iteration count). During a 563-pass
steady-state sample, throughput was 9.68 iterations/s. This is one adapter,
not a cross-platform performance guarantee.

A controlled four-way probe of 100 warmed-up passes (starting after pass 100)
on the same adapter measured:

| Host/device policy | Zeta projection | Iterations/s |
| --- | --- | ---: |
| Reference host transfers | Direct DFT | 4.28 |
| Reference host transfers | Mixed-radix FFT | 4.28 |
| Resident field edges | Direct DFT | 9.95 |
| Resident field edges | Mixed-radix FFT | 9.98 |

These short samples do not establish a speed difference between FFT and DFT.
The measured improvement is predominantly residency: uploads dropped from
about 159.7 MB/pass to 14.1 MB/pass, and readbacks from about 68.3 MB/pass to
37.6 MB/pass (decimal MB; sample boundaries can include a partial operator).
The independent library benchmark also finds that its strict paired-f32 FFT
can be slower than its workgroup-local DFT at N=36; see the dependency's
`docs/qualification.md`. FFT availability is not itself a performance claim.

The resident direct-projection full run converged in the original 2812
iterations with the same reported residual triple
`(9.985e-13, 2.130e-13, 1.955e-13)` and output size. Wall time was 296.7
seconds: 53% less than the 627.2-second baseline, and 17% less than the FFT
run. Consequently the example defaults to resident direct projection, while
`?solve=w7x&fft=1` explicitly exercises the qualified FFT path. Neither the
small-transform benchmark nor this one-case comparison justifies a universal
FFT/DFT crossover threshold.

The rebuilt conformance page passed all operator checks and the 327-iteration
Solovev multigrid test after integration. Keep browser tests in the foreground:
Chrome background-tab timer throttling can dominate small-grid callback chains.

### FFT trajectory diagnostics

`&trace=1` records per-pass residuals, search-direction and state fingerprints,
damping, restart anchors, and refresh decisions in `window.cumesDiagnostics`.
`&compare_fft=1` additionally runs shadow forward transforms on **identical
resident input fields**, on iterations 1–3 and multiples of 100. Shadow outputs
never enter the solver. The variants are 0: generic FFT, 1: legacy direct DFT,
2: direct DFT with canonically split double-precision zeta roots. These extra
dispatches are for numerical diagnosis, not benchmarking.

`&fft_kernel=generic` retains the pre-optimization FFT arithmetic as a reference.
`&basis=canonical` selects the corrected zeta table for direct forward
projection only; the poloidal table, inverse synthesis, grid, tolerance, and
controller are unchanged. This is an explicit diagnostic option, not a claim
that a more accurate table must converge faster.

Capture and compare traces with:

```sh
CUMES_CDP_TARGET=YOUR_TEST_TAB node scripts/webgpu_cdp.mjs eval 'window.cumesDiagnostics' > ../tmp/direct-trace.json
# Repeat for the FFT run, saving ../tmp/fft-trace.json.
node scripts/webgpu_compare_traces.mjs ../tmp/direct-trace.json ../tmp/fft-trace.json
```

The Chrome/D3D12 RTX 3060 Ti investigation established:

- Both paths start from identical paired state. On the first force projection,
  generic FFT versus direct DFT differs by `1.21e-14` relative L2
  (`1.82e-12` maximum absolute coefficient difference). The first constrained
  projection differs by `2.50e-14` relative L2. This is small numerical error,
  not a transform sign, grid, or normalization mismatch.
- The legacy table uses a float-angle sine/cosine high word and a float
  correction toward double trigonometry. A host reproduction on the W7-X
  zeta grid gives up to `1.74e-13` reconstruction error; directly splitting
  the double root gives `8.88e-16`. Correcting only this table reduces the
  first force discrepancy to `7.84e-15` relative L2 and the constrained one
  to `2.11e-14`, but does **not** remove the discrepancy. Different summation
  and cancellation order therefore remains important.
- The first preconditioned search-direction fingerprint differs on iteration
  1; both state-word fingerprints differ by iteration 2. The preconditioner
  operates on scalar-f32 forces, so tiny cancellation-sensitive coefficients
  can already have different high words. Paired state evolution and paired
  invariant norms do not make this search direction bit-identical.
- Double damping differs on iteration 2; after the actual f32 conversion
  used by descent, `fac` first differs on iteration 35 and `b1` on iteration
  40. The two runs still have identical time steps, restart anchors, restart
  reasons, and preconditioner-refresh schedules through the direct run's
  2812 effective iterations. The derived m=1 gauge schedule also matches.
  A discrete restart, step-size, or gauge branch mismatch is not the cause.

These observations explain how roundoff enters and propagates into different
nonlinear trajectories. They do not assign every additional iteration to one
coefficient, nor imply that FFT is intrinsically less accurate. Accuracy is
qualified against references and convergence gates; runtime is measured over
the complete solve. Hashes locate divergence but are not error bounds.

The traced direct control repeated the established 2812-iteration solution
with `FSQR=9.985159710508547e-13` (291.9 seconds navigation to result).
Through iteration 750 the two FSQR histories remain close: the FFT/direct
ratio is 1.00033 there. The large difference in final iteration count is not
an immediate large force error at startup.
The full generic replay reproduced 3450 iterations and the original residual
triple. It also matched every available controller field and fingerprint in
the earlier 781-iteration shadow-instrumented capture: the extra diagnostic
dispatches did not perturb that trajectory.

All three complete runs have their last restart anchor at iteration 93 and
retain `delta=0.5246427169237352` afterwards. Over iterations 2000–2800,
mean `dtau=1-b1` is `0.002130` direct, `0.002588` generic FFT, and `0.002113`
optimized FFT. The generic run thus has about 22% more damping in this
window. At iteration 2800, FSQR is respectively `1.021e-12`, `1.480e-12`,
and `1.450e-12`. The evolved state/search direction also matters; damping
alone is not a complete causal attribution. Generic and optimized checkpoint
refresh choices first differ at attempt 520, but there are no later restores
to consume those checkpoints.

### Specialized paired FFT performance

The paired small-transform implementation now uses radix-2 sum/difference
butterflies, a radix-3 sum/difference and real-scale formula, exact trivial
twiddles, and direct natural-order stores from the final stage. The final
shared-memory write/read and two barriers are removed. Explicit rounding
barriers inside paired arithmetic are retained. Scalar-f32 keeps the original
generator: the specialization regressed at the full solver batch size, so it
is not enabled for that precision.

On the same foreground Chrome / RTX 3060 Ti session, N=36 and the actual
`20*99*16 = 31,680` solver batch size, seven samples of 20 repeated dispatches
gave the following median queue-completion times (setup, upload and readback
excluded; command encoding and submission included):

| Standalone paired transform | ms/pass |
| --- | ---: |
| Generic FFT | 3.736 |
| Specialized FFT | 1.851 |
| Standalone full-complex direct DFT | 10.152 |

The FFT improvement is **2.02×**. The standalone DFT computes all 36 complex
bins; it is not the solver's truncated 13-bin real projection and must not be
used to claim a 5.5× solver speedup.

The optimized cuMES operator conformance and 327-iteration Solovev regression
passed with same-input shadow diagnostics enabled. The dependency separately
passed 496 small-transform GPU/CPU comparisons plus 248 same-input
generic/optimized comparisons, 197 real/large-transform cases, and 12 actual
Emdawnwebgpu C++ runtime cases. See its `docs/qualification.md` for accuracy
and timing details.

A warmed 287-pass optimized W7-X profile measured 100.3 ms/iteration. The two
forward-projection readback waits totalled about 18.1 ms/iteration and remained
the largest aggregate wait category. These waits include queued pack, FFT,
poloidal projection, and synchronization work, not isolated kernel times.
Doubling FFT throughput cannot double the whole solver's throughput.
A separate warmed generic sample measured 100.2 ms/iteration over 723 passes,
with 10.38 ms per forward-projection wait versus 9.07 ms optimized. Thus the
forward waits improved, but these samples **do not establish an end-to-end
per-iteration speedup**: total pass throughput was essentially unchanged.

The optimized full single-grid W7-X run converged in **3091** effective
iterations, with residual `(9.987756002533206e-13, 2.0790797273186487e-13,
1.725519314229383e-13)`, in **311.8 seconds** navigation to result. The
unchanged direct-projection control took 291.9 seconds / 2812 iterations in
this session, so direct projection remains the default. `?solve=w7x&fft=1`
selects the optimized FFT; append `&fft_kernel=generic` for the reference.
The optimization changes arithmetic ordering as well as execution cost;
changes in full-solve wall time cannot be attributed solely to kernel speed.

The complete same-session comparison (foreground, resident fields, unchanged
single-grid input and `1e-12` tolerance) is:

| Forward transform | Effective iterations | Wall seconds | Final FSQR |
| --- | ---: | ---: | ---: |
| Direct projection | 2812 | 291.9 | 9.985e-13 |
| Generic FFT replay | 3450 | 348.3 | 9.980e-13 |
| Optimized FFT | 3091 | 311.8 | 9.988e-13 |

The optimized FFT run takes 10.5% less wall time than the generic replay,
predominantly from fewer iterations; it remains 6.8% slower than direct
projection overall. These are one-case, one-adapter measurements, not a
universal FFT/DFT crossover rule. Traces add optional host fingerprinting;
the short optimized and generic profiles additionally include measurement
hooks. No shadow dispatches were enabled in these three complete timings.

## Backend boundary

### Earlier queue-gap reduction without changing arithmetic

The first resident production 3-D optimization passed both forward-projection outputs
directly to residual decomposition on the device. The original high words
are copied alongside the decomposed output into one combined readback, so the
original finite/nonzero guards are still checked before the controller or
state update. This removes two readback/upload fences per iteration without
changing WGSL arithmetic or the host double-norm accumulation order.
`&fences=1` restores the original spectral readbacks for comparison;
`&compare_fft=1` also retains them for the same-input transform diagnostics.

Resident field copies in geometry, magnetic field, force, constraint, and
forward projection are encoded with their consuming compute pass instead of
being submitted individually. This eliminates 22 copy-only submissions per
ordinary W7-X pass. Copies still handle non-storage-aligned high/low offsets.
This is not full residency: Jacobian checks, profiles, norms, constraint
references, and state updates still require host interaction.

The full direct-projection `ns=99` run on the same foreground Chrome /
RTX 3060 Ti completed in **259.9 seconds**, versus **291.9 seconds** before
this change (11.0% less wall time). It retained 2812 effective iterations,
`FSQR=9.985159710508547e-13`, and the same final residual triple. All 2817
recorded controller records and state fingerprints matched the original
trace; this improvement does not rely on a different convergence trajectory.

A warmed 1178-pass profile measured 93.1 ms/iteration, 13.09 explicit queue
submissions, and 9.09 map completions per iteration (fractional counts include
periodic preconditioner refresh and sample boundaries). Uploads were about
12.66 MB/pass. Relative to the previous spectral handoffs, exactly 1,482,624
upload bytes and 741,312 readback bytes per ordinary pass are eliminated.
The browser profiler now counts explicit `GPUQueue.submit` calls as well.

Full conformance and the 327-iteration Solovev regression passed. New resident
residual tests compare both words and host norms against the host-fed GPU path
for f32 and paired inputs, exercise unaligned source offsets, and verify zero
and NaN original-input guards. The user's utilization reading motivated this
work; no new Windows GPU-utilization percentage was measured from DevTools.
The remaining readback boundaries motivated the next change below; these
measurements do not imply a speedup proportional to the reported idle percentage.

### One readback mapping per production 3-D iteration

`ReadbackBatch` collects all force-evaluation readbacks in a reusable mapped
buffer. Each producer copies its output into a disjoint, eight-byte-aligned
slice before another pass can overwrite its scratch. Geometry, magnetic
fields, preconditioner elements/matrices, constraint head/filter/tail, and
decomposed residuals now feed dependent kernels through device handles.
Periodic preconditioner refreshes add slices, not mappings. Unused individual
readback buffers are not allocated on this path.

The descent update depends on the host damping decision. Its results therefore
join the **next** iteration's batch, while the inverse transform immediately
consumes its device state. Axis extrapolation is implemented by selecting the
same neighboring coefficient bits on device and updating the host mirror at
the fence; no floating-point operation is introduced by this handoff.

```text
descent(k-1) -> inverse / force evaluation(k) -> one map
                                              -> checks / controller(k)
                                              -> enqueue descent(k)
```

Force evaluation is speculative with respect to the Jacobian gate. The host
still processes the collected results through the original checks, double
reductions, checkpoint rules, and controller in their original order. Invalid
Jacobian attempts discard the remaining evaluation results. Accepted host
preconditioner caches prevent a speculative refresh from replacing a valid
cache on rollback. Pending descent state is validated and committed before
checking the next Jacobian. A terminal iteration-limit stop drains pending
descent reads; a normally converged run needs no extra descent mapping.

The foreground Chrome / RTX 3060 Ti direct W7-X `ns=99` run completed in
**141.1 seconds**, versus **259.9 seconds** before iteration batching
(45.7% less elapsed time, about 1.84x faster). All 2817 recorded controller
entries matched the previous run, including both state-word fingerprints,
restart/checkpoint decisions, and normalized residuals. It retained 2812
effective iterations and the final residual triple
`(9.985159710508547e-13, 2.1300946362442957e-13, 1.9552385266042975e-13)`.
The complete schema-v8 output remained 11,809,091 bytes.

A late-run profile captured 731 mappings, **all** labelled
`cuMES iteration readback batch`: one per evaluated pass, with no separate
descent mapping. The corresponding 730 controller intervals averaged
49.56 ms. Submissions remained about 13.08/pass; uploads were about
10.36 MB/pass and batched readback copies 36.90 MB/pass. This change reduces
sequential host waits, not the readback payload itself. Map-wait measurements
include queued GPU work and are not isolated copy or kernel timings.
These are observations from the user's PC, not a multi-device confidence
interval or a new GPU-utilization percentage.

FFT mode also completed with the one-map pipeline: **173.1 seconds**, retaining
its previously qualified 3091 iterations and final residual triple
`(9.987756002533206e-13, 2.0790797273186487e-13, 1.725519314229383e-13)`.
All 3096 recorded controller entries matched the saved optimized-FFT trace.
The earlier fully timed FFT run was 311.8 seconds, before both this batching
change and the preceding spectral-handoff optimization. Direct projection
remains the default; batching does not change either transform's trajectory.

Conformance covers f32/paired residual batches, original zero/NaN guards,
non-storage-aligned source offsets, snapshot survival across producer reuse,
mapped-buffer reuse, device-state axis extrapolation in both precisions, and
empty/overflow batch errors. Error reporting uses
callbacks and does not require Wasm C++ exception catching. The complete
conformance suite and 327-iteration Solovev regression passed.

`&fences=1`, `&resident=0`, and `&compare_fft=1` retain sequential diagnostic
paths. Axisymmetric solves and operator conformance also retain their existing
readbacks; this optimization targets resident production 3-D solves.

### Per-pass GPU timestamps and CPU sampling

For deeper profiling, `scripts/webgpu_deep_profile.mjs` creates its own
foreground Chrome target and injects `scripts/webgpu_timestamps.js` before
the application requests its device. It requests the optional `timestamp-query`
feature and records beginning/end timestamps for each compute pass. A pass
containing multiple pipelines (currently constraint de-aliasing) is reported
as one combined interval, not as individually timed dispatches.

```sh
node scripts/test_webgpu_timestamps.mjs
node scripts/webgpu_deep_profile.mjs \
  'http://localhost:6969/magnetic-equilibrium-solver/tmp/cumes-build-webgpu-ds/webgpu/cumes_webgpu.html' \
  ../tmp/w7x-deep-direct
# Add ?fft=1 to the URL and choose a different output prefix for FFT.
node scripts/webgpu_profile_summary.mjs ../tmp/w7x-deep-direct
```

Use the production resident single-grid path, not `resident=0`, `fences=1`,
axisymmetric mode, or conformance. The probe assumes cuMES's one-device,
single-flight, submitted-before-map ordering. It warms 250 controller records,
times 256 iteration batches, restores API hooks, and then collects a separate
10-second CPU sample at 1 ms sampling intervals. It waits for convergence,
saves the full controller trace, and preserves its result tab. Failed diagnostic
solves are navigated to a blank page so they do not continue consuming the GPU.
Keep the test tab visible and do not run concurrent GPU work.

Timestamp results are copied immediately after the used payload in the same
iteration readback buffer; its allocation reserves 32 KiB for profiling.
There is **no additional map or host fence**. Profiling adds one query-resolve
submission per sampled iteration. It also adds timestamp writes, API wrappers,
and decoding work, so captured elapsed times are not uninstrumented benchmarks.
Buffers and shaders in the normal served build are unchanged. The final harness
maps only the used payload plus timestamps, not the batch's unused capacity.

Enable Chrome's `chrome://flags/#enable-webgpu-developer-features` in the test
profile for unquantized timestamps. Ordinary timestamps are rounded to 100 us;
the developer flag removes that rounding, not hardware timestamp granularity.
Restore the flag afterward because finer timing has privacy implications.
See [Chrome's developer-feature documentation](https://developer.chrome.com/docs/web-platform/webgpu/developer-features).

The capture writes `-gpu.json`, `-cpu.json`, `-windows.json`, `-result.json`, and
`-trace.json`. The summarizer writes `-summary.json`. Supplying an Emscripten
`--emit-symbol-map` file as its second argument additionally writes a
`-cpu-symbolized.cpuprofile` that Chrome DevTools can import. Generate the map
with the same link inputs/options and verify that its Wasm binary is
**byte-identical** to the captured build before assigning function names.

GPU pass sums measure compute execution only. First-to-last GPU timestamp spans
also include intervening commands and scheduling gaps; between-batch gaps
include host work and transfers outside those timestamps. Neither spans nor
renderer idle samples measure hardware occupancy. Map waits include queued GPU
work and overlap host submission/execution; **do not add them to pass sums or
CPU sample percentages**. CPU percentages use the entire sampled interval,
including renderer idle, and concern the renderer thread, not Chrome's GPU
process or driver threads.

### Measured GPU/CPU profile (2026-09-06)

The user's foreground Windows Chrome 152 / RTX 3060 Ti / Dawn D3D12 device
reported driver `32.0.16.1047`. The solver was the unchanged `0c19912` build;
profiling code is separate JavaScript, not part of that executable. An
`--emit-symbol-map` relink produced byte-identical Wasm (SHA-256
`49d7d3bb3a545aa00aa825c703d156db019714522094a669c11401486a86b81d`),
allowing the sampled Wasm function indices to be assigned verified C++ names.

The finalized direct-path capture contains 256 visible iteration batches:

| GPU compute work | Mean ms/iteration |
| --- | ---: |
| Toroidal forward projections, both force evaluations | 3.705 |
| Prescribed-current surface reduction | 2.161 |
| Poloidal inverse synthesis | 1.586 |
| Poloidal forward projections, both evaluations | 1.195 |
| Preconditioner apply | 0.832 |
| MHD force | 0.639 |
| Base geometry | 0.520 |
| Toroidal inverse synthesis | 0.419 |
| Other passes, including amortized preconditioner refresh | 0.622 |
| **All compute passes** | **11.679** |

The mean completion-to-completion iteration interval was **50.459 ms**
(median 46.365, p95 71.435). First-to-last compute timestamps spanned
19.059 ms; the gap from the previous batch's final compute timestamp to the
next batch's first was 31.407 ms. Thus compute-pass execution accounts for
about 23% of the iteration interval, **not a measurement of SM occupancy or
Windows Task Manager's utilization counter**. Copies, uploads, host work,
and scheduling occupy or overlap the remaining time. Mean map wait was
10.979 ms and must not be added to those intervals.

The separate 10-second renderer CPU sample attributed:

| Sampled host activity | Share of sampled interval, including idle |
| --- | ---: |
| Float-vector assignment/insertion, self time | 29.49% |
| `GPUQueue.writeBuffer`, self time | 11.94% |
| Emdawn mapped-range handling / copy into Wasm, self time | 8.97% |
| Shader-source loading, identified inclusive stacks | 3.35% |
| Other host work | 23.77% |
| Renderer idle | 22.49% |

The hot vector-assignment callers are the host geometry/magnetic/constraint
handoffs and the readback slice decoders. This is repeated materialization of
large arrays, not mainly scalar convergence checks. The four constraint
state/filter high/low uploads alone accounted for **7.50 ms/iteration** of
host `writeBuffer` call time and 7.70 MB/iteration. Such API time can include
staging or backpressure; it is not isolated PCIe transfer duration. There were
about 36.90 MB of readback copies and 151.94 MB of device copies per iteration.
Submission calls themselves took only about 0.175 ms/iteration, including the
profiler's extra submission, so API call count alone is not the dominant
measured CPU cost. It does not capture all submission/driver scheduling costs.

Source inspection identifies another GPU optimization candidate:
`magnetic_field_double_single.wgsl::finalize_current` assigns one invocation
to each of 98 half-surfaces and launches just **two 64-thread workgroups**.
Each invocation performs a serial angular integral. Parallelizing its
integrand evaluation is worth testing, but changing the reduction order could
change paired-precision results and requires renewed trajectory qualification.
Shader sources are also loaded before cached-pipeline lookup; cached pipeline
creation does not eliminate this warmed-up host cost.

A separate FFT capture measured 11.646 ms of compute and 49.550 ms/iteration.
Its two FFT executions took 4.360 ms/iteration, packing/unpacking 0.233 ms,
and poloidal forward projection 1.056 ms: **5.649 ms for the forward path**,
versus 4.900 ms for the direct path in the repeated direct sample. Shared
kernels ran somewhat faster in the FFT capture too, so these sequential
captures are **not** a controlled clock-matched A/B comparison. They establish
that host data handling remains important in both paths, not a new winner
for total solve time. The preliminary FFT harness mapped unused batch capacity
as well as the timestamp tail; the finalized direct harness avoids this extra
mapping. Treat FFT's host/map timings as instrumented observations only.

Both transform modes retained their previously qualified convergence traces:
2812 direct / 3091 FFT effective iterations at `1e-12`, with matching residuals,
controller decisions, and high/low state fingerprints against their respective
saved runs. Deterministic profiler tests cover feature negotiation, query reuse,
single-map behavior, disjoint mapped ranges, unchanged payload, and hook
restoration. These are profiling findings, **not solver optimizations**.

Prioritize removing redundant host vector copies and constraint round trips,
then reducing full-field readback to genuinely needed control/output data.
Cache shader-source construction outside the loop. GPU kernel work should
target the current reduction and forward transforms after those host costs
are addressed, with precision/convergence checks maintained.

Raw captures and Chrome-importable symbolized CPU profiles are saved under
`../tmp/w7x-deep-direct-final-*` and `../tmp/w7x-deep-fft-*`; the earlier
independent direct sample is `../tmp/w7x-deep-direct-*`. The timing summaries
use `-summary.json`; `-cpu-symbolized.cpuprofile` can be imported into Chrome
DevTools. These local artifacts are not required by the application.

### Host data-path optimization after profiling (2026-09-06)

`30418b8` caches immutable embedded shader text (including paired-precision
prelude assembly) and populates constraint input buffers plane by plane from
their actual host/device owners. Previously, large zero-filled placeholders
were uploaded and then immediately overwritten with device copies. Removing
those transfers saves **5,987,520 upload bytes per ns=99 iteration**. The
f32 bandpass output's low plane is explicitly cleared on the GPU; no shader
arithmetic or reduction order changes.

`31e0d69` skips duplicate host input construction while consuming already
collected iteration results. The primary inverse geometry, base geometry, and
magnetic arrays remain available for finite/Jacobian checks, force
normalization, and final output. The original validation/controller callback
chain remains in order. Accepted constraint arrays are moved into their host
owner rather than copied. Sequential/conformance paths still construct their
ordinary inputs, and rejected-pass/preconditioner ownership is unchanged.

An uninstrumented A/B/B/A comparison used the retained `0c19912` executable
and the optimized executable in the same foreground Chrome/RTX 3060 Ti
session. Each launch warmed 250 controller records and measured the same
next 256 intervals. No timestamp/API/CPU profiling hooks were installed in
these four runs; `trace=1` was enabled identically for timing and equivalence.

| Run | Before, ms/iteration | Optimized, ms/iteration |
| --- | ---: | ---: |
| First sample | 49.899 | 35.111 |
| Second sample | 48.895 | 35.610 |
| **Mean** | **49.397** | **35.361** |

This is **28.4% less elapsed time per iteration** (1.40x throughput). Every
sampled controller record, including state fingerprints, matched between
builds. `scripts/webgpu_compare_builds.mjs` reproduces this probe, saves its
records, and closes only its own test tab:

```sh
node scripts/webgpu_compare_builds.mjs OLD_BUILD_URL NEW_BUILD_URL \
  ../tmp/w7x-opt-abba.json
```

The full optimized direct solve with a short GPU sample and a separate CPU
sample completed in **99.7 s**, retaining 2812 effective iterations and the
entire saved 2817-record trajectory. Its final residuals remain
`(9.985159710508547e-13, 2.1300946362442957e-13, 1.9552385266042975e-13)`.
The 11,809,091-byte output file before/after the host-copy change was
byte-identical, including derived fields (SHA-256
`435743355022d3919e8ef1db33240c2cc2e52c16835d017c7724f6a0e749ed4d`;
these captures had identical build-provenance strings).

The optimized FFT solve completed in **113.8 s** without profiling hooks,
retaining 3091 effective iterations, all 3096 saved controller records, and
final residuals
`(9.987756002533206e-13, 2.0790797273186487e-13, 1.725519314229383e-13)`.
The full trace matched the previously qualified FFT trajectory exactly.

The warmed instrumented sample measured **4.37 MB uploads/iteration**, down
from 10.36 MB, while readback remains about 36.90 MB. Vector assignment/
insertion self time fell from 29.49% to 15.53% of a separate 10-second CPU
sample; `writeBuffer` self time fell from 11.94% to 4.69%. Shader-source I/O
no longer appeared in the warmed CPU sample. These percentages include idle
time and use separate sampling windows; use the A/B/B/A timings above for the
performance claim. Unchanged shader execution times varied substantially
between captures, so the earlier standalone profile times are not substituted
for a contemporaneous baseline. The longer map wait in the optimized sample
does not imply a regression: the CPU reaches its one fence sooner.

The native shader-cache test checks exact text/prelude assembly, cache object
identity, and failed-load retry. The full Chrome conformance suite with FFT
enabled passed, including the paired W7-X slice and the 327-iteration Solovev
regression. The browser artifact CTest gate passed as well. Captures are saved
under `../tmp/w7x-opt-transfer-cache-*`, `../tmp/w7x-opt-host-copies-*`,
`../tmp/w7x-opt-abba.json`, `../tmp/w7x-opt-fft-*`, and
`../tmp/w7x-opt-conformance.json`.

Further work remains: reducing full-field readback, retaining persistent
constraint/descent data on the device, and accelerating the serial current
integral. This change does not claim full residency or change paired
arithmetic to ordinary f32.

The WebGPU implementation lives under these paths:

```text
include/cumes/webgpu/          public WebGPU operator contracts
src/webgpu/                    emdawnwebgpu host implementation
src/webgpu/shaders/            WGSL compute kernels
webgpu/                        Emscripten target, browser bridge, and webapp
```

WebGPU code does not include CUDA compatibility shims. Buffers are WebGPU
objects rather than emulated pointers, work is recorded into command encoders,
and completion is callback-driven. This makes synchronization and ownership
visible instead of attempting to reproduce CUDA stream behavior through a
source-level macro layer.

## Capability boundary and future optimization

The following are follow-on optimizations or optional backend expansions, not
completion gates for the fixed-boundary WebGPU port:

1. reduce the remaining host validation/reduction payload and combine the
   individual operator submissions (production 3-D operator dependencies now
   stay on device, with one batched mapping per evaluated pass);
2. port the optional free-boundary/NESTOR dependency as a separate WebGPU
   project if browser free-boundary equilibria are required.

`deps/vacuum-field` is itself a CUDA solver and is intentionally outside the
CUDA-free browser target; inputs with `lfreeb=true` therefore fail validation
instead of silently using fixed-boundary physics. NetCDF/HDF5 and the
magnetic-coordinate CUDA postprocessor are likewise host/native extensions,
not browser solver requirements. The browser publishes the complete native
binary schema through MEMFS and a JavaScript Blob download adapter, including
spectral state, scientific fields, multigrid history, provenance, and the
normalized input record.

## Verification levels

- `cmake --build --preset webgpu`: compiles C++ against the installed
  emdawnwebgpu headers and links the embedded WGSL/browser bundle.
- `ctest ...`: verifies non-empty `.html`, `.js`, and `.wasm` artifacts.
- browser self-test: compiles WGSL on the selected adapter, dispatches both
  prolongation modes and the complete direct axisymmetric and 3-D transform
  paths, maps results, compares every value with the C++ references, runs two
  complete W7-X controller passes, then converges all three Solovev multigrid
  stages (operator tolerances `4e-6` through `1e-3`, solver tolerance `1e-6`).
- W7-X integration gate: the browser controller trajectory through effective
  iteration 3 must match native CUDA mixed-float, including the invariant
  residual triple. The selected `?solve=w7x` path must additionally converge
  its final `ns=99` grid with the input's unmodified `1e-12` tolerance on a
  physical WebGPU adapter; `?solve=w7x&grids=3` is the full multigrid gate.
