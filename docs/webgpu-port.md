# WebGPU port

## Status

The WebGPU backend is an additive, experimental backend. CUDA remains the
default and the only backend with free-boundary coupling and production
performance qualification. WebGPU now implements the complete fixed-boundary
iteration DAG for axisymmetric and folded 3-D equilibria.

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
  avoiding adapter-dependent WGSL transcendental approximations; direct 3-D
  analysis and synthesis use compensated f32 accumulation, which restores the
  qualified W7-X stage trajectory despite WebGPU's lack of f64;
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
  pivot guards, paired R/Z Thomas solves, axis boundary zeroing, and lambda
  diagonal scaling; the browser gate applies it to the constrained Solovev
  residual and requires zero pivot breakdowns;
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
  against the independent C++ reference; WebGPU converges in `72 -> 32 -> 182`
  effective iterations (286 total) with final residual
  `(9.831e-07, 3.438e-07, 3.011e-10)`, while native CUDA mixed-float converges
  in `72 -> 31 -> 182` (285 total) with
  `(9.696e-07, 4.486e-07, 3.081e-10)`;
- the complete controller-driven 3-D W7-X stage path: inverse transform,
  geometry, prescribed-current closure, MHD force, constraint bandpass,
  toroidal projection, residual decomposition, `(m,n)` preconditioner, and
  descent. The browser and native CUDA mixed-float paths agree at effective
  iteration 3 on `(FSQR,FSQZ,FSQL) = (1.141e+01, 7.079e+00, 1.012e-01)`;
- a selectable `?solve=w7x` browser mode that runs the complete three-stage
  W7-X multigrid path and publishes the same schema-v8 result form; at the
  qualified mixed-float tolerance, both native CUDA and WebGPU converge in
  `82 -> 30 -> 20` effective iterations (132 total), with WebGPU's final
  residual `(7.346e-03, 2.026e-03, 1.347e-05)`;
- cached immutable toroidal basis buffers plus persistent, grow-only operator
  scratch/readback buffers and compute pipelines; this removes hot-loop shader
  recompilation/allocation and keeps the complete high-resolution W7-X run
  stable on Chromium's software adapter;
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

The default self-test parses both embedded inputs, runs a controller-complete
two-pass W7-X slice, then converges all three Solovev stages. The separate W7-X
solve entry point is integrated and convergence-qualified end to end on
Chromium SwiftShader. Hardware WebGPU performance remains unqualified on this
host because its headless browser cannot acquire the NVIDIA Vulkan adapter;
native CUDA verification does use the TITAN Xp.

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

Serve the generated files over HTTP; browsers do not reliably initialize
WebGPU from `file://` URLs:

```bash
python3 -m http.server 8000 \
  --directory ../tmp/cumes-build-webgpu/webgpu
```

Open `http://localhost:8000/cumes_webgpu.html`. A successful run reports both
interpolation cases followed by:

```text
cuMES WebGPU self-test: PASS
```

Open `http://localhost:8000/cumes_webgpu.html?solve=w7x` to run the complete
fixed-boundary W7-X multigrid solver instead of the conformance suite. This
path can be slow on software WebGPU adapters because the current 3-D transform
is a direct DFT.

The page also publishes `data-cumes-webgpu="pass|fail"` and a diagnostic
`data-cumes-detail` on `<body>` so browser automation can inspect the result.
Long W7-X runs additionally publish the current stage, iteration, FSQR, and
last-progress timestamp as `data-cumes-stage`, `data-cumes-iteration`,
`data-cumes-fsqr`, and `data-cumes-last-progress-at`.

## Precision policy

Core WGSL exposes `f32` but not `f64`. Consequently:

- the WebGPU preset requires `mixed-float` and `CUMES_USE_FLOAT=ON`;
- configuring WebGPU with a double policy is a hard error;
- CUDA Class A byte identity is not a WebGPU acceptance criterion;
- operator acceptance uses the existing float tolerances and CPU-reference
  comparisons;
- full-solver tolerances must be achievable in mixed-float. Solovev qualifies
  at `1e-6`; prescribed-current W7-X has a measured CUDA float floor near
  `3e-3`, so both the CUDA float capture and the browser's selected W7-X solve
  use `1e-2`.

The host uses double for the invariant residual reduction after mapped WebGPU
readback. Each squared residual pair is evaluated in `f32` before the sum is
promoted, matching the CUDA mixed-float expression, and the long accumulation
is `double`. Future device-only reductions must use `f32` or a documented
compensated/multiword representation.

## Backend boundary

The WebGPU implementation lives under these paths:

```text
include/cumes/webgpu/          public WebGPU operator contracts
src/webgpu/                    emdawnwebgpu host implementation
src/webgpu/shaders/            WGSL compute kernels
webgpu/                        Emscripten target and browser shell
```

WebGPU code does not include CUDA compatibility shims. Buffers are WebGPU
objects rather than emulated pointers, work is recorded into command encoders,
and completion is callback-driven. This makes synchronization and ownership
visible instead of attempting to reproduce CUDA stream behavior through a
source-level macro layer.

## Remaining port sequence

The remaining qualification and optimization order is:

1. qualify the complete W7-X convergence run on a hardware WebGPU adapter;
2. replace the compensated direct DFT with a WebGPU FFT;
3. retain spectral/real-space fields on device across adjacent operators and
   batch each device-only segment into one command submission (buffers and
   pipelines are persistent today, but mapped host results still connect the
   operator APIs);
4. integrate free-boundary/NESTOR support.

Free-boundary support is last because `deps/vacuum-field` is itself CUDA-based
and includes host/device coupling beyond the main operator DAG. NetCDF/HDF5 and
the magnetic-coordinate CUDA postprocessor are also excluded from the browser
target. The browser publishes the complete native binary schema through MEMFS
and a JavaScript Blob download adapter, including spectral state, scientific
fields, multigrid history, provenance, and the normalized input record.

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
  residual triple. The selected `?solve=w7x` path uses the CUDA mixed-float
  qualification tolerance (`1e-2`) for all three stages.
