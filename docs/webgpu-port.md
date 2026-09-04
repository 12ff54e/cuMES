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
  avoiding adapter-dependent WGSL transcendental approximations; direct 3-D
  analysis and synthesis use compensated f32 accumulation, which stabilizes
  the qualified W7-X solve despite WebGPU's lack of f64;
- a two-dispatch separable 3-D inverse transform: a short compensated toroidal
  synthesis feeds the poloidal synthesis through persistent storage. For the
  W7-X shape this replaces a 156-term mode loop per real-space point with
  13-term toroidal and 12-term poloidal loops, while improving the hardware
  conformance error from `2.861e-06` to `1.431e-06`;
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
- a selectable `?solve=w7x` browser mode that runs the complete three-stage
  W7-X multigrid path and publishes the same schema-v8 result form; at the
  qualified mixed-float tolerance, the TITAN Xp Vulkan WebGPU path converges in
  `82 -> 35 -> 25` effective iterations (142 total), with final residual
  `(8.893e-03, 2.347e-03, 7.882e-06)` and an 11,809,203-byte result; the
  sequential end-to-end browser run takes 47.1 seconds including page/Wasm
  startup; SwiftShader converges independently in `82 -> 30 -> 20` (132 total)
  with `(7.692e-03, 2.082e-03, 1.347e-05)`;
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
solve entry point is integrated and convergence-qualified end to end on both
Chromium SwiftShader and the physical NVIDIA TITAN Xp through Chrome/Dawn's
Vulkan backend. The page requests the high-performance adapter and publishes
its device/type/backend metadata for automation. Chrome's privacy-reduced
adapter name is the PCI device id `0x1b02`; `chrome://gpu` and Vulkan enumerate
that id as the TITAN Xp, and `nvidia-smi` observes the browser GPU process.

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

Append `?mode=test` for the full GPU/CPU operator conformance suite and
stricter Solovev convergence gate. A successful run finishes with:

```text
cuMES WebGPU self-test: PASS
```

Append `?solve=w7x` to run the complete fixed-boundary W7-X multigrid solver
instead of the conformance suite. This path can be slow on software WebGPU
adapters because the current 3-D transform is a direct DFT.

The page also publishes `data-cumes-webgpu="pass|fail"`, a diagnostic
`data-cumes-detail`, and `data-cumes-adapter`, `data-cumes-adapter-type`, and
`data-cumes-adapter-backend` on `<body>` so browser automation can inspect both
the result and the selected device.
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

1. retain spectral/real-space fields on device across adjacent operators and
   batch each device-only segment into one command submission (buffers and
   pipelines are persistent today, but mapped host results still connect the
   operator APIs);
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
  residual triple. The selected `?solve=w7x` path uses the CUDA mixed-float
  qualification tolerance (`1e-2`) for all three stages.
