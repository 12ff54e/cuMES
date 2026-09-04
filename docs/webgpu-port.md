# WebGPU port

## Status

The WebGPU backend is an additive, experimental backend. CUDA remains the
default and the only complete equilibrium solver.

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
- host-generated `f32` Fourier tables, matching the CUDA operator contract and
  avoiding adapter-dependent WGSL transcendental approximations;
- the existing JSON mapper, validator, boundary folder, and resolution logic
  compiled into Wasm, with production-shaped axisymmetric cold-start and
  radial-profile initialization;
- the shipped Solovev input parsed in-browser, relaxed to the documented f32
  tolerance floor, initialized, and passed through the WGSL inverse transform;
- the staggered half-grid base-geometry operator in WGSL: parity recombination,
  radial interpolation and derivatives, Jacobian, and covariant metric;
- fixed-iota contravariant/covariant magnetic fields and total pressure, with
  the CUDA path's finite-Jacobian division guard;
- axisymmetric radial/poloidal MHD weak-form forces and hybrid λ force on the
  full grid;
- first-pass Solovev force projection into the six-family spectral residual
  slab (the initial constraint multiplier is zero, matching CUDA/vmecpp);
- odd-m residual decomposition, the m=1 force gauge, fixed-boundary LCFS norm
  exclusion, and CUDA-compatible `f32` products accumulated into `double` host
  residual sums;
- the later-pass axisymmetric spectral-condensation constraint: LCFS-volume
  reference reset, current-preconditioner `tcon` refresh, `gConEff`, direct
  poloidal bandpass, MHD-force injection, `frcon`/`fzcon`, and constrained
  residual projection;
- fixed-boundary radial preconditioner element assembly from half-grid force
  Hessian surface integrals, including parity factors, full-grid diagonal
  averaging, and the LCFS pedestal; these live Solovev `ard`/`azd` values feed
  the constraint multiplier test;
- an independent C++ float reference evaluated by the browser self-test.

This milestone parses and initializes the embedded Solovev input, but does
**not** yet run equilibrium iterations. It is a functioning compute-backend
slice, not a complete cuMES WebGPU solver.

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

The page also publishes `data-cumes-webgpu="pass|fail"` and a diagnostic
`data-cumes-detail` on `<body>` so browser automation can inspect the result.

## Precision policy

Core WGSL exposes `f32` but not `f64`. Consequently:

- the WebGPU preset requires `mixed-float` and `CUMES_USE_FLOAT=ON`;
- configuring WebGPU with a double policy is a hard error;
- CUDA Class A byte identity is not a WebGPU acceptance criterion;
- operator acceptance uses the existing float tolerances and CPU-reference
  comparisons;
- full-solver inputs will need `ftol_array >= 1e-6`, consistent with the CUDA
  mixed-float policy.

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

The recommended dependency order is:

1. reusable buffer/pipeline/bind-group ownership and a stage command encoder;
2. persistent spectral/real-space stage storage and profile GPU buffers (input
   parsing, cold seeding, and host profile evaluation are complete);
3. prescribed-current magnetic-field closure (fixed-iota field evaluation and
   base geometry are complete);
4. tridiagonal/lambda preconditioner assembly and application, then descent
   shaders (the radial element cache, core axisymmetric MHD force, later-pass
   constraint force, and invariant residual decomposition are complete), each
   compared with the existing CPU/CUDA references;
5. a fixed-boundary axisymmetric stage loop and relaxed-float Solovev gate;
6. generic 3-D transforms, using a WebGPU FFT implementation or a direct DFT
   correctness path before optimization;
7. 3-D W7-X qualification and only then free-boundary/NESTOR integration.

Free-boundary support is last because `deps/vacuum-field` is itself CUDA-based
and includes host/device coupling beyond the main operator DAG. NetCDF/HDF5 and
the magnetic-coordinate CUDA postprocessor are also excluded from the browser
target; browser output should first use the native binary schema through the
virtual filesystem or a JavaScript download adapter.

## Verification levels

- `cmake --build --preset webgpu`: compiles C++ against the installed
  emdawnwebgpu headers and links the embedded WGSL/browser bundle.
- `ctest ...`: verifies non-empty `.html`, `.js`, and `.wasm` artifacts.
- browser self-test: compiles WGSL on the selected adapter, dispatches both
  prolongation modes and the complete direct axisymmetric transform path, maps
  results, compares every value with the C++ references, then parses and
  initializes the embedded Solovev case and evaluates its half-grid geometry
  plus first- and later-pass force/residual paths (tolerances `4e-6` through
  `1e-3`).
- future operator gates: compare full typed views against existing test
  references before wiring the operator into the stage DAG.
