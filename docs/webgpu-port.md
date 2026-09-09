# WebGPU port

## Status

The WebGPU backend is an additive browser backend. CUDA remains the default. Free-boundary browser solves combine WebGPU plasma
operators with the vacuum-field library using WebGPU or HOST/WebAssembly. The WebGPU port
of cuMES's fixed-boundary solver is complete: it implements and hardware-
qualifies the entire iteration DAG for axisymmetric and folded 3-D equilibria,
multigrid control, native binary result publication, and an interactive
fixed/free-boundary editor with axisymmetric and 3-D previews.

### Unified boundary editor

The Boundary editor handles both fixed and free equilibria. Select Solovev or
W7-X under **Fixed boundary**, or select the coil configuration under **Free
boundary**. There is one Run/Stop workflow, precision control, residual plot,
and result download. Historical `?solve=w7x` links open the fixed W7-X editor;
`?preset=w7x` is the current link. Neither starts a solve until Run is clicked.
The fixed W7-X preset keeps its single-grid default and combined iteration
budget; **Resolution, profiles and input JSON** also offers multigrid.

Run, Stop and edit, Reset, and Download share a toolbar above the workspace
that remains visible while scrolling. During a solve it shows recent completed
iterations per second, updated about twice per second from the existing pass
timing events. The rate includes GPU waits and vacuum work, counts completed
passes through restarts, and resets at each grid so startup, MAKEGRID, and grid
setup do not enter the estimate. It remains available with `timing=0` and adds
no GPU readbacks. Completion, failure, and returning to setup clear the rate.

For 3-D boundaries and free-boundary initial guesses, select a signed toroidal
mode `n` and edit the RBC/ZBS coefficients by poloidal mode `m`. A toroidal-angle
slider sweeps one field period. The R-Z cross-section and orange section on the
orbitable boundary preview update together, using the solver's six-family
Fourier convention. Coefficient edits update the input JSON directly; no
projection or rounding to the axisymmetric editor's `m <= 5` basis is applied.
The fixed Solovev editor retains its existing Fourier sliders and contour mode.

Boundary previews are labeled separately from converged flux surfaces. The
result's **2D cut** view has its own toroidal-angle slider for 3-D equilibria,
covering `0–360°/NFP`. It reconstructs cuts from the returned Fourier coefficients
without another solve and keeps the angle when switching between 2D and 3D.
A small 3-D inset in the top-right corner highlights the selected cut in orange
and displays its zeta angle. It reuses the full view's WebGPU canvas, geometry
buffers, camera, and coil visibility; the slider updates only the section line.
The inset can still be orbited and zoomed. The **3D inset** button toggles it
and remembers the choice across reloads. Axisymmetric results show neither
the inset nor its toggle or angle control.

The JSON panel retains access to profiles, field periods, angular/radial resolution,
and iteration budgets. W7-X fixed-boundary edits and free-boundary setups are
stored separately so mode/precision changes retain them. The browser now uses
the existing interactive solver entry point for W7-X as well; its scalar
radius-reference and compensated-geometry options are preserved for fixed 3-D
inputs. The separate W7-X startup implementation has been removed.

### 3-D rendering

`webgpu/orbit_renderer.js` renders the boundary preview and solved flux
surfaces with WebGPU. A compute shader reconstructs the full torus from the
six physical Fourier families and reduces its bounding radius. Geometry,
bounds, and line indices stay on the GPU; changing the camera uploads one
32-byte uniform. Coefficient edits regenerate geometry, while moving the
section slider updates only the highlighted line. Every supplied surface is
drawn, without the previous CPU renderer's surface subsampling.

Vertex and fragment shaders draw instanced line ribbons with consistent screen
widths, transparent colors, and 4x multisampling. The view retains transparent
wireframe semantics. Render targets, including a depth attachment, are reused
until resize; buffers are reused until more capacity is needed. Frames are
submitted only when the view changes. The renderer owns a separate device from
the solver and releases resources on page exit. A renderer failure is reported
in the view; 2-D cuts remain available. Device loss releases the renderer and
asks for a reload or browser restart, since Firefox can stop delivering
animation callbacks even after a replacement device has been initialized.

Free-boundary views draw the selected coils in copper around both the initial
boundary and solved flux surfaces. The **Coils** toggle above the 3-D view
retains its setting across reloads and remains available during a solve. Coil
positions and indices use the same resident line layer as the surfaces;
toggling visibility reuses those buffers and preserves the camera framing.
Solovev's effectively infinite central conductor is clipped for display so
its remote return path does not shrink the plasma out of view. Full-device
coil files are drawn once, without replication by field period.

`webgpu/coil_geometry.js` loads a small Wasm build of vacuum-field's existing
`coils-convert` tool to normalize both preset and uploaded geometry. This
reuses the solver's coils-dot and JSON parsing and validation. The preview
and solver share the original selected coil bytes; MAKEGRID runs only when
the solve starts. Fixed-boundary setup does not load the coil parser or assets.

Opaque meshes and volume passes can share the camera and depth target;
volume rendering is not yet implemented. Display geometry uses WGSL f32;
this does not change equilibrium arithmetic or downloaded scientific data.
CPU Fourier evaluation remains only for the 2-D cuts and section highlight.
The production 3-D path performs no GPU readback; the geometry validation
script explicitly reads vertices to compare them with independent harmonics.

### Free-boundary browser setup

The browser runtime is built with Emscripten pthreads. Serve it with
`Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp` so shared Wasm memory is available.
Free-boundary setup prewarms up to 16 CPU threads, bounded by the browser's
reported concurrency, and uses the existing vacuum-field MAKEGRID parallel
loop. `?boundary=free&makegrid_threads=1` retains sequential generation for
comparisons; other positive counts are capped at the available pool size.
Fixed-boundary runs do not create MAKEGRID workers. Coil sums retain their
original order within each grid point, independent of the thread count.

Free-boundary 3-D and axisymmetric solves in both precisions reuse the resident
plasma operators in two batches:
inverse/geometry/magnetic/MHD force, then projection/constraint/preconditioning.
Spectral state and descent velocity remain on the GPU across normal iterations.
The preceding descent's state snapshot joins the next prefix map, avoiding a
separate descent wait; velocity retains a compact finite check. The host
commits the pending state, accepts the geometry, updates the existing double
NESTOR solve, and uploads only the four corrected LCFS force rows before the
suffix. The force readback contains only those four rows plus flags from a
finite scan of every force word. Rejected geometry discards the pending
continuation before the vacuum update. The coupling reconstructs only the
axis, boundary coefficients and outer rows it needs; full geometry/magnetic
readbacks and the native bridge kernels' reduction order remain.
`&field_readbacks=full` restores full force and velocity snapshots.
`&resident=0`, `&fences=1`, or `&compare_fft=1` selects the separate-dispatch
free-boundary path for trajectory comparisons. Axisymmetric constraint
filtering retains its existing shader; buffer copies adapt its force-plane
layout to the shared projector. Transfer counts and qualification are recorded
in [the optimization assessment](webgpu-optimization-assessment.md#free-boundary-residency-and-compact-force-readbacks).

Fixed-boundary 3-D and axisymmetric solves use the same resident iteration
pipeline, with one batched mapping per evaluated pass. The transform layer
shares cached basis tables, buffers, device-state handoffs, and readback
handling. Scalar axisymmetric transforms select the existing direct poloidal
shaders, skipping the toroidal pass and its scratch allocation. Their original
summation order and host-rounded derivative tables are preserved. Paired
axisymmetric transforms retain the separable shaders with `ntor=0, nzeta=1`.
`&resident=0` restores separate dispatches for trajectory comparisons.

Choose **Fixed boundary** or **Free boundary** above the editor. Free-boundary
setup defaults to paired precision and offers Solovev, W7-X (vacuum), and
cth_like coil configurations, plus
MAKEGRID coils-dot / cumes-coils-v1 JSON uploads. Currents, grid parameters, and
initial equilibrium JSON are editable. Uploaded geometry stays in IndexedDB;
setup choices stay in localStorage. Switching modes or **Stop and edit** ends
an active worker and returns to setup. Precision changes restart the solve.
`?boundary=free&coils=w7x` opens a named preset; append `&run=1` to run it.

The build links vacuum-field's WebGPU backend and HOST-double reference with
NetCDF disabled. Append `&vacuum=webgpu` to select paired-f32 vacuum kernels;
`&vacuum=host` is the reference path and remains the default.
Coil parsing and MAKEGRID generation stay in Wasm double. Only coil geometry and small input
JSON files are served as lazy preset assets; no field grids are shipped.
`src/webgpu/vacuum.cpp` compiles the existing cuMES vacuum state machine and
bridge kernels for Wasm memory and converts WebGPU high/low words at the
handover. The WebGPU backend keeps geometry, fields, integrals and the Laplace
assembly resident, maps matrix/RHS together for Wasm-double LU, uploads the
potential, then maps the reconstructed outputs for the existing LCFS coupling.
Full/partial update reuse and activation/restart state are shared. Asyncify yields
the worker's host controller while the two GPU batches complete. The HOST path
runs its kernels in the same worker.

Vacuum activation, edge force/preconditioning, constraint decay, soft restarts,
and multigrid persistence follow the CUDA coupling. The three bundled presets
pass full paired-precision solves in Chrome, including the 3-D W7-X vacuum
and cth_like cases. The shared bridge has analytic CUDA/WebAssembly tests for
pressure symmetry, surface averages, and boundary-only force updates;
vacuum-field's 19 operator/reference tests also pass under WebAssembly, native
HOST, and CUDA. Headless Firefox also passes the full paired Solovev solve
with HOST vacuum.
The opt-in GPU vacuum passes paired Solovev/W7-X/cth_like and scalar Solovev
on Chrome/RTX 3060 Ti. GPU versus HOST vacuum can change paired trajectories.
The subsequent integral optimizations preserve the original GPU arithmetic
and controller trajectories while accelerating singular and 3-D regularized
work. [ADR-0019](adr/0019-webgpu-vacuum-backend.md) records numerical comparisons,
measured performance and the retained HOST default.

Scalar-float Solovev works, but W7-X can stall above its tolerance; single
precision remains experimental for free-boundary work. Preset provenance is
recorded in `webgpu/presets/README.md`.

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
`mpol`, `ntor`, and `nfp`; the renderer's compute shader reconstructs the full
torus locally. Drag the canvas to orbit and use the wheel to zoom. W7-X uses
the same result panel in the boundary editor.

Both the boundary editor and W7-X show a live **Residual history** canvas with
FSQR, FSQZ, and FSQL on a logarithmic vertical axis and the current tolerance
as a dashed horizontal line. The horizontal axis counts attempted iterations
cumulatively across grids; the caption also shows the controller's effective
iteration, which can reset. Vertical dashed lines separate grids, without
connecting their curves. Pink solid vertical lines mark actual controller
restarts, including early Jacobian/nonfinite and maintenance restores that
produce no residual sample; the caption gives the restart count. These come
from the controller's restart-event history, not residual spikes or inferred
counter changes, and remain visible without `trace=1`.
The markers are also available as `window.cumesResidualPlot.report().restarts`.
Both W7-X precisions retain their exact qualified controller traces with nine
markers: five post-descent restarts and four early rejected passes.
Nonpositive/nonfinite values remain in the diagnostic
history but are omitted from the log plot, not clamped to a false residual floor.

The scalar progress hook reuses normalized residuals already available to the
host controller: no extra GPU readback, numerical change, `EM_JS`, or full
state-fingerprint tracing is needed. JavaScript retains every sample and
coalesces drawing to at most 10 Hz (plus the final flush); per-pixel
first/min/max/last envelopes preserve spikes. Hidden tabs skip rendering.
`window.cumesResidualPlot.report()` exposes samples and cumulative canvas-draw
time for diagnostics; `residual_plot=0` disables the chart. Verification keeps
its compact check summary instead of plotting its diagnostic solver run.

`ctest` includes canvas/scalar-bridge tests for log coordinates, zeros/nonfinite
values, stage changes, controller restarts, and coalesced/final rendering.
The editor precision and W7-X Start browser smoke tests check live plotting
before completion; the validation harness checks every plotted residual against
the controller trace when `trace=1` is requested.

Chrome/RTX 3060 Ti qualification retained exact pre-chart controller traces:
editor paired/single 507/73 records, W7-X paired/single 2817/1261 records
(2812/1256 effective iterations), and verification 329 records. In the paired
W7-X run, cumulative canvas drawing took 379 ms over 39.8 s page elapsed time;
single took 116 ms over 14.7 s. These are drawing costs, not an end-to-end
on/off slowdown measurement.

Append `?mode=test` for the full GPU/CPU operator conformance suite and
stricter Solovev convergence gate. Verification runs in a dedicated Web Worker:
Wasm, CPU references, and the WebGPU device stay off the UI thread. The page
receives batched logs/diagnostics and final timing/output messages. Verification
still checks output serialization and reports its byte count, but does not
create a download link. Downloads are only offered for editor and W7-X solves.
Leaving the page also
terminates the worker. Editor and W7-X solves keep their existing main-thread
orchestration. `?mode=test&worker=0` is a diagnostic opt-out, and
`data-cumes-execution="worker|main"` identifies the selected path.
Serve `browser_ui.js` and `verification_worker.js` alongside the generated
HTML/JS/Wasm files; the build copies and content-versions these assets.

The W7-X integration passes and Solovev convergence loop reuse the solver's
batched iteration dispatch. Verification snapshots every force, projection,
and constraint intermediate before its GPU scratch is reused, then runs all
the original CPU comparisons after one batch readback. Descent retains its
separate readback and comparison. This preserves the checked values,
tolerances, controller decisions, and full log while reducing GPU/host waits.
`?mode=test&resident=0` retains separate dispatches for comparison. Normal
solves keep their compact readbacks and do not collect these extra snapshots.

A successful run finishes with:

```text
cuMES WebGPU self-test: PASS
```

The verification page shows a compact summary: GPU setup, operator and W7-X
integration results, one convergence line per Solovev grid, output checks, and
the final result. A live status line replaces per-iteration log spam. Warnings
and failures are always shown. Expand **Detailed log** for every original
comparison and error value; collapsed details are buffered in memory without
rendering thousands of lines. Browser automation can read the unchanged full
stream using `window.cumesVerificationLog.text()` (the validation script does
this automatically). Editor and W7-X solver logs are unchanged.

Append `?preset=w7x` to open the fixed-boundary W7-X editor. Choose precision
and click **Run equilibrium** to load Wasm, request the GPU, and start solving.
Opening setup does not request the GPU; `&run=1` starts the edited input.
Historical `?solve=w7x` links migrate to setup and discard `run=1`.
**Stop and edit** returns to setup with the input retained. Changing precision
during a solve restarts it using the saved input, as in the other editor modes.
The run deadline starts with the solve. Browser validation/profiling tools
click Run for automated runs. The default example solves directly on its
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

## Headless Firefox qualification (2026-09-07)

The local Firefox 155.0.1 / Linux / NVIDIA TITAN Xp configuration requires
`dom.webgpu.enabled=true` and `gfx.webgpu.ignore-blocklist=true` to expose a
WebGPU adapter. With default preferences `navigator.gpu` is absent; enabling
only `dom.webgpu.enabled` still returns no adapter. These are browser/driver
availability restrictions that application code cannot change. The test
harness sets preferences only in a disposable profile.

Two compatibility fixes address failures after an adapter is available:

- The scalar forward shader uses named accumulators instead of passing
  pointers into local arrays. The latter crashed Firefox during compilation
  of `poloidal_stage`, despite WGSL validation succeeding.
- Precision kernels use `atomicExchange` followed by `atomicAdd(..., 0u)`
  for their f32 rounding boundary. The original store/load sequence lost
  low-word corrections on this configuration; changing only the read was
  insufficient for single-invocation workgroups. The pinned FFT generator
  receives the same fix through `src/webgpu/fft_shader.hpp`; the wrapper
  checks the expected upstream helper before replacing it.

The single-grid W7-X shortcut also retains the input's total iteration
budget (`3000 + 4000 + 5000 = 12000` attempts). Starting cold at `ns=99` had
kept only the 5000-attempt allowance intended for the final multigrid stage;
with corrected arithmetic this adapter reached `FSQR=1.088294e-12` before
that limit, still above the required `1e-12`. Chromium matched Firefox's first
3805 controller records exactly, including state hashes. Keeping the skipped
stages' budget allows the cold start to finish without changing the tolerance,
arithmetic, or restart policy. The three-grid route keeps its original stage
budgets. Startup logs report the active budget, and exhausted-limit errors
print residuals in scientific notation instead of rounding them to zero.

Verification now tests the actual embedded rounding functions from twelve
solver shaders and the generated FFT. It checks 128 exact f64-reference
sum/product cases per shader with both 1 and 32 invocations per workgroup,
including nonzero low words and cancellation, before operator conformance.
The existing operator tolerances and solver convergence targets are unchanged.

The full operator/Solovev gate passes in both the diagnostic main-thread path
and the default worker path with timing enabled. Both converge in
`72 → 32 → 247` effective iterations (351 total), final residuals
`(9.725e-7, 2.053e-7, 3.624e-10)`, and publish a 118,736-byte result.
Their 353 controller trace records match exactly. The default worker run takes
about 480 seconds on this headless setup; this is a correctness qualification,
not a cross-browser performance benchmark.

The default paired W7-X single-grid solve, with tracing and timing enabled,
converges in **5167 effective iterations / 5176 attempts**, final residuals
`(9.974830e-13, 2.089052e-13, 1.551586e-13)`, and publishes an
11,809,155-byte result in about 521 seconds. All 4996 controller records from
the earlier capped run are retained exactly; the additional 176 attempts
complete convergence. Timing records cover all attempts with no timing errors.
Median device compute is 14.882 ms/attempt and median readback wait is
71.600 ms/attempt on this setup.

The default float boundary editor also passes with timing enabled in
`54 → 17 → 2` iterations (73 total), with residuals
`(8.300579e-6, 4.700206e-6, 3.305034e-8)` and a 118,877-byte result in about
98 seconds. Its controller trace matches the pre-fix Firefox editor exactly.
Chromium 152.0.7977.64 on the same adapter passes the final conformance build
in about 15 seconds; all 353 controller records match Firefox. All seven
browser artifact/frontend CTest checks pass.

Start the installed geckodriver with a profile directory accessible to Firefox
(use a directory under `~/snap/firefox/common` for Snap Firefox):

```bash
mkdir -p ~/snap/firefox/common/cumes-test-profiles
geckodriver --host 127.0.0.1 --port 4445 \
  --profile-root ~/snap/firefox/common/cumes-test-profiles
```

In another terminal, run the gates sequentially, substituting the served build
URL. Node.js 22 or newer is sufficient; no browser automation package is needed.

```bash
cumes_test_url=http://localhost:6969/magnetic-equilibrium-solver/tmp/cumes-build-webgpu-firefox/webgpu/cumes_webgpu.html
cumes_test_prefs='{"dom.webgpu.enabled":true,"gfx.webgpu.ignore-blocklist":true}'
node scripts/webgpu_firefox_validate.mjs "$cumes_test_url?mode=test" \
  ../tmp/firefox/conformance "$cumes_test_prefs"
node scripts/webgpu_firefox_validate.mjs "$cumes_test_url?solve=w7x&trace=1" \
  ../tmp/firefox/w7x "$cumes_test_prefs"
```

The harness checks the terminal result and plotted convergence, records browser
capabilities, preferences, logs, residuals, timing, trace, and a screenshot,
and closes its own session on completion or failure. `GECKODRIVER_URL`
overrides the endpoint; `CUMES_FIREFOX_TIMEOUT_MS` overrides the 30-minute gate
limit. Omitting the preference argument tests an unmodified Firefox profile.

## Precision policy

### Scalar-f32 radius reference and selective geometry correction

The W7-X page has a **Single / Double** precision switch above the log;
the boundary editor has the same switch alongside its run controls.
Single selects scalar f32 (`1e-5`); Double selects the existing paired-f32
mode (`1e-12`), not native IEEE fp64. Switching restarts an existing editor solve
(an idle editor stays idle); W7-X returns to setup and waits for **Start**.
Both preserve the other URL options, including radial grids and FFT selection.
The current precision is also published as `data-cumes-precision` on the
page body. The boundary editor defaults to Single and W7-X defaults to Double.
Editor harmonics and contour points are retained when switching precision.
For the default editor boundary, Chrome qualification gives 507 effective
iterations in Double with final FSQR `9.375e-13`, and 73 in Single with FSQR
`8.515e-6`. These use different tolerances and are not a speed comparison.
The isolated end-to-end test changes precision using the actual buttons,
checks both converged results, and verifies that saved boundary data is intact:

```bash
node scripts/webgpu_editor_precision_smoke.mjs \
  'http://localhost:6969/magnetic-equilibrium-solver/tmp/cumes-build-webgpu-ds/webgpu/cumes_webgpu.html?timing=0&trace=1' \
  ../tmp/editor-precision
```

`?solve=w7x&precision=float` runs the single-grid example with scalar-f32
state and `ftol=1e-5`. It ports main's [radius-reference
representation](adr/0014-float-radius-reference.md) and [selective odd R/Z
reconstruction](w7x-float-float.md). Both are enabled for this explicit float
example; `&radius_reference=0` and `&geometry=native` independently disable
them for diagnostics. `&grids=3` retains the three radial grids. This is not
a scalar-f32 `1e-12` claim; the ordinary `?solve=w7x` paired-f32 route keeps
that tolerance and is unchanged numerically.

The float state stores all m=0 Rcc coefficients as small displacements from
the immutable double-precision boundary coefficients. Cold-start subtraction
happens before conversion to f32. The inverse retains displaced even radius,
but restores the reference before toroidal differentiation. Geometry and
forces restore absolute radius where needed; radial differences never
subtract two large reconstructed radii. Refinement and controller rollback
keep the displacement representation. Force normalization, downloaded
coefficients, derived fields, and both visualizations restore physical radius.

Only odd R/Z poloidal products, sums, and final radial scaling use compensated
pairs. Scalar toroidal intermediates, basis, angular derivatives, lambda,
constraints, and state storage retain their previous precision. Workgroup
atomic rounding boundaries prevent backend optimization from erasing the
compensation. Host setup caches the immutable angular reference in f32; no
f64 shader arithmetic is introduced. This ports the geometry ideas, not the
entire CUDA float controller/reduction policy, so CUDA trajectory identity
is not expected.

On the exposed Chrome/RTX 3060 Ti, the initial single-grid qualification
converged in **1256 effective iterations**, final residual
`(9.946e-6, 4.505e-6, 1.773e-9)`. Three runs reproduced the trajectory.
Instrumented page completion was about **13.7–14.4 s**, including startup
and output; these timings are not an equal-tolerance comparison with the
paired-f32 solver.

The three-grid float run also passes: **148 → 280 → 625** effective
iterations (1053 total), final residual `(9.316e-6, 4.295e-6, 2.119e-9)`.
Both prolongation comparisons pass with maximum absolute GPU/CPU difference
`1.192e-7`; the independently checked rendered LCFS is unchanged.

Initial single-grid opt-outs, measured at `1e-5` with the original
5000 attempted-pass budget:

| Float geometry options | Outcome |
| --- | --- |
| Neither fix | Exhausted budget; best FSQR `6.185e-4`, final `1.300e-3` |
| Radius reference only (`geometry=native`) | 2969 effective iterations; FSQR `9.694e-6` |
| Radius reference + compensation | 1256 effective iterations; FSQR `9.946e-6` |

Thus reference-only also converges on this Chrome backend, unlike the
qualified CUDA experiment; the two backends do not share identical f32
rounding or convergence trajectories. Compensation is beneficial here, not
a universal requirement inferred from the CUDA result.

The existing paired-f32 direct route, tested with its matching baseline
options `trace=1&gpu_norms=1&gpu_control=jacobian&timing=0`, retains all **2817
controller records exactly**, including state hashes and controller decisions.
It converges in 2812 effective iterations with residual
`(9.985e-13, 2.130e-13, 1.955e-13)`.

The conformance suite includes cancellation-sensitive odd reconstruction,
bit-identical untouched inverse fields, nonconstant-reference toroidal
derivatives, sub-ULP radial differences, absolute metric/force terms, and
all-stage seed/LCFS invariants. The rendered W7-X LCFS is independently
checked against the signed input harmonics, and the axis must collapse to
a curve:

```bash
node scripts/webgpu_validate_geometry.mjs CHROME_TARGET_ID ../tmp/w7x-render.png
```

The renderer uses physical state coefficients directly: the inverse-work
`1/sqrt(s)` odd regularization must not be applied to plotted coefficients.
The extrapolated odd axis row is omitted from the physical axis. The geometry
gate reads the shader-generated vertex buffer and checks both the LCFS and
poloidal axis spread within `2e-5 m`, allowing for display-only f32 arithmetic.

### Paired-f32 path

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
axisymmetric solves default to scalar-f32 at their responsive `1e-5`
tolerance; the precision switch enables paired-f32 at `1e-12`. The paired
axisymmetric path uses the same separable transforms with `ntor=0, nzeta=1`
(no toroidal FFT is needed), retaining low words through geometry, forces,
constraints, and descent. Constraint planes are remapped from the compact
axisymmetric layout to the shared projection layout without rounding.

## Browser performance

### Verification responsiveness

Chrome on the user's RTX 3060 Ti exposed two main-thread bottlenecks in
verification: synchronous log append/scroll on every line, and a CPU constraint
de-aliasing reference that repeated analysis for each synthesis point. Logs now
flush at most every 100 ms (plus a final flush), keeping all lines in one text
node. The CPU reference computes each surface/mode projection once, then reuses
the unscaled sums, retaining the original compensated accumulation and synthesis
order. No GPU shader or solver arithmetic changed.

The same default `?mode=test` route, with Chrome CPU sampling and frame/long-task
observation enabled in both captures, measured:

| Measurement | Before | Batched logs + cached reference + worker |
| --- | ---: | ---: |
| Completion time | 99.6 s | 17.5 s |
| Renderer layout time | 42.50 s | 0.324 s |
| Renderer layouts | 3685 | 117 |
| Main-thread tasks over 50 ms | 17 | 0 |
| Worst main-thread task | 14.4 s | None over 50 ms |

These are individual local captures, not cross-device guarantees. The page had
no Wasm heap in the worker capture, and no animation-frame gaps over 50 ms were
observed. The unchanged 3684 printed lines were retained. All 3677 PASS lines
match the earlier conformance output, and the worker/main-thread controller
traces match exactly across 329 records (327 effective Solovev iterations).
The output checks retain the same 118736-byte result; verification no longer
exposes that test artifact as a download.
Both editor precision modes were rechecked: 507 paired and 73 scalar iterations,
with the same final residuals and boundary retained across precision switches.

Reproduce the worker/main-thread numerical comparison on the exposed Chrome:

```bash
node scripts/webgpu_validate_run.mjs \
  'http://localhost:6969/magnetic-equilibrium-solver/tmp/cumes-build-webgpu-ds/webgpu/cumes_webgpu.html?mode=test&worker=0&trace=1' \
  ../tmp/verification-main
node scripts/webgpu_validate_run.mjs \
  'http://localhost:6969/magnetic-equilibrium-solver/tmp/cumes-build-webgpu-ds/webgpu/cumes_webgpu.html?mode=test&trace=1' \
  ../tmp/verification-worker ../tmp/verification-main-trace.json
```

Local profiling evidence: `../tmp/verification-lag-{result,cpu}.json` (before)
and `../tmp/verification-lag-worker-{result,cpu}.json` (after). Node tests cover
batched log ordering/final flush, DOM-free worker messages, transferred output
ownership, query propagation, runtime selection, errors, and worker cleanup.

### W7-X shader performance history

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

The later [causal convergence audit](webgpu-fft-convergence-audit.md) includes
a reproducible one-iteration FFT intervention (2812 → 3100 iterations), sham
controls, late route switches, and identical-input 113-bit projection oracles.
It establishes sensitivity experimentally rather than attributing iteration
counts to transform accuracy alone.

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

### One readback mapping per production fixed-boundary iteration

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
paths. Operator conformance retains its individual readbacks; production
axisymmetric and 3-D solves share the batched pipeline in both precisions.

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

## Persistent iteration state and GPU residual norms (2026-09-06)

The next two residency changes remove **99.5% of the remaining warmed
uploads**, from 4,373,773 to **19,933 bytes/iteration** on the same Chrome /
RTX 3060 Ti, ns=99 W7-X case:

- `9357194`: accepted constraint references and preconditioner caches have
  independent GPU snapshots. A speculative refresh can overwrite operator
  scratch without corrupting the accepted version on a rejected Jacobian.
- `0b3906f`: descent consumes device state, velocity and preconditioned
  directions. Paired state axis extrapolation matches the existing host
  operation, and an f32 direction's correction plane is cleared on-device.
  Checkpoint restores still use the host snapshot; the reduction above is
  for warmed steady-state passes, not initialization/restarts.

All 2817 direct W7-X controller records, including state/low-word hashes,
residuals and restart decisions, matched the prior CPU-controller trajectory
exactly. The profiled full run took **95.6 s** and 2812 effective iterations,
reaching the unchanged residual triple
`(9.985159710508547e-13, 2.1300946362442957e-13, 1.9552385266042975e-13)`.
These wall times are observations, not a new controlled A/B speedup claim.

`893d02b` adds deterministic two-dispatch paired GPU residual norms. Add
`&gpu_norms=shadow` to compare every reduction against the host double
reference without changing controller inputs. The FFT shadow run's maximum
relative difference was **3.158034e-14**. Shadow mode retained every direct
and FFT controller record exactly. `norm-shadow` diagnostics retain the
maximum error independently of the UI's progress-log filtering.

Add **`&gpu_norms=1`** to use GPU norms and omit the decomposed-residual vector
readbacks/CPU reductions. This remains opt-in; omission or `gpu_norms=0`
retains the reference CPU reductions. Qualification results:

| Path | Observed time | Effective iterations | Qualification |
| --- | ---: | ---: | --- |
| Direct, GPU norms (profiled) | 92.3 s | 2812 | All state hashes, restart decisions and f32 shader control parameters unchanged |
| FFT, GPU norms | 104.0 s | 3091 | Same state/decision gate against the qualified FFT trajectory |

Both paths reach `1e-12`. Direct normalized residual differences were at most
`2.679e-14` relative. Full host-double telemetry is not bit-identical when
GPU reductions are authoritative; this is not a Class-A reduction change.
The direct profile measured **19,981 upload bytes** and **35.42 MB readback**
per iteration (down from 36.90 MB). The reduction result retains a validity
flag: nonfinite inputs, overflow and complete squared-norm underflow must not
be classified as convergence. Original forward-input validity checks remain.

Browser conformance covers device-only descent with unaligned source planes,
paired/f32 directions, axis remapping, snapshot isolation, and residual norms
with awkward sizes, zero inputs, both LCFS policies and invalid/range cases.
`scripts/webgpu_validate_run.mjs URL PREFIX BASELINE_TRACE` requires exact
controller records; its explicit `paired-reductions` comparison instead
requires identical state hashes, integer decisions and f32 control parameters
plus a `2e-12` relative residual bound. It never treats a mere iteration-count
match as trajectory equivalence.

**The controller is still on the CPU.** GPU geometry/validity reductions and
force normalization, persistent GPU checkpoint/rollback, shader damping and
restart control, then bounded multi-iteration dispatch batches remain to be
implemented. Most full-field readbacks and the per-iteration host fence have
not yet been removed. The scalar radial/profile/parameter uploads also remain.

Evidence: `../tmp/w7x-resident-caches-*`, `w7x-resident-descent-*`,
`resident-descent-conformance-*`, `w7x-gpu-norm-shadow-*`, `w7x-gpu-norm-fft-*`,
`gpu-norm-conformance-*`, `w7x-device-norms-*` and `w7x-device-norms-fft-*`.

### Compact inverse readbacks (2026-09-06)

Resident production iterations now scan the inverse geometry's high words on
GPU instead of downloading all 20 high/low planes. The integer exponent scan
preserves the existing finite gate, including NaN and infinity rejection,
without changing transform arithmetic. It returns one flag per 256 values.
Geometry is downloaded once at final convergence for derived-field output;
the controller and physics are not advanced to construct that snapshot.
`field_readbacks=full` restores the full-field diagnostic path.

On the same Chrome/RTX 3060 Ti, single-grid paired W7-X with `gpu_norms=1`
completed in **71.27 s**, versus the preceding **91.92 s** observation. All
2,817 controller records (including raw control values and state hashes) were
exactly identical. This is an observed whole-run comparison, not a controlled
multi-trial benchmark. SHA-256 hashes of both the spectral payload and all 13
derived fields were unchanged; `scripts/webgpu_output_digest.js` computes
these independently of revision/provenance metadata in retained result tabs.

The full browser conformance suite passed (90.74 s), including twelve finite
scan cases covering unaligned binding offsets, partial blocks, signed zeros,
subnormals, maximal finite values, NaNs/infinities and malformed ranges.
Evidence: `../tmp/w7x-compact-inverse-*`, `compact-inverse-conformance-*`.

### Compact magnetic readbacks (2026-09-06)

Between preconditioner refreshes, magnetic fields now stay on device too. A
bitwise finite scan checks every high/low field and radial profile; only the
small `chip_h`/`iota_h` profile vectors and scan flags return to the CPU. Full
fields still return on refresh passes for the unchanged CPU force-normalization
reduction. Final output downloads any missing fields in one additional batch,
without advancing the controller. The `field_readbacks=full` diagnostic switch
disables both inverse and magnetic compaction; the optimization is otherwise
enabled for resident production solves, independent of `gpu_norms`.

The direct GPU-norm W7-X run completed in **64.79 s**, with all 2,817 controller
records exactly matching the prior GPU-norm baseline, including both state
hashes, residual triples and control parameters. Spectral and derived-field
SHA-256 digests also match. A sequential 256-iteration GPU timestamp window
and separate 10-second CPU profile measured:

| Measurement | Before field compaction | After |
| --- | ---: | ---: |
| Readback bytes/iteration | 35.42 MB | 14.29 MB |
| Mean iteration interval | 33.00 ms | 22.46 ms |
| Compute shader sum | 8.727 ms | 8.987 ms |
| GPU batch span | 13.314 ms | 12.014 ms |
| Between GPU batches | 19.708 ms | 10.440 ms |
| Map wait (overlaps GPU execution) | 17.110 ms | 13.226 ms |

The new finite scans together cost **0.035 ms/iteration**. Shader arithmetic
and the costly direct forward projections/current solve remain unchanged.
The total scan-parameter upload is only 31 bytes/iteration amortized. Base
geometry/Jacobian checks still download full fields, and GPU checkpointing,
controller logic and multi-iteration batching remain follow-on work.
Evidence: `../tmp/w7x-compact-fields-{gpu,cpu,windows,result,trace,summary}.json`.

An uninstrumented, visible A/B/B/A benchmark used the **same build**, toggling
only `field_readbacks=full` with `gpu_norms=1` held constant. Each run warmed
250 controller records then measured 256 intervals:

| Run | Mean iteration time |
| --- | ---: |
| Full readbacks A1 | 34.827 ms |
| Compact readbacks B1 | 23.441 ms |
| Compact readbacks B2 | 23.295 ms |
| Full readbacks A2 | 33.805 ms |

The two-run averages are **34.316 → 23.368 ms**, a **31.90% reduction**.
All sampled controller records match exactly. The benchmark now preserves
feature flags from its input URLs instead of silently dropping them.
`../tmp/w7x-compact-fields-abba.json` contains the samples and trajectories.
The complete conformance suite passed in 90.74 s, including dedicated f32 and
paired comparisons of compact radial profiles against full readbacks
(`../tmp/compact-fields-conformance-*`).

The FFT route also passed: **73.98 s**, 3,091 effective iterations and all
3,096 controller records exactly equal to its qualified GPU-norm baseline.
Its spectral and derived-field digests are unchanged too
(`../tmp/w7x-compact-fields-fft-*`). These changes accelerate each route
without trying to make the direct and FFT convergence trajectories equal.

### Shader Jacobian control (2026-09-06)

`gpu_control=jacobian` opts resident production solves into the first
shader-controlled acceptance gate. A two-pass reduction scans the ten base
geometry fields, checks the existing high-Jacobian/low-word/axisymmetric
validity guards, and classifies the oriented-Jacobian restart predicate. It
returns a **32-byte status record**, including min/max pairs, the earliest
minimum index, guard flags and the restart verdict. The relative threshold
comes from the host policy constant; the existing browser axis-exemption
cutoff is preserved. No transform or force arithmetic changes.

Ordinary passes retain base geometry on device. Refresh passes still download
it for unchanged CPU force normalization and check the reduced statistics
against the full CPU scan. `field_readbacks=full&gpu_control=jacobian` retains
this full comparison on every pass. The compact scalar predicate is checked
on every GPU-controlled pass before its verdict is applied; disagreement is
an error, never silent acceptance. Accepted final fields are downloaded once
for publication without advancing physics or the controller.

Near-threshold, underflow/overflow and ambiguous earliest-index comparisons
fall back to the original full host gate. Only ambiguity in the surviving
minimum propagates; maximum-value ordering is monotone under double rounding
and needs no tie-index fallback. A fallback can add a second map to that
pass. Completion diagnostics report GPU-controlled and fallback pass counts.

This is **partial controller migration**, not a shader-resident iteration
loop: damping/log history, residual convergence classification, restart
counters and checkpoint restore still run on the CPU, and there is still a
host fence per iteration. Moving those requires persistent device controller
and checkpoint state before multi-iteration dispatch can be enabled safely.

Host controller tests verify exact restart bookkeeping for a device verdict.
Browser conformance includes 16 geometry-control cases: f32/paired data,
unaligned plane offsets, partial blocks, first/interior sign flips, zero and
nonfinite guards, axis exemption, low-word minima, ambiguous ties, threshold
and subnormal fallback, and malformed buffer ranges.

Chrome/RTX 3060 Ti qualification: the refined direct GPU-norm run reached
`1e-12` in **54.92 s**, with all 2,817 controller records and scientific-output
SHA-256 digests identical to the preceding compact-field baseline. All 2,821
attempted passes used the GPU gate (zero fallbacks). The 256-iteration profile
measured **6.15 MB readbacks/pass**, down from 14.29 MB, and **0.0205 ms/pass**
for the two new shaders. Mean profiled interval was 20.76 ms (previous capture
22.46 ms); whole-run timing is observational, not a controlled speedup claim.
The host controller regression and full browser conformance suite passed.
Evidence: `../tmp/w7x-gpu-jacobian-refined-*`, `gpu-jacobian-conformance-*`.

A visible, uninstrumented same-build A/B/B/A comparison (GPU norms enabled,
only `gpu_control=jacobian` toggled) measured host-gate runs of 23.791/22.207
ms and shader-gate runs of 16.986/17.718 ms per warmed iteration. The averages
are **22.999 → 17.352 ms**, a **24.55% reduction**, with exact sampled
controller-record equality (`../tmp/w7x-gpu-jacobian-abba.json`). Timestamp
instrumentation adds overhead; these values should not be mixed with the
profiled 20.76 ms interval when computing a speedup.

An additional full-readback run compared GPU min/max values, earliest indices
and decisions with the CPU scan on **every pass**, not only refreshes: all
2,821 passed, with zero fallbacks and exact 2,817-record trajectory equality
(`../tmp/w7x-gpu-jacobian-full-check-*`). The host controller suite also passed
with ASan/UBSan enabled.

The FFT route qualified in **58.00 s**, retaining 3,091 effective iterations,
all 3,096 exact controller records and the original spectral/derived-field
digests (`../tmp/w7x-gpu-jacobian-fft-*`).

## Compact validation and retained velocity (2026-09-06)

The resident route now keeps descent velocity and constraint intermediates on
device. Only the radial constraint coefficient vector is downloaded; accepted
constraint-reference snapshots remain independent of speculative scratch.
Finite scans replace velocity/intermediate downloads. Original spectral-force
validation uses a compact finite/nonzero flag instead of downloading both
source arrays. Integer magnitude tests distinguish signed zero from subnormal
nonzero values even on flush-to-zero GPUs. Copy-only input buffers retain the
reference download path. `field_readbacks=full` retains velocity/constraint
snapshots for diagnostics.

Chrome/RTX 3060 Ti: single-grid direct W7-X with GPU norms and the GPU Jacobian
gate converged in **42.74 s**, with all **2,817 controller records** and both
scientific-output SHA-256 digests exactly unchanged. Full browser conformance
passed, including compact velocity versus full device snapshots, copy-only
and storage-backed residual validation, and signed-zero/subnormal flag cases.
An uninstrumented visible A/B/B/A comparison measured **17.326 → 14.767 ms**
per warmed iteration (**14.77% lower**), with exact sampled trajectories.
Evidence: `../tmp/w7x-compact-control-*`, `compact-control-conformance-*`.

The follow-up 256-pass profile measured **1.692 MB/readback per iteration**,
down from 6.151 MB. The new validity scans cost 0.063 ms/pass in total;
Emdawn mapped-data copying dropped to 0.84% of the separate CPU sample.
Evidence: `../tmp/w7x-compact-control-profile-*`.

## Parallel current integrands with ordered sums (2026-09-06)

The paired prescribed-current path evaluates its expensive angular integrands
in the existing point-parallel magnetic pass. Two otherwise-unused field
planes hold the exact terms until field finalization overwrites them. The
surface reduction retains the original zeta/theta summation order, including
the invalid-Jacobian skip; raw term words are loaded without an additional
normalization. One workgroup per surface distributes those ordered sums
across the GPU instead of packing 64 surfaces into each of only two W7-X
workgroups. No reduction tree or tolerance change is introduced.

The serial-current kernel fell from **1.689 to 0.380 ms/iteration** in separate
256-pass captures. A visible uninstrumented A/B/B/A test measured
**14.232 → 12.817 ms/iteration (9.94% lower)** with exact sampled controller
records. The complete direct W7-X solve took **38.92 s**, preserving all
2,817 controller records, `1e-12` convergence and both scientific-output
digests. Evidence: `../tmp/w7x-parallel-current-*`.

A theta-contiguous invocation mapping was also tested for direct toroidal
projections. It retained the sampled trajectory but showed no useful speedup:
12.901 ms baseline versus 12.947 ms candidate in A/B/B/A. The experiment was
reverted (`../tmp/w7x-forward-layout-abba.json`). The profiled remaining device
leaders are forward projections (3.63 ms), inverse transforms (1.48 ms), and
preconditioner application (0.61 ms); this is not a claim of global optimality.

## Standalone FFT optimization integrated into W7-X (2026-09-06)

The browser build now uses `webgpu-fft` revision `c985220`, including
whole-butterfly ownership for the paired N=36 transform. No solver physics,
controller settings, transform grid or tolerance changed. The larger-length
FFT improvements are also available but do not apply to this N=36 case.

Fresh visible Chrome/RTX 3060 Ti full solves, in FFT/direct/direct/FFT order,
used `solve=w7x&trace=1&gpu_norms=1&gpu_control=jacobian&timing=0` with
`fft=1` or `fft=0`. Profiling timestamps were disabled; controller tracing
was enabled identically for both routes. Times include startup and output:

| Route | Full solve seconds | Mean seconds | Effective iterations |
| --- | --- | ---: | ---: |
| New optimized FFT | 41.094, 41.566 | 41.330 | 3091 |
| Direct projection | 37.775, 37.771 | 37.773 | 2812 |

FFT remains **9.4% slower end-to-end** in this comparison. Mean intervals
between recorded controller entries were nearly equal: 12.892 ms FFT versus
12.921 ms direct. FFT still requires 9.9% more effective iterations, so a
faster standalone FFT does not establish a faster complete equilibrium solve.
Direct projection remains the default.

Both FFT runs exactly matched all 3096 controller records in the previously
qualified FFT trace; both direct runs matched their own 2817-record trace.
Final FFT residuals were `(9.988e-13, 2.079e-13, 1.726e-13)`; direct retained
`(9.985e-13, 2.130e-13, 1.955e-13)`. The old/new FFT scientific-output SHA-256
digests were identical (provenance excluded):

```text
state  06df977e16ba76b510af822498b29574630d111a7dfb1b235176d0797b3fbb5e
fields 7bc4e3e710c5036e09bec954bbfcf66be8ac467830b5f6e55262a3d9cc656259
```

The preserved old build's first full solve took 62.975 s, but its first
controller record appeared only at 18.469 s and its warmed controller sample
was 13.810 ms/iteration. Do not use that cold full-run outlier against the new
runs to claim a kernel speedup. A separate warmed old/new/new/old comparison
(256 controller intervals per run) measured **13.870 → 12.698 ms/iteration**,
an **8.45% reduction** in FFT-route iteration time, with exact sampled
controller records. Evidence: `../tmp/w7x-fft-owner-abba.json` and
`../tmp/w7x-fft-owner-{old,new-1,new-2,direct-1,direct-2}-{result,trace}.json`.

Full browser conformance with `mode=test&fft=1&gpu_norms=1&gpu_control=jacobian`
passed, including all three Solovev stages with timing instrumentation
enabled (`../tmp/fft-owner-integration-conformance-*`). Artifact CTest and
the frontend iteration/timestamp tests also passed. The served
`../tmp/cumes-build-webgpu-ds/webgpu/cumes_webgpu.html` was rebuilt with this
dependency revision; `fft=1` selects it without changing the direct default.

## Iteration statistics in the webpage log (2026-09-06)

Production runs append minimum, maximum, median and average iteration times
to the end of the solver log, with sample counts. This includes all attempted
passes (cold compilation, refreshes and rejected candidates), not only the
effective iteration counter. Startup before the first pass and final output
publication are excluded.

- **Wall / iteration:** the host-observed iteration interval.
- **Host non-wait elapsed:** JavaScript/Wasm orchestration and callback
  scheduling, excluding time awaiting `mapAsync`; not sampled CPU execution.
- **Readback wait:** elapsed mapping wait, which overlaps GPU execution.
- **Device compute:** sum of compute-pass GPU timestamp intervals, excluding
  GPU copies and gaps between passes. It is not added to host/wait time.

`timestamp-query` is requested when the adapter supports it. Query results
share each existing mapping's reserved tail: no extra map or host fence.
Unavailable or incomplete device samples are reported as unavailable, never
estimated from wall time. Statistics, raw samples and availability are also
accessible through `cumesIterationTiming.report()` in the browser console.
All frontend instrumentation lives in `webgpu/iteration_timing.js`, with only
iteration-boundary notifications in C++; no `EM_JS` is used.

Use `&timing=0` for uninstrumented throughput measurements. The deep-profile
and A/B harnesses set this themselves to avoid stacking timestamp hooks.
`node scripts/test_webgpu_iteration_timing.mjs` tests exact host/wait timing,
timestamp decoding, median calculation, unsupported adapters, reset and opt-out.
The Chrome GPU run retained all 2,817 controller records and produced timing
samples for all 2,821 passes (`../tmp/w7x-iteration-timing-*`).
Conformance exposed and now guards against the reserved timestamp tail
enlarging the logical payload capacity. `ReadbackBatch` enforces its declared
payload budget independently of physical buffer size. Full conformance passed
with instrumentation enabled (`../tmp/timing-capacity-conformance-*`).

The WebGPU implementation lives under these paths:

```text
include/cumes/webgpu/          public WebGPU operator contracts
src/webgpu/                    emdawnwebgpu host implementation
src/webgpu/shaders/            WGSL kernels and shared precision templates
webgpu/                        Emscripten target, browser bridge, and webapp
```

WebGPU code does not include CUDA compatibility shims. Buffers are WebGPU
objects rather than emulated pointers, work is recorded into command encoders,
and completion is callback-driven. This makes synchronization and ownership
visible instead of attempting to reproduce CUDA stream behavior through a
source-level macro layer.

Numerical shader families use shared `.wgsl.in` templates. CMake runs a small
JavaScript preprocessor before embedding the generated WGSL: `Real` types,
arithmetic intrinsics, split constants, and optional low-word bindings select
scalar or pair-single precision. Explicit `Pair` operations retain compensated
geometry and norm accumulation inside scalar solves. The evaluation order is
shared across precisions; scalar rounding trajectories can therefore differ
from the earlier separate shaders. See
[`src/webgpu/shaders/README.md`](../src/webgpu/shaders/README.md) for syntax and
[ADR-0016](adr/0016-webgpu-precision-templates.md) for numerical qualification.

## Capability boundary and future optimization

The following are follow-on optimizations or optional backend expansions, not
completion gates for the fixed-boundary WebGPU port:

1. reduce the remaining host validation/reduction payload and combine the
   individual operator submissions (production 3-D operator dependencies now
   stay on device, with one batched mapping per evaluated pass);
2. reduce final vacuum readbacks by moving the remaining LCFS coupling to GPU;
   `vacuum=webgpu` already executes the vacuum kernel pipeline on WebGPU.

The [cuMES 1.5 optimization assessment](webgpu-optimization-assessment.md)
records the implemented weighted scalar axisymmetric cache, vacuum kernel backend,
and separately qualified opt-in Newton–GMRES and scalar m=1 geometry compensation.
Use `newton=1&precision=double` for the fixed-axisymmetric experiment, or
`geometry=compensated-m1&precision=float` for scalar 3-D geometry.

`deps/vacuum-field` provides WebGPU and HOST backends for browser free-boundary
solves. The CUDA and HOST builds share C++ kernel bodies; WebGPU uses corresponding
WGSL kernels with independent reference gates. NetCDF/HDF5 and the
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
