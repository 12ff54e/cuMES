// Rendering owns its device and buffers; it never uses the solver's device.
// Positions and line indices stay resident. Orbiting uploads only the camera.
const ORBIT_RENDER_SHADER = `
struct Camera { rotation: vec4f, viewport: vec4f }
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> radius_bits: array<u32>;
@group(1) @binding(0) var<storage, read> points: array<vec4f>;
@group(1) @binding(1) var<storage, read> indices: array<u32>;

fn project(point: vec3f) -> vec4f {
  let radius = max(1.0, bitcast<f32>(radius_bits[0]));
  let rotation = camera.rotation;
  let x = rotation.x * point.x - rotation.y * point.y;
  let y = rotation.y * point.x + rotation.x * point.y;
  let depth = rotation.z * y - rotation.w * point.z;
  let vertical = rotation.w * y + rotation.z * point.z;
  let distance = 4.5 * radius + depth;
  let scale = min(camera.viewport.x, camera.viewport.y) * 0.86 * camera.viewport.z / radius;
  let xy = vec2f(x, vertical) * scale * (4.5 * radius) / camera.viewport.xy;
  let near = 0.1 * radius;
  let far = 9.0 * radius;
  return vec4f(xy, far / (far - near) * (distance - near), distance);
}

struct Vertex { @builtin(position) position: vec4f, @location(0) color: vec4f }
@vertex fn vertex_main(@builtin(vertex_index) vertex: u32,
                       @builtin(instance_index) segment: u32) -> Vertex {
  let first = points[indices[2u * segment]];
  let second = points[indices[2u * segment + 1u]];
  let a = project(first.xyz);
  let b = project(second.xyz);
  let direction = (b.xy / b.w - a.xy / a.w) * camera.viewport.xy;
  let normal = vec2f(-direction.y, direction.x) / max(length(direction), 1e-6);
  let corner = array<u32, 6>(0u, 1u, 2u, 2u, 1u, 3u)[vertex];
  var position = select(a, b, corner >= 2u);
  let radial = first.w;
  let width = select(0.6 + 0.75 * radial, 2.5, radial < 0.0) * camera.viewport.w;
  position = vec4f(position.xy + normal * select(-1.0, 1.0, (corner & 1u) != 0u) *
                   width / camera.viewport.xy * position.w, position.zw);
  let color = mix(vec3f(0.22, 0.84, 0.88), vec3f(0.52, 0.71, 1.0), radial);
  let alpha = select(0.12 + 0.35 * radial, 0.78, radial >= 1.0);
  return Vertex(position, select(vec4f(color, alpha), vec4f(1.0, 0.714, 0.365, 1.0), radial < 0.0));
}
@fragment fn fragment_main(vertex: Vertex) -> @location(0) vec4f { return vertex.color; }
`;

class CumesOrbitRenderer {
  static async create(canvas) {
    const adapter = await navigator.gpu?.requestAdapter({powerPreference: 'high-performance'});
    if (!adapter) throw Error('WebGPU adapter unavailable');
    const device = await adapter.requestDevice({label: 'cuMES 3D viewer'});
    const renderer = new CumesOrbitRenderer(canvas, device);
    try { await renderer.initialize(); return renderer; }
    catch (error) { renderer.destroy(); throw error; }
  }

  constructor(canvas, device) {
    this.canvas = canvas; this.device = device; this.layers = [];
    this.buffers = new Set(); this.camera = new Float32Array(8);
  }

  buffer(previous, bytes, usage, label) {
    if (bytes > this.device.limits.maxStorageBufferBindingSize) throw Error('3D geometry exceeds the GPU buffer limit');
    if (previous?.size >= bytes) return previous;
    if (previous) { previous.destroy(); this.buffers.delete(previous); }
    const buffer = this.device.createBuffer({size: Math.max(16, bytes), usage, label});
    this.buffers.add(buffer); return buffer;
  }

  async initialize() {
    const device = this.device;
    this.context = this.canvas.getContext('webgpu');
    if (!this.context) throw Error('WebGPU canvas unavailable');
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({device, format: this.format, alphaMode: 'premultiplied'});
    const module = device.createShaderModule({code: ORBIT_RENDER_SHADER, label: 'cuMES orbit projection and lines'});
    this.pipeline = await device.createRenderPipelineAsync({
      label: 'cuMES wireframe', layout: 'auto',
      vertex: {module, entryPoint: 'vertex_main'},
      fragment: {module, entryPoint: 'fragment_main', targets: [{format: this.format, blend: {
        color: {srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha'},
        alpha: {srcFactor: 'one', dstFactor: 'one-minus-src-alpha'}
      }}]},
      primitive: {topology: 'triangle-list'},
      // Transparent wireframes reveal interior surfaces. Future opaque passes
      // can write this depth attachment before the transparent line passes.
      depthStencil: {format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less-equal'},
      multisample: {count: 4}
    });
    this.cameraBuffer = this.buffer(null, 32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'orbit camera');
    this.radiusBuffer = this.buffer(null, 16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, 'orbit bounds');
    this.cameraGroup = device.createBindGroup({layout: this.pipeline.getBindGroupLayout(0), entries: [
      {binding: 0, resource: {buffer: this.cameraBuffer}}, {binding: 1, resource: {buffer: this.radiusBuffer}}
    ]});
    this.surface = {}; this.section = {}; this.layers.push(this.surface, this.section);
  }

  setLines(layer, positions, indices) {
    const previousPositions = layer.positions, previousIndices = layer.indices;
    layer.positions = this.buffer(layer.positions, positions.byteLength,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, 'orbit positions');
    layer.indices = this.buffer(layer.indices, indices.byteLength,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, 'orbit line indices');
    this.device.queue.writeBuffer(layer.positions, 0, positions);
    this.device.queue.writeBuffer(layer.indices, 0, indices);
    layer.count = indices.length / 2;
    if (previousPositions !== layer.positions || previousIndices !== layer.indices)
      layer.group = this.device.createBindGroup({layout: this.pipeline.getBindGroupLayout(1), entries: [
        {binding: 0, resource: {buffer: layer.positions}}, {binding: 1, resource: {buffer: layer.indices}}
      ]});
  }

  setFourier(fourier) {
    const mesh = equilibriumMesh(fourier), vertices = mesh.surfaces[0].points.length / 3;
    const positions = new Float32Array(4 * vertices * mesh.surfaces.length), indices = [];
    let radius = 1;
    mesh.surfaces.forEach((surface, s) => {
      for (let i = 0; i < vertices; i++) {
        positions.set(surface.points.subarray(3 * i, 3 * i + 3), 4 * (s * vertices + i));
        positions[4 * (s * vertices + i) + 3] = surface.radial;
        radius = Math.max(radius, Math.hypot(...surface.points.subarray(3 * i, 3 * i + 3)));
      }
      const row = mesh.thetaSegments + 1, offset = s * vertices;
      for (let p = 0; p < mesh.phiSegments; p += 4)
        for (let t = 0; t < mesh.thetaSegments; t++) indices.push(offset + p * row + t, offset + p * row + t + 1);
      for (let t = 0; t < mesh.thetaSegments; t += 4)
        for (let p = 0; p < mesh.phiSegments; p++) indices.push(offset + p * row + t, offset + (p + 1) * row + t);
    });
    this.setLines(this.surface, positions, new Uint32Array(indices));
    this.device.queue.writeBuffer(this.radiusBuffer, 0, new Float32Array([radius]));
    this.fourier = fourier;
  }

  setSection(section) {
    if (!section?.length) { this.section.count = 0; this.sectionSource = section; return; }
    const count = section.length / 3, positions = new Float32Array(4 * count), indices = new Uint32Array(2 * (count - 1));
    for (let i = 0; i < count; i++) {
      positions.set(section.slice(3 * i, 3 * i + 3), 4 * i); positions[4 * i + 3] = -1;
      if (i + 1 < count) { indices[2 * i] = i; indices[2 * i + 1] = i + 1; }
    }
    this.setLines(this.section, positions, indices); this.sectionSource = section;
  }

  resize(width, height) {
    if (this.color && this.canvas.width === width && this.canvas.height === height) return;
    this.color?.destroy(); this.depth?.destroy();
    this.canvas.width = width; this.canvas.height = height;
    const descriptor = {size: [width, height], sampleCount: 4, usage: GPUTextureUsage.RENDER_ATTACHMENT};
    this.color = this.device.createTexture({...descriptor, format: this.format, label: 'orbit multisampling'});
    this.depth = this.device.createTexture({...descriptor, format: 'depth24plus', label: 'orbit depth'});
    this.colorView = this.color.createView(); this.depthView = this.depth.createView();
  }

  draw(state) {
    const rect = this.canvas.getBoundingClientRect();
    if (rect.width < 2 || rect.height < 2 || this.canvas.hidden) return;
    if (this.fourier !== state.fourier) this.setFourier(state.fourier);
    if (this.sectionSource !== state.section) this.setSection(state.section);
    const limit = this.device.limits.maxTextureDimension2D;
    const dpr = Math.min(2, devicePixelRatio || 1, limit / rect.width, limit / rect.height);
    const width = Math.max(1, Math.round(rect.width * dpr)), height = Math.max(1, Math.round(rect.height * dpr));
    this.resize(width, height);
    this.camera.set([Math.cos(state.yaw), Math.sin(state.yaw), Math.cos(state.pitch), Math.sin(state.pitch), width, height, state.zoom, dpr]);
    this.device.queue.writeBuffer(this.cameraBuffer, 0, this.camera);
    const encoder = this.device.createCommandEncoder({label: 'orbit frame'});
    const pass = encoder.beginRenderPass({
      colorAttachments: [{view: this.colorView, resolveTarget: this.context.getCurrentTexture().createView(),
        clearValue: {r: 0, g: 0, b: 0, a: 0}, loadOp: 'clear', storeOp: 'discard'}],
      depthStencilAttachment: {view: this.depthView, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'discard'}
    });
    pass.setPipeline(this.pipeline); pass.setBindGroup(0, this.cameraGroup);
    for (const layer of this.layers) if (layer.count) {
      pass.setBindGroup(1, layer.group); pass.draw(6, layer.count);
    }
    pass.end(); this.device.queue.submit([encoder.finish()]);
    this.canvas.dataset.renderer = 'webgpu';
  }

  destroy() {
    for (const buffer of this.buffers) buffer.destroy();
    this.buffers.clear(); this.color?.destroy(); this.depth?.destroy();
    this.context?.unconfigure(); this.device.destroy();
  }
}

function orbitMessage(canvas, message) {
  const hint = canvas.parentElement.querySelector('.orbit-hint');
  if (hint) { hint.textContent = message; hint.hidden = false; }
}

function startOrbitRenderer(canvas, state) {
  orbitMessage(canvas, 'Preparing 3D view…');
  state.ready = CumesOrbitRenderer.create(canvas).then(renderer => {
    if (state.disposed) { renderer.destroy(); return; }
    state.renderer = renderer;
    renderer.device.addEventListener('uncapturederror', event => {
      canvas.dataset.rendererError = event.error.message;
      orbitMessage(canvas, '3D view unavailable. 2D cross-sections remain available.');
    });
    renderer.device.lost.then(() => {
      if (state.disposed || state.renderer !== renderer) return;
      state.renderer = null; renderer.destroy();
      if (state.recoveries++ === 0) startOrbitRenderer(canvas, state);
      else orbitMessage(canvas, '3D device lost. Reload to restore the view.');
    });
    orbitMessage(canvas, 'Drag to orbit · wheel to zoom');
    queueOrbitDraw(canvas);
  }).catch(error => {
    canvas.dataset.rendererError = String(error);
    orbitMessage(canvas, '3D view unavailable. 2D cross-sections remain available.');
  });
}

function installOrbitRenderer(canvas, fourier) {
  if (!canvas || !fourier?.surfaces?.length) return;
  if (!canvas.cumesOrbit) {
    const state = {yaw: -.55, pitch: .55, zoom: 1, drag: null, frame: 0, recoveries: 0};
    canvas.cumesOrbit = state;
    canvas.addEventListener('pointerdown', event => {
      state.drag = {id: event.pointerId, x: event.clientX, y: event.clientY, yaw: state.yaw, pitch: state.pitch};
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener('pointermove', event => {
      if (!state.drag || state.drag.id !== event.pointerId) return;
      state.yaw = state.drag.yaw + (event.clientX - state.drag.x) * .008;
      state.pitch = Math.max(-1.35, Math.min(1.35, state.drag.pitch + (event.clientY - state.drag.y) * .008));
      queueOrbitDraw(canvas);
    });
    const release = event => { if (state.drag?.id === event.pointerId) state.drag = null; };
    canvas.addEventListener('pointerup', release); canvas.addEventListener('pointercancel', release);
    canvas.addEventListener('lostpointercapture', release);
    canvas.addEventListener('wheel', event => {
      state.zoom = Math.max(.45, Math.min(3, state.zoom * Math.exp(-event.deltaY * .001)));
      queueOrbitDraw(canvas); event.preventDefault();
    }, {passive: false});
    const observer = new ResizeObserver(() => queueOrbitDraw(canvas)); observer.observe(canvas);
    addEventListener('pagehide', event => {
      cancelAnimationFrame(state.frame); state.frame = 0;
      if (!event.persisted) { state.disposed = true; observer.disconnect(); state.renderer?.destroy(); }
    });
    addEventListener('pageshow', () => queueOrbitDraw(canvas));
    startOrbitRenderer(canvas, state);
  }
  canvas.cumesOrbit.fourier = fourier;
  queueOrbitDraw(canvas);
}

function queueOrbitDraw(canvas) {
  const state = canvas?.cumesOrbit;
  if (!state || state.frame || state.disposed || canvas.hidden || !state.renderer) return;
  state.frame = requestAnimationFrame(() => { state.frame = 0; drawOrbit(canvas, state); });
}

function drawOrbit(canvas, state) {
  try { state.renderer?.draw(state); }
  catch (error) {
    canvas.dataset.rendererError = String(error);
    orbitMessage(canvas, '3D view unavailable. 2D cross-sections remain available.');
  }
}
