// Validate the published W7-X mesh against the input harmonics, not a solver
// residual. Usage: node scripts/webgpu_validate_geometry.mjs TARGET [SCREENSHOT]
import {readFile, writeFile} from 'node:fs/promises';
const [target, screenshot] = process.argv.slice(2);
const spec = JSON.parse(await readFile(new URL('../inputs/w7x.json', import.meta.url)));
const pages = await (await fetch('http://127.0.0.1:9333/json/list')).json();
const page = pages.find(p => p.id === target);
if (!page) throw Error('Pass the ID of a completed W7-X browser tab');
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });
let id = 0;
const requests = new Map();
ws.onmessage = event => {
  const reply = JSON.parse(event.data), request = requests.get(reply.id);
  if (!request) return;
  requests.delete(reply.id); clearTimeout(request.timer);
  if (reply.error || reply.result?.exceptionDetails)
    request.reject(Error(JSON.stringify(reply.error || reply.result.exceptionDetails)));
  else request.resolve(reply.result);
};
function call(method, params = {}) {
  return new Promise((resolve, reject) => {
    const current = ++id;
    const timer = setTimeout(() => reject(Error(`Timeout: ${method}`)), 30000);
    requests.set(current, {resolve, reject, timer});
    ws.send(JSON.stringify({id: current, method, params}));
  });
}
try {
  const {result} = await call('Runtime.evaluate', {
    expression: `(async () => {
      if (document.body.dataset.cumesWebgpu !== 'pass') throw Error('Solve has not passed');
      const canvas = document.getElementById('result-3d'), state = canvas.cumesOrbit;
      await state.ready;
      showResultMode('3d'); drawOrbit(canvas, state);
      const renderer = state.renderer, {device, mesh} = renderer;
      const bytes = mesh.vertices * mesh.surfaces * 16;
      const readback = device.createBuffer({size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
      try {
        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(renderer.surface.positions, 0, readback, 0, bytes);
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        if (canvas.dataset.rendererError) throw Error(canvas.dataset.rendererError);
        return {...mesh, points: [...new Float32Array(readback.getMappedRange())]};
      } finally { readback.unmap(); readback.destroy(); }
    })()`, returnByValue: true, awaitPromise: true});
  const mesh = result.value, edge = mesh.points.slice((mesh.surfaces - 1) * mesh.vertices * 4), axis = mesh.points;
  if (!mesh.points.every(Number.isFinite)) throw Error('Nonfinite rendered surface geometry');
  const row = mesh.thetaSegments + 1;
  let edgeError = 0, axisSpread = 0;
  for (let p = 0; p <= mesh.phiSegments; ++p) {
    const phi = 2 * Math.PI * p / mesh.phiSegments;
    for (let t = 0; t <= mesh.thetaSegments; ++t) {
      const theta = 2 * Math.PI * t / mesh.thetaSegments;
      const r = spec.rbc.reduce((sum, h) => sum + h.value * Math.cos(h.m * theta - h.n * spec.nfp * phi), 0);
      const z = spec.zbs.reduce((sum, h) => sum + h.value * Math.sin(h.m * theta - h.n * spec.nfp * phi), 0);
      const expected = [r * Math.cos(phi), r * Math.sin(phi), z];
      for (let c = 0; c < 3; ++c) {
        const i = (p * row + t) * 4 + c;
        edgeError = Math.max(edgeError, Math.abs(edge[i] - expected[c]));
        axisSpread = Math.max(axisSpread, Math.abs(axis[i] - axis[p * row * 4 + c]));
      }
    }
  }
  // WGSL reconstructs display geometry in f32; this is well below a pixel at
  // the viewer's maximum zoom. The equilibrium solve retains its own precision.
  if (!(edgeError < 2e-5 && axisSpread < 2e-5))
    throw Error(`Rendered geometry mismatch: LCFS=${edgeError}, axis=${axisSpread}`);
  console.log(JSON.stringify({result: 'PASS', target, edgeError, axisSpread}));
  if (screenshot) {
    await call('Runtime.evaluate', {expression: `document.getElementById('result-3d').scrollIntoView()`});
    const capture = await call('Page.captureScreenshot', {format: 'png'});
    await writeFile(screenshot, Buffer.from(capture.data, 'base64'));
  }
} finally { ws.close(); }
