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
    expression: `(() => {
      if (document.body.dataset.cumesWebgpu !== 'pass') throw Error('Solve has not passed');
      const mesh = document.getElementById('legacy-result-3d').cumesOrbit.mesh;
      return {...mesh, surfaces: mesh.surfaces.map(s => ({...s, points: [...s.points]}))};
    })()`, returnByValue: true});
  const mesh = result.value, edge = mesh.surfaces.at(-1).points, axis = mesh.surfaces[0].points;
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
        const i = (p * row + t) * 3 + c;
        edgeError = Math.max(edgeError, Math.abs(edge[i] - expected[c]));
        axisSpread = Math.max(axisSpread, Math.abs(axis[i] - axis[p * row * 3 + c]));
      }
    }
  }
  if (!(edgeError < 3e-6 && axisSpread < 3e-6))
    throw Error(`Rendered geometry mismatch: LCFS=${edgeError}, axis=${axisSpread}`);
  console.log(JSON.stringify({result: 'PASS', target, edgeError, axisSpread}));
  if (screenshot) {
    await call('Runtime.evaluate', {expression: `document.getElementById('legacy-viewer').scrollIntoView()`});
    const capture = await call('Page.captureScreenshot', {format: 'png'});
    await writeFile(screenshot, Buffer.from(capture.data, 'base64'));
  }
} finally { ws.close(); }
