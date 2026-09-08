// Controlled W7-X A/B probe. Uses and closes its own Chrome target, preserving
// the user's result tab. Do not run other GPU work during measurements.
// Usage: node scripts/webgpu_benchmark.mjs APP_URL [CDP_PORT=9333]
import {readFile} from 'node:fs/promises';

const [appUrl, port = '9333'] = process.argv.slice(2);
if (!appUrl) throw Error('Pass the served cumes_webgpu.html URL');
const base = `http://127.0.0.1:${port}`;
const page = await (await fetch(`${base}/json/new?about:blank`, {method: 'PUT'})).json();
const profiler = await readFile(new URL('./webgpu_profile.js', import.meta.url), 'utf8');
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {ws.onopen = resolve; ws.onerror = reject;});
let nextId = 0;
const pending = new Map();
ws.onmessage = event => {
  const reply = JSON.parse(event.data), request = pending.get(reply.id);
  if (!request) return;
  pending.delete(reply.id); clearTimeout(request.timer);
  if (reply.error || reply.result?.exceptionDetails)
    request.reject(Error(JSON.stringify(reply.error || reply.result.exceptionDetails)));
  else request.resolve(reply.result);
};
function call(method, params) {
  return new Promise((resolve, reject) => {
    const id = ++nextId;
    const timer = setTimeout(() => {pending.delete(id); reject(Error(`CDP timeout: ${method}`));}, 60000);
    pending.set(id, {resolve, reject, timer});
    ws.send(JSON.stringify({id, method, params}));
  });
}
const evaluate = expression => call('Runtime.evaluate', {expression, returnByValue: true, awaitPromise: true});
try {
  for (const [resident, fft] of [[0, 0], [0, 1], [1, 0], [1, 1]]) {
    const url = new URL(appUrl);
    url.search = new URLSearchParams({solve: 'w7x', resident, fft}).toString();
    await call('Page.navigate', {url: url.href});
    await call('Page.bringToFront', {});
    // Wait for the new document, then warm the real full-size solver.
    await new Promise(resolve => setTimeout(resolve, 500));
    await evaluate(`new Promise((resolve,reject)=>{const t=setInterval(()=>{
      if(document.body.dataset.cumesExecution==='idle')(document.getElementById('w7x-actions')?.hidden===false?document.getElementById('w7x-start'):document.getElementById('run'))?.click();
      if(document.body.dataset.cumesWebgpu==='fail'){clearInterval(t);reject(Error(document.body.dataset.cumesDetail));}
      else if(document.body.innerText.includes('iter=100')){clearInterval(t);resolve(true);}
    },100)})`);
    await evaluate(profiler);
    const result = await evaluate(`new Promise(resolve=>{const t=setInterval(()=>{
      const p=cumesGpuProfile.report();
      const inverse=p.records.find(r=>r.kind==='map-wait'&&r.label==='cuMES toroidal inverse readback');
      if(inverse&&inverse.calls>=100){clearInterval(t);cumesGpuProfile.stop();resolve({
        seconds:p.seconds,iterations:inverse.calls,iterationsPerSecond:inverse.calls/p.seconds,
        totals:p.records.reduce((o,r)=>{const x=o[r.kind]??={bytes:0,calls:0,milliseconds:0};
          x.bytes+=r.bytes;x.calls+=r.calls;x.milliseconds+=r.milliseconds;return o;},{}),
        waits:p.records.filter(r=>r.kind==='map-wait')});}
    },5)})`);
    console.log(JSON.stringify({resident: Boolean(resident), fft: Boolean(fft), ...result.result.value}));
  }
} finally {
  ws.close();
  await fetch(`${base}/json/close/${page.id}`);
}
