// Uninstrumented warmed W7-X A/B/B/A comparison. Preserves URL feature flags.
// Uses and closes only
// its own test tab; do not run another solve concurrently.
// Usage: node scripts/webgpu_compare_builds.mjs BASE_URL NEW_URL OUTPUT_JSON [PORT]
import {writeFile} from 'node:fs/promises';
const [baseline, candidate, output, port = '9333'] = process.argv.slice(2);
if (!baseline || !candidate || !output) throw Error('Pass both build URLs and an output path');
const base = `http://127.0.0.1:${port}`;
const page = await (await fetch(`${base}/json/new?about:blank`, {method: 'PUT'})).json();
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {ws.onopen = resolve; ws.onerror = reject;});
let nextId = 0;
const pending = new Map(), runs = [];
ws.onmessage = event => {
  const reply = JSON.parse(event.data), request = pending.get(reply.id);
  if (!request) return;
  pending.delete(reply.id); clearTimeout(request.timer);
  if (reply.error || reply.result?.exceptionDetails)
    request.reject(Error(JSON.stringify(reply.error || reply.result.exceptionDetails)));
  else request.resolve(reply.result);
};
function call(method, params = {}) {
  return new Promise((resolve, reject) => {
    const id = ++nextId;
    const timer = setTimeout(() => {pending.delete(id); reject(Error(`CDP timeout: ${method}`));}, 30000);
    pending.set(id, {resolve, reject, timer}); ws.send(JSON.stringify({id, method, params}));
  });
}
const evaluate = async expression => (await call('Runtime.evaluate', {
  expression, returnByValue: true})).result.value;
try {
  await call('Page.enable');
  for (const [kind, source] of [['baseline', baseline], ['candidate', candidate],
    ['candidate', candidate], ['baseline', baseline]]) {
    const url = new URL(source);
    url.searchParams.set('solve', 'w7x');
    url.searchParams.set('trace', '1');
    if (!url.searchParams.has('fft')) url.searchParams.set('fft', '0');
    await call('Page.navigate', {url: 'about:blank'});
    await call('Page.navigate', {url: url.href});
    await call('Page.bringToFront');
    const deadline = Date.now() + 180000;
    let ready = false;
    while (Date.now() < deadline) {
      const state = await evaluate(`({count: window.cumesDiagnostics?.length || 0,
        status: document.body?.dataset.cumesWebgpu, detail: document.body?.dataset.cumesDetail})`);
      if (state.status === 'fail') throw Error(state.detail);
      if (state.count >= 507) { ready = true; break; }
      await new Promise(resolve => setTimeout(resolve, 250));
    }
    if (!ready) throw Error('Warmup/sample timeout');
    const trace = await evaluate('window.cumesDiagnostics.filter(r => r.kind === "controller").slice(250, 507)');
    const visibility = await evaluate('document.visibilityState');
    if (trace.length !== 257 || visibility !== 'visible') throw Error('Incomplete or hidden sample');
    const states = trace.map(({milliseconds, ...state}) => state);
    if (runs.length && JSON.stringify(states) !== JSON.stringify(runs[0].states))
      throw Error('A/B trajectory mismatch');
    const intervals = trace.slice(1).map((row, i) => row.milliseconds - trace[i].milliseconds).sort((a, b) => a - b);
    const result = {kind, url: url.href, millisecondsPerIteration:
      (trace.at(-1).milliseconds - trace[0].milliseconds) / 256,
      median: intervals[128], p95: intervals[243], states};
    runs.push(result);
    await writeFile(output, JSON.stringify(runs));
    console.log(JSON.stringify({...result, states: undefined}));
  }
} finally {
  ws.close();
  await fetch(`${base}/json/close/${page.id}`);
}
