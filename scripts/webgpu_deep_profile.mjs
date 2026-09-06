// Usage: node scripts/webgpu_deep_profile.mjs APP_URL OUTPUT_PREFIX [CDP_PORT]
// Creates and preserves its own result tab. Captures 256 warmed GPU iterations,
// then a separate 10-second CPU sample; waits for convergence and saves the trace.
import {readFile, writeFile} from 'node:fs/promises';

const [appUrl, prefix, port = '9333'] = process.argv.slice(2);
if (!appUrl || !prefix) throw Error('Pass APP_URL and OUTPUT_PREFIX');
const base = `http://127.0.0.1:${port}`;
const page = await (await fetch(`${base}/json/new?about:blank`, {method: 'PUT'})).json();
const source = await readFile(new URL('./webgpu_timestamps.js', import.meta.url), 'utf8');
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {ws.onopen = resolve; ws.onerror = reject;});
const pending = new Map();
let nextId = 0;
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
    pending.set(id, {resolve, reject, timer});
    ws.send(JSON.stringify({id, method, params}));
  });
}
const evaluate = async expression => (await call('Runtime.evaluate', {
  expression, returnByValue: true, awaitPromise: true})).result.value;
const save = (suffix, value) => writeFile(`${prefix}-${suffix}.json`, JSON.stringify(value));
const marker = () => evaluate(`({milliseconds: performance.now(),
  controllers: window.cumesDiagnostics?.length || 0, visible: document.visibilityState})`);
let succeeded = false;
async function until(expression, label) {
  const deadline = Date.now() + 600000;
  while (Date.now() < deadline) {
    const status = await evaluate(`({ready: Boolean(${expression}),
      status: document.body?.dataset.cumesWebgpu, detail: document.body?.dataset.cumesDetail})`);
    if (status.status === 'fail') throw Error(status.detail);
    if (status.ready) return;
    if (status.status === 'pass') throw Error(`Run finished before ${label}`);
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
  throw Error(`Timed out waiting for ${label}`);
}
try {
  await call('Page.enable');
  await call('Page.addScriptToEvaluateOnNewDocument', {source});
  const url = new URL(appUrl);
  url.searchParams.set('solve', 'w7x'); url.searchParams.set('trace', '1');
  console.log(JSON.stringify({target: page.id, url: url.href}));
  await call('Page.navigate', {url: url.href});
  await call('Page.bringToFront');
  await until('window.cumesTimestampProfile', 'profiling hook installation');
  await until('(window.cumesDiagnostics?.length || 0) >= 250', 'warmup');
  const gpuStart = await marker();
  await evaluate('cumesTimestampProfile.start(256)');
  console.log('GPU timestamp sample started');
  await until('window.cumesTimestampProfile && !cumesTimestampProfile.report().active', 'GPU sample');
  const gpu = await evaluate('cumesTimestampProfile.report()');
  await save('gpu', gpu);
  const gpuEnd = await marker();
  if (gpu.errors.length || gpu.batches.length !== 256) throw Error(JSON.stringify(gpu.errors));
  console.log(JSON.stringify({batches: gpu.batches.length, gpu: gpu.gpu.slice(0, 8)}));
  await evaluate('cumesTimestampProfile.restore()');
  await call('Profiler.enable');
  await call('Profiler.setSamplingInterval', {interval: 1000});
  const cpuStart = await marker();
  await call('Profiler.start');
  await new Promise(resolve => setTimeout(resolve, 10000));
  const {profile} = await call('Profiler.stop');
  await call('Profiler.disable');
  const cpuEnd = await marker();
  await save('cpu', profile);
  await save('windows', {gpuStart, gpuEnd, cpuStart, cpuEnd});
  console.log('CPU sample saved; waiting for convergence');
  await until("document.body?.dataset.cumesWebgpu === 'pass'", 'convergence');
  const result = await evaluate(`({dataset: {...document.body.dataset},
    log: document.getElementById('log')?.textContent || document.body.innerText,
    userAgent: navigator.userAgent})`);
  await save('result', result);
  await save('trace', await evaluate('window.cumesDiagnostics'));
  succeeded = true;
  console.log(JSON.stringify(result));
} finally {
  // Do not leave a failed diagnostic solve consuming the user's GPU.
  if (!succeeded) await call('Page.navigate', {url: 'about:blank'}).catch(() => {});
  ws.close();
}
