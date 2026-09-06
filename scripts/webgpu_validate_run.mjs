// Run a browser correctness gate in its own tab; preserve a successful result.
// Usage: node scripts/webgpu_validate_run.mjs APP_URL OUTPUT_PREFIX [BASELINE_TRACE]
// APP_URL chooses the solve/conformance mode. Never runs concurrent GPU solves.
import {readFile, writeFile} from 'node:fs/promises';
import {isDeepStrictEqual} from 'node:util';

const [url, prefix, baselinePath] = process.argv.slice(2);
if (!url || !prefix) throw Error('Pass APP_URL and OUTPUT_PREFIX');
const base = 'http://127.0.0.1:9333';
const page = await (await fetch(`${base}/json/new?about:blank`, {method: 'PUT'})).json();
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
function call(method, params = {}) {
  return new Promise((resolve, reject) => {
    const id = ++nextId;
    const timer = setTimeout(() => {
      pending.delete(id); reject(Error(`CDP timeout: ${method}`));
    }, 30000);
    pending.set(id, {resolve, reject, timer});
    ws.send(JSON.stringify({id, method, params}));
  });
}
const evaluate = async expression => (await call('Runtime.evaluate', {
  expression, returnByValue: true})).result.value;
let finished = false;
try {
  await call('Page.enable');
  await call('Page.navigate', {url});
  await call('Page.bringToFront');
  console.log(JSON.stringify({target: page.id, url}));
  const deadline = Date.now() + 600000;
  while (Date.now() < deadline) {
    const status = await evaluate('document.body?.dataset.cumesWebgpu');
    if (status === 'pass' || status === 'fail') {
      const result = await evaluate(`({dataset: {...document.body.dataset},
        log: document.getElementById('log')?.textContent || document.body.innerText})`);
      const trace = await evaluate('window.cumesDiagnostics || []');
      await writeFile(`${prefix}-result.json`, JSON.stringify(result));
      await writeFile(`${prefix}-trace.json`, JSON.stringify(trace));
      if (status === 'fail') throw Error(result.dataset.cumesDetail);
      if (baselinePath) {
        const baseline = JSON.parse(await readFile(baselinePath, 'utf8'));
        const states = rows => rows.filter(row => row.kind === 'controller')
          .map(({milliseconds, ...state}) => state);
        const expected = states(baseline), actual = states(trace);
        if (!expected.length || expected.length !== actual.length)
          throw Error(`Controller trace length: ${actual.length}/${expected.length}`);
        const mismatch = actual.findIndex((row, i) => !isDeepStrictEqual(row, expected[i]));
        if (mismatch >= 0) throw Error(`Controller trace mismatch at record ${mismatch}`);
        console.log(`Exact controller trace: PASS (${actual.length} records)`);
      }
      console.log(JSON.stringify(result.dataset));
      finished = true;
      break;
    }
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
  if (!finished) throw Error('Browser gate timed out');
} finally {
  if (!finished) await call('Page.navigate', {url: 'about:blank'}).catch(() => {});
  ws.close();
}
