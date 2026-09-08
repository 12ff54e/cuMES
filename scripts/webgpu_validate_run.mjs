// Run a browser correctness gate in a new tab with tab-scoped test settings.
// Usage: node scripts/webgpu_validate_run.mjs APP_URL OUTPUT_PREFIX [BASELINE_TRACE]
// APP_URL chooses the solve/conformance mode. Never runs concurrent GPU solves.
import {readFile, writeFile} from 'node:fs/promises';
import {isDeepStrictEqual} from 'node:util';
import assert from 'node:assert/strict';

const [url, prefix, baselinePath, comparison = 'exact'] = process.argv.slice(2);
if (!url || !prefix) throw Error('Pass APP_URL and OUTPUT_PREFIX');
if (!['exact', 'paired-reductions'].includes(comparison))
  throw Error('Comparison must be exact or paired-reductions');
const base = 'http://127.0.0.1:' + (process.env.CUMES_CDP_PORT || '9333');
const browser = await (await fetch(`${base}/json/version`)).json();
const ws = new WebSocket(browser.webSocketDebuggerUrl);
let session, page;
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
    ws.send(JSON.stringify({id, method, params, sessionId: method.startsWith('Target.') ? undefined : session}));
  });
}
const evaluate = async expression => (await call('Runtime.evaluate', {
  expression, returnByValue: true})).result.value;
let finished = false;
try {
  page = {id: (await call('Target.createTarget', {url: 'about:blank', newWindow: false})).targetId};
  session = (await call('Target.attachToTarget', {targetId: page.id, flatten: true})).sessionId;
  await call('Page.enable');
  await call('Page.addScriptToEvaluateOnNewDocument', {source:
    `Object.defineProperty(window, 'localStorage', {get: () => window.sessionStorage});`});
  await call('Page.navigate', {url});
  await call('Page.bringToFront');
  console.log(JSON.stringify({target: page.id, url}));
  const deadline = Date.now() + 600000;
  while (Date.now() < deadline) {
    const status = await evaluate(`(()=>{
      if(document.body?.dataset.cumesExecution==='idle')(document.getElementById('w7x-actions')?.hidden===false?document.getElementById('w7x-start'):document.getElementById('run'))?.click();
      return document.body?.dataset.cumesWebgpu;
    })()`);
    if (status === 'pass' || status === 'fail') {
      const result = await evaluate(`({dataset: {...document.body.dataset},
        plot: window.cumesResidualPlot?.report(),
        log: window.cumesVerificationLog?.text() || document.getElementById('log')?.textContent || document.body.innerText})`);
      const trace = await evaluate('window.cumesDiagnostics || []');
      await writeFile(`${prefix}-result.json`, JSON.stringify(result));
      await writeFile(`${prefix}-trace.json`, JSON.stringify(trace));
      if (status === 'fail') throw Error(result.dataset.cumesDetail);
      if (result.plot) {
        const samples = result.plot.samples, states = trace.filter(row => row.kind === 'controller');
        assert.ok(samples.length > 0);
        assert.equal(samples.at(-1).converged, true);
        assert.ok(samples.at(-1).fsq.every(value => value < samples.at(-1).tolerance));
        if (states.length) {
          assert.equal(samples.length, states.length);
          const expectedRestarts = [];
          let stage = 0, previousAttempt = 0;
          samples.forEach((row, i) => {
            assert.deepEqual(row.fsq, states[i].fsq);
            assert.equal(row.attempt, states[i].attempt);
            assert.equal(row.iteration, states[i].iter);
            if (row.stage !== stage) { stage = row.stage; previousAttempt = 0; }
            // Early rejected passes have no residual; post-descent restores
            // carry a nonzero decision in the independent controller trace.
            for (let attempt = previousAttempt + 1; attempt < row.attempt; ++attempt)
              expectedRestarts.push({stage, attempt});
            if (states[i].restart || states[i].vacuum_restart) expectedRestarts.push({stage, attempt: row.attempt});
            previousAttempt = row.attempt;
          });
          assert.deepEqual(result.plot.restarts.map(({stage, attempt}) => ({stage, attempt})), expectedRestarts);
        }
        console.log(`Residual plot: PASS (${samples.length} samples; ${result.plot.restarts.length} restart markers; ${result.plot.drawMilliseconds.toFixed(1)} ms drawing)`);
      }
      if (new URL(url).searchParams.get('boundary') === 'free') {
        assert.equal(result.dataset.cumesExecution, 'worker');
        const states = trace.filter(row => row.kind === 'controller');
        if (states.length) {
          assert.ok(states.some(row => row.vacuum_restart), 'vacuum activation must request a restart');
          assert.equal(states.at(-1).vacuum_state, 2, 'vacuum pressure must be active at convergence');
        }
      }
      if (baselinePath) {
        const baseline = JSON.parse(await readFile(baselinePath, 'utf8'));
        const states = rows => rows.filter(row => row.kind === 'controller')
          .map(({milliseconds, ...state}) => state);
        const expected = states(baseline), actual = states(trace);
        if (!expected.length || expected.length !== actual.length)
          throw Error(`Controller trace length: ${actual.length}/${expected.length}`);
        const equivalent = (a, b) => {
          if (comparison === 'exact') return isDeepStrictEqual(a, b);
          const reduced = row => {
            const {fsq, preconditioned, delta, b1, fac, ...rest} = row;
            return {...rest, delta: Math.fround(delta), b1: Math.fround(b1), fac: Math.fround(fac)};
          };
          const close = (x, y) => Number.isFinite(x) && Number.isFinite(y) &&
            Math.abs(x - y) <= 2e-12 * Math.max(Math.abs(x), Math.abs(y));
          return isDeepStrictEqual(reduced(a), reduced(b)) &&
            ['fsq', 'preconditioned'].every(key => a[key].length === b[key].length &&
              a[key].every((value, i) => close(value, b[key][i])));
        };
        const mismatch = actual.findIndex((row, i) => !equivalent(row, expected[i]));
        if (mismatch >= 0) throw Error(`Controller trace mismatch at record ${mismatch}`);
        console.log(`${comparison} controller trace: PASS (${actual.length} records)`);
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
  if (page && (!finished || process.env.CUMES_CLOSE_TEST_TAB === '1'))
    await call('Target.closeTarget', {targetId: page.id});
  ws.close();
}
