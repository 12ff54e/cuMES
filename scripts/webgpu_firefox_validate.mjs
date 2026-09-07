// Headless Firefox gate using a local geckodriver WebDriver endpoint.
// Usage: node scripts/webgpu_firefox_validate.mjs APP_URL OUTPUT_PREFIX [PREFS_JSON]
// Preferences apply only to geckodriver's disposable profile. GPU solves run serially.
import {mkdir, writeFile} from 'node:fs/promises';
import {dirname} from 'node:path';

const [url, prefix, prefsJson = '{}'] = process.argv.slice(2);
if (!url || !prefix) throw Error('Pass APP_URL OUTPUT_PREFIX [PREFS_JSON]');
const prefs = JSON.parse(prefsJson);
const endpoint = process.env.GECKODRIVER_URL || 'http://127.0.0.1:4445';
const timeout = Number(process.env.CUMES_FIREFOX_TIMEOUT_MS || 1800000);
if (!Number.isFinite(timeout) || timeout <= 0) throw Error('Invalid Firefox timeout');
await mkdir(dirname(prefix), {recursive: true});
async function request(path, body, method = 'POST') {
  const response = await fetch(endpoint + path, {
    method, headers: {'Content-Type': 'application/json'},
    ...(body === undefined ? {} : {body: JSON.stringify(body)}),
    signal: AbortSignal.timeout(60000)
  });
  const data = await response.json();
  if (!response.ok || data.value?.error) throw Error(JSON.stringify(data.value));
  return data.value;
}
const session = await request('/session', {capabilities: {alwaysMatch: {
  browserName: 'firefox', 'moz:firefoxOptions': {args: ['-headless'], prefs},
  timeouts: {pageLoad: 60000, script: 30000}
}}});
const base = '/session/' + session.sessionId;
const execute = script => request(base + '/execute/sync', {script, args: []});
const result = {url, prefs, capabilities: session.capabilities};
const save = () => writeFile(prefix + '-result.json', JSON.stringify(result, null, 2));
let interrupted = false;
for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => {interrupted = true;});
try {
  // This route stays idle until Start is clicked, so the probe starts no solver.
  const idle = new URL(url); idle.search = '?solve=w7x';
  await request(base + '/url', {url: idle.href});
  result.probe = await request(base + '/execute/async', {script: `
    const done = arguments[arguments.length - 1];
    (async () => {
      const adapter = await navigator.gpu?.requestAdapter({powerPreference: 'high-performance'});
      return {gpu: !!navigator.gpu, secure: isSecureContext, isolated: crossOriginIsolated,
        adapter: adapter ? {
          info: Object.fromEntries(['vendor','architecture','device','description','isFallbackAdapter']
            .map(key => [key, adapter.info?.[key] ?? adapter[key] ?? null])),
          maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
          features: [...adapter.features]
        } : null};
    })().then(done, error => done({error: String(error)}));`, args: []});
  await save();
  console.log(JSON.stringify({browser: session.capabilities.browserVersion, prefs, probe: result.probe}));
  if (!result.probe.adapter) throw Error('No WebGPU adapter in this Firefox profile');
  await request(base + '/url', {url});
  const started = Date.now(), deadline = started + timeout;
  let lastReport = 0, terminal = false;
  while (!interrupted && Date.now() < deadline) {
    const status = await execute(`
      if (document.body?.dataset.cumesExecution === 'idle') document.getElementById('w7x-start')?.click();
      if (document.body?.dataset.cumesWebgpu === 'ready' && !new URLSearchParams(location.search).has('solve'))
        document.getElementById('run')?.click();
      return {dataset: {...document.body?.dataset},
        progress: document.getElementById('verification-progress')?.textContent};`);
    result.lastStatus = status;
    const state = status.dataset.cumesWebgpu;
    if (Date.now() - lastReport > 15000 || state === 'pass' || state === 'fail') {
      console.log(JSON.stringify({seconds: (Date.now() - started) / 1000, ...status}));
      lastReport = Date.now();
      await save();
    }
    if (state === 'pass' || state === 'fail') {terminal = true; break;}
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
  result.elapsedSeconds = (Date.now() - started) / 1000;
  Object.assign(result, await execute(`return {dataset: {...document.body.dataset},
    plot: window.cumesResidualPlot?.report() || null,
    timing: window.cumesIterationTiming?.report() || null,
    log: window.cumesVerificationLog?.text() || document.getElementById('output')?.textContent || document.getElementById('legacy-output')?.textContent || document.body.innerText};`));
  await save();
  await writeFile(prefix + '-trace.json', JSON.stringify(await execute('return window.cumesDiagnostics || [];')));
  const png = await request(base + '/screenshot', undefined, 'GET');
  await writeFile(prefix + '.png', Buffer.from(png, 'base64'));
  if (interrupted) throw Error('Firefox gate interrupted');
  if (!terminal) throw Error('Firefox gate timed out');
  if (result.dataset.cumesWebgpu !== 'pass') throw Error(result.dataset.cumesDetail);
  if (result.plot) {
    const last = result.plot.samples.at(-1);
    if (!last?.converged || !last.fsq.every(value => Number.isFinite(value) && value < last.tolerance))
      throw Error('Final plotted residuals did not converge');
  }
  console.log('PASS: ' + url);
} catch (error) {
  result.error = String(error);
  await save();
  console.error(result.error);
  process.exitCode = 1;
} finally {
  await request(base, undefined, 'DELETE').catch(error => {
    console.error('Firefox session cleanup failed: ' + error);
    process.exitCode = 1;
  });
}
