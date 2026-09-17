// Headless Firefox gate using a local geckodriver WebDriver endpoint.
// Usage: node scripts/webgpu_firefox_validate.mjs APP_URL OUTPUT_PREFIX [PREFS_JSON]
// Preferences apply only to geckodriver's disposable profile. GPU solves run serially.
import {mkdir, readFile, writeFile} from 'node:fs/promises';
import {dirname} from 'node:path';

const [url, prefix, prefsJson = '{}'] = process.argv.slice(2);
if (!url || !prefix) throw Error('Pass APP_URL OUTPUT_PREFIX [PREFS_JSON]');
const prefs = JSON.parse(prefsJson);
const input = process.env.CUMES_INPUT_JSON ? JSON.parse(await readFile(process.env.CUMES_INPUT_JSON, 'utf8')) : undefined;
let inputStorage;
if (input) {
  const query = new URL(url).searchParams;
  if (input.lfreeb) {
    const preset = input.coils_file?.match(/coils\.(solovev|w7x|cth_like)$/)?.[1];
    if (!preset || query.get('boundary') !== 'free' || query.has('coils'))
      throw Error('Custom free input requires a bundled coil file and ?boundary=free without coils=');
    inputStorage = {key: 'cumes.free.v1', value: {preset,
      input: {...input, coils_file: '/inputs/coils.' + preset}}};
  } else {
    if (query.get('preset') !== 'w7x')
      throw Error('Custom fixed input requires the advanced editor (?preset=w7x)');
    inputStorage = {key: 'cumes.fixed.w7x.v1', value: input};
  }
}
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
  // This route stays idle until Run is clicked, so the probe starts no solver.
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
  if (inputStorage) await execute(`localStorage.setItem(${JSON.stringify(inputStorage.key)}, ${JSON.stringify(JSON.stringify(inputStorage.value))});`);
  await request(base + '/url', {url});
  const started = Date.now(), deadline = started + timeout;
  let lastReport = 0, terminal = false;
  while (!interrupted && Date.now() < deadline) {
    const status = await execute(`
      if (document.body?.dataset.cumesExecution === 'idle')
        (document.getElementById('w7x-actions')?.hidden === false ? document.getElementById('w7x-start') : document.getElementById('run'))?.click();
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
  if (process.env.CUMES_CAPTURE_OUTPUT === '1' && result.dataset.cumesWebgpu === 'pass') {
    const encoded = await request(base + '/execute/async', {script: `
      const done = arguments[arguments.length - 1];
      fetch(window.cumesOutputUrl).then(r => r.arrayBuffer()).then(bytes =>
        done(btoa(Array.from(new Uint8Array(bytes), value => String.fromCharCode(value)).join(''))));`, args: []});
    await writeFile(prefix + '-output.bin', Buffer.from(encoded, 'base64'));
  }
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
