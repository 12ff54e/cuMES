import assert from 'node:assert/strict';
import {mkdir, mkdtemp, readFile, readdir, rm, writeFile} from 'node:fs/promises';
import {join} from 'node:path';
import {fileURLToPath} from 'node:url';
import vm from 'node:vm';
import {packageWebgpuPages} from './package_webgpu_pages.mjs';

const bootstrap = await readFile(new URL('../webgpu/pages_bootstrap.js', import.meta.url), 'utf8');
const upstream = await readFile(new URL('../webgpu/vendor/coi-serviceworker/coi-serviceworker.js', import.meta.url), 'utf8');
function isolation({isolated = false, supported = true, stored = new Map(), registrationError} = {}) {
  const listeners = new Map(), registrations = [], timers = new Map();
  let reloads = 0;
  const serviceWorker = {controller: null, ready: Promise.resolve(),
    async register(...args) { registrations.push(args); if (registrationError) throw registrationError; },
    addEventListener(name, handler) { listeners.set(name, handler); },
    removeEventListener(name) { listeners.delete(name); }};
  const context = vm.createContext({URL, Promise, navigator: {serviceWorker: supported ? serviceWorker : undefined},
    crossOriginIsolated: isolated, isSecureContext: true,
    document: {currentScript: {src: 'https://example.test/cuMES/pages_bootstrap.js'}},
    sessionStorage: {getItem: key => stored.get(key), setItem: (key, value) => stored.set(key, value), removeItem: key => stored.delete(key)},
    location: {reload() { reloads++; }},
    setTimeout(handler) { timers.set(1, handler); return 1; }, clearTimeout: id => timers.delete(id)});
  vm.runInContext(bootstrap, context);
  return {ready: context.cumesIsolationReady, registrations, serviceWorker, listeners, timers, stored,
    get reloads() { return reloads; }};
}
const tick = () => new Promise(resolve => setImmediate(resolve));
const first = isolation(); let started = false;
first.ready.then(() => { started = true; });
await tick();
assert.equal(first.registrations[0][0], 'https://example.test/cuMES/coi-serviceworker.js');
assert.equal(first.registrations[0][1].updateViaCache, 'none');
assert.equal(first.reloads, 0); assert.equal(started, false);
first.serviceWorker.controller = {scriptURL: 'https://example.test/another-worker.js'};
first.listeners.get('controllerchange')();
await tick();
assert.equal(first.reloads, 0, 'a worker with a broader scope must not trigger the isolated reload');
first.serviceWorker.controller = {scriptURL: first.registrations[0][0]};
first.listeners.get('controllerchange')();
await tick();
assert.equal(first.reloads, 1); assert.equal(started, false);
assert.equal(first.timers.size, 0);
assert.equal(first.listeners.size, 0);
const isolated = isolation({isolated: true, stored: first.stored});
await isolated.ready;
assert.equal(isolated.stored.size, 0); assert.equal(isolated.registrations.length, 0);
assert.equal(isolated.reloads, 0);
const repeated = isolation({stored: new Map([['cumes.pages.isolation:/cuMES/coi-serviceworker.js', '1']])});
await assert.rejects(repeated.ready, /isolation could not be enabled/);
assert.equal(repeated.reloads, 0); assert.equal(repeated.registrations.length, 0);
await assert.rejects(isolation({supported: false}).ready, /service workers enabled/);
await assert.rejects(isolation({registrationError: Error('permission denied')}).ready, /permission denied/);
const timeout = isolation(); await tick(); timeout.timers.get(1)();
await assert.rejects(timeout.ready, /timed out/);
let cancelCoils;
const coilContext = vm.createContext({cumesIsolationReady: new Promise((_, reject) => { cancelCoils = reject; })});
vm.runInContext(await readFile(new URL('../webgpu/coil_geometry.js', import.meta.url), 'utf8'), coilContext);
const preview = coilContext.readCumesCoils({name: 'coils.solovev', bytes: new Uint8Array()});
let previewFinished = false;
preview.finally(() => { previewFinished = true; }).catch(() => {});
await tick();
assert.equal(previewFinished, false, 'coil conversion must wait before importing shared-memory Wasm');
cancelCoils(Error('isolation failed'));
await assert.rejects(preview, /isolation failed/);

// Execute the vendored fetch handler: Wasm MIME and bytes survive header injection.
const workerEvents = new Map();
vm.runInNewContext(upstream, {self: {addEventListener: (name, callback) => workerEvents.set(name, callback)},
  Request, Response, Headers, console,
  fetch: async () => new Response(new Uint8Array([0, 97, 115, 109]), {headers: {'Content-Type': 'application/wasm'}})});
let reply;
workerEvents.get('fetch')({request: new Request('https://example.test/cuMES/cumes_webgpu.wasm'), respondWith: promise => { reply = promise; }});
const response = await reply;
assert.equal(response.headers.get('Cross-Origin-Opener-Policy'), 'same-origin');
assert.equal(response.headers.get('Cross-Origin-Embedder-Policy'), 'require-corp');
assert.equal(response.headers.get('Content-Type'), 'application/wasm');
assert.deepEqual(new Uint8Array(await response.arrayBuffer()), new Uint8Array([0, 97, 115, 109]));

const parent = fileURLToPath(new URL('../../tmp/', import.meta.url));
await mkdir(parent, {recursive: true});
const scratch = await mkdtemp(join(parent, 'cumes-pages-test-'));
try {
  const build = join(scratch, 'build'), output = join(scratch, 'site');
  await mkdir(join(build, 'presets'), {recursive: true});
  const assets = ['cumes_webgpu.js', 'cumes_webgpu.wasm', 'cumes_coils.mjs', 'cumes_coils.wasm',
    'browser_ui.js', 'verification_worker.js', 'residual_plot.js', 'free_boundary.js', 'equilibrium_view.js',
    'orbit_renderer.js', 'boundary_editor.js'];
  for (const name of assets) await writeFile(join(build, name), 'bytes:' + name);
  await writeFile(join(build, 'coil_geometry.js'), await readFile(new URL('../webgpu/coil_geometry.js', import.meta.url)));
  const html = (await readFile(new URL('../webgpu/shell.html', import.meta.url), 'utf8'))
    .replace('{{{ SCRIPT }}}', '<script src="cumes_webgpu.js?v=abcdef"></script>');
  await writeFile(join(build, 'cumes_webgpu.html'), html);
  for (const name of ['solovev.json', 'w7x.json', 'cth_like.json', 'fixed-w7x.json', 'coils.solovev', 'coils.w7x', 'coils.cth_like', 'LICENSE.vmecpp', 'README.md'])
    await writeFile(join(build, 'presets', name), 'fixture:' + name);
  await writeFile(join(build, 'CTestTestfile.cmake'), 'do not publish');
  await writeFile(join(build, 'presets', 'unwanted.nc'), 'do not publish');
  const inputsFile = join(scratch, 'package-inputs.json');
  await packageWebgpuPages(build, output, inputsFile);
  const inputs = JSON.parse(await readFile(inputsFile, 'utf8'));
  for (const name of ['pages_bootstrap.js', 'vendor/coi-serviceworker/coi-serviceworker.js',
    'vendor/coi-serviceworker/LICENSE', '../LICENSE', '../deps/webgpu-fft/LICENSE', '../deps/vacuum-field/LICENSE'])
    assert(inputs.includes(fileURLToPath(new URL('../webgpu/' + name, import.meta.url))),
      'packaging dependencies must follow the actual reads and copies: ' + name);
  for (const name of assets) assert(inputs.includes(join(build, name)));
  const index = await readFile(join(output, 'index.html'), 'utf8');
  assert.equal(index, await readFile(join(output, 'cumes_webgpu.html'), 'utf8'));
  assert(index.includes('cumes_webgpu.js?v=abcdef'));
  assert(/src="pages_bootstrap\.js\?v=[0-9a-f]{64}"/.test(index));
  assert(index.indexOf('src="pages_bootstrap.js?') < index.indexOf('src="browser_ui.js"'));
  for (const name of assets) assert.deepEqual(await readFile(join(output, name)), await readFile(join(build, name)));
  assert.equal(await readFile(join(output, 'coi-serviceworker.js'), 'utf8'), upstream);
  assert((await readFile(join(output, 'licenses/coi-serviceworker/LICENSE'), 'utf8')).includes('MIT License'));
  for (const name of ['cuMES', 'webgpu-fft', 'vacuum-field'])
    assert((await readFile(join(output, 'licenses', name + '.txt'), 'utf8')).includes('MIT License'));
  assert(!(await readdir(output)).includes('CTestTestfile.cmake'));
  assert(!(await readdir(join(output, 'presets'))).includes('unwanted.nc'));
  await assert.rejects(packageWebgpuPages(build, output), /must be empty/);
  await assert.rejects(packageWebgpuPages(build, build), /must be separate/);
  await assert.rejects(packageWebgpuPages(build, join(build, '..nested')), /must be separate/);
  await assert.rejects(packageWebgpuPages(build, scratch), /must be separate/);
  await writeFile(join(build, 'cumes_webgpu.html'), html.replace('await globalThis.cumesIsolationReady;', ''));
  await assert.rejects(packageWebgpuPages(build, join(scratch, 'stale')), /Rebuild WebGPU/);
} finally { await rm(scratch, {recursive: true, force: true}); }
console.log('PASS: Pages isolation startup/reload/failure, worker headers, and complete versioned artifact packaging');
