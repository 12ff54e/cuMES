// Deterministic frontend timing tests; browser qualification covers Dawn.
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';
const source = await readFile(new URL('../webgpu/iteration_timing.js', import.meta.url), 'utf8');
function fixture(supported = true, search = '', worker = false) {
  let now = 0, gpuNow = 0n;
  const maps = [], log = [], speeds = [];
  class GPUBuffer {
    constructor(d) { this.size = d.size; this.data = new ArrayBuffer(d.size); }
    mapAsync() { return new Promise(resolve => maps.push(resolve)); }
    getMappedRange(offset, bytes) { return this.data.slice(offset, offset + bytes); }
  }
  class GPUCommandEncoder {
    beginComputePass(d = {}) {
      if (d.timestampWrites) {
        const t = d.timestampWrites;
        t.querySet.data[t.beginningOfPassWriteIndex] = gpuNow;
        gpuNow += 2000000n;
        t.querySet.data[t.endOfPassWriteIndex] = gpuNow;
      }
      return {};
    }
    resolveQuerySet(q, start, count, b, offset) {
      new BigUint64Array(b.data, offset, count).set(q.data.slice(start, start + count));
    }
    copyBufferToBuffer(a, ao, b, bo, bytes) {
      new Uint8Array(b.data, bo, bytes).set(new Uint8Array(a.data, ao, bytes));
    }
    finish() { return {}; }
  }
  class GPUDevice {
    queue = {submit() {}};
    createBuffer(d) { return new GPUBuffer(d); }
    createCommandEncoder() { return new GPUCommandEncoder(); }
    createQuerySet(d) { return {data: new BigUint64Array(d.count)}; }
  }
  class GPUAdapter {
    features = new Set(supported ? ['timestamp-query'] : []);
    async requestDevice() { return new GPUDevice(); }
  }
  const context = vm.createContext({URLSearchParams, location: {search},
    ...(worker ? {cumesVisibility: 'visible', cumesSearch: search} : {document: {visibilityState: 'visible'}}),
    performance: {now: () => now},
    GPUAdapter, GPUDevice, GPUCommandEncoder, GPUBuffer, BigUint64Array,
    GPUBufferUsage: {MAP_READ: 1, QUERY_RESOLVE: 2, COPY_SRC: 4}, GPUMapMode: {READ: 1},
    cumesAppendLog: line => log.push(line), cumesBrowser: {speed: value => speeds.push(value)}});
  vm.runInContext(source, context);
  return {api: context.cumesIterationTiming, adapter: new GPUAdapter(), log, speeds,
    time: t => { now = t; }, resolve: () => maps.shift()()};
}
for (const supported of [true, false]) {
  const f = fixture(supported), device = await f.adapter.requestDevice();
  const buffer = device.createBuffer({size: 32, usage: 1});
  f.api.event(0, 0);
  f.time(10); f.api.event(1, 1);
  device.createCommandEncoder().beginComputePass();
  f.time(13); const mapped = buffer.mapAsync(1, 0, 16);
  f.time(20); f.resolve(); await mapped;
  f.time(22); f.api.event(2, 1);
  const report = f.api.finish();
  assert.equal(report.stats.wall.average, 12);
  assert.equal(report.stats.host.average, 5);
  assert.equal(report.stats.wait.average, 7);
  assert.equal(report.stats.device?.average ?? null, supported ? 2 : null);
  assert(f.log.some(line => line.includes('median')));
  if (!supported) assert(f.log.some(line => line.includes('unavailable')));
  assert.equal(f.api.statistics([9, 1, 5]).median, 5);
  assert.equal(f.api.statistics([9, 1, 5, 3]).median, 4);
  assert.equal(f.api.statistics([]), null);
  f.api.event(0, 0);
  assert.equal(f.api.report().iterations.length, 0);
}
const disabled = fixture(true, '?timing=0');
disabled.api.event(1, 1);
assert.equal(disabled.api.finish().iterations.length, 0);
assert(disabled.log[0].includes('disabled'));
const worker = fixture(true, '', true);
worker.api.event(1, 1);worker.time(12);worker.api.event(2, 1);
assert.equal(worker.api.report().iterations[0].visible, 'visible');
assert.equal(worker.api.finish().stats.wall.average, 12);
const overflow = fixture(), device = await overflow.adapter.requestDevice();
overflow.api.event(1, 1);
for (let i = 0; i < 514; ++i) device.createCommandEncoder().beginComputePass();
const buffer = device.createBuffer({size: 32, usage: 1});
const mapped = buffer.mapAsync(1, 0, 16);
overflow.resolve(); await mapped;
overflow.api.event(2, 1);
assert.equal(overflow.api.report().stats.device, null);
assert.equal(overflow.api.report().errors.length, 1);
console.log('Frontend iteration timing: PASS (host/wait, timestamps, unsupported, reset, median, disabled)');

// The live rate uses completed-pass wall time even with profiling disabled.
const speed = fixture(true, '?timing=0');
speed.time(10000); speed.api.event(0, 0);
speed.api.event(2, 1); // no pass has started
const pass = (start, end, stage = 1) => {
  speed.time(start); speed.api.event(1, stage);
  speed.time(end); speed.api.event(2, stage);
  speed.api.event(2, stage); // duplicate final/maintenance callback
};
for (let i = 0; i < 25; i++) pass(20000 + i * 20, 20020 + i * 20);
assert.deepEqual(speed.speeds, [null, null, 50], 'exclude startup and publish once per half second');
pass(20500, 20800); pass(20800, 21100);
assert.equal(speed.speeds.at(-1), 2 / .6, 'follow recent throughput, including slow vacuum/refresh passes');
pass(100000, 100200, 2); pass(100200, 100500, 2);
assert.deepEqual(speed.speeds.slice(-2), [null, 4], 'reset at a new grid and exclude grid setup');
speed.time(200000); speed.api.event(0, 0); pass(300000, 301000);
assert.deepEqual(speed.speeds.slice(-3), [null, null, 1], 'new runs cannot inherit previous samples');
assert.equal(speed.api.report().iterations.length, 0, 'live speed must not enable detailed profiling');
console.log('Live iteration speed: PASS (wall time, recent windows, restarts, grids, setup exclusion, profiling disabled)');
