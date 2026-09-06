// Deterministic frontend timing tests; browser qualification covers Dawn.
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';
const source = await readFile(new URL('../webgpu/iteration_timing.js', import.meta.url), 'utf8');
function fixture(supported = true, search = '') {
  let now = 0, gpuNow = 0n;
  const maps = [], log = [];
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
    document: {visibilityState: 'visible'}, performance: {now: () => now},
    GPUAdapter, GPUDevice, GPUCommandEncoder, GPUBuffer, BigUint64Array,
    GPUBufferUsage: {MAP_READ: 1, QUERY_RESOLVE: 2, COPY_SRC: 4}, GPUMapMode: {READ: 1},
    cumesAppendLog: line => log.push(line)});
  vm.runInContext(source, context);
  return {api: context.cumesIterationTiming, adapter: new GPUAdapter(), log,
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
