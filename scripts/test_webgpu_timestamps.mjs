// Deterministic harness checks; not a substitute for the real browser GPU run.
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {runInNewContext} from 'node:vm';

const source = await readFile(new URL('./webgpu_timestamps.js', import.meta.url), 'utf8');
async function fixture(supported = true) {
  let mapCalls = 0, submissions = 0, request;
  class GPUBuffer {
    constructor(descriptor) { Object.assign(this, descriptor); this.data = new ArrayBuffer(this.size); }
    async mapAsync(mode, offset, size) { ++mapCalls; this.mapped = [offset, offset + size]; this.ranges = []; }
    getMappedRange(offset, size) {
      assert.ok(offset >= this.mapped[0] && offset + size <= this.mapped[1]);
      assert.ok(!this.ranges.some(([a, b]) => offset < b && offset + size > a));
      this.ranges.push([offset, offset + size]);
      return this.data.slice(offset, offset + size);
    }
    unmap() { this.mapped = null; }
  }
  class GPUComputePassEncoder {
    constructor(descriptor) { this.descriptor = descriptor; }
    setPipeline() {}
    dispatchWorkgroups() {}
    end() {
      const writes = this.descriptor.timestampWrites;
      if (!writes) return;
      const {querySet, beginningOfPassWriteIndex: begin, endOfPassWriteIndex: end} = writes;
      querySet.values[begin] = BigInt(1000000 + begin * 1000000);
      querySet.values[end] = querySet.values[begin] + 500000n;
    }
  }
  class GPUCommandEncoder {
    beginComputePass(descriptor) { return new GPUComputePassEncoder(descriptor); }
    resolveQuerySet(query, first, count, target, offset) {
      new BigUint64Array(target.data, offset, count).set(query.values.slice(first, first + count));
    }
    copyBufferToBuffer(source, sourceOffset, target, targetOffset, size) {
      new Uint8Array(target.data, targetOffset, size).set(new Uint8Array(source.data, sourceOffset, size));
    }
    finish() { return {}; }
  }
  class GPUQueue { submit() { ++submissions; } writeBuffer() {} }
  class GPUDevice {
    constructor() { this.queue = new GPUQueue(); this.lost = new Promise(() => {}); }
    createBuffer(descriptor) { return new GPUBuffer(descriptor); }
    createQuerySet(descriptor) { return {values: new BigUint64Array(descriptor.count)}; }
    createCommandEncoder() { return new GPUCommandEncoder(); }
    addEventListener() {}
  }
  class GPUAdapter {
    constructor() { this.features = new Set(supported ? ['timestamp-query'] : []); }
    async requestDevice(descriptor) { request = descriptor; return new GPUDevice(); }
  }
  const context = {GPUAdapter, GPUDevice, GPUQueue, GPUBuffer, GPUCommandEncoder, GPUComputePassEncoder,
    GPUBufferUsage: {QUERY_RESOLVE: 1, COPY_SRC: 2, COPY_DST: 4, MAP_READ: 8}, GPUMapMode: {READ: 1},
    performance, document: {visibilityState: 'visible'}};
  const original = GPUBuffer.prototype.mapAsync;
  runInNewContext(source, context);
  const device = await new GPUAdapter().requestDevice({requiredFeatures: ['timestamp-query']});
  return {device, profile: context.cumesTimestampProfile, original, GPUBuffer,
    counts: () => ({mapCalls, submissions, features: [...request.requiredFeatures]})};
}
const test = await fixture();
const buffer = test.device.createBuffer({label: 'cuMES iteration readback batch', size: 96, usage: 12});
new Uint8Array(buffer.data, 0, 32).fill(42);
assert.equal(buffer.size, 96 + 4096 * 8);
assert.throws(() => test.profile.start(0), /positive/);
test.profile.start(2);
assert.throws(() => test.profile.restore(), /Wait/);
for (let i = 0; i < 2; ++i) {
  const encoder = test.device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline({label: 'test kernel'}); pass.dispatchWorkgroups(1); pass.end();
  test.device.queue.submit([encoder.finish()]);
  await buffer.mapAsync(1, 0, 32);
  assert.deepEqual(buffer.mapped, [0, 48]);
  assert.deepEqual([...new Uint8Array(buffer.getMappedRange(0, 32))], Array(32).fill(42));
  buffer.unmap();
}
const report = test.profile.report();
assert.equal(report.active, false);
assert.equal(report.errors.length, 0);
assert.equal(report.batches.length, 2);
assert.equal(report.gpu[0].calls, 2);
assert.equal(report.gpu[0].milliseconds, 1);
assert.equal(report.batches[0].timeline[0].end, 0.5);
assert.deepEqual(test.counts(), {mapCalls: 2, submissions: 4, features: ['timestamp-query']});
assert.throws(() => test.profile.start(), /Reload/);
test.profile.restore();
assert.equal(test.GPUBuffer.prototype.mapAsync, test.original);
const unsupported = await fixture(false);
assert.match(unsupported.profile.report().errors[0], /does not support/);
assert.throws(() => unsupported.profile.start(), /No healthy/);
unsupported.profile.restore();
console.log('PASS: timestamp feature negotiation, query reuse, one map per batch, disjoint payload/timestamp ranges, unchanged payload, and hook restoration');
