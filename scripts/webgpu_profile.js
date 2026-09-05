// Run through webgpu_cdp.mjs eval-file after shader warmup. Query
// cumesGpuProfile.report() later. Mapping latency includes queued GPU work;
// it is NOT a measurement of copy time or isolated kernel time.
(() => {
  if (window.cumesGpuProfile) return 'already profiling';
  const started = performance.now(), records = new Map();
  const add = (kind, label, bytes, milliseconds = 0) => {
    const key = `${kind}: ${label}`;
    const row = records.get(key) || {kind, label, calls: 0, bytes: 0, milliseconds: 0};
    ++row.calls; row.bytes += bytes; row.milliseconds += milliseconds;
    records.set(key, row);
  };
  const write = GPUQueue.prototype.writeBuffer;
  GPUQueue.prototype.writeBuffer = function(buffer, offset, data, dataOffset = 0, size) {
    const unit = data.BYTES_PER_ELEMENT || 1;
    const bytes = size === undefined ? data.byteLength - dataOffset * unit : size * unit;
    const start = performance.now();
    const result = write.apply(this, arguments);
    add('upload', buffer.label, bytes, performance.now() - start);
    return result;
  };
  const map = GPUBuffer.prototype.mapAsync;
  GPUBuffer.prototype.mapAsync = function(mode, offset = 0, size = this.size - offset) {
    const start = performance.now();
    return map.apply(this, arguments).then(result => {
      add('map-wait', this.label, size, performance.now() - start);
      return result;
    });
  };
  const copy = GPUCommandEncoder.prototype.copyBufferToBuffer;
  GPUCommandEncoder.prototype.copyBufferToBuffer = function(source, sourceOffset, target, targetOffset, size) {
    const result = copy.apply(this, arguments);
    add(target.usage & GPUBufferUsage.MAP_READ ? 'readback-copy' : 'device-copy', target.label, size);
    return result;
  };
  window.cumesGpuProfile = {
    report: () => ({seconds: (performance.now() - started) / 1000,
      records: [...records.values()].sort((a, b) => b.milliseconds - a.milliseconds)}),
    stop: () => {
      GPUQueue.prototype.writeBuffer = write;
      GPUBuffer.prototype.mapAsync = map;
      GPUCommandEncoder.prototype.copyBufferToBuffer = copy;
    }
  };
  return 'profiling started';
})();
