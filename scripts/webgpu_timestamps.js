// Inject before the application loads (Page.addScriptToEvaluateOnNewDocument).
// Opt-in diagnostics for cuMES's single-flight iteration readback. No EM_JS,
// shader changes, extra mapAsync, or extra host fence. One profiling-only queue
// submission resolves the pass timestamps into a reserved readback-buffer tail.
(() => {
  if (globalThis.cumesTimestampProfile || !globalThis.GPUAdapter) return;
  const CAPACITY = 4096;
  const BATCH_LABEL = 'cuMES iteration readback batch';
  const passes = new WeakMap();
  const rows = new Map(), batches = [], errors = [], cpu = new Map(), adapters = [];
  let active = false, remaining = 0, lastCompletion = null, lastGpuEnd = null;
  const count = (label, milliseconds, bytes = 0) => {
    const row = cpu.get(label) || {label, calls: 0, milliseconds: 0, bytes: 0};
    ++row.calls; row.milliseconds += milliseconds; row.bytes += bytes;
    cpu.set(label, row);
  };
  const capture = globalThis.createCumesTimestampCapture(CAPACITY, {
    query: 'cuMES profiling timestamps', resolve: 'cuMES profiling query resolve',
    encoder: 'cuMES profiling resolve'
  }, descriptor => descriptor.label === BATCH_LABEL, (adapter, device, supported) => {
    if (!supported) {
      errors.push('Adapter does not support timestamp-query');
      return;
    }
    adapters.push(Object.fromEntries(['vendor', 'architecture', 'device', 'description',
      'driver', 'backend', 'type'].map(key => [key, adapter.info?.[key]])));
    device.addEventListener('uncapturederror', event => errors.push(event.error.message));
    device.lost.then(info => errors.push(`Device lost: ${info.reason}: ${info.message}`));
  });
  const {patch} = capture;
  patch(GPUCommandEncoder.prototype, 'beginComputePass', original => function(descriptor = {}) {
    const state = capture.state(this);
    if (!active || !state || descriptor.timestampWrites) return original.call(this, descriptor);
    if (state.pending.length * 2 + 2 > CAPACITY) {
      errors.push('Timestamp capacity exhausted; sample is incomplete');
      active = false;
      return original.call(this, descriptor);
    }
    const row = {label: descriptor.label || '', pipelines: [], dispatches: 0};
    const pass = original.call(this, {...descriptor, timestampWrites: capture.writes(state, row)});
    passes.set(pass, row);
    return pass;
  });
  patch(GPUComputePassEncoder.prototype, 'setPipeline', original => function(pipeline) {
    const row = passes.get(this);
    if (row) row.pipelines.push(pipeline.label || '(unlabelled pipeline)');
    return original.apply(this, arguments);
  });
  patch(GPUComputePassEncoder.prototype, 'dispatchWorkgroups', original => function() {
    const row = passes.get(this);
    if (row) ++row.dispatches;
    return original.apply(this, arguments);
  });
  patch(GPUQueue.prototype, 'writeBuffer', original => function(buffer, offset, data, dataOffset = 0, size) {
    if (!active) return original.apply(this, arguments);
    const start = performance.now(), unit = data.BYTES_PER_ELEMENT || 1;
    const result = original.apply(this, arguments);
    count(`upload: ${buffer.label}`, performance.now() - start,
      size === undefined ? data.byteLength - dataOffset * unit : size * unit);
    return result;
  });
  patch(GPUQueue.prototype, 'submit', original => function() {
    if (!active) return original.apply(this, arguments);
    const start = performance.now(), result = original.apply(this, arguments);
    count('submit (including profiling resolve)', performance.now() - start);
    return result;
  });
  patch(GPUCommandEncoder.prototype, 'copyBufferToBuffer', original => function(source, sourceOffset, target, targetOffset, size) {
    const start = active ? performance.now() : 0;
    const result = original.apply(this, arguments);
    if (active) count(target.usage & GPUBufferUsage.MAP_READ ? 'readback copy encoding' : 'device copy encoding',
      performance.now() - start, size);
    return result;
  });
  patch(GPUBuffer.prototype, 'getMappedRange', original => function() {
    const start = active ? performance.now() : 0;
    const result = original.apply(this, arguments);
    if (active) count('getMappedRange (excludes subsequent Wasm heap copy)', performance.now() - start, result.byteLength);
    return result;
  });
  patch(GPUBuffer.prototype, 'mapAsync', original => function(mode, offset = 0, size = this.size - offset) {
    const info = capture.buffer(this);
    if (!active || !info || !info.state.pending.length) return original.apply(this, arguments);
    if (offset !== 0 || size > info.tail || mode !== GPUMapMode.READ) {
      errors.push('Unsupported iteration mapping range; profiling stopped');
      active = false;
      return original.apply(this, arguments);
    }
    const {state} = info;
    const pending = state.pending.splice(0);
    const start = performance.now();
    const {tail, bytes} = capture.resolve(this, state, pending, size);
    return original.call(this, mode, 0, tail + bytes).then(result => {
      const completed = performance.now();
      const timestamps = new BigUint64Array(this.getMappedRange(tail, bytes));
      let sum = 0, earliest = null, latest = null;
      const timeline = [];
      for (let i = 0; i < pending.length; ++i) {
        const begin = timestamps[2 * i], end = timestamps[2 * i + 1];
        if (!begin || end < begin) { errors.push('Invalid GPU timestamp pair'); continue; }
        const milliseconds = Number(end - begin) / 1e6;
        const label = pending[i].pipelines.join(' → ') || pending[i].label || '(unlabelled pass)';
        const row = rows.get(label) || {label, calls: 0, dispatches: 0, milliseconds: 0,
          minMilliseconds: Infinity, maxMilliseconds: 0};
        ++row.calls; row.dispatches += pending[i].dispatches; row.milliseconds += milliseconds;
        row.minMilliseconds = Math.min(row.minMilliseconds, milliseconds);
        row.maxMilliseconds = Math.max(row.maxMilliseconds, milliseconds);
        rows.set(label, row);
        sum += milliseconds;
        earliest = earliest === null || begin < earliest ? begin : earliest;
        latest = latest === null || end > latest ? end : latest;
        timeline.push({label, begin: Number(begin - timestamps[0]) / 1e6,
          end: Number(end - timestamps[0]) / 1e6});
      }
      batches.push({passes: pending.length, computeMilliseconds: sum,
        gpuSpanMilliseconds: earliest === null ? null : Number(latest - earliest) / 1e6,
        mapWaitMilliseconds: completed - start,
        intervalMilliseconds: lastCompletion === null ? null : completed - lastCompletion,
        gpuBetweenBatchesMilliseconds: lastGpuEnd === null || earliest === null ? null : Number(earliest - lastGpuEnd) / 1e6,
        visible: document.visibilityState, timeline});
      lastCompletion = completed;
      lastGpuEnd = latest;
      if (--remaining === 0) active = false;
      return result;
    }, error => { active = false; errors.push(String(error)); throw error; });
  });
  globalThis.cumesTimestampProfile = {
    start(count = 256) {
      if (active || batches.length) throw Error('Reload before starting another timestamp sample');
      if (!Number.isInteger(count) || count < 1) throw Error('Expected a positive sample count');
      if (!adapters.length || errors.length) throw Error('No healthy timestamp-enabled device');
      remaining = count; active = true;
    },
    report: () => ({active, remaining, adapters: [...adapters], errors: [...errors], batches: [...batches],
      gpu: [...rows.values()].sort((a, b) => b.milliseconds - a.milliseconds),
      cpu: [...cpu.values()].sort((a, b) => b.milliseconds - a.milliseconds)}),
    restore() {
      if (active) throw Error('Wait for the profiling sample to complete before restoring hooks');
      capture.restore();
    }
  };
})();
