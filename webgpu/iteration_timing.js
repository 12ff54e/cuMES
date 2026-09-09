// Frontend-only iteration timing. GPU timestamps share the solver's existing
// mapAsync: no additional map or host fence. ?timing=0 disables detailed
// profiling; the lightweight live iteration rate remains available.
(() => {
  const enabled = new URLSearchParams(globalThis.cumesSearch ?? location.search).get('timing') !== '0';
  let speedStage = 0, speedStart = 0, speedCount = 0, speedActive = false;
  function updateSpeed(kind, stage) {
    const now = performance.now();
    if (kind === 0 || (kind === 1 && stage !== speedStage)) {
      speedStage = stage; speedStart = now; speedCount = 0; speedActive = false;
      globalThis.cumesBrowser?.speed?.(null);
    }
    if (kind === 1) speedActive = true;
    // An end event can occur without a started pass (maintenance or a final
    // stage callback). Count actual completed passes, including restarts.
    if (kind === 2 && speedActive) {
      speedActive = false; speedCount++;
      const elapsed = now - speedStart;
      if (elapsed >= 500) {
        globalThis.cumesBrowser?.speed?.(1000 * speedCount / elapsed);
        speedStart = now; speedCount = 0;
      }
    }
  }
  const rows = [], errors = [];
  const warn = message => { if (!errors.includes(message)) errors.push(message); };
  let current = null, waiting = 0, lastTick = performance.now();
  let deviceAvailable = false;
  const tick = () => {
    const now = performance.now();
    if (current) current[waiting ? 'wait' : 'host'] += now - lastTick;
    lastTick = now;
    return now;
  };
  const end = () => {
    const now = tick();
    if (current) current.wall = now - current.start;
    current = null;
  };
  const statistics = values => {
    const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
    const n = sorted.length;
    return n ? {count: n, min: sorted[0], max: sorted[n - 1],
      median: (sorted[Math.floor((n - 1) / 2)] + sorted[Math.floor(n / 2)]) / 2,
      average: sorted.reduce((a, b) => a + b, 0) / n} : null;
  };
  const report = () => ({enabled, deviceAvailable, errors: [...errors],
    iterations: rows.map(row => ({...row})),
    stats: Object.fromEntries(['wall', 'host', 'wait', 'device'].map(key => [key,
      statistics(rows.filter(row => Number.isFinite(row.wall) &&
        (key !== 'device' || row.gpuPasses > 0 && !row.gpuMissing)).map(row => row[key]))]))});
  globalThis.cumesIterationTiming = {
    event(kind, stage) {
      updateSpeed(kind, stage);
      if (!enabled) return;
      if (kind === 0) { end(); rows.length = 0; errors.length = 0; }
      if (kind === 2) end();
      if (kind === 1) {
        end();
        current = {stage, start: performance.now(), host: 0, wait: 0,
          device: 0, gpuPasses: 0, gpuMissing: false,
          visible: globalThis.cumesVisibility ?? globalThis.document?.visibilityState ?? 'unknown'};
        lastTick = current.start;
        rows.push(current);
      }
    },
    report,
    statistics,
    finish() {
      end();
      const result = report();
      const append = globalThis.cumesAppendLog;
      if (!append) return result;
      if (!enabled) { append('Iteration timing disabled (?timing=0).'); return result; }
      if (!rows.length) return result;
      append('\nIteration timing (ms; all attempted passes, including refresh/restarts):');
      append('                              min        max     median    average      count');
      for (const [key, label] of [['wall', 'Wall / iteration'], ['host', 'Host non-wait elapsed'],
        ['wait', 'Readback wait'], ['device', 'Device compute (GPU)']]) {
        const s = result.stats[key];
        append(label.padEnd(27) + (s ? [s.min, s.max, s.median, s.average]
          .map(value => value.toFixed(3).padStart(11)).join('') + String(s.count).padStart(11)
          : 'unavailable (timestamp-query not available or sample incomplete)'));
      }
      append('Host = JS/Wasm orchestration + callback scheduling, excluding map waits; not CPU-profiler time.');
      append('Device = sum of timestamped compute passes; excludes copies/queue gaps. Wait overlaps device work; do not add them.');
      append('Includes cold iterations; timestamps add overhead. Use ?timing=0 for uninstrumented benchmarks.');
      const hidden = rows.filter(row => row.visible !== 'visible').length;
      if (hidden) append(`Warning: ${hidden} iterations started in a hidden tab; browser throttling may affect timing.`);
      for (const error of errors) append(`Timing warning: ${error}`);
      return result;
    }
  };
  if (!enabled || !globalThis.GPUAdapter) return;

  const CAPACITY = 1024, TAIL_BYTES = CAPACITY * 8;
  const devices = new WeakMap(), buffers = new WeakMap(), encoders = new WeakMap();
  const patch = (prototype, name, wrap) => { prototype[name] = wrap(prototype[name]); };
  patch(GPUAdapter.prototype, 'requestDevice', original => async function(descriptor = {}) {
    const supported = this.features.has('timestamp-query');
    const device = await original.call(this, supported ? {...descriptor,
      requiredFeatures: [...new Set([...(descriptor.requiredFeatures || []), 'timestamp-query'])]}
      : descriptor);
    if (supported) {
      deviceAvailable = true;
      devices.set(device, {device, pending: [], queries: device.createQuerySet({
        label: 'cuMES iteration timestamps', type: 'timestamp', count: CAPACITY}),
        resolve: device.createBuffer({label: 'cuMES iteration timestamp resolve', size: TAIL_BYTES,
          usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC})});
    }
    return device;
  });
  patch(GPUDevice.prototype, 'createBuffer', original => function(descriptor) {
    const state = devices.get(this);
    const tail = Math.ceil(descriptor.size / 8) * 8;
    const extend = state && (descriptor.usage & GPUBufferUsage.MAP_READ) && !descriptor.mappedAtCreation;
    const buffer = original.call(this, extend ? {...descriptor, size: tail + TAIL_BYTES} : descriptor);
    if (extend) buffers.set(buffer, {state, tail});
    return buffer;
  });
  patch(GPUDevice.prototype, 'createCommandEncoder', original => function(...args) {
    const encoder = original.apply(this, args);
    encoders.set(encoder, devices.get(this));
    return encoder;
  });
  patch(GPUCommandEncoder.prototype, 'beginComputePass', original => function(descriptor = {}) {
    const state = encoders.get(this);
    if (!current || !state) return original.call(this, descriptor);
    if (descriptor.timestampWrites || state.pending.length * 2 + 2 > CAPACITY) {
      current.gpuMissing = true;
      warn(descriptor.timestampWrites ? 'Another profiler owns some compute timestamps.'
        : 'Timestamp capacity exceeded; incomplete device samples are omitted.');
      return original.call(this, descriptor);
    }
    const index = state.pending.length * 2;
    state.pending.push(current);
    return original.call(this, {...descriptor, timestampWrites: {
      querySet: state.queries, beginningOfPassWriteIndex: index, endOfPassWriteIndex: index + 1}});
  });
  patch(GPUBuffer.prototype, 'mapAsync', original => function(mode, offset = 0, size = this.size - offset) {
    const info = buffers.get(this);
    let pending = [], tail = 0;
    if (info?.state.pending.length) {
      const {state} = info;
      pending = state.pending.splice(0);
      if (offset !== 0 || size > info.tail || mode !== GPUMapMode.READ) {
        for (const row of pending) row.gpuMissing = true;
        warn('Unsupported mapping range; incomplete device samples are omitted.');
        pending = [];
      } else {
        tail = Math.ceil(size / 8) * 8;
        const encoder = state.device.createCommandEncoder();
        encoder.resolveQuerySet(state.queries, 0, pending.length * 2, state.resolve, 0);
        encoder.copyBufferToBuffer(state.resolve, 0, this, tail, pending.length * 16);
        state.device.queue.submit([encoder.finish()]);
      }
    }
    tick(); ++waiting;
    let promise;
    try { promise = original.call(this, mode, offset, pending.length ? tail + pending.length * 16 : size); }
    catch (error) { tick(); --waiting; throw error; }
    return promise.then(value => {
      tick(); --waiting;
      if (pending.length) {
        try {
          const times = new BigUint64Array(this.getMappedRange(tail, pending.length * 16));
          for (let i = 0; i < pending.length; ++i) {
            const begin = times[2 * i], end = times[2 * i + 1], row = pending[i];
            if (end < begin) {
              row.gpuMissing = true; warn('Invalid device timestamp pair; sample omitted.'); continue;
            }
            row.device += Number(end - begin) / 1e6;
            ++row.gpuPasses;
          }
        } catch (error) {
          for (const row of pending) row.gpuMissing = true;
          warn(`Timestamp decoding failed: ${error}`);
        }
      }
      return value;
    }, error => { tick(); --waiting; for (const row of pending) row.gpuMissing = true; throw error; });
  });
})();
