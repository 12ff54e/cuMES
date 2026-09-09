// Passive until a caller installs hooks. The profiler can inject this before
// the runtime embeds it again; existing captures must keep their original helper.
globalThis.createCumesTimestampCapture ||= function(capacity, labels, selectBuffer, deviceReady) {
  const originals = [], devices = new WeakMap(), buffers = new WeakMap(), encoders = new WeakMap();
  const patch = (prototype, name, wrap) => {
    const original = prototype[name];
    originals.push(() => { prototype[name] = original; });
    prototype[name] = wrap(original);
  };
  patch(GPUAdapter.prototype, 'requestDevice', original => async function(descriptor = {}) {
    const supported = this.features.has('timestamp-query');
    if (!supported) { deviceReady(this, undefined, false); return original.call(this, descriptor); }
    const device = await original.call(this, {...descriptor,
      requiredFeatures: [...new Set([...(descriptor.requiredFeatures || []), 'timestamp-query'])]});
    devices.set(device, {device, pending: [], queries: device.createQuerySet({
      label: labels.query, type: 'timestamp', count: capacity}),
      resolve: device.createBuffer({label: labels.resolve, size: capacity * 8,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC})});
    deviceReady(this, device, true);
    return device;
  });
  patch(GPUDevice.prototype, 'createBuffer', original => function(descriptor) {
    const state = devices.get(this), tail = Math.ceil(descriptor.size / 8) * 8;
    const extend = state && selectBuffer(descriptor) && !descriptor.mappedAtCreation;
    const buffer = original.call(this, extend ? {...descriptor, size: tail + capacity * 8} : descriptor);
    if (extend) buffers.set(buffer, {state, tail});
    return buffer;
  });
  patch(GPUDevice.prototype, 'createCommandEncoder', original => function(...args) {
    const encoder = original.apply(this, args);
    encoders.set(encoder, devices.get(this));
    return encoder;
  });
  return {
    patch,
    state: encoder => encoders.get(encoder),
    buffer: buffer => buffers.get(buffer),
    writes(state, row) {
      const index = state.pending.length * 2;
      state.pending.push(row);
      return {querySet: state.queries, beginningOfPassWriteIndex: index, endOfPassWriteIndex: index + 1};
    },
    resolve(buffer, state, pending, size) {
      // Map the used payload and timestamps, excluding spare readback capacity.
      const tail = Math.ceil(size / 8) * 8, bytes = pending.length * 16;
      const encoder = state.device.createCommandEncoder(labels.encoder ? {label: labels.encoder} : undefined);
      encoder.resolveQuerySet(state.queries, 0, pending.length * 2, state.resolve, 0);
      encoder.copyBufferToBuffer(state.resolve, 0, buffer, tail, bytes);
      state.device.queue.submit([encoder.finish()]);
      return {tail, bytes};
    },
    restore() { for (const restore of originals.reverse()) restore(); }
  };
};
