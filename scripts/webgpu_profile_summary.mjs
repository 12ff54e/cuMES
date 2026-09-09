// Usage: node scripts/webgpu_profile_summary.mjs OUTPUT_PREFIX [SYMBOL_MAP]
// Only use a symbol map from a byte-identical Wasm binary. Writes a compact
// report and (when symbols are available) a Chrome-importable CPU profile.
import {readFile, writeFile} from 'node:fs/promises';

const [prefix, symbolPath] = process.argv.slice(2);
if (!prefix) throw Error('Pass the capture output prefix');
const gpu = JSON.parse(await readFile(`${prefix}-gpu.json`, 'utf8'));
const profile = JSON.parse(await readFile(`${prefix}-cpu.json`, 'utf8'));
const symbols = new Map(symbolPath ? (await readFile(symbolPath, 'utf8')).trim().split('\n').map(line => {
  const index = line.indexOf(':');
  return [line.slice(0, index), line.slice(index + 1)];
}) : []);
for (const node of profile.nodes) {
  const index = node.callFrame.functionName.match(/^wasm-function\[(\d+)\]$/)?.[1];
  if (symbols.has(index)) node.callFrame.functionName = symbols.get(index);
}
const nodes = new Map(profile.nodes.map(node => [node.id, node]));
const parents = new Map(profile.nodes.flatMap(node => (node.children || []).map(id => [id, node.id])));
const self = new Map(), categories = new Map();
let duration = 0;
for (let i = 0; i < profile.samples.length; ++i) {
  const id = profile.samples[i], milliseconds = profile.timeDeltas[i] / 1000;
  const name = nodes.get(id).callFrame.functionName;
  duration += milliseconds;
  self.set(name, (self.get(name) || 0) + milliseconds);
  let shaderLoading = false;
  for (let ancestor = id; ancestor !== undefined; ancestor = parents.get(ancestor)) {
    if (/::(?:read_shader|load_shader|load_inverse_shader|load_forward_shader|load_dealias_shader)\(/.test(
      nodes.get(ancestor).callFrame.functionName)) shaderLoading = true;
  }
  const category = shaderLoading ? 'shader-source loading (inclusive)' :
    /vector<float.*__(assign|insert)_with_size/.test(name) ? 'float vector assignment/insertion (self)' :
    name === '_emwgpuBufferGetConstMappedRange' ? 'Emdawn mapped readback to Wasm (self)' :
    name === 'writeBuffer' ? 'WebGPU writeBuffer (self)' :
    name === '(idle)' ? 'renderer idle' : 'other host work';
  categories.set(category, (categories.get(category) || 0) + milliseconds);
}
const sorted = map => [...map].map(([label, milliseconds]) => ({label, milliseconds,
  percent: 100 * milliseconds / duration})).sort((a, b) => b.milliseconds - a.milliseconds);
const stats = key => {
  const values = gpu.batches.map(row => row[key]).filter(value => Number.isFinite(value)).sort((a, b) => a - b);
  if (!values.length) return null;
  return {mean: values.reduce((a, b) => a + b, 0) / values.length,
    median: values[Math.floor(values.length / 2)], p95: values[Math.floor(values.length * 0.95)]};
};
const count = gpu.batches.length;
const summary = {
  adapters: gpu.adapters, errors: gpu.errors, batches: count,
  visibleBatches: gpu.batches.filter(row => row.visible === 'visible').length,
  milliseconds: Object.fromEntries(['computeMilliseconds', 'gpuSpanMilliseconds',
    'mapWaitMilliseconds', 'intervalMilliseconds', 'gpuBetweenBatchesMilliseconds'].map(key => [key, stats(key)])),
  gpu: gpu.gpu.map(row => ({label: row.label, millisecondsPerIteration: row.milliseconds / count,
    callsPerIteration: row.calls / count, millisecondsPerCall: row.milliseconds / row.calls})),
  hostApi: gpu.cpu.map(row => ({label: row.label, millisecondsPerIteration: row.milliseconds / count,
    callsPerIteration: row.calls / count, bytesPerIteration: row.bytes / count})),
  cpuSampleMilliseconds: duration, cpuCategories: sorted(categories), cpuSelf: sorted(self).slice(0, 30)
};
await writeFile(`${prefix}-summary.json`, JSON.stringify(summary, null, 2));
if (symbolPath) await writeFile(`${prefix}-cpu-symbolized.cpuprofile`, JSON.stringify(profile));
console.log(JSON.stringify({prefix, batches: count, milliseconds: summary.milliseconds,
  gpu: summary.gpu, cpuCategories: summary.cpuCategories}, null, 2));
