import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
import vm from 'node:vm';

const {default: createModule} = await import(pathToFileURL(process.argv[2]).href);
const errors = [], module = await createModule({print() {}, printErr: line => errors.push(line)});
const context = vm.createContext({});
for (const file of ['equilibrium_view.js', 'coil_geometry.js'])
  vm.runInContext(readFileSync(new URL('../webgpu/' + file, import.meta.url), 'utf8'), context);
function normalize(path, bytes) {
  module.FS.writeFile(path, bytes);
  errors.length = 0;
  const code = module.callMain([path, '/normalized.json']);
  if (code) throw Error(errors.join('\n'));
  return JSON.parse(module.FS.readFile('/normalized.json', {encoding: 'utf8'}));
}
for (const [name, count] of [['solovev', 13], ['w7x', 70], ['cth_like', 9]]) {
  const input = JSON.parse(readFileSync(new URL(`../webgpu/presets/${name}.json`, import.meta.url)));
  const source = name === 'w7x' ? '../webgpu/presets/coils.w7x' : `../deps/vacuum-field/tests/data/coils.${name}`;
  const normalized = normalize('/input.coils', readFileSync(new URL(source, import.meta.url)));
  const lines = context.coilLines(normalized, context.boundaryFourier(input));
  assert.equal(lines.filaments, count, 'full-device geometry must not be replicated by field_periods');
  assert.equal(lines.clipped, name === 'solovev');
  assert(lines.positions.every(Number.isFinite));
  assert(lines.indices.every(index => index < lines.positions.length / 4));
  assert(lines.radius > 0 && lines.radius < 20);
  if (name === 'solovev') {
    assert.equal(lines.indices.length / 2, 12 * 160 + 1, 'remote closure leaves one central conductor and twelve circles');
    assert.deepEqual([...lines.positions.slice(0, 2)], [0, 0]);
    assert.deepEqual([...lines.positions.slice(4, 6)], [0, 0]);
    assert(lines.positions[2] < -5 && lines.positions[6] > 5);
  }
}
const simple = readFileSync(new URL('../deps/vacuum-field/tests/data/coils.simple.json', import.meta.url));
const normalized = normalize('/input.json', simple);
const fourier = {mpol: 1, ntor: 0, surfaces: [{coefficients: [2, 0, 0, 0, 0, 0]}]};
const lines = context.coilLines(normalized, fourier);
assert.equal(lines.indices.length / 2, 3 + 160);
assert.deepEqual([...lines.positions.slice(0, 3)], [1, 0, 0]);
assert.deepEqual([...lines.positions.slice(20, 23)], [1, 0, 0], 'polygon closing segment is explicit only in render data');
assert(Math.abs(lines.radius - Math.hypot(2.25, .75)) < 1e-12);
assert.throws(() => normalize('/bad.json', '{"schema":"wrong"}'), /coil|schema/);
assert.equal(normalize('/input.json', simple).circuits.length, 2, 'a rejected upload must not poison the reader');
console.log('PASS: shared Wasm coil parser, all presets, circles, closed polygons, full-device scope, remote returns, invalid upload recovery');
