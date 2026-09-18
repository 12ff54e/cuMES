import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

const context = vm.createContext({});
for (const name of ['equilibrium_view', 'boundary_editor', 'free_boundary'])
  vm.runInContext(readFileSync(new URL(`../webgpu/${name}.js`, import.meta.url), 'utf8'), context);
const copy = value => JSON.parse(JSON.stringify(value));
const input = JSON.parse(readFileSync(new URL('../inputs/solovev.json', import.meta.url), 'utf8'));
const source = JSON.stringify(input);
const loaded = context.readCumesInputJSON('\uFEFF' + source, 'float');
assert.deepEqual(copy(loaded.input), input, 'Import must retain all coefficients, profiles and stage settings');
assert.equal(loaded.precision, 'double', 'Strict input tolerances select paired precision without relaxing the input');
const scalar = {...input, ftol_array:[1e-3,1e-4,1e-5]};
assert.equal(context.readCumesInputJSON(JSON.stringify(scalar), 'float').precision, 'float');
assert.equal(context.readCumesInputJSON(JSON.stringify(scalar), 'double').precision, 'double');
for (const bad of ['{', 'null', '[]', '42', '{}', JSON.stringify({...input, lfreeb:'true'}),
  JSON.stringify({...input, ns_array:[5,5,55]}), JSON.stringify({...input, rbc:null})])
  assert.throws(() => context.readCumesInputJSON(bad, 'float'));

const free = JSON.parse(readFileSync(new URL('../webgpu/presets/solovev.json', import.meta.url), 'utf8'));
for (const path of ['../data/coils.solovev', 'C:\\inputs\\coils.solovev', '/inputs/coils.solovev']) {
  const selected = context.cumesFreeInputConfig({...free, coils_file:path});
  assert.equal(selected.preset, 'solovev');
  assert.deepEqual(copy(selected.input), free, 'Only the coil path is mapped to the browser file');
}
const saved = {preset:'upload', coilName:'my-coils.json', input:{coils_file:'/inputs/coils.json'}};
for (const path of ['data/my-coils.json', '/inputs/coils.json']) {
  const selected = context.cumesFreeInputConfig({...free, coils_file:path}, saved);
  assert.equal(selected.preset, 'upload');
  assert.equal(selected.coilName, saved.coilName);
  assert.deepEqual(copy(selected.input), {...free, coils_file:'/inputs/coils.json'});
}
for (const bad of [{...free, coils_file:'unrelated.coils'}, {...free, extcur:[null]},
  {...free, makegrid_parameters:null}, {...free, makegrid_parameters:[]},
  {...free, makegrid_parameters_file:'grid.json'}, {...free, mgrid_file:'mgrid.nc'}])
  assert.throws(() => context.cumesFreeInputConfig(bad, saved));
assert.equal(JSON.stringify(input), source);
console.log('PASS: complete JSON imports, syntax/shape gates, unchanged tolerances and browser coil mapping');
