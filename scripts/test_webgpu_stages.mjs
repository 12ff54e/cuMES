import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

const context = vm.createContext({});
vm.runInContext(readFileSync(new URL('../webgpu/boundary_editor.js', import.meta.url), 'utf8'), context);
const copy = value => JSON.parse(JSON.stringify(value));
const input = {ns_array:[5,11,55], niter_array:[1200,2500,3000], ftol_array:[1e-3,1e-4,1e-5], am:[1,-1]};
const stages = context.cumesValidateStages(input, 'float');
assert.deepEqual(copy(stages), {ns_array:input.ns_array, niter_array:input.niter_array, ftol_array:input.ftol_array});
stages.ns_array[0] = 7;
assert.equal(input.ns_array[0], 5, 'Reading the schedule must not alias its input');
assert.deepEqual(copy(context.cumesStageArrays({})), {ns_array:[11], niter_array:[1000], ftol_array:[1e-16]});
for (const bad of [
  {...input, ns_array:[]}, {...input, niter_array:[10]}, {...input, ftol_array:null},
  {...input, ns_array:[2,11,55]}, {...input, ns_array:[5,11,513]},
  {...input, ns_array:[5,5,55]}, {...input, ns_array:[5,55,11]},
  {...input, ns_array:[5,11.5,55]}, {...input, niter_array:[0,100,100]},
  {...input, niter_array:[1,2.5,3]}, {...input, niter_array:[1,2,2147483648]},
  {...input, ftol_array:[1e-3,NaN,1e-5]}, {...input, ftol_array:[1e-3,Infinity,1e-5]},
  {...input, ftol_array:[1e-3,0,1e-5]}, {...input, ftol_array:[1e-3,1e-7,1e-5]}
]) assert.throws(() => context.cumesValidateStages(bad, 'float'));
assert.doesNotThrow(() => context.cumesValidateStages({...input, ftol_array:[1e-6,1e-6,1e-6]}, 'float'));
assert.doesNotThrow(() => context.cumesValidateStages({...input, ftol_array:[1e-12,1e-14,1e-16]}, 'double'));
assert.throws(() => context.cumesValidateStages({...input, ftol_array:[1e-12,1e-14,1e-17]}, 'double'));

const one = context.cumesResizeStages(input, 1);
assert.deepEqual(copy(one), {ns_array:[55], niter_array:[3000], ftol_array:[1e-5]});
const three = context.cumesResizeStages(one, 3);
assert.deepEqual(copy(three.ns_array), [18,37,55]);
assert.deepEqual(copy(three.niter_array), [3000,3000,3000]);
assert.deepEqual(copy(three.ftol_array), [1e-5,1e-5,1e-5]);
assert.deepEqual(copy(context.cumesResizeStages(input, 3)), {ns_array:input.ns_array, niter_array:input.niter_array, ftol_array:input.ftol_array});
for (const count of [0,1.5,54,NaN]) assert.throws(() => context.cumesResizeStages(input, count));
for (const count of [1,2,3,7,53]) {
  const resized = context.cumesResizeStages(input, count);
  assert.equal(resized.ns_array.length, count);
  assert.equal(resized.ns_array.at(-1), 55, 'Stage count must retain the final resolution');
  assert.equal(resized.niter_array.at(-1), 3000);
  context.cumesValidateStages(resized, 'float');
}

const paired = context.cumesStagePrecision(input, 'float', 'double');
assert.deepEqual(copy(paired.ftol_array), [1e-3,1e-4,1e-12], 'Only the default tolerance follows precision');
assert.deepEqual(copy(context.cumesStagePrecision(paired, 'double', 'float').ftol_array), input.ftol_array);
assert.throws(() => context.cumesStagePrecision({...input, ftol_array:[1e-3,1e-8,1e-12]}, 'double', 'float'),
  /at least 0.000001/, 'Custom tolerances must not be silently relaxed');
assert.deepEqual(input.am, [1,-1]);
console.log('PASS: stage array contracts, precision floors, final-grid preservation and explicit tolerance retention');
