import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

const source=readFileSync(new URL('../webgpu/runtime_threads.js',import.meta.url),'utf8');
function configured(search,cores){
  const context=vm.createContext({URLSearchParams,cumesSearch:search,navigator:{hardwareConcurrency:cores},Module:{preRun:[]},ENV:{}});
  vm.runInContext(source,context);
  const count=context.cumesThreadCount();
  for(const prepare of context.Module.preRun)prepare();
  assert.equal(context.ENV.VFIELD_MAKEGRID_THREADS,String(count),'MAKEGRID and the prewarmed pool must agree');
  return count;
}
for(const search of ['', '?preset=w7x&run=1', '?mode=test&boundary=free'])assert.equal(configured(search,16),1);
assert.equal(configured('?boundary=free',16),16);
assert.equal(configured('?boundary=free',64),16);
assert.equal(configured('?boundary=free',undefined),1);
for(const [requested,expected] of [[1,1],[4,4],[128,16],[0,16],[-1,16],[1.5,16],['invalid',16]])
  assert.equal(configured('?boundary=free&makegrid_threads='+requested,16),expected);
assert.equal(configured('?boundary=free&makegrid_threads=16',4),4);
console.log('Browser thread configuration: PASS (pool bounds, sequential diagnostic, fixed-mode scope)');
