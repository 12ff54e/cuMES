import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
const source=readFileSync(new URL('../webgpu/verification_worker.js',import.meta.url),'utf8');
const librarySource=readFileSync(new URL('../webgpu/browser_bridge.js',import.meta.url),'utf8');
function fixture(failImport=false){
  const messages=[],jobs=new Map(),listeners={},library={};let id=0,imports=0,now=123;
  const context=vm.createContext({URL,URLSearchParams,location:{search:'?wrong=worker-script-url'},
    performance:{now:()=>now},
    setTimeout(fn){jobs.set(++id,fn);return id},clearTimeout(id){jobs.delete(id)},
    postMessage(message,transfer=[]){messages.push(structuredClone(message,{transfer}))},
    addEventListener(name,handler){listeners[name]=handler},
    importScripts(){imports++;if(failImport)throw Error('blocked runtime')},
    LibraryManager:{library},mergeInto:Object.assign,UTF8ToString:value=>value,
    FS:{readFile:()=>new Uint8Array([1,2,3,4])}});
  vm.runInContext(source,context);vm.runInContext(librarySource,context);
  const start=()=>context.onmessage({data:{kind:'start',runtimeUrl:'https://example.test/cumes_webgpu.js?v=abc',
    search:'?mode=test&precision=double&trace=1&timing=0',visibility:'visible'}});
  return{context,messages,jobs,listeners,library,start,imports:()=>imports,time:value=>{now=value}};
}
const f=fixture();f.start();f.start();
assert.equal(f.imports(),1);
assert.equal(f.library.requested_app_mode(),false);
assert.equal(f.library.requested_double_solve(),1);
assert.equal(f.library.requested_solver_trace(),true);
assert.equal(f.library.requested_webgpu_vacuum(),false);
assert.equal(f.library.requested_webgpu_vacuum_lu(),false);
assert.equal(f.library.requested_device_vacuum_force(),false);
{
  const saved=f.context.cumesSearch;
  f.context.cumesSearch='?boundary=free&vacuum=webgpu';
  assert.equal(f.library.requested_webgpu_vacuum(),true);
  assert.equal(f.library.requested_webgpu_vacuum_lu(),false);
  assert.equal(f.library.requested_device_vacuum_force(),true);
  f.context.cumesSearch='?boundary=free&vacuum=webgpu&vacuum_force=host';
  assert.equal(f.library.requested_device_vacuum_force(),false);
  f.context.cumesSearch='?boundary=free&vacuum=webgpu&vacuum_lu=webgpu';
  assert.equal(f.library.requested_webgpu_vacuum_lu(),true);
  f.context.cumesSearch='?boundary=free&vacuum=webgpu&vacuum_lu=host';
  assert.equal(f.library.requested_webgpu_vacuum_lu(),false);
  f.context.cumesSearch='?boundary=free&vacuum=host';
  assert.equal(f.library.requested_webgpu_vacuum(),false);
  assert.equal(f.library.requested_device_vacuum_force(),false);
  f.context.cumesSearch='?boundary=free&vacuum=host&vacuum_force=webgpu';
  assert.equal(f.library.requested_device_vacuum_force(),true);
  f.context.cumesSearch=saved;
}
assert.equal(f.library.requested_newton_solve(),false);
assert.equal(f.library.requested_newton_step(),1e-6);
{
  const saved=f.context.cumesSearch;
  f.context.cumesSearch='?newton=1&newton_step=0.00003&newton_probe=1';
  assert.equal(f.library.requested_newton_solve(),true);
  assert.equal(f.library.requested_newton_step(),0.00003);
  assert.equal(f.library.requested_newton_probe(),true);
  f.context.cumesSearch='?newton_step=invalid';
  assert.ok(Number.isNaN(f.library.requested_newton_step()));
  f.context.cumesSearch=saved;
}
for(const [geometry,expected] of [['',1],['native',0],['compensated',1],['compensated-m1',2]]){
  const saved=f.context.cumesSearch;
  f.context.cumesSearch=`?geometry=${geometry}`;
  assert.equal(f.library.requested_compensated_geometry(),expected);
  f.context.cumesSearch=saved;
}
assert.equal(f.context.Module.locateFile('cumes_webgpu.wasm'),'https://example.test/cumes_webgpu.wasm?v=abc');
assert.equal(f.context.cumesVisibility,'visible');
f.context.onmessage({data:{kind:'visibility',value:'hidden'}});
assert.equal(f.context.cumesVisibility,'hidden');
for(let i=0;i<1000;i++)f.context.Module.print(`line ${i}`);
f.library.publish_browser_diagnostic('{"kind":"controller","iter":1}');
assert.equal(f.messages.length,0);assert.equal(f.jobs.size,1);
// A transferred output buffer is detached in the worker: return its saved size.
assert.equal(f.library.publish_browser_output('/output.bin'),4);
assert.deepEqual([...f.messages[0].bytes],[1,2,3,4]);
f.library.publish_browser_adapter('gpu','discrete','webgpu');
f.library.publish_browser_equilibrium('{"surfaces":[]}');
const timing={stats:{wall:{average:2}}};
f.context.cumesIterationTiming={finish(){f.context.cumesAppendLog('timing summary');return timing}};
f.library.publish_browser_result(1,'converged');
assert.equal(f.jobs.size,0);
assert.deepEqual(f.messages.map(row=>row.kind),['output','adapter','equilibrium','logs','diagnostics','result']);
assert.equal(f.messages[3].lines.length,1001);assert.equal(f.messages[3].lines.at(-1),'timing summary');
assert.equal(f.messages[4].rows[0].iter,1);
assert.deepEqual(f.messages.at(-1).args,[true,'converged',timing]);
f.context.Module.print('late log');assert.equal(f.jobs.size,0);
f.listeners.error({message:'late error'});assert.equal(f.messages.length,6);
for(const failure of ['import','abort','rejection']){
  const f=fixture(failure==='import');f.start();
  if(failure==='abort')f.context.Module.onAbort('test');
  if(failure==='rejection')f.listeners.unhandledrejection({reason:Error('test')});
  assert.equal(f.messages.at(-1).kind,'result');assert.equal(f.messages.at(-1).args[0],false);
}
console.log('PASS: DOM-free worker bridge, query propagation, batched messages, final flush, output transfer, errors');

const files=fixture();
const written=new Map();
files.context.FS.mkdirTree=()=>{};
files.context.FS.writeFile=(path,bytes)=>written.set(path,bytes);
files.context.onmessage({data:{kind:'start',runtimeUrl:'https://example.test/cumes_webgpu.js',
  search:'?boundary=free&run=1',visibility:'visible',files:[
    {path:'/inputs/interactive.json',bytes:new Uint8Array([123,125])},
    {path:'/inputs/coils.upload',bytes:new Uint8Array([1,2,3])}]}});
for(const preRun of files.context.Module.preRun)preRun();
assert.equal(written.size,2);
assert.deepEqual([...written.get('/inputs/coils.upload')],[1,2,3]);
assert.equal(files.library.requested_app_mode(),true);
console.log('PASS: interactive input and coil bytes reach the worker filesystem');

assert.equal(files.library.requested_double_solve(),1);
files.context.cumesSearch='?boundary=free&run=1&precision=float';
assert.equal(files.library.requested_double_solve(),0);
files.context.cumesSearch='?run=1';
assert.equal(files.library.requested_double_solve(),0);
console.log('PASS: worker precision agrees with free-boundary paired and fixed-boundary float defaults');

const speed=fixture();speed.start();
vm.runInContext(readFileSync(new URL('../webgpu/timestamp_capture.js',import.meta.url),'utf8'),speed.context);
vm.runInContext(readFileSync(new URL('../webgpu/iteration_timing.js',import.meta.url),'utf8'),speed.context);
speed.time(10000);speed.library.publish_browser_iteration_timing(1,1);
speed.time(10500);speed.library.publish_browser_iteration_timing(2,1);
assert.deepEqual(speed.messages,[{kind:'speed',value:null},{kind:'speed',value:2}]);
assert.equal(speed.context.cumesIterationTiming.report().enabled,false);
console.log('PASS: live speed crosses the worker bridge with detailed profiling disabled');
