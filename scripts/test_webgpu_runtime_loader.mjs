import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
const source=readFileSync(new URL('../webgpu/browser_ui.js',import.meta.url),'utf8');
function fixture(search,unavailable=false){
  const sent=[],scripts=[],logs=[],results=[],listeners=new Map();let worker;
  const events={addEventListener:(name,handler)=>listeners.set(name,handler),removeEventListener:name=>listeners.delete(name)};
  const runtime={src:'cumes_webgpu.js?v=abc',getAttribute:()=> 'cumes_webgpu.js?v=abc'};
  const document={...events,visibilityState:'visible',body:{dataset:{},append:script=>scripts.push(script)},
    getElementById:()=>({content:{querySelector:()=>runtime}}),createElement:()=>({})};
  const context=vm.createContext({URL,URLSearchParams,document,...events,
    location:{search,href:'https://example.test/app/cumes_webgpu.html'+search},
    Module:{print:line=>logs.push(line)},cumesAppendLog:line=>logs.push(line),
    cumesBrowser:{result:(...args)=>results.push(args),diagnostic(){},output(){},adapter(){},ready(){},equilibrium(){},error(){}},
    Worker:class{
      constructor(url){if(unavailable)throw Error('unavailable');worker=this;this.url=String(url);this.stopped=false}
      postMessage(message){sent.push(message)}
      terminate(){this.stopped=true}
    }});
  vm.runInContext(source,context);context.startCumesRuntime();
  return{context,document,worker,sent,scripts,logs,results,listeners};
}
for(const query of ['', '?run=1&precision=double', '?solve=w7x', '?mode=test&solve=w7x', '?mode=test&worker=0']){
  const f=fixture(query);
  assert.equal(f.document.body.dataset.cumesExecution,'main');assert.equal(f.worker,undefined);
  assert.equal(f.scripts.length,1);assert.equal(f.scripts[0].src,'https://example.test/app/cumes_webgpu.js?v=abc');
  f.scripts[0].onerror();assert.equal(f.results[0][0],false);
}
const f=fixture('?mode=test&trace=1');
assert.equal(f.scripts.length,0);assert.equal(f.document.body.dataset.cumesExecution,'worker');
assert.equal(f.worker.url,'https://example.test/app/verification_worker.js?v=abc');
assert.equal(f.sent[0].runtimeUrl,'https://example.test/app/cumes_webgpu.js?v=abc');
assert.equal(f.sent[0].search,'?mode=test&trace=1');
f.worker.onmessage({data:{kind:'logs',lines:['first','second']}});assert.deepEqual(f.logs,['first','second']);
f.document.visibilityState='hidden';f.listeners.get('visibilitychange')();assert.equal(f.sent.at(-1).value,'hidden');
f.worker.onmessage({data:{kind:'result',args:[true,'PASS',{}]}});
assert.equal(f.results[0][0],true);assert(f.worker.stopped);assert.equal(f.worker.onmessage,null);assert.equal(f.listeners.size,0);
for(const cause of ['pagehide','error','messageerror','unexpected']){
  const f=fixture('?mode=test');
  if(cause==='pagehide')f.listeners.get('pagehide')();
  if(cause==='error')f.worker.onerror({message:'test',preventDefault(){}});
  if(cause==='messageerror')f.worker.onmessageerror();
  if(cause==='unexpected')f.worker.onmessage({data:{kind:'bad'}});
  assert(f.worker.stopped);assert.equal(f.listeners.size,0);
  if(cause!=='pagehide')assert.equal(f.results[0][0],false);
}
const missing=fixture('?mode=test',true);assert.equal(missing.results[0][0],false);assert.equal(missing.scripts.length,0);
console.log('PASS: verification-only worker, versioned URLs, main-thread opt-out, messages, errors, navigation cleanup');
