// Dependency-free canvas/bridge checks; real GPU coverage lives in the smoke tests.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
const source=readFileSync(new URL('../webgpu/residual_plot.js',import.meta.url),'utf8');
const jobs=new Map(),frames=new Map(),calls=[],events={},outputs=Array.from({length:3},()=>({textContent:''}));
let id=0,resize;
const context=new Proxy({}, {get:(_,method)=>(...args)=>{
  for(const arg of args)if(typeof arg==='number')assert.ok(Number.isFinite(arg),`${method}: ${arg}`);
  calls.push([method,...args]);
},set:()=>true});
const canvas={width:0,height:0,getContext:()=>context,getBoundingClientRect:()=>({width:640,height:250}),setAttribute(){}};
const caption={textContent:''};
const doc={visibilityState:'visible',addEventListener(name,fn){events[name]=fn}};
const panel={ownerDocument:doc,querySelector:s=>s==='canvas'?canvas:caption,querySelectorAll:()=>outputs};
const sandbox=vm.createContext({panel,performance:{now:()=>0},devicePixelRatio:2,
  setTimeout(fn,delay){assert.equal(delay,100);jobs.set(++id,fn);return id},clearTimeout:id=>jobs.delete(id),
  requestAnimationFrame(fn){frames.set(++id,fn);return id},cancelAnimationFrame:id=>frames.delete(id),
  ResizeObserver:class {constructor(fn){resize=fn}observe(target){assert.equal(target,canvas)}}});
vm.runInContext(source,sandbox);
const envelope=(rows,projectX=()=>0)=>sandbox.cumesResidualEnvelope(rows,0,projectX);
const sample=(attempt,fsq=[1e-4,1e-8,1e-12],stage=1,iteration=attempt)=>({stage,attempt,iteration,fsq,tolerance:1e-12,converged:false});
const bucket=[8,2,9,5,6].map((v,i)=>({...sample(i+1,[v,v,v]),x:i+1}));
assert.deepEqual(Array.from(envelope(bucket),r=>r?.x??null),[null,1,2,3,5],'first/min/max/last kept in sample order');
const many=Array.from({length:10000},(_,i)=>({...sample(i+1,[1+(i%17),1,1]),x:i}));
assert.ok(envelope(many,x=>x/100).length<=401,'draw work bounded by pixel buckets');
const gaps=[1,0,2,-1,3,NaN,4,Infinity,5].map((v,i)=>({...sample(i+1,[v,v,v]),x:i+1}));
assert.deepEqual(Array.from(envelope(gaps),r=>r?.x??null),[null,1,null,3,null,5,null,7,null,9]);
assert.deepEqual(Array.from(envelope([bucket[0],{...bucket[1],stage:2}]),r=>r?.x??null),[null,1,null,2],'do not connect grids');
const plot=vm.runInContext('createCumesResidualPlot(panel, 1e-12)',sandbox);
const tick=()=>{
  for(const [key,fn]of [...jobs]){jobs.delete(key);fn()}
  for(const [key,fn]of [...frames]){frames.delete(key);fn()}
};
tick();assert.equal(canvas.width,1280);assert.equal(canvas.height,500);
assert.ok(calls.some(c=>c[0]==='fillText'&&c[1]==='1e-13'));
assert.ok(calls.some(c=>c[0]==='fillText'&&c[1]==='target 1e-12'));
const original=sample(1);plot.append(original);original.fsq[0]=999;
for(let i=2;i<=100;i++)plot.append(sample(i));
assert.equal(jobs.size,1,'only one repaint for a burst of samples');
assert.equal(plot.report().drawCount,1,'append never draws synchronously');
assert.equal(plot.report().samples[0].fsq[0],1e-4,'copy incoming values');
tick();assert.match(caption.textContent,/Live/);assert.equal(outputs[2].textContent,'1.000e-12');
const ys=[1e-4,1e-8,1e-12].map(value=>212-(Math.log10(value)+13)/13*182);
const dots=calls.filter(c=>c[0]==='arc').slice(-3);
dots.forEach((c,i)=>assert.ok(Math.abs(c[2]-ys[i])<1e-10));
assert.ok(Math.abs((dots[1][2]-dots[0][2])-(dots[2][2]-dots[1][2]))<1e-10,'equal decades occupy equal vertical distances');
plot.append(sample(101,[1e-20,0,Infinity],1,2)); // effective-iteration restart
plot.append(sample(1,[2,1e-10,1e-14],2));
plot.append(sample(1,[3,2,1],2)); // duplicate
plot.append(sample(999,[1,2,3],1)); // stale stage
plot.append(sample(0));plot.append(sample(3,[1,2]));
assert.equal(plot.report().samples.length,102);
assert.deepEqual(Array.from(plot.report().samples.slice(-2),r=>r.x),[101,102]);
assert.equal(plot.report().samples[100].iteration,2);
assert.equal(plot.report().minLog,-20);assert.equal(plot.report().maxLog,1);
assert.deepEqual(Array.from(plot.report().stages,r=>r.x),[1,102]);
const snapshot=plot.report();snapshot.samples[0].fsq[0]=999;
assert.equal(plot.report().samples[0].fsq[0],1e-4,'report cannot mutate the plot');
doc.visibilityState='hidden';tick();const before=plot.report().drawCount;
plot.append(sample(2,[1e-13,1e-14,1e-15],2));plot.finish(true);
assert.equal(plot.report().drawCount,before,'no hidden-tab redraws, even on completion');
assert.equal(jobs.size,0);doc.visibilityState='visible';events.visibilitychange();tick();
assert.match(caption.textContent,/Converged · grid 2/);assert.equal(plot.report().drawCount,before+1);
resize();assert.equal(jobs.size,1);plot.finish(false);
assert.equal(jobs.size,0);assert.equal(frames.size,0);assert.match(caption.textContent,/Stopped/);
// Exercise the actual Emscripten scalar import and its payload (not log parsing).
const library={};let emitted;
vm.runInNewContext(readFileSync(new URL('../webgpu/browser_bridge.js',import.meta.url),'utf8'),{
  LibraryManager:{library},mergeInto:Object.assign,cumesBrowser:{residual:row=>{emitted=row}}});
library.publish_browser_residual(2,19,17,1e-13,2e-14,3e-15,1e-12,1);
assert.deepEqual(JSON.parse(JSON.stringify(emitted)),{stage:2,attempt:19,iteration:17,fsq:[1e-13,2e-14,3e-15],tolerance:1e-12,converged:true});
console.log('PASS: log coordinates, scalar bridge, exact samples, spikes/gaps, stage/restart axes, coalesced/hidden/final redraws');
