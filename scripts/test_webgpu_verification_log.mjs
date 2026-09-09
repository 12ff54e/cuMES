import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
const source=readFileSync(new URL('../webgpu/browser_ui.js',import.meta.url),'utf8');
function fixture(){
  const jobs=new Map();let id=0;
  const context=vm.createContext({setTimeout(fn){jobs.set(++id,fn);return id},clearTimeout:id=>jobs.delete(id)});
  vm.runInContext(source,context);
  const element=()=>{
    const nodes=[];
    return {ownerDocument:{createTextNode:()=>({data:'',appendData(text){this.data+=text}})},
      append(node){nodes.push(node)},get textContent(){return nodes.map(node=>node.data).join('')},scrollHeight:100};
  };
  const output=element(),full=element(),label={},progress={};
  const details={open:false,querySelector:tag=>tag==='pre'?full:label,addEventListener(event,fn){assert.equal(event,'toggle');this.toggle=fn}};
  const log=context.createCumesVerificationLog(output,details,progress);
  return {log,output,full,label,progress,details,jobs};
}
const f=fixture(),raw=[];
const append=line=>{raw.push(line);f.log.append(line)};
append('adapter selected: RTX (discrete, webgpu)');
append('  linear: PASS (max |GPU-CPU| = 4.768e-07)');
append('  parsed W7-X 3-D cold start: PASS (ns=33)');
append('  W7-X controller-complete two-pass slice: PASS (FSQR=1.141e+01)');
for(let i=0;i<4000;i++)append('  Solovev MHD force: PASS (max |GPU-CPU| = 0.000e+00)');
append('  host controller: PASS (iter=10, FSQR=1.000e-05, delta=9.000e-01)');
assert.match(f.progress.textContent,/Solovev · grid 1\/3 · iteration 10/);
for(const [stage,iterations] of [[1,72],[2,31],[3,224]])
  append(`  Solovev stage ${stage}/3 converged: iter=${iterations} residual=(9.869e-07, 2.739e-07, 4.241e-10)`);
append('  published schema-v8 output: 118736 bytes');
append('cuMES WebGPU self-test: PASS — Solovev converged');
f.log.finish(true,'Solovev converged');
assert.equal(f.jobs.size,0);assert.equal(f.full.textContent,'','closed details must not render thousands of lines');
assert.equal(f.output.textContent.trim().split('\n').length,8);
assert.match(f.output.textContent,/GPU operator checks: PASS/);
assert.match(f.output.textContent,/W7-X integration: PASS/);
assert.match(f.output.textContent,/Solovev grid 3\/3: PASS — 224 iterations/);
assert.equal(f.progress.textContent,'All checks passed.');
assert.equal(f.log.text(),raw.join('\n')+'\n');
assert.equal(f.label.textContent,`Detailed log (${raw.length} lines)`);
f.details.open=true;f.details.toggle();
assert.equal(f.full.textContent,f.log.text());
f.details.open=false;f.details.toggle();append('WARNING: diagnostic warning');f.log.flush();
assert.match(f.output.textContent,/WARNING: diagnostic warning/);
assert.notEqual(f.full.textContent,f.log.text());
f.details.open=true;f.details.toggle();f.details.toggle();
assert.equal(f.full.textContent,f.log.text(),'reopening must not duplicate lines');
const failed=fixture();failed.log.append('FATAL: GPU lost');failed.log.finish(false,'Device lost');
assert.match(failed.output.textContent,/FATAL: GPU lost/);assert.match(failed.output.textContent,/Verification: FAIL — Device lost/);
assert(!failed.output.textContent.includes('PASS'));assert.equal(failed.progress.textContent,'Verification failed.');
console.log('PASS: concise phases, live progress, full lazy details, ordered replay, warnings, startup failures, final flush');
