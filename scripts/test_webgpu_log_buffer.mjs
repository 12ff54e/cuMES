import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
const context=vm.createContext({});
vm.runInContext(readFileSync(new URL('../webgpu/browser_ui.js',import.meta.url),'utf8'),context);
let data='',layouts=0,appends=0,next=0;
const jobs=new Map();
const output={ownerDocument:{createTextNode:()=>({appendData(value){data+=value;appends++}})},
  append(){},get scrollHeight(){layouts++;return 500}};
const buffer=context.createCumesLogBuffer(output,(callback,delay)=>{assert.equal(delay,100);jobs.set(++next,callback);return next},id=>jobs.delete(id));
for(let i=0;i<4000;i++)buffer.append(`line ${i}`);
assert.equal(jobs.size,1);assert.equal(layouts,0);assert.equal(data,'');
[...jobs.values()][0]();
assert.equal(jobs.size,0);assert.equal(layouts,1);assert.equal(appends,1);
assert.equal(data,Array.from({length:4000},(_,i)=>`line ${i}\n`).join(''));
buffer.append('final');buffer.flush();
assert.equal(jobs.size,0);assert.equal(layouts,2);assert(data.endsWith('final\n'));
buffer.flush();assert.equal(layouts,2);
console.log('PASS: batched ordered logs, one layout per flush, terminal flush, no dropped lines');
