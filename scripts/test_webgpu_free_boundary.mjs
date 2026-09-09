import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
const source=readFileSync(new URL('../webgpu/free_boundary.js',import.meta.url),'utf8');
const presets=Object.fromEntries(['solovev','w7x','cth_like'].map(name=>[
  name,JSON.parse(readFileSync(new URL(`../webgpu/presets/${name}.json`,import.meta.url),'utf8'))]));
function fixture(search,saved){
  const nodes=new Map(),storage=new Map(saved?[['cumes.free.v1',JSON.stringify(saved)]]:[]),requests=[],parsed=[],uploads=[];
  const get=id=>{if(!nodes.has(id))nodes.set(id,{value:'',textContent:'',hidden:false,
    classList:{toggle(){}},setAttribute(){},addEventListener(){},click(){}});return nodes.get(id)};
  const location={href:'https://example.test/app/cumes_webgpu.html'+search,search,assign(url){this.next=url}};
  const context=vm.createContext({URL,URLSearchParams,Uint8Array,location,
    document:{body:{dataset:{}},getElementById:get},
    localStorage:{getItem:key=>storage.get(key),setItem:(key,value)=>storage.set(key,value)},
    async readCumesCoils(file){parsed.push(file.name);return{name:file.name,circuits:[]}},
    async fetch(url){requests.push(String(url));const name=String(url).split('/').at(-1);
      return{ok:true,json:async()=>structuredClone(presets[name.replace('.json','')]),
        arrayBuffer:async()=>new Uint8Array([1,2,3]).buffer}}});
  vm.runInContext(source,context);
  context.cumesCoilStore=async value=>{if(value)uploads.push(value);return uploads.at(-1)};
  return{app:context.installCumesBoundaryMode(),get,location,storage,requests,parsed,uploads};
}
const f=fixture('?boundary=free&coils=cth_like&run=1',
  {preset:'solovev',input:presets.solovev});
await f.app.ready;
assert.equal(f.app.input().nfp,5,'an explicit preset must override the previous setup');
assert.equal(f.get('coil-preset').value,'cth_like');
assert.equal(f.requests.length,2,'free-boundary setup loads geometry for its preview');
assert.equal(f.app.geometry().name,'coils.cth_like');
const files=await f.app.files();
assert.equal(files[0].path,'/inputs/coils.cth_like');
assert.equal(f.requests.length,2,'the solve must reuse the preview coil bytes');
assert.equal(f.parsed.length,1,'preview geometry is parsed once per selection');
assert.equal(files[0].bytes.length,3);
f.get('coil-currents').value='[1200, -500]';
assert.deepEqual([...f.app.input().extcur],[1200,-500]);
f.get('boundary-fixed').onclick();
const next=new URL(f.location.next);
assert.equal(next.searchParams.get('boundary'),'fixed');
assert.equal(next.searchParams.has('run'),false);
assert.equal(next.searchParams.has('coils'),false);
assert.deepEqual(JSON.parse(f.storage.get('cumes.free.v1')).input.extcur,[1200,-500]);
f.get('coil-currents').value='[null]';
assert.throws(()=>f.app.input(),/finite coil currents/);
const fixed=fixture('?boundary=fixed');await fixed.app.ready;
assert.equal(fixed.requests.length,0,'fixed-boundary startup must not fetch coil assets');
console.log('PASS: lazy presets, explicit selection, editable currents, validation, and mode-switch persistence');

await f.get('coil-preset').onchange({target:{value:'w7x'}});
assert.equal(f.app.geometry().name,'coils.w7x');
assert.equal((await f.app.files())[0].path,'/inputs/coils.w7x');
assert.equal(f.requests.length,4);
assert.equal(f.parsed.length,2);
console.log('PASS: preview and solver replace their geometry together on preset changes');

const previous=f.app.geometry();
const upload={target:{files:[{name:'custom.json',arrayBuffer:async()=>new Uint8Array([4,5,6]).buffer}]}};
f.get('coil-currents').value='[null]';
await f.get('coil-upload').onchange(upload);
assert.equal(f.uploads.length,0,'invalid settings must not overwrite the saved upload');
assert.equal(f.app.geometry(),previous,'a rejected upload keeps the last valid preview');
f.get('coil-currents').value='[1200]';
await f.get('coil-upload').onchange(upload);
assert.equal(f.get('coil-preset').value,'upload');
assert.equal(f.app.geometry().name,'coils.json');
assert.equal(f.app.input().coils_file,'/inputs/coils.json');
assert.deepEqual([...(await f.app.files())[0].bytes],[4,5,6]);
assert.equal(f.uploads.length,1);
console.log('PASS: uploaded coils update preview and solver only after validation');
