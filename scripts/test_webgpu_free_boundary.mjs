import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
const source=readFileSync(new URL('../webgpu/free_boundary.js',import.meta.url),'utf8');
const presets=Object.fromEntries(['solovev','w7x','cth_like'].map(name=>[
  name,JSON.parse(readFileSync(new URL(`../webgpu/presets/${name}.json`,import.meta.url),'utf8'))]));
function fixture(search,saved){
  const nodes=new Map(),storage=new Map(saved?[['cumes.free.v1',JSON.stringify(saved)]]:[]),requests=[];
  const get=id=>{if(!nodes.has(id))nodes.set(id,{value:'',textContent:'',hidden:false,
    classList:{toggle(){}},setAttribute(){},addEventListener(){},click(){}});return nodes.get(id)};
  const location={href:'https://example.test/app/cumes_webgpu.html'+search,search,assign(url){this.next=url}};
  const context=vm.createContext({URL,URLSearchParams,Uint8Array,location,
    document:{body:{dataset:{}},getElementById:get},
    localStorage:{getItem:key=>storage.get(key),setItem:(key,value)=>storage.set(key,value)},
    async fetch(url){requests.push(String(url));const name=String(url).split('/').at(-1);
      return{ok:true,json:async()=>structuredClone(presets[name.replace('.json','')]),
        arrayBuffer:async()=>new Uint8Array([1,2,3]).buffer}}});
  vm.runInContext(source,context);
  return{app:context.installCumesBoundaryMode(),get,location,storage,requests};
}
const f=fixture('?boundary=free&coils=cth_like&run=1',
  {preset:'solovev',input:presets.solovev});
await f.app.ready;
assert.equal(f.app.input().nfp,5,'an explicit preset must override the previous setup');
assert.equal(f.get('coil-preset').value,'cth_like');
assert.equal(f.requests.length,1,'setup should load only the small configuration');
const files=await f.app.files();
assert.equal(files[0].path,'/inputs/coils.cth_like');
assert.equal(f.requests.length,2,'geometry should be fetched only when requested');
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
