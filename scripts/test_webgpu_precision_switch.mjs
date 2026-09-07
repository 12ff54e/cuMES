import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

const shell=readFileSync(new URL('../webgpu/shell.html',import.meta.url),'utf8');
const begin=shell.indexOf('    function installPrecisionSwitch(');
const end=shell.indexOf('    installPrecisionSwitch(document,location,',begin);
assert(begin>=0&&end>begin);
const context=vm.createContext({URL});
vm.runInContext(shell.slice(begin,end),context);

function setup(search){
  const nodes=new Map();
  for(const id of ['', 'editor-'].flatMap(prefix=>['precision-control','precision-single','precision-double','precision-tolerance'].map(id=>prefix+id))){
    const node={hidden:true,attributes:{},active:false,
      setAttribute(key,value){this.attributes[key]=value},
      addEventListener(event,callback){assert.equal(event,'click');this.click=callback}};
    node.classList={toggle(key,value){assert.equal(key,'active');node.active=value}};
    nodes.set(id,node);
  }
  const document={body:{dataset:{}},getElementById:id=>nodes.get(id)};
  const location={href:`https://example.test/cumes.html${search}`,assign(url){this.assigned=url}};
  context.installPrecisionSwitch(document,location,()=>{location.prepared=true});
  return{nodes,document,location};
}
for(const initial of ['', '&precision=double','&precision=float']){
  const {nodes,document,location}=setup(`?solve=w7x&grids=3&fft=1&gpu_norms=1${initial}#result`);
  const single=initial.endsWith('float');
  assert.equal(nodes.get('precision-control').hidden,false);
  assert.equal(document.body.dataset.cumesPrecision,single?'float':'double');
  assert.equal(nodes.get('precision-tolerance').textContent,`Tolerance: ${single?'1e-5':'1e-12'}`);
  for(const id of ['precision-single','precision-double'])
    assert.equal(nodes.get(id).attributes['aria-pressed'],String((id==='precision-single')===single));
  nodes.get(single?'precision-single':'precision-double').click();
  assert.equal(location.assigned,undefined,'active mode must not restart');
  assert.equal(location.prepared,undefined,'active mode must not rewrite input');
  nodes.get(single?'precision-double':'precision-single').click();
  const next=new URL(location.assigned);
  assert.equal(location.prepared,true,'save current input before navigating');
  assert.equal(next.searchParams.get('precision'),single?'double':'float');
  assert.equal(next.searchParams.get('grids'),'3');
  assert.equal(next.searchParams.get('fft'),'1');
  assert.equal(next.searchParams.get('gpu_norms'),'1');
  assert.equal(next.hash,'#result');
}
for(const search of ['','?run=1','?precision=double','?run=1&precision=double']){
  const {nodes,location,document}=setup(search);
  const selected=search.includes('double')?'double':'float';
  assert.equal(nodes.get('editor-precision-control').hidden,false);
  assert.equal(nodes.get('precision-control').hidden,true);
  assert.equal(document.body.dataset.cumesPrecision,selected);
  nodes.get(selected==='float'?'editor-precision-double':'editor-precision-single').click();
  const next=new URL(location.assigned);
  assert.equal(next.searchParams.get('precision'),selected==='float'?'double':'float');
  assert.equal(next.searchParams.get('run'),search.includes('run=1')?'1':null);
}
for(const search of ['?mode=test']){
  const {nodes}=setup(search);
  assert.equal(nodes.get('precision-control').hidden,true);
  assert.equal(nodes.get('precision-single').click,undefined);
}
console.log('Precision switch: PASS (defaults, both directions, active no-op, URL preservation, scope)');
