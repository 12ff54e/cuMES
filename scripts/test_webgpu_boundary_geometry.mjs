import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
const context=vm.createContext({});
vm.runInContext(readFileSync(new URL('../webgpu/equilibrium_view.js',import.meta.url),'utf8'),context);
const close=(actual,expected)=>assert.ok(Math.abs(actual-expected)<2e-13,`${actual} != ${expected}`);
// Independent signed-n VMEC formula versus folded solver-output representation.
for(const name of ['w7x','cth_like','solovev']){
  const input=JSON.parse(readFileSync(new URL(`../webgpu/presets/${name}.json`,import.meta.url),'utf8'));
  const fourier=context.boundaryFourier(input),c=fourier.surfaces[0].coefficients;
  for(const theta of [0,.2,1.3,Math.PI,5.2])for(const phi of [0,.137,.41,1.12]){
    const actual=context.fourierPoint(fourier,c,theta,phi);
    for(const [i,family,trig] of [[0,'rbc',Math.cos],[1,'zbs',Math.sin]]){
      const reference=input[family].filter(e=>e.m<input.mpol&&Math.abs(e.n)<=input.ntor)
        .reduce((sum,e)=>sum+e.value*trig(e.m*theta-e.n*(input.nfp||1)*phi),0);
      close(actual[i],reference);
    }
    const periodic=context.fourierPoint(fourier,c,theta,phi+2*Math.PI/fourier.nfp);
    const mirrored=context.fourierPoint(fourier,c,-theta,-phi);
    close(periodic[0],actual[0]);close(periodic[1],actual[1]);
    close(mirrored[0],actual[0]);close(mirrored[1],-actual[1]);
  }
  const section=context.fourierSections(fourier,.2)[0];
  close(section[0][0],section.at(-1)[0]);close(section[0][1],section.at(-1)[1]);
  const mesh=context.equilibriumMesh(fourier);
  assert(mesh.surfaces[0].points.every(Number.isFinite));
}
assert.throws(()=>context.boundaryFourier({mpol:1e9}),/valid mpol/);
console.log('PASS: VMEC signed-n geometry, stellarator symmetry, field periodicity, closed sections and finite meshes');
vm.runInContext(readFileSync(new URL('../webgpu/boundary_editor.js',import.meta.url),'utf8'),context);
context.URL=URL;
let migrated;
context.migrateCumesEditorUrl({href:'https://example.test/app?solve=w7x&grids=3&precision=float&fft=1&run=1#result'},
  {replaceState(_state,_title,url){migrated=new URL(url)}});
assert.equal(migrated.searchParams.get('preset'),'w7x');
assert.equal(migrated.searchParams.get('boundary'),'fixed');
assert(!migrated.searchParams.has('solve'));assert(!migrated.searchParams.has('run'));
assert.equal(migrated.searchParams.get('grids'),'3');assert.equal(migrated.searchParams.get('fft'),'1');
assert.equal(migrated.searchParams.get('precision'),'float');assert.equal(migrated.hash,'#result');
console.log('PASS: legacy W7-X links enter the editor without losing precision/grid/FFT settings or starting a solve');
