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
}
// Complementary signed-n harmonics, including vertical offset and m=0,
// must follow the full VMEC formula without reflecting the section.
const asymmetric = {mpol:4, ntor:2, nfp:3, lasym:true,
  rbc:[{m:0,n:0,value:4},{m:1,n:0,value:1}],
  zbs:[{m:1,n:0,value:1.4}],
  rbs:[{m:1,n:0,value:.1},{m:2,n:-1,value:.07},{m:0,n:2,value:.03}],
  zbc:[{m:0,n:0,value:.2},{m:1,n:1,value:.08},{m:2,n:-2,value:.04}]};
const full = context.boundaryFourier(asymmetric);
assert.equal(full.surfaces[0].coefficients.length, 12 * 4 * 3);
for (const theta of [.2, 1.3, 4.2]) for (const phi of [0, .17, .41]) {
  const actual = context.fourierPoint(full, full.surfaces[0].coefficients, theta, phi);
  for (const [i, even, odd] of [[0,'rbc','rbs'],[1,'zbc','zbs']]) {
    const phase = h => h.m * theta - h.n * asymmetric.nfp * phi;
    const expected = asymmetric[even].reduce((v,h)=>v+h.value*Math.cos(phase(h)),0) +
      asymmetric[odd].reduce((v,h)=>v+h.value*Math.sin(phase(h)),0);
    close(actual[i], expected);
  }
}
assert.notEqual(context.fourierPoint(full,full.surfaces[0].coefficients,.4,0)[1],
  -context.fourierPoint(full,full.surfaces[0].coefficients,-.4,0)[1]);
console.log('PASS: asymmetric signed-n boundary, vertical offset and all complementary families');
assert.throws(()=>context.boundaryFourier({mpol:1e9}),/valid mpol/);
console.log('PASS: VMEC signed-n geometry, stellarator symmetry, field periodicity and closed sections');
vm.runInContext(readFileSync(new URL('../webgpu/boundary_editor.js',import.meta.url),'utf8'),context);
// Analytic planar curve: recover both parities, its vertical offset, and modes
// above the symmetric demo's m=5 cap at the requested resolution.
const curve = theta => [4 + .9*Math.cos(theta) + .06*Math.sin(theta) + .02*Math.sin(7*theta),
  .12 + 1.3*Math.sin(theta) + .08*Math.cos(theta) - .025*Math.cos(9*theta)];
const fit = context.fitTokamakContour(curve, 10, true);
const expected = {rbc:{0:4,1:.9}, zbs:{1:1.3}, rbs:{1:.06,7:.02}, zbc:{0:.12,1:.08,9:-.025}};
for (const family of Object.keys(expected)) for (let m=0;m<10;m++)
  close(fit[family][m], expected[family][m] || 0);
const symmetricFit = context.fitTokamakContour(curve, 10, false);
assert.deepEqual(Object.keys(symmetricFit), ['rbc','zbs']);
for (const family of ['rbc','zbs']) for (let m=0;m<10;m++)
  close(symmetricFit[family][m], expected[family][m] || 0);
const points = Array.from({length:16},(_,i)=>curve(2*Math.PI*i/16));
for (const index of [0,3,8]) {
  const moved = structuredClone(points), target = [4.8,.37];
  context.moveContourPoint(moved,index,target,true);
  for (let i=0;i<points.length;i++) assert.deepEqual([...moved[i]],i===index?target:points[i]);
}
const mirrored = Array.from({length:16},(_,i)=>[4+Math.cos(2*Math.PI*i/16),Math.sin(2*Math.PI*i/16)]);
context.moveContourPoint(mirrored,3,[4.8,.37],false);
close(mirrored[3][0],mirrored[13][0]);close(mirrored[3][1],-mirrored[13][1]);
context.moveContourPoint(mirrored,0,[5.1,.37],false);
close(mirrored[0][1],0);
console.log('PASS: contour parity, vertical offset, selected Fourier resolution and independent asymmetric handles');
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
