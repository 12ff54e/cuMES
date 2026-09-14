// Drag real handles, retain edits across navigation, and solve both tokamak parities.
// Usage: node scripts/webgpu_contour_smoke.mjs APP_URL OUTPUT_PREFIX
import {writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
import {connectCdp} from './include/webgpu_cdp.mjs';

const [base, prefix] = process.argv.slice(2);
if (!base || !prefix) throw Error('Pass APP_URL and OUTPUT_PREFIX');
const browser = await (await fetch(`http://127.0.0.1:${process.env.CUMES_CDP_PORT || 9333}/json/version`)).json();
const cdp = await connectCdp(browser.webSocketDebuggerUrl);
let target;
try {
  target = (await cdp.call('Target.createTarget', {url:'about:blank', newWindow:false})).targetId;
  const session = (await cdp.call('Target.attachToTarget', {targetId:target, flatten:true})).sessionId;
  const call = (method, params = {}) => cdp.call(method, params, session);
  const evaluate = async expression => (await call('Runtime.evaluate', {
    expression, returnByValue:true, awaitPromise:true})).result.value;
  const wait = async (expression, timeout = 30000) => {
    const deadline = Date.now() + timeout;
    while (Date.now() < deadline) {
      try { const value = await evaluate(expression); if (value) return value; }
      catch (error) {
        if (!/context|Cannot find|Inspected target navigated/i.test(String(error))) throw error;
      }
      await new Promise(resolve => setTimeout(resolve, 500));
    }
    throw Error(`Timed out waiting for ${expression}`);
  };
  const ready = () => wait(`document.body?.dataset.cumesWebgpu==='ready' && !document.getElementById('run').disabled`);
  const snapshot = () => evaluate(`({input:JSON.parse(inputJSON()), points:shape.contour,
    mode:editorMode, dataset:{...document.body.dataset}, error:document.getElementById('boundary-error').textContent})`);
  const resize = async count => {
    const before = await snapshot();
    await evaluate(`document.getElementById('contour-points').value=${count};document.getElementById('contour-points').dispatchEvent(new Event('input',{bubbles:true}))`);
    const after = await snapshot();
    assert.equal(after.points.length,count);
    assert.equal(await evaluate(`document.querySelectorAll('.handle').length`),count);
    assert.equal(await evaluate(`document.getElementById('contour-points-value').textContent`),String(count));
    assert.deepEqual(after.input,before.input,'Changing handle count must not change the fitted boundary');
  };
  const drag = async (index, dx, dy) => {
    const {x,y} = await evaluate(`(() => {
      const handle=document.querySelector('.handle[data-index="${index}"]');
      handle.scrollIntoView({block:'center'});
      const r=handle.getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2};
    })()`);
    await call('Input.dispatchMouseEvent', {type:'mouseMoved', x, y});
    await call('Input.dispatchMouseEvent', {type:'mousePressed', x, y, button:'left', buttons:1, clickCount:1});
    await call('Input.dispatchMouseEvent', {type:'mouseMoved', x:x+dx, y:y+dy, button:'left', buttons:1});
    await call('Input.dispatchMouseEvent', {type:'mouseReleased', x:x+dx, y:y+dy, button:'left', buttons:0, clickCount:1});
  };
  const solve = async name => {
    const before = await snapshot();
    await evaluate(`document.getElementById('run').click()`);
    await wait(`document.body?.dataset.cumesWebgpu==='pass'||document.body?.dataset.cumesWebgpu==='fail'`, 600000);
    const result = await evaluate(`({dataset:{...document.body.dataset}, input:JSON.parse(inputJSON()),
      points:shape.contour, mode:editorMode, plot:window.cumesResidualPlot.report(),
      log:document.getElementById('output').textContent})`);
    await writeFile(`${prefix}-${name}.json`, JSON.stringify(result));
    assert.equal(result.dataset.cumesWebgpu, 'pass', result.dataset.cumesDetail);
    assert.equal(result.mode, 'contour');
    assert.deepEqual(result.input, before.input, 'Run must use the edited boundary');
    assert.deepEqual(result.points, before.points, 'Run must retain contour handles');
    const last = result.plot.samples.at(-1);
    assert.equal(last.converged, true);
    assert.ok(last.fsq.every(value => Number.isFinite(value) && value < last.tolerance));
    const shot = await call('Page.captureScreenshot', {format:'png'});
    await writeFile(`${prefix}-${name}.png`, Buffer.from(shot.data, 'base64'));
    console.log(JSON.stringify({name, ...result.dataset, residuals:last.fsq}));
  };
  await call('Page.enable');
  await call('Page.addScriptToEvaluateOnNewDocument', {source:
    `Object.defineProperty(window, 'localStorage', {get: () => window.sessionStorage});`});
  const url = new URL(base);
  url.searchParams.delete('run');
  url.searchParams.set('preset','solovev'); url.searchParams.set('precision','float');
  await call('Page.navigate', {url:url.href});
  await call('Page.bringToFront');
  await ready();
  await evaluate(`document.getElementById('mode-contour').click()`);
  await resize(20);
  const symmetricBefore = await snapshot();
  await drag(3, 5, -4);
  const symmetricAfter = await snapshot();
  const mirror = symmetricAfter.points.length-3;
  assert.equal(symmetricAfter.points[3][0], symmetricAfter.points[mirror][0]);
  assert.equal(symmetricAfter.points[3][1], -symmetricAfter.points[mirror][1]);
  assert.notDeepEqual(symmetricAfter.points[3], symmetricBefore.points[3]);
  await solve('symmetric');

  url.searchParams.set('preset','asymmetric'); url.searchParams.set('precision','double');
  await call('Page.navigate', {url:url.href});
  await ready();
  const initial = await snapshot();
  assert.equal(initial.input.lasym, true);
  await evaluate(`document.getElementById('mode-contour').click()`);
  await resize(12);
  await resize(64);
  await resize(24);
  const before = await snapshot();
  assert.deepEqual(before.input, initial.input, 'Selecting Contour must not refit the input');
  await drag(3, 5, -4);
  const edited = await snapshot();
  for (let i=0;i<before.points.length;i++) {
    if (i===3) assert.notDeepEqual(edited.points[i],before.points[i]);
    else assert.deepEqual(edited.points[i],before.points[i],'An asymmetric drag must move only one point');
  }
  const physics = ({rbc,zbs,rbs,zbc,...rest}) => rest;
  assert.deepEqual(physics(edited.input),physics(before.input),'Contour fitting changed profiles, axis or resolution');
  assert.ok(edited.input.zbc.find(h=>h.m===0&&h.n===0).value > .1,'Vertical offset was lost');
  for (const family of ['rbc','zbs','rbs','zbc']) assert.ok(edited.input[family].some(h=>h.value!==0));
  assert.equal(edited.error,'');
  await call('Page.reload');
  await ready();
  let retained = await snapshot();
  assert.deepEqual(retained.input,edited.input); assert.deepEqual(retained.points,edited.points);
  assert.equal(retained.mode,'contour');
  for (const precision of ['float','double']) {
    await evaluate(`document.getElementById('editor-precision-${precision==='float'?'single':'double'}').click()`);
    await wait(`document.body?.dataset.cumesPrecision==='${precision}' && document.body.dataset.cumesWebgpu==='ready' && !document.getElementById('run').disabled`);
    retained = await snapshot();
    assert.deepEqual(retained.points,edited.points); assert.equal(retained.mode,'contour');
    assert.deepEqual({...retained.input,ftol_array:edited.input.ftol_array},edited.input);
  }
  await resize(16);
  // A 3-D JSON edit must retain its toroidal harmonics and leave planar editing.
  const threeD = structuredClone(edited.input);
  threeD.ntor = 1; threeD.rbs.push({m:1,n:1,value:.003});
  await evaluate(`document.getElementById('coil-equilibrium').value=${JSON.stringify(JSON.stringify(threeD))};document.getElementById('apply-equilibrium').click()`);
  assert.equal(await evaluate(`document.getElementById('boundary-editor-mode').hidden`),true);
  assert.equal((await snapshot()).mode,'fourier');
  assert.deepEqual((await snapshot()).input,threeD);
  // Changing the selected poloidal resolution must update the contour basis.
  const refined = structuredClone(edited.input);
  refined.mpol = 10; refined.zbc.push({m:9,n:0,value:.004});
  await evaluate(`document.getElementById('coil-equilibrium').value=${JSON.stringify(JSON.stringify(refined))};document.getElementById('apply-equilibrium').click();document.getElementById('mode-contour').click()`);
  assert.equal((await snapshot()).points.length,20);
  await drag(3,2,-2);
  assert.ok((await snapshot()).input.zbc.some(h=>h.m===9&&Math.abs(h.value)>1e-4));
  await evaluate(`document.getElementById('coil-equilibrium').value=${JSON.stringify(JSON.stringify(edited.input))};document.getElementById('apply-equilibrium').click()`);
  assert.equal((await snapshot()).points.length,16);
  console.log('Contour editing: PASS (point count, independent handles, vertical offset, profiles, reload, precision and JSON resolution)');
  await solve('asymmetric');
} finally {
  if (target) await cdp.call('Target.closeTarget',{targetId:target});
  cdp.close();
}
