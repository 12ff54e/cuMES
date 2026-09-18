// Exercise stage controls and their actual solver inputs in one disposable tab.
// Usage: node scripts/webgpu_stages_smoke.mjs APP_URL OUTPUT_PREFIX
import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
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
  const evaluate = async expression => (await call('Runtime.evaluate', {expression, returnByValue:true, awaitPromise:true})).result.value;
  const wait = async (expression, timeout = 30000) => {
    const deadline = Date.now() + timeout;
    while (Date.now() < deadline) {
      try { const result = await evaluate(expression); if (result) return result; }
      catch (error) { if (!/context|Cannot find|Inspected target navigated/i.test(String(error))) throw error; }
      await new Promise(resolve => setTimeout(resolve, 150));
    }
    throw Error('Timed out waiting for ' + expression);
  };
  const ready = () => wait(`document.body?.dataset.cumesWebgpu === 'ready' && !document.getElementById('run').disabled`);
  const navigate = async search => {
    const url = new URL(base); url.search = search;
    await call('Page.navigate', {url:url.href}); await ready();
  };
  const schedule = () => evaluate(`(() => {const {ns_array,niter_array,ftol_array}=JSON.parse(inputJSON());return {ns_array,niter_array,ftol_array}})()`);
  const set = (key, index, value) => evaluate(`(() => {
    const input=document.querySelectorAll('[data-stage-key="${key}"]')[${index}];
    input.value=${JSON.stringify(String(value))};input.dispatchEvent(new Event('input',{bubbles:true}));
  })()`);
  const count = number => evaluate(`document.getElementById('stage-count').value=${number};document.getElementById('stage-count').dispatchEvent(new Event('change',{bubbles:true}))`);
  const sync = () => evaluate(`document.getElementById('stage-sync-limits').click()`);
  const reload = async () => { await call('Page.reload'); await ready(); };
  const changePrecision = async precision => {
    await evaluate(`document.getElementById('editor-precision-${precision === 'float' ? 'single' : 'double'}').click()`);
    await wait(`document.body?.dataset.cumesPrecision === '${precision}' && document.body.dataset.cumesWebgpu === 'ready'`);
  };
  const photograph = async name => {
    await evaluate(`document.getElementById('stage-settings').scrollIntoView({block:'center'})`);
    const shot = await call('Page.captureScreenshot', {format:'png'});
    await writeFile(`${prefix}-${name}.png`, Buffer.from(shot.data, 'base64'));
  };
  const solve = async (name, expected, success = true) => {
    await evaluate(`document.getElementById('run').click()`);
    await wait(`document.body?.classList.contains('busy')`, 30000);
    assert.equal(await evaluate(`document.getElementById('stage-settings').disabled &&
      [...document.querySelectorAll('#stage-settings input')].every(input=>input.matches(':disabled'))`), true);
    await wait(`['pass','fail'].includes(document.body?.dataset.cumesWebgpu)`, 300000);
    const result = await evaluate(`({dataset:{...document.body.dataset}, input:JSON.parse(inputJSON()),
      plot:cumesResidualPlot.report(), ns:resultData?.fourier?.ns, log:document.getElementById('output').textContent})`);
    await writeFile(`${prefix}-${name}.json`, JSON.stringify(result));
    assert.equal(result.dataset.cumesWebgpu, success ? 'pass' : 'fail', result.dataset.cumesDetail);
    assert.deepEqual(await schedule(), expected);
    assert.equal(await evaluate(`document.getElementById('stage-settings').disabled`), false);
    for (const row of result.plot.samples) assert.equal(row.tolerance, expected.ftol_array[row.stage - 1]);
    if (success) {
      assert.equal(result.plot.stages.length, expected.ns_array.length);
      assert.equal(result.ns, expected.ns_array.at(-1));
      for (let stage = 1; stage <= expected.ns_array.length; stage++) {
        const last = result.plot.samples.filter(row => row.stage === stage).at(-1);
        assert(last.converged && last.fsq.every(value => value < last.tolerance));
      }
    } else {
      assert.match(result.dataset.cumesDetail, /exhausted its iteration limit/);
      assert(result.plot.samples.every(row => row.attempt <= expected.niter_array[row.stage - 1]));
    }
    console.log(`PASS: ${name}`, expected);
  };

  await call('Page.enable');
  await call('Page.addScriptToEvaluateOnNewDocument', {source:
    `Object.defineProperty(window, 'localStorage', {get: () => window.sessionStorage});`});
  await call('Emulation.setDeviceMetricsOverride', {width:1280,height:1000,deviceScaleFactor:1,mobile:false});
  await call('Page.bringToFront');
  await navigate('?precision=float');
  assert.equal(await evaluate(`document.getElementById('stage-limits').open`), false);
  assert.deepEqual(await schedule(), {ns_array:[5,11,55],niter_array:[1200,2500,3000],ftol_array:[1e-5,1e-5,1e-5]});
  const physics = await evaluate(`(() => {const {ns_array,niter_array,ftol_array,...physics}=JSON.parse(inputJSON());return physics})()`);
  await count(1); assert.deepEqual((await schedule()).ns_array, [55]);
  await count(2); assert.deepEqual((await schedule()).ns_array, [28,55]);
  await set('ns_array',0,7); await set('ns_array',1,19);
  await evaluate(`document.getElementById('stage-limits').open=true`);
  await set('niter_array',0,1200); await set('niter_array',1,1800);
  await set('ftol_array',0,1e-3); await set('ftol_array',1,2e-5);
  await sync();
  assert.deepEqual(await schedule(), {ns_array:[7,19],niter_array:[1200,1200],ftol_array:[1e-3,1e-3]});
  await set('niter_array',1,1600); await set('ftol_array',1,2e-4);
  const linked = {ns_array:[7,19],niter_array:[1600,1600],ftol_array:[2e-4,2e-4]};
  assert.deepEqual(await schedule(), linked);
  await reload(); assert.deepEqual(await schedule(), linked);
  assert.equal(await evaluate(`document.getElementById('stage-sync-limits').checked`), true);
  assert.equal(await evaluate(`document.getElementById('stage-limits').open`), false);
  await evaluate(`document.getElementById('stage-limits').open=true`); await sync();
  await set('ftol_array',1,1e-5);
  await changePrecision('double'); assert.deepEqual((await schedule()).ftol_array, [2e-4,1e-12]);
  await changePrecision('float'); assert.deepEqual((await schedule()).ftol_array, [2e-4,1e-5]);
  assert.deepEqual(await evaluate(`(() => {const {ns_array,niter_array,ftol_array,...physics}=JSON.parse(inputJSON());return physics})()`), physics);

  for (const [key, value, repaired] of [['ns_array',7,19],['niter_array',0,1600],['ftol_array',1e-9,1e-5]]) {
    const before = await schedule(); await set(key,1,value);
    await evaluate(`document.getElementById('run').click()`);
    assert.equal(await evaluate(`document.body.dataset.cumesExecution`), 'idle');
    assert.equal(await evaluate(`document.body.dataset.cumesWebgpu`), 'ready');
    assert(await evaluate(`document.getElementById('stage-error').textContent.length > 0`));
    assert.deepEqual(await schedule(), before, 'Invalid edits must not replace the saved schedule');
    await set(key,1,repaired);
  }
  await set('ftol_array',0,1e-3); await set('ftol_array',1,2e-5);
  await photograph('desktop');
  await solve('custom-scalar-stages', {ns_array:[7,19],niter_array:[1600,1600],ftol_array:[1e-3,2e-5]});
  await navigate('?precision=double');
  await set('ftol_array',0,2e-7); await set('ftol_array',1,3e-8);
  await solve('custom-paired-stages', {ns_array:[7,19],niter_array:[1600,1600],ftol_array:[2e-7,3e-8]});
  await navigate('?precision=float');
  await set('ftol_array',0,1e-5); await set('ftol_array',1,1e-5);
  await count(1); await set('niter_array',0,1);
  await solve('one-step-cap', {ns_array:[19],niter_array:[1],ftol_array:[1e-5]}, false);

  for (const preset of ['w7x','asymmetric']) {
    await navigate(`?preset=${preset}&precision=double`);
    const original = await evaluate('JSON.parse(inputJSON())');
    const imported = {...original, ns_array:[9,17],niter_array:[700,800],ftol_array:[1e-5,1e-6]};
    await evaluate(`document.getElementById('coil-equilibrium').value=${JSON.stringify(JSON.stringify(imported))};document.getElementById('apply-equilibrium').click()`);
    assert.deepEqual(await evaluate(`stageEditor.read()`), {ns_array:[9,17],niter_array:[700,800],ftol_array:[1e-5,1e-6]});
    await set('ns_array',0,11);
    assert.deepEqual(await evaluate('JSON.parse(document.getElementById("coil-equilibrium").value)'), {...imported,ns_array:[11,17]});
    await reload(); assert.deepEqual((await schedule()).ns_array,[11,17]);
  }
  await navigate('?boundary=free&coils=solovev&precision=double');
  assert.deepEqual((await schedule()).ns_array,[16,32]);
  await count(1); await set('ns_array',0,8); await set('ftol_array',0,1e-4); await set('niter_array',0,2000);
  const free = {ns_array:[8],niter_array:[2000],ftol_array:[1e-4]};
  await reload(); assert.deepEqual(await schedule(),free);
  await solve('custom-free-stage', free);
  assert.equal(await evaluate('document.body.dataset.cumesExecution'),'worker');
  await navigate('?boundary=free');
  await evaluate(`document.getElementById('coil-preset').value='cth_like';document.getElementById('coil-preset').dispatchEvent(new Event('change',{bubbles:true}))`);
  await wait(`document.getElementById('coil-status').textContent.startsWith('Coils ready') && document.getElementById('stage-count').value === '1'`);
  assert.deepEqual((await schedule()).ns_array,[15]);

  await call('Emulation.setDeviceMetricsOverride', {width:390,height:1000,deviceScaleFactor:1,mobile:false});
  await evaluate(`document.getElementById('stage-limits').open=true;document.getElementById('theme-toggle').click()`);
  assert.equal(await evaluate(`document.documentElement.scrollWidth > document.documentElement.clientWidth`),false);
  await photograph('mobile-dark');
  console.log('PASS: stage count, independent/synced limits, JSON round trips, persistence, precision, invalid-input gates, fixed/free solves and step caps');
} finally {
  if (target) await cdp.call('Target.closeTarget', {targetId:target});
  cdp.close();
}
