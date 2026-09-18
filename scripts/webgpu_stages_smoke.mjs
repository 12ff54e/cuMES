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
  const evaluate = async expression => {
    const result = await call('Runtime.evaluate', {expression, returnByValue:true, awaitPromise:true});
    if (result.exceptionDetails) throw Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text);
    return result.result.value;
  };
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
  const selectedTab = () => evaluate(`document.querySelector('.editor-tabs [aria-selected="true"]').id.replace('editor-tab-','')`);
  const tab = async name => {
    await evaluate(`document.getElementById('editor-tab-${name}').click()`);
    assert.equal(await selectedTab(), name);
    assert.equal(await evaluate(`document.getElementById('editor-panel-${name}').hidden`), false);
    assert.equal(await evaluate(`document.getElementById('editor-panel-${name === 'boundary' ? 'stages' : 'boundary'}').getClientRects().length`), 0);
    assert.equal(await evaluate(`document.getElementById('editor-precision-control').getClientRects().length > 0`), true);
  };
  const key = async name => {
    const params = {key:name,code:name,windowsVirtualKeyCode:{ArrowLeft:37,ArrowRight:39,Home:36,End:35,Tab:9}[name]};
    await call('Input.dispatchKeyEvent', {type:'keyDown',...params});
    await call('Input.dispatchKeyEvent', {type:'keyUp',...params});
  };
  const set = async (key, index, value) => {
    await tab('stages');
    if (key !== 'ns_array') await evaluate(`document.getElementById('stage-limits').open=true`);
    return evaluate(`(() => {
    const input=document.querySelectorAll('[data-stage-key="${key}"]')[${index}];
    input.focus();input.value=${JSON.stringify(String(value))};input.dispatchEvent(new Event('input',{bubbles:true}));
    if(document.activeElement!==input)throw Error('Validation moved focus away from the edited input');
  })()`);
  };
  const count = async number => {
    await tab('stages');
    return evaluate(`document.getElementById('stage-count').value=${number};document.getElementById('stage-count').dispatchEvent(new Event('change',{bubbles:true}))`);
  };
  const sync = () => evaluate(`document.getElementById('stage-sync-limits').click()`);
  const reload = async () => { await call('Page.reload'); await ready(); };
  const changePrecision = async precision => {
    await evaluate(`document.getElementById('editor-precision-${precision === 'float' ? 'single' : 'double'}').click()`);
    await wait(`document.body?.dataset.cumesPrecision === '${precision}' && document.body.dataset.cumesWebgpu === 'ready'`);
  };
  const selectPreset = async preset => {
    await tab('boundary');
    await evaluate(`window.cumesPresetNavigating=true;document.getElementById('fixed-preset').value='${preset}';document.getElementById('fixed-preset').dispatchEvent(new Event('change',{bubbles:true}))`);
    await wait(`!window.cumesPresetNavigating && document.body?.dataset.cumesWebgpu === 'ready' && !document.getElementById('run').disabled`);
    assert.equal(await selectedTab(), 'boundary', 'Selecting a boundary preset must retain the boundary tab');
    assert.equal(await evaluate(`document.getElementById('fixed-preset').value`), preset);
  };
  const photograph = async name => {
    await tab('stages');
    await evaluate(`document.getElementById('stage-settings').scrollIntoView({block:'center'})`);
    const shot = await call('Page.captureScreenshot', {format:'png'});
    await writeFile(`${prefix}-${name}.png`, Buffer.from(shot.data, 'base64'));
  };
  const solve = async (name, expected, success = true) => {
    await evaluate(`document.getElementById('run').click()`);
    await wait(`document.body?.classList.contains('busy')`, 30000);
    assert.equal(await evaluate(`document.getElementById('stage-settings').disabled &&
      [...document.querySelectorAll('#stage-settings input')].every(input=>input.matches(':disabled'))`), true);
    assert.equal(await selectedTab(), 'stages', 'Run must retain the selected setup tab');
    if (name === 'custom-paired-stages') {
      await tab('boundary'); await tab('stages');
      assert.deepEqual(await schedule(), expected, 'Viewing another tab must not change the running input');
    }
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
  await evaluate('document.fonts.ready');
  assert.equal(await selectedTab(), 'boundary');
  const originalInput = await evaluate('inputJSON()');
  const boundaryHeight = await evaluate(`document.querySelector('.panel.editor').getBoundingClientRect().height`);
  await evaluate(`document.getElementById('editor-tab-boundary').focus()`);
  for (const [pressed, expected] of [['ArrowRight','stages'],['ArrowLeft','boundary'],['End','stages'],['Home','boundary'],['ArrowLeft','stages']]) {
    await key(pressed); assert.equal(await selectedTab(), expected);
    assert.equal(await evaluate('document.activeElement.id'), 'editor-tab-' + expected);
  }
  await key('Tab'); assert.equal(await evaluate('document.activeElement.id'), 'stage-count');
  assert.equal(await evaluate('inputJSON()'), originalInput, 'Tab navigation must preserve the complete setup');
  const stageHeight = await evaluate(`document.querySelector('.panel.editor').getBoundingClientRect().height`);
  assert(stageHeight < boundaryHeight, 'Showing stages must remove the boundary editor from the layout');
  console.log(`PASS: setup tabs and keyboard navigation; boundary ${boundaryHeight}px, stages ${stageHeight}px`);
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
  await tab('boundary'); await tab('stages'); assert.deepEqual(await schedule(), linked);
  await reload(); assert.deepEqual(await schedule(), linked);
  assert.equal(await selectedTab(), 'stages');
  assert.equal(await evaluate(`document.getElementById('stage-sync-limits').checked`), true);
  assert.equal(await evaluate(`document.getElementById('stage-limits').open`), false);
  await evaluate(`document.getElementById('stage-limits').open=true`); await sync();
  await set('ftol_array',1,1e-5);
  await changePrecision('double'); assert.deepEqual((await schedule()).ftol_array, [2e-4,1e-12]);
  await changePrecision('float'); assert.deepEqual((await schedule()).ftol_array, [2e-4,1e-5]);
  assert.deepEqual(await evaluate(`(() => {const {ns_array,niter_array,ftol_array,...physics}=JSON.parse(inputJSON());return physics})()`), physics);

  for (const [key, value, repaired] of [['ns_array',7,19],['niter_array',0,1600],['ftol_array',1e-9,1e-5]]) {
    const before = await schedule(); await set(key,1,value);
    await tab('boundary');
    await evaluate(`document.getElementById('run').click()`);
    assert.equal(await evaluate(`document.body.dataset.cumesExecution`), 'idle');
    assert.equal(await evaluate(`document.body.dataset.cumesWebgpu`), 'ready');
    assert(await evaluate(`document.getElementById('stage-error').textContent.length > 0`));
    assert.equal(await selectedTab(), 'stages', 'Run must reveal invalid settings in the hidden stage tab');
    assert.deepEqual(await schedule(), before, 'Invalid edits must not replace the saved schedule');
    await set(key,1,repaired);
    assert.equal(await evaluate(`document.getElementById('stage-error').textContent + document.getElementById('boundary-error').textContent`), '', 'Corrected stage errors must clear');
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

  await selectPreset('asymmetric');
  assert.equal(await evaluate(`document.getElementById('stage-error').textContent`), '');
  await changePrecision('double');
  const asymmetricStages = await schedule();
  await selectPreset('solovev');
  await changePrecision('float');
  await selectPreset('asymmetric');
  assert.deepEqual(await schedule(), asymmetricStages, 'Preset selection must preserve saved tolerances');
  assert.match(await evaluate(`document.getElementById('stage-error').textContent`), /at least 0.000001 for single precision/);
  await reload();
  assert.equal(await selectedTab(), 'boundary', 'Loading saved stages must not change tabs');
  await evaluate(`document.getElementById('run').click()`);
  assert.equal(await selectedTab(), 'stages', 'Run must still reveal incompatible saved tolerances');
  assert.equal(await evaluate('document.body.dataset.cumesExecution'), 'idle');
  await changePrecision('double');
  assert.equal(await evaluate(`document.getElementById('stage-error').textContent`), '');
  assert.deepEqual(await schedule(), asymmetricStages);
  await selectPreset('solovev');
  await selectPreset('asymmetric');
  console.log('PASS: fresh and saved asymmetric presets retain the boundary tab; incompatible stages are revealed on Run');

  for (const preset of ['w7x','asymmetric']) {
    await navigate(`?preset=${preset}&precision=double`);
    await tab('boundary');
    const original = await evaluate('JSON.parse(inputJSON())');
    const imported = {...original, ns_array:[9,17],niter_array:[700,800],ftol_array:[1e-5,1e-6]};
    const invalid = {...imported, ns_array:[17,9]};
    await evaluate(`document.getElementById('equilibrium-settings').open=true;const source=document.getElementById('coil-equilibrium');source.focus();source.value=${JSON.stringify(JSON.stringify(invalid))};source.dispatchEvent(new Event('input',{bubbles:true}))`);
    assert.equal(await selectedTab(), 'boundary', 'Live JSON validation must not switch away from the text being edited');
    assert.equal(await evaluate('document.activeElement.id'), 'coil-equilibrium');
    await evaluate(`document.getElementById('run').click()`);
    assert.equal(await selectedTab(), 'stages');
    assert.equal(await evaluate('document.body.dataset.cumesExecution'), 'idle');
    await tab('boundary');
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
  await tab('boundary');
  await evaluate(`document.getElementById('coil-preset').value='cth_like';document.getElementById('coil-preset').dispatchEvent(new Event('change',{bubbles:true}))`);
  await wait(`document.getElementById('coil-status').textContent.startsWith('Coils ready') && document.getElementById('stage-count').value === '1'`);
  assert.deepEqual((await schedule()).ns_array,[15]);

  await call('Emulation.setDeviceMetricsOverride', {width:390,height:1000,deviceScaleFactor:1,mobile:false});
  await evaluate(`document.getElementById('stage-limits').open=true;document.getElementById('theme-toggle').click()`);
  assert.equal(await evaluate(`document.documentElement.scrollWidth > document.documentElement.clientWidth`),false);
  await photograph('mobile-dark');
  console.log('PASS: setup tabs, stage count, independent/synced limits, JSON round trips, persistence, precision, invalid-input gates, fixed/free solves and step caps');
} finally {
  if (target) await cdp.call('Target.closeTarget', {targetId:target});
  cdp.close();
}
