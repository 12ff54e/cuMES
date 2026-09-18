// Exercise input preview, copying and File uploads, including fixed/free solves.
// Usage: node scripts/webgpu_input_upload_smoke.mjs APP_URL OUTPUT_PREFIX
import assert from 'node:assert/strict';
import {readFile, writeFile} from 'node:fs/promises';
import {connectCdp} from './include/webgpu_cdp.mjs';

const [url, prefix] = process.argv.slice(2);
if (!url || !prefix) throw Error('Pass APP_URL and OUTPUT_PREFIX');
const fixture = async path => JSON.parse(await readFile(new URL(path, import.meta.url), 'utf8'));
const solovev = {...await fixture('../inputs/solovev.json'), ns_array:[7,19], niter_array:[1600,1600], ftol_array:[1e-3,2e-5]};
const w7x = await fixture('../inputs/w7x.json');
const asymmetric = await fixture('../inputs/asymmetric_tokamak.json');
const free = {...await fixture('../webgpu/presets/solovev.json'), ns_array:[8], niter_array:[2000], ftol_array:[1e-4]};
const browser = await (await fetch(`http://127.0.0.1:${process.env.CUMES_CDP_PORT || 9333}/json/version`)).json();
const cdp = await connectCdp(browser.webSocketDebuggerUrl);
let target;
try {
  target = (await cdp.call('Target.createTarget', {url:'about:blank',newWindow:false})).targetId;
  const session = (await cdp.call('Target.attachToTarget', {targetId:target,flatten:true})).sessionId;
  const call = (method, params = {}) => cdp.call(method, params, session);
  const evaluate = async expression => {
    const result = await call('Runtime.evaluate', {expression,returnByValue:true,awaitPromise:true});
    if (result.exceptionDetails) throw Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text);
    return result.result.value;
  };
  const wait = async (expression, timeout = 30000) => {
    const deadline = Date.now() + timeout;
    while (Date.now() < deadline) {
      try { const result = await evaluate(expression); if (result) return result; }
      catch (error) { if (!/context|Cannot find|navigated/i.test(String(error))) throw error; }
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    throw Error('Timed out waiting for ' + expression);
  };
  const ready = () => wait(`document.body?.dataset.cumesWebgpu==='ready' && !document.getElementById('run').disabled`);
  const input = () => evaluate('JSON.parse(inputJSON())');
  const preview = async () => {
    const expected = await input();
    await evaluate(`document.getElementById('equilibrium-settings').open=true`);
    await wait(`document.getElementById('coil-equilibrium').value && !document.getElementById('copy-input-json').disabled`);
    const shown = await evaluate(`(() => {const field=document.getElementById('coil-equilibrium');return {source:field.value,readOnly:field.readOnly,disabled:field.disabled,visible:field.getClientRects().length>0}})()`);
    assert(shown.readOnly && !shown.disabled && shown.visible);
    assert.equal(shown.source,JSON.stringify(expected,null,2),'Preview must contain the complete current solver input');
    await evaluate(`window.cumesCopiedJSON=null;document.getElementById('copy-input-json').click()`);
    await wait(`typeof window.cumesCopiedJSON==='string' && document.getElementById('input-copy-status').textContent==='Copied to clipboard.'`);
    assert.equal(await evaluate('window.cumesCopiedJSON'),shown.source);
    assert.deepEqual(await input(),expected,'Preview and copying must preserve solver input');
  };
  const upload = async (name, source, valid = true) => {
    await evaluate(`(() => {
      const file=new File([${JSON.stringify(typeof source === 'string' ? source : JSON.stringify(source))}],${JSON.stringify(name)},{type:'application/json'});
      const transfer=new DataTransfer();transfer.items.add(file);
      const control=document.getElementById('input-upload');control.files=transfer.files;
      control.dispatchEvent(new Event('change',{bubbles:true}));
    })()`);
    if (valid) {
      await wait(`document.getElementById('input-upload-status')?.textContent === ${JSON.stringify('Loaded ' + name)}`);
      await ready();
      assert.equal(await evaluate('document.body.dataset.cumesExecution'), 'idle');
      assert.equal(await evaluate(`document.getElementById('editor-panel-boundary').hidden`), false, 'Uploads open the boundary tab');
      assert.equal(await evaluate(`new URL(location.href).searchParams.has('run')`), false);
      await preview();
    } else {
      await wait(`document.getElementById('input-upload-status')?.classList.contains('error') && !inputLoading`);
      assert.equal(await evaluate(`document.getElementById('input-upload').value`), '');
      assert.equal(await evaluate(`document.getElementById('input-upload-button').disabled`), false);
    }
  };
  const preset = async name => {
    await evaluate(`document.getElementById('editor-tab-boundary').click()`);
    await evaluate(`document.getElementById('fixed-preset').value=${JSON.stringify(name)};document.getElementById('fixed-preset').dispatchEvent(new Event('change',{bubbles:true}))`);
    await wait(`new URL(location.href).searchParams.get('preset')===${JSON.stringify(name)}`); await ready();
  };
  const solve = async name => {
    await evaluate(`document.getElementById('run').click()`);
    await wait(`document.body?.classList.contains('busy')`);
    assert.equal(await evaluate(`document.getElementById('input-upload-button').disabled && document.getElementById('input-upload').disabled`), true);
    await preview();
    await wait(`['pass','fail'].includes(document.body?.dataset.cumesWebgpu)`, 300000);
    const result = await evaluate(`({dataset:{...document.body.dataset},input:JSON.parse(inputJSON()),plot:cumesResidualPlot.report(),ns:resultData?.fourier?.ns})`);
    await writeFile(`${prefix}-${name}.json`, JSON.stringify(result));
    assert.equal(result.dataset.cumesWebgpu, 'pass', result.dataset.cumesDetail);
    assert.equal(result.ns, result.input.ns_array.at(-1));
    for (const row of result.plot.samples) assert.equal(row.tolerance, result.input.ftol_array[row.stage - 1]);
    assert.equal(await evaluate(`document.getElementById('input-upload-button').disabled`), false);
    console.log('PASS: uploaded ' + name + ' converged with its configured stages');
  };

  await call('Page.enable');
  await call('Page.addScriptToEvaluateOnNewDocument', {source:`
    Object.defineProperty(window,'localStorage',{get:()=>window.sessionStorage});
    // Capture writes in this tab instead of replacing the user's clipboard.
    Object.defineProperty(navigator,'clipboard',{value:{async writeText(text){
      if(window.cumesCopyFailure)throw Error('Clipboard denied');
      if(window.cumesCopyPending)await new Promise(resolve=>window.cumesResolveCopy=resolve);
      window.cumesCopiedJSON=text;
    }}});
  `});
  await call('Emulation.setDeviceMetricsOverride', {width:1280,height:1000,deviceScaleFactor:1,mobile:false});
  await call('Page.navigate', {url}); await call('Page.bringToFront'); await ready();
  assert.equal(await evaluate(`document.getElementById('equilibrium-settings').open`),false);
  await preview();
  const initialPreview=await evaluate(`document.getElementById('coil-equilibrium').value`);
  await evaluate(`document.getElementById('coil-equilibrium').focus()`);
  await call('Input.insertText',{text:'unexpected edit'});
  assert.equal(await evaluate(`document.getElementById('coil-equilibrium').value`),initialPreview,'JSON preview must be read-only');
  await evaluate(`window.cumesCopyFailure=true;document.getElementById('copy-input-json').click()`);
  await wait(`document.getElementById('input-copy-status').textContent.startsWith('Clipboard unavailable.')`);
  assert.equal(await evaluate(`(() => {const field=document.getElementById('coil-equilibrium');return document.activeElement===field && field.selectionStart===0 && field.selectionEnd===field.value.length})()`),true);
  await evaluate(`window.cumesCopyFailure=false;window.cumesCopyPending=true;document.getElementById('copy-input-json').click();document.getElementById('rbc-1').value='1.11';document.getElementById('rbc-1').dispatchEvent(new Event('input',{bubbles:true}));document.getElementById('pressure').value='0.15';document.getElementById('pressure').dispatchEvent(new Event('input',{bubbles:true}))`);
  assert.equal(await evaluate(`document.getElementById('copy-input-json').disabled`),true,'Edits must not start overlapping clipboard writes');
  await evaluate(`window.cumesCopyPending=false;window.cumesResolveCopy()`);
  await wait(`!document.getElementById('copy-input-json').disabled`);
  assert.equal(await evaluate('window.cumesCopiedJSON'),initialPreview,'Copy uses the input shown when the button was pressed');
  await preview();
  await evaluate(`(() => {document.getElementById('editor-tab-stages').click();const field=document.querySelector('[data-stage-key="ns_array"]');field.value='35';field.dispatchEvent(new Event('input',{bubbles:true}))})()`);
  assert.equal(await evaluate(`document.getElementById('coil-equilibrium').value`),'');
  assert.equal(await evaluate(`document.getElementById('copy-input-json').disabled`),true);
  await evaluate(`(() => {const field=document.querySelector('[data-stage-key="ns_array"]');field.value='7';field.dispatchEvent(new Event('input',{bubbles:true}))})()`);
  await preview();
  await evaluate(`document.getElementById('editor-tab-boundary').click()`);
  console.log('PASS: basic fixed preview, live controls, both tabs, read-only JSON, clipboard success/failure and invalid-edit recovery');
  const original = await input();
  const savedEditor = await evaluate(`localStorage.getItem('cumes.editor.v1')`);
  for (const source of ['{', '[]', JSON.stringify({...solovev,ns_array:[19,7]})]) {
    await upload('input.json', source, false);
    assert.deepEqual(await input(), original, 'Rejected uploads must preserve the setup');
    assert.equal(await evaluate(`localStorage.getItem('cumes.fixed.upload.v1')`), null);
  }
  await evaluate(`document.getElementById('input-upload').files=new DataTransfer().files;document.getElementById('input-upload').dispatchEvent(new Event('change'))`);
  assert.deepEqual(await input(), original, 'Canceling the file chooser leaves the input alone');
  await upload('input.json', '\uFEFF' + JSON.stringify(solovev));
  assert.deepEqual(await input(), {...solovev,lfreeb:false});
  assert.equal(await evaluate(`localStorage.getItem('cumes.editor.v1')`), savedEditor);
  assert.equal(await evaluate(`document.getElementById('fixed-preset').value`), 'upload');
  assert.equal(await evaluate('document.body.dataset.cumesPrecision'), 'float');
  assert.equal(await evaluate('contourAvailable()'), true);
  assert.equal(await evaluate(`document.querySelectorAll('#boundary-coefficients input[data-family="rbc"]').length`), solovev.mpol);
  await evaluate(`document.getElementById('editor-tab-stages').click();const control=document.querySelector('[data-stage-key="ns_array"]');control.value='8';control.dispatchEvent(new Event('input',{bubbles:true}))`);
  await preview();
  await call('Page.reload'); await ready(); assert.deepEqual((await input()).ns_array, [8,19]);
  await preset('w7x'); await preset('upload'); assert.deepEqual((await input()).ns_array, [8,19]);
  await evaluate(`document.getElementById('reset').click()`); await ready();
  assert.deepEqual(await input(), {...solovev,lfreeb:false});
  await solve('fixed');

  await upload('w7x.json', w7x);
  assert.deepEqual(await input(), {...w7x,lfreeb:false});
  assert.equal(await evaluate('document.body.dataset.cumesPrecision'), 'double');
  assert.equal(await evaluate('surfaceEditor.fourier.ntor'), w7x.ntor);
  assert.equal(await evaluate('contourAvailable()'), false);
  await preset('solovev');
  await evaluate(`document.getElementById('editor-precision-single').click()`);
  await wait(`document.body?.dataset.cumesPrecision==='float' && document.body.dataset.cumesWebgpu==='ready'`);
  await preset('upload');
  assert.equal(await evaluate('document.body.dataset.cumesPrecision'), 'double');
  assert.deepEqual((await input()).ftol_array, w7x.ftol_array);
  await evaluate(`document.getElementById('editor-tab-stages').click();for(const control of document.querySelectorAll('[data-stage-key="ftol_array"]')){control.value='1e-5';control.dispatchEvent(new Event('input'))}`);
  await evaluate(`document.getElementById('editor-precision-single').click()`);
  await wait(`document.body?.dataset.cumesPrecision==='float' && document.body.dataset.cumesWebgpu==='ready'`);
  await evaluate(`document.getElementById('reset').click()`);
  await wait(`document.body?.dataset.cumesPrecision==='double' && document.body.dataset.cumesWebgpu==='ready'`);
  assert.deepEqual(await input(), {...w7x,lfreeb:false});
  await upload('asymmetric.json', asymmetric);
  assert.deepEqual(await input(), {...asymmetric,lfreeb:false});
  assert.equal(await evaluate(`document.getElementById('allow-asymmetry').checked`), true);
  assert.equal(await evaluate('contourAvailable()'), true);
  const before = await input();
  await upload('missing-coils.json', {...free,coils_file:'coils.missing'}, false);
  assert.deepEqual(await input(), before);

  await upload('free.json', {...free,coils_file:'../data/coils.solovev'});
  assert.deepEqual(await input(), free);
  assert.equal(await evaluate('document.body.dataset.boundaryMode'), 'free');
  assert.deepEqual(await evaluate(`JSON.parse(document.getElementById('coil-currents').value)`), free.extcur);
  assert.deepEqual(await evaluate(`JSON.parse(document.getElementById('coil-grid').value)`), free.makegrid_parameters);
  for (const [id,changed] of [
    ['coil-currents',free.extcur.map((current,i)=>i ? current : current+100)],
    ['coil-grid',{...free.makegrid_parameters,number_of_r_grid_points:203}]
  ]) {
    const source=await evaluate(`document.getElementById('${id}').value`);
    await evaluate(`document.getElementById('${id}').value=${JSON.stringify(JSON.stringify(changed))};document.getElementById('${id}').dispatchEvent(new Event('input',{bubbles:true}))`);
    await preview();
    await evaluate(`document.getElementById('${id}').value='{';document.getElementById('${id}').dispatchEvent(new Event('input',{bubbles:true}))`);
    assert.equal(await evaluate(`document.getElementById('coil-equilibrium').value`),'');
    assert.equal(await evaluate(`document.getElementById('copy-input-json').disabled`),true);
    await evaluate(`document.getElementById('${id}').value=${JSON.stringify(source)};document.getElementById('${id}').dispatchEvent(new Event('input',{bubbles:true}))`);
    await preview();
  }
  await call('Page.reload'); await ready(); assert.deepEqual(await input(), free);
  await solve('free');
  assert.equal(await evaluate('document.body.dataset.cumesExecution'), 'worker');
  await evaluate(`document.getElementById('coil-preset').value='cth_like';document.getElementById('coil-preset').dispatchEvent(new Event('change',{bubbles:true}))`);
  await wait(`document.getElementById('coil-status').textContent.startsWith('Coils ready') && JSON.parse(inputJSON()).coils_file==='/inputs/coils.cth_like'`);
  await preview();

  const name = 'uploaded <input> with a long filename.json';
  await upload(name, solovev);
  assert.equal(await evaluate(`document.querySelector('#input-upload-status input')`), null);
  await call('Emulation.setDeviceMetricsOverride', {width:390,height:1000,deviceScaleFactor:1,mobile:false});
  await evaluate(`document.getElementById('theme-toggle').click();document.getElementById('equilibrium-settings').scrollIntoView({block:'center'})`);
  assert.equal(await evaluate('document.documentElement.scrollWidth > document.documentElement.clientWidth'), false);
  const screenshot = await call('Page.captureScreenshot', {format:'png'});
  await writeFile(`${prefix}-mobile.png`, Buffer.from(screenshot.data,'base64'));
  console.log('PASS: JSON preview/copy, invalid/canceled uploads, full fixed/3-D/asymmetric/free input retention, precision selection, presets, reload/reset, busy controls and mobile layout');
} finally {
  if (target) await cdp.call('Target.closeTarget', {targetId:target});
  cdp.close();
}
