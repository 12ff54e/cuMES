// Exercise W7-X setup/start/reset without completing a long equilibrium run.
// Uses isolated storage and closes only its own browser context.
// Usage: node scripts/webgpu_w7x_start_smoke.mjs APP_URL
import assert from 'node:assert/strict';
const [url]=process.argv.slice(2);
const version=await(await fetch('http://127.0.0.1:9333/json/version')).json();
const ws=new WebSocket(version.webSocketDebuggerUrl);
await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject});
let id=0;const pending=new Map();
ws.onmessage=event=>{
  const reply=JSON.parse(event.data),request=pending.get(reply.id);if(!request)return;
  pending.delete(reply.id);clearTimeout(request.timer);
  if(reply.error||reply.result?.exceptionDetails)request.reject(Error(JSON.stringify(reply.error||reply.result.exceptionDetails)));
  else request.resolve(reply.result);
};
function call(method,params={},sessionId){return new Promise((resolve,reject)=>{
  const current=++id,timer=setTimeout(()=>{pending.delete(current);reject(Error(`CDP timeout: ${method}`))},30000);
  pending.set(current,{resolve,reject,timer});ws.send(JSON.stringify({id:current,method,params,sessionId}));
})}
let context;
try{
  context=(await call('Target.createBrowserContext',{disposeOnDetach:true})).browserContextId;
  const target=(await call('Target.createTarget',{url:'about:blank',browserContextId:context})).targetId;
  const session=(await call('Target.attachToTarget',{targetId:target,flatten:true})).sessionId;
  const evaluate=async expression=>(await call('Runtime.evaluate',{expression,returnByValue:true},session)).result.value;
  const wait=async expression=>{
    for(let i=0;i<300;i++){
      try{if(await evaluate(expression))return}catch(error){if(!/context|Cannot find|navigated/i.test(String(error)))throw error}
      await new Promise(resolve=>setTimeout(resolve,100));
    }
    throw Error(`Timeout: ${expression}`);
  };
  const idle=`document.body?.dataset.cumesExecution==='idle'&&document.body.dataset.cumesWebgpu==='ready'`;
  await call('Page.enable',{},session);await call('Network.enable',{},session);
  for(const precision of ['double','float']){
    const next=new URL(url);next.search=new URLSearchParams({solve:'w7x',precision,grids:'3',run:'1',timing:'0'});
    await call('Page.navigate',{url:next.href},session);await call('Page.bringToFront',{},session);
    await wait(idle+`&&document.body.dataset.cumesPrecision==='${precision}'`);
    assert.equal(await evaluate(`!new URL(location.href).searchParams.has('solve')&&!new URL(location.href).searchParams.has('run')&&
      document.querySelector('nav [aria-current=page]').dataset.view==='editor'&&typeof HEAPU8==='undefined'`),true);
    const input=await evaluate('inputJSON()');
    assert.deepEqual(JSON.parse(input).ns_array,[33,66,99]);
    assert.equal(await evaluate(`new URL(location.href).searchParams.has('grids')`),false);
    // A one-time grid URL must not overwrite subsequent JSON edits on reload.
    const custom={...JSON.parse(input),ns_array:[55],niter_array:[123],ftol_array:[1e-5]};
    await evaluate(`document.getElementById('coil-equilibrium').value=${JSON.stringify(JSON.stringify(custom))};document.getElementById('apply-equilibrium').click()`);
    assert.equal(await evaluate(`document.getElementById('grid-sequence').value`),'custom');
    // A signed-n coefficient edit updates the exact solver input and both previews.
    await evaluate(`document.getElementById('toroidal-mode').value='-1';document.getElementById('toroidal-mode').dispatchEvent(new Event('change'))`);
    const edit=`document.querySelector('#boundary-coefficients input[data-family="rbc"][data-m="1"]')`;
    const original=await evaluate(`${edit}.valueAsNumber`);
    const before=await evaluate(`document.getElementById('boundary-plot').querySelector('path').getAttribute('d')`);
    await evaluate(`${edit}.value=${original}+.0001;${edit}.dispatchEvent(new Event('input'))`);
    assert.notEqual(await evaluate(`document.getElementById('boundary-plot').querySelector('path').getAttribute('d')`),before);
    assert.equal(await evaluate(`JSON.parse(inputJSON()).rbc.find(e=>e.m===1&&e.n===-1).value`),original+.0001);
    const changed=await evaluate('inputJSON()');
    await evaluate(`document.getElementById('slice-angle').value='37';document.getElementById('slice-angle').dispatchEvent(new Event('input'))`);
    assert.equal(await evaluate('inputJSON()'),changed,'view changes do not edit coefficients');
    const other=precision==='double'?'single':'double';
    await evaluate(`document.getElementById('editor-precision-${other}').click()`);
    await wait(idle+`&&document.body.dataset.cumesPrecision!=='${precision}'`);
    assert.deepEqual(JSON.parse(await evaluate('inputJSON()')).rbc,JSON.parse(changed).rbc);
    assert.deepEqual(JSON.parse(await evaluate('inputJSON()')).ns_array,[55]);
    assert.deepEqual(JSON.parse(await evaluate('inputJSON()')).niter_array,[123]);
    assert.equal(await evaluate(`document.getElementById('grid-sequence').value`),'custom');
    await evaluate(`document.getElementById('editor-precision-${precision==='double'?'double':'single'}').click()`);
    await wait(idle+`&&document.body.dataset.cumesPrecision==='${precision}'`);
    // Restore the preset coefficient before testing startup.
    await evaluate(`document.getElementById('coil-equilibrium').value=${JSON.stringify(input)};document.getElementById('apply-equilibrium').click()`);
    await evaluate(`document.getElementById('run').click();document.getElementById('run').click()`);
    await wait(`document.body?.dataset.cumesExecution==='main'&&window.cumesResidualPlot?.report().samples.length>=3`);
    assert.equal(await evaluate(`window.cumesResidualPlot.report().samples.every(row=>row.tolerance===${precision==='double'?'1e-12':'1e-5'})`),true);
    assert.equal(await evaluate(`document.getElementById('run').disabled&&document.querySelectorAll('script[src*="cumes_webgpu.js"]').length===1`),true);
    await evaluate(`document.getElementById('stop-run').click()`);await wait(idle);
    assert.deepEqual(JSON.parse(await evaluate('inputJSON()')).rbc,JSON.parse(input).rbc);
    console.log(`PASS: ${precision} unified W7-X setup, signed-n edit, linked previews, precision persistence, Run and Stop`);
  }
  await call('Network.setBlockedURLs',{urls:['*cumes_webgpu.js*']},session);
  await evaluate(`document.getElementById('run').click()`);
  await wait(`document.body?.dataset.cumesWebgpu==='fail'&&!document.getElementById('run').disabled`);
  await evaluate(`document.getElementById('stop-run').click()`);await wait(idle);
  console.log('PASS: runtime load failure and Stop return to editable setup');
}finally{
  if(context)await call('Target.disposeBrowserContext',{browserContextId:context});
  ws.close();
}
