// End-to-end precision switching with tab-scoped settings, preserving user edits.
// Usage: node scripts/webgpu_editor_precision_smoke.mjs APP_URL OUTPUT_PREFIX
import {writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
import {connectCdp} from './include/webgpu_cdp.mjs';
const [url,prefix]=process.argv.slice(2);
const version=await(await fetch('http://127.0.0.1:9333/json/version')).json();
const {call, close} = await connectCdp(version.webSocketDebuggerUrl);
let target;
try{
  target=(await call('Target.createTarget',{url:'about:blank',newWindow:false})).targetId;
  const session=(await call('Target.attachToTarget',{targetId:target,flatten:true})).sessionId;
  await call('Page.addScriptToEvaluateOnNewDocument',{source:
    `Object.defineProperty(window, 'localStorage', {get: () => window.sessionStorage});`},session);
  const evaluate=async expression=>(await call('Runtime.evaluate',{expression,returnByValue:true},session)).result.value;
  const wait=async(expression,timeout=600000)=>{
    const deadline=Date.now()+timeout;
    while(Date.now()<deadline){
      try{const value=await evaluate(expression);if(value)return value}catch(error){
        if(!/context|Cannot find|Inspected target navigated/i.test(String(error)))throw error;
      }
      await new Promise(resolve=>setTimeout(resolve,500));
    }
    throw Error(`Timed out waiting for ${expression}`);
  };
  await call('Page.enable',{},session);
  await call('Page.navigate',{url},session);
  await call('Page.bringToFront',{},session);
  console.log(JSON.stringify({target,url,storage:'session'}));
  await wait(`document.body?.dataset.cumesWebgpu==='ready'`,30000);
  assert.equal(await evaluate('document.body.dataset.cumesPrecision'),'float');
  const boundary=await evaluate(`localStorage.getItem('cumes.editor.v1')`);
  await evaluate(`document.getElementById('editor-precision-double').click()`);
  await wait(`document.body?.dataset.cumesWebgpu==='ready'&&document.body.dataset.cumesPrecision==='double'`,30000);
  assert.equal(await evaluate(`localStorage.getItem('cumes.editor.v1')`),boundary);
  await evaluate(`document.getElementById('run').click()`);
  for(const precision of ['double','float']){
    await wait(`document.body?.dataset.cumesPrecision==='${precision}'&&window.cumesResidualPlot?.report().samples.length>=3`);
    assert.equal(await evaluate(`document.body.dataset.cumesWebgpu==='pass'`),false,'residuals should be published before completion');
    await wait(`document.querySelector('.residual-caption')?.textContent.startsWith('Live')`);
    await wait(`document.body?.dataset.cumesPrecision==='${precision}'&&['pass','fail'].includes(document.body.dataset.cumesWebgpu)`);
    const result=await evaluate(`({dataset:{...document.body.dataset},url:location.href,
      boundary:localStorage.getItem('cumes.editor.v1'),log:document.getElementById('output').textContent,
      trace:window.cumesDiagnostics||[],plot:window.cumesResidualPlot.report()})`);
    await writeFile(`${prefix}-${precision}.json`,JSON.stringify(result));
    assert.equal(result.dataset.cumesWebgpu,'pass',result.dataset.cumesDetail);
    assert.equal(result.boundary,boundary,'precision switch changed boundary');
    assert.match(result.log,new RegExp(precision==='double'?'double-single ftol=1e-12':'float ftol=1e-05'));
    const samples=result.plot.samples,trace=result.trace.filter(row=>row.kind==='controller');
    assert.ok(samples.length>=Number(result.dataset.cumesIteration));
    assert.equal(samples.at(-1).converged,true);
    assert.ok(samples.at(-1).fsq.every(value=>value<samples.at(-1).tolerance));
    if(trace.length){
      assert.equal(samples.length,trace.length,'every classified iteration is plotted');
      for(let i=0;i<trace.length;i++)assert.deepEqual(samples[i].fsq,trace[i].fsq,'plotted values match controller scalars exactly');
    }
    const shot=await call('Page.captureScreenshot',{format:'png'},session);
    await writeFile(`${prefix}-${precision}.png`,Buffer.from(shot.data,'base64'));
    console.log(JSON.stringify({precision,...result.dataset,samples:samples.length,draws:result.plot.drawCount,drawMilliseconds:result.plot.drawMilliseconds}));
    if(precision==='double')await evaluate(`document.getElementById('editor-precision-single').click()`);
  }
  console.log('Editor precision switching: PASS (paired and scalar convergence, retained boundary, selected tolerance)');
}finally{
  if(target)await call('Target.closeTarget',{targetId:target});
  close();
}
