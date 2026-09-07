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
    const next=new URL(url);next.search=new URLSearchParams({solve:'w7x',precision,run:'1',timing:'0'});
    await call('Page.navigate',{url:next.href},session);await call('Page.bringToFront',{},session);
    await wait(idle+`&&document.body.dataset.cumesPrecision==='${precision}'`);
    await new Promise(resolve=>setTimeout(resolve,1000));
    assert.equal(await evaluate(`!window.cumesRuntimeStarted&&!window.cumesKeepAlive&&!document.body.dataset.cumesAdapter&&typeof HEAPU8==='undefined'`),true);
    // Precision changes must remain idle, even with a historical run=1 URL.
    const other=precision==='double'?'single':'double';
    await evaluate(`document.getElementById('precision-${other}').click()`);
    await wait(idle+`&&document.body.dataset.cumesPrecision!=='${precision}'`);
    await evaluate(`document.getElementById('precision-${precision==='double'?'double':'single'}').click()`);
    await wait(idle+`&&document.body.dataset.cumesPrecision==='${precision}'`);
    await evaluate(`document.getElementById('w7x-start').click();document.getElementById('w7x-start').click()`);
    await wait(`document.body?.dataset.cumesExecution==='main'&&Number(document.body.dataset.cumesIteration)>=2`);
    assert.equal(await evaluate(`document.getElementById('w7x-start').disabled&&document.querySelectorAll('script[src*="cumes_webgpu.js"]').length===1`),true);
    console.log(`PASS: ${precision} idle without GPU/Wasm, precision setup, explicit Start, real solver progress, duplicate click guarded`);
    await call('Page.reload',{},session);await wait(idle);
    assert.equal(await evaluate('!document.body.dataset.cumesAdapter'),true);
  }
  await call('Network.setBlockedURLs',{urls:['*cumes_webgpu.js*']},session);
  await evaluate(`document.getElementById('w7x-start').click()`);
  await wait(`document.body?.dataset.cumesWebgpu==='fail'&&!document.getElementById('w7x-start').disabled`);
  assert.equal(await evaluate(`document.getElementById('w7x-start').textContent`),'Reset run');
  await evaluate(`document.getElementById('w7x-start').click()`);await wait(idle);
  console.log('PASS: reload remains idle; load failure offers Reset run, which returns to idle setup');
}finally{
  if(context)await call('Target.disposeBrowserContext',{browserContextId:context});
  ws.close();
}
