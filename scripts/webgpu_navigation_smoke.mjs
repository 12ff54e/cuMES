// Check the shared navigation without launching GPU solves or touching user tabs/storage.
// Usage: node scripts/webgpu_navigation_smoke.mjs APP_URL [SCREENSHOT_PREFIX]
import assert from 'node:assert/strict';
import {writeFile} from 'node:fs/promises';
const [url,prefix]=process.argv.slice(2);
if(!url)throw Error('Pass APP_URL');
const version=await(await fetch('http://127.0.0.1:9333/json/version')).json();
const ws=new WebSocket(version.webSocketDebuggerUrl);
await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject});
let id=0;
const pending=new Map();
ws.onmessage=event=>{
  const reply=JSON.parse(event.data),request=pending.get(reply.id);
  if(!request)return;
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
  await call('Network.enable',{},session);
  // Only the frontend is in scope: keep verification/W7-X from starting a solve.
  await call('Network.setBlockedURLs',{urls:['*cumes_webgpu.js*']},session);
  await call('Page.enable',{},session);
  for(const width of [1280,390]){
    await call('Emulation.setDeviceMetricsOverride',{width,height:900,deviceScaleFactor:1,mobile:false},session);
    let expectedStyle;
    for(const [search,view] of [['','editor'],['?mode=test','verification'],['?solve=w7x','w7x'],['?solve=w7x&precision=float','w7x']]){
      const next=new URL(url);next.search=search;
      await call('Page.navigate',{url:next.href},session);
      let ready=false;
      for(let attempt=0;attempt<100&&!ready;++attempt){
        try{ready=await evaluate(`location.search===${JSON.stringify(search)}&&document.querySelector('nav [aria-current="page"]')?.dataset.view===${JSON.stringify(view)}&&!!window.cumesAppReady`)}catch{}
        if(!ready)await new Promise(resolve=>setTimeout(resolve,100));
      }
      assert(ready,`frontend did not initialize: ${search}`);
      const state=await evaluate(`(()=>{
        const nav=document.querySelector('nav'),links=[...nav.querySelectorAll('a')];
        const styles=links.map(link=>{const s=getComputedStyle(link);return [s.fontSize,s.padding,s.textDecorationLine,s.borderRadius]});
        return {navCount:document.querySelectorAll('nav').length,shared:!nav.closest('#app,#legacy'),
          links:links.map(link=>[link.textContent,link.getAttribute('href')]),styles,
          visible:links.every(link=>{const r=link.getBoundingClientRect();return r.width>0&&r.height>0&&r.left>=0&&r.right<=innerWidth&&r.top>=0&&r.bottom<=innerHeight}),
          active:links.filter(link=>link.getAttribute('aria-current')==='page').map(link=>link.dataset.view),
          precision:document.body.dataset.cumesPrecision};
      })()`);
      assert.equal(state.navCount,1);assert(state.shared);assert(state.visible);
      assert.deepEqual(state.links,[['Boundary editor','?'],['GPU verification','?mode=test'],['W7-X','?solve=w7x']]);
      assert.deepEqual(state.active,[view]);
      if(expectedStyle)assert.deepEqual(state.styles,expectedStyle);else expectedStyle=state.styles;
      if(view==='w7x')assert.equal(state.precision,search.includes('float')?'float':'double');
      if(prefix&&!search.includes('float')){
        const shot=await call('Page.captureScreenshot',{format:'png'},session);
        await writeFile(`${prefix}-${view}-${width}.png`,Buffer.from(shot.data,'base64'));
      }
      console.log(`PASS: ${view} ${width}px ${state.precision||''}`);
    }
  }
}finally{
  if(context)await call('Target.disposeBrowserContext',{browserContextId:context});
  ws.close();
}
