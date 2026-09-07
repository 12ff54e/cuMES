// Diagnostic-only route switching at completed controller records. No solver
// state, damping or input changes. Successful result tabs remain inspectable.
// Usage: node ... URL PREFIX '[{"after":1,"fft":1},{"after":2,"fft":0}]' [SETUP_JS]
import {readFile, writeFile} from 'node:fs/promises';
const [url, prefix, scheduleText = '[]', setupPath] = process.argv.slice(2);
if (!url || !prefix) throw Error('Pass URL and output prefix');
const schedule = JSON.parse(scheduleText);
if (!Array.isArray(schedule) || schedule.some(s => !Number.isInteger(s.after) ||
    s.after < 1 || ![0, 1].includes(s.fft))) throw Error('Invalid switch schedule');
const base = 'http://127.0.0.1:9333';
const page = await (await fetch(`${base}/json/new?about:blank`, {method:'PUT'})).json();
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {ws.onopen=resolve;ws.onerror=reject;});
let nextId=0;const pending=new Map();
ws.onmessage=event=>{
  const reply=JSON.parse(event.data),r=pending.get(reply.id);if(!r)return;
  pending.delete(reply.id);clearTimeout(r.timer);
  if(reply.error||reply.result?.exceptionDetails)r.reject(Error(JSON.stringify(reply.error||reply.result.exceptionDetails)));
  else r.resolve(reply.result);
};
function call(method,params={}){return new Promise((resolve,reject)=>{
  const id=++nextId,timer=setTimeout(()=>{pending.delete(id);reject(Error(`CDP timeout: ${method}`));},60000);
  pending.set(id,{resolve,reject,timer});ws.send(JSON.stringify({id,method,params}));
});}
const evaluate=async expression=>(await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true})).result.value;
let finished=false;
try{
  await call('Page.enable');
  const source=`(() => {
    const schedule=${JSON.stringify(schedule)}, events=[];
    let index=0;window.cumesRouteEvents=events;window.cumesDiagnostics=[];
    window.cumesDiagnostics.push=function(...rows){
      const count=Array.prototype.push.apply(this,rows);
      for(const row of rows)if(row.kind==='controller'){
        while(index<schedule.length && row.iter>=schedule[index].after){
          const step=schedule[index++],url=new URL(location.href);
          const before=url.searchParams.get('fft');url.searchParams.set('fft',step.fft);
          history.replaceState(null,'',url.href);
          events.push({after:row.iter,attempt:row.attempt,before,fft:step.fft});
        }
      }
      return count;
    };
  })();`;
  await call('Page.addScriptToEvaluateOnNewDocument',{source});
  if(setupPath)await call('Page.addScriptToEvaluateOnNewDocument',{source:await readFile(setupPath,'utf8')});
  await call('Page.navigate',{url});await call('Page.bringToFront');
  console.log(JSON.stringify({target:page.id,url,schedule}));
  const deadline=Date.now()+900000;
  while(Date.now()<deadline){
    const status=await evaluate(`(()=>{
      if(document.body?.dataset.cumesExecution==='idle')document.getElementById('w7x-start')?.click();
      return document.body?.dataset.cumesWebgpu;
    })()`);
    if(status==='pass'||status==='fail'){
      const result=await evaluate('({dataset:{...document.body.dataset},events:window.cumesRouteEvents,log:document.body.innerText})');
      await writeFile(`${prefix}-result.json`,JSON.stringify(result));
      await writeFile(`${prefix}-trace.json`,JSON.stringify(await evaluate('window.cumesDiagnostics||[]')));
      if(status==='fail')throw Error(result.dataset.cumesDetail);
      if(result.events.length!==schedule.length)throw Error('Not all route switches executed');
      if(setupPath){
        await evaluate('Promise.all(window.cumesTransformCaptures?.map(c=>c.ready)||[])');
        const metadata=await evaluate('(window.cumesTransformCaptures||[]).map(({bytes,ready,...metadata})=>metadata)');
        await writeFile(`${prefix}-captures.json`,JSON.stringify(metadata));
        for(let i=0;i<metadata.length;i++){
          const size=await evaluate(`window.cumesTransformCaptures[${i}].bytes.length`);
          const chunks=[];
          for(let at=0;at<size;at+=262144){
            const encoded=await evaluate(`btoa(Array.from(window.cumesTransformCaptures[${i}].bytes.subarray(${at},${at+262144}),v=>String.fromCharCode(v)).join(''))`);
            chunks.push(Buffer.from(encoded,'base64'));
          }
          await writeFile(`${prefix}-capture-${i}.bin`,Buffer.concat(chunks));
        }
        console.log(JSON.stringify({captures:metadata.length}));
      }
      console.log(JSON.stringify({...result.dataset,events:result.events}));finished=true;break;
    }
    await new Promise(resolve=>setTimeout(resolve,1000));
  }
  if(!finished)throw Error('Experiment timed out');
}finally{
  if(!finished)await call('Page.navigate',{url:'about:blank'}).catch(()=>{});
  ws.close();
}
