// Reusable diagnostics for the user-exposed Chrome DevTools tunnel.
import {readFile} from 'node:fs/promises';
const [command, argument, port = '9333'] = process.argv.slice(2);
const expression = command === 'eval-file' ? await readFile(argument, 'utf8') : argument;
const pages = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
const page = pages.find(page => page.type === 'page' && page.url.includes('magnetic-equilibrium-solver'));
if (!page) throw Error('No project tab is exposed');
const ws = new WebSocket(page.webSocketDebuggerUrl);
const timeout = setTimeout(() => {console.error('CDP timeout');process.exit(2);}, 30000);
ws.onopen = () => ws.send(JSON.stringify({id:1,
  method:command==='navigate'?'Page.navigate':'Runtime.evaluate',
  params:command==='navigate'?{url:argument}:{expression,returnByValue:true,awaitPromise:true}}));
ws.onmessage = event => {const reply=JSON.parse(event.data);if(reply.id!==1)return;
  console.log(JSON.stringify(reply.result||reply.error));clearTimeout(timeout);ws.close();};
