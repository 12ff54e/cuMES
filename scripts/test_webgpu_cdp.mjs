import assert from 'node:assert/strict';
import {mock} from 'node:test';
import {connectCdp} from './include/webgpu_cdp.mjs';

const originalWebSocket = globalThis.WebSocket;
class FakeWebSocket {
  constructor(url) { this.url = url; this.sent = []; this.closed = false; FakeWebSocket.last = this; }
  send(message) { if (this.sendError) throw this.sendError; this.sent.push(JSON.parse(message)); }
  reply(message) { this.onmessage({data: JSON.stringify(message)}); }
  close() { this.closed = true; this.onclose(); }
}
globalThis.WebSocket = FakeWebSocket;
mock.timers.enable({apis: ['setTimeout']});
async function fixture(timeout) {
  const connection = connectCdp('ws://example.test/devtools/browser/test', timeout);
  const socket = FakeWebSocket.last;
  socket.onopen();
  return {socket, ...await connection};
}
try {
  const f = await fixture();
  const first = f.call('Target.createTarget', {url: 'about:blank', newWindow: false});
  const second = f.call('Runtime.evaluate', {expression: '42', awaitPromise: true}, 'tab-session');
  assert.deepEqual(f.socket.sent, [
    {id: 1, method: 'Target.createTarget', params: {url: 'about:blank', newWindow: false}},
    {id: 2, method: 'Runtime.evaluate', params: {expression: '42', awaitPromise: true}, sessionId: 'tab-session'}
  ]);
  f.socket.reply({method: 'Page.loadEventFired', params: {timestamp: 1}});
  f.socket.reply({id: 2, result: {result: {value: 42}}});
  f.socket.reply({id: 1, result: {targetId: 'test-tab'}});
  assert.deepEqual(await first, {targetId: 'test-tab'});
  assert.deepEqual(await second, {result: {value: 42}});
  for (const response of [{error: {message: 'Unknown method'}},
    {result: {exceptionDetails: {text: 'Execution context was destroyed'}}}]) {
    const rejected = assert.rejects(f.call('Runtime.evaluate'), /Unknown method|Execution context/);
    f.socket.reply({id: f.socket.sent.at(-1).id, ...response});
    await rejected;
  }
  for (const timeout of [undefined, 60000]) {
    const timed = await fixture(timeout);
    let settled = false;
    const call = timed.call('Runtime.evaluate').finally(() => { settled = true; });
    const rejected = assert.rejects(call, /CDP timeout: Runtime.evaluate/);
    mock.timers.tick((timeout ?? 30000) - 1); await Promise.resolve();
    assert.equal(settled, false);
    mock.timers.tick(1); await rejected;
    timed.socket.reply({id: 1, result: {late: true}});
    const next = timed.call('Page.enable');
    timed.socket.reply({id: 2, result: {}}); await next;
    timed.close();
  }
  f.socket.sendError = Error('send failed');
  await assert.rejects(f.call('Page.enable'), /send failed/);
  delete f.socket.sendError;
  const pending = assert.rejects(f.call('Runtime.evaluate'), /connection closed/);
  f.close(); await pending;
  assert.equal(f.socket.closed, true);
  await assert.rejects(f.call('Page.enable'), /connection closed/);
  for (const event of ['onerror', 'onclose']) {
    const failed = await fixture();
    const rejected = assert.rejects(failed.call('Page.enable'), /CDP connection/);
    failed.socket[event](); await rejected; failed.close();
    const opening = assert.rejects(connectCdp('ws://example.test/failure'), /CDP connection/);
    FakeWebSocket.last[event](); await opening;
  }
  mock.timers.tick(60000);
  console.log('PASS: CDP sessions, out-of-order replies, protocol errors, timeouts, and connection cleanup');
} finally {
  globalThis.WebSocket = originalWebSocket;
  mock.timers.reset();
}
