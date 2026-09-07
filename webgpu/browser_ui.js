// UI-only helpers. The numerical runtime may live on the page or in a worker.
function createCumesLogBuffer(output, schedule = setTimeout, cancel = clearTimeout) {
  let pending = [], timer = null;
  const text = output.ownerDocument.createTextNode('');
  output.append(text);
  function flush() {
    if (timer !== null) cancel(timer);
    timer = null;
    if (!pending.length) return;
    text.appendData(pending.join(''));
    pending = [];
    output.scrollTop = output.scrollHeight;
  }
  return {
    append(line) {
      pending.push(line + '\n');
      if (timer === null) timer = schedule(flush, 100);
    },
    flush
  };
}

function installCumesBrowser() {
  globalThis.cumesBrowser = {
    result(success, detail, timing) {
      if (timing && globalThis.cumesVerificationWorker) {
        globalThis.cumesIterationTiming = {report: () => timing};
      }
      globalThis.cumesFlushLog?.();
      document.body.dataset.cumesWebgpu = success ? 'pass' : 'fail';
      document.body.dataset.cumesDetail = detail;
      document.body.dataset.cumesFinishedMilliseconds = String(performance.now());
      clearTimeout(globalThis.cumesDeadline);
      clearInterval(globalThis.cumesKeepAlive);
    },
    output(bytes) {
      document.body.dataset.cumesOutputBytes = String(bytes.length);
      const query = new URLSearchParams(location.search);
      if (query.get('mode') === 'test' && query.get('solve') !== 'w7x') return;
      const blob = new Blob([bytes], {type: 'application/octet-stream'});
      if (globalThis.cumesOutputUrl) URL.revokeObjectURL(globalThis.cumesOutputUrl);
      globalThis.cumesOutputUrl = URL.createObjectURL(blob);
      const app = document.getElementById('app');
      const link = document.getElementById(app && app.hidden ? 'legacy-download' : 'download');
      link.href = globalThis.cumesOutputUrl;
      link.download = 'cumes-webgpu-output.bin';
      link.hidden = false;
    },
    error(key, message) { document.body.dataset[key] = message; },
    diagnostic(row) { (globalThis.cumesDiagnostics ||= []).push(row); },
    ready() {
      document.body.dataset.cumesWebgpu = 'ready';
      globalThis.cumesAppReady?.();
    },
    equilibrium(result) { globalThis.cumesPublishEquilibrium?.(result); },
    adapter(device, type, backend) {
      document.body.dataset.cumesAdapter = device;
      document.body.dataset.cumesAdapterType = type;
      document.body.dataset.cumesAdapterBackend = backend;
    }
  };
}

function startCumesRuntime(userStarted = false) {
  if (globalThis.cumesRuntimeStarted) return;
  const runtime = document.getElementById('cumes-runtime').content.querySelector('script');
  // Template contents have an inert owner document; resolve against the page.
  const runtimeUrl = new URL(runtime.getAttribute('src'), location.href);
  const query = new URLSearchParams(location.search);
  if (query.get('solve') === 'w7x' && !userStarted) {
    document.body.dataset.cumesExecution = 'idle';
    globalThis.cumesBrowser.ready();
    return;
  }
  globalThis.cumesRuntimeStarted = true;
  const verification = query.get('mode') === 'test' && query.get('solve') !== 'w7x';
  // Diagnostic opt-out permits exact main-thread/worker trajectory comparisons.
  if (!verification || query.get('worker') === '0') {
    document.body.dataset.cumesExecution = 'main';
    const script = document.createElement('script');
    script.src = runtimeUrl.href;
    script.onerror = () => globalThis.cumesBrowser.result(false, 'Could not load WebAssembly runtime script');
    document.body.append(script);
    return;
  }
  document.body.dataset.cumesExecution = 'worker';
  const workerUrl = new URL('verification_worker.js', runtimeUrl);
  workerUrl.search = runtimeUrl.search;
  let worker;
  const fail = message => {
    globalThis.cumesAppendLog('ERROR: ' + message);
    globalThis.cumesBrowser.result(false, message);
    worker?.terminate();
  };
  try { worker = new Worker(workerUrl); }
  catch (error) { fail('Could not start verification worker: ' + error); return; }
  globalThis.cumesVerificationWorker = worker;
  const visibility = () => worker.postMessage({kind: 'visibility', value: document.visibilityState});
  const stop = () => {
    worker.terminate();
    worker.onmessage = worker.onerror = worker.onmessageerror = null;
    document.removeEventListener('visibilitychange', visibility);
    globalThis.removeEventListener('pagehide', stop);
  };
  worker.onmessage = ({data}) => {
    try {
      switch (data.kind) {
        case 'logs': for (const line of data.lines) Module.print(line); break;
        case 'result': globalThis.cumesBrowser.result(...data.args); stop(); break;
        case 'output': globalThis.cumesBrowser.output(data.bytes); break;
        case 'diagnostics': for (const row of data.rows) globalThis.cumesBrowser.diagnostic(row); break;
        case 'adapter': globalThis.cumesBrowser.adapter(...data.args); break;
        case 'ready': globalThis.cumesBrowser.ready(); break;
        case 'equilibrium': globalThis.cumesBrowser.equilibrium(data.value); break;
        case 'error': globalThis.cumesBrowser.error(...data.args); break;
        default: throw Error('Unknown verification worker message: ' + data.kind);
      }
    } catch (error) { fail(String(error)); stop(); }
  };
  worker.onerror = event => { event.preventDefault(); fail('Verification worker failed: ' + event.message); stop(); };
  worker.onmessageerror = () => { fail('Could not decode verification worker message'); stop(); };
  document.addEventListener('visibilitychange', visibility);
  globalThis.addEventListener('pagehide', stop);
  worker.postMessage({kind: 'start', runtimeUrl: runtimeUrl.href,
    search: location.search, visibility: document.visibilityState});
}
