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

function createCumesVerificationLog(output, details, progress) {
  const summary = createCumesLogBuffer(output);
  const full = createCumesLogBuffer(details.querySelector('pre'));
  const label = details.querySelector('summary');
  const lines = [];
  let cursor = 0, timer = null, phase = 'operators', grid = 1, terminal = false;
  function flushDetails() {
    if (timer !== null) clearTimeout(timer);
    timer = null;
    label.textContent = `Detailed log (${lines.length} lines)`;
    if (details.open && cursor < lines.length) {
      full.append(lines.slice(cursor).join('\n'));
      cursor = lines.length;
      full.flush();
    }
  }
  details.addEventListener('toggle', flushDetails);
  return {
    append(raw) {
      lines.push(raw);
      if (timer === null) timer = setTimeout(flushDetails, 100);
      const line = raw.trim();
      const result = line.match(/^cuMES WebGPU self-test: (PASS|FAIL)(.*)$/);
      if (result) {
        terminal = true;
        summary.append(`Verification: ${result[1]}${result[2]}`);
        progress.textContent = result[1] === 'PASS' ? 'All checks passed.' : 'Verification failed.';
        return;
      }
      if (/\b(FAIL|FATAL|ERROR|WARNING)\b/i.test(line)) { summary.append(line); return; }
      if (line.startsWith('adapter selected:')) {
        summary.append(line.replace('adapter selected:', 'GPU:')); return;
      }
      if (line.startsWith('parsed W7-X 3-D cold start: PASS')) {
        phase = 'w7x';
        summary.append('GPU operator checks: PASS');
        progress.textContent = 'Checking W7-X initialization, paired precision, and solver integration…';
        return;
      }
      if (line.startsWith('W7-X controller-complete two-pass slice: PASS')) {
        phase = 'solovev';
        summary.append('W7-X integration: PASS (two solver passes)');
        progress.textContent = 'Solovev convergence check · grid 1/3';
        return;
      }
      const stage = line.match(/^Solovev stage (\d+)\/(\d+) converged: iter=(\d+) residual=\(([^,]+),/);
      if (stage) {
        summary.append(`Solovev grid ${stage[1]}/${stage[2]}: PASS — ${stage[3]} iterations, FSQR ${stage[4]}`);
        grid = Math.min(Number(stage[1]) + 1, Number(stage[2]));
        return;
      }
      const iteration = line.match(/^host controller: PASS \(iter=(\d+), FSQR=([^,]+)/);
      if (iteration) {
        progress.textContent = `${phase === 'solovev' ? 'Solovev · grid ' + grid + '/3' : 'W7-X'} · iteration ${iteration[1]} · FSQR ${iteration[2]}`;
        return;
      }
      if (line.startsWith('published schema-')) { summary.append('Output serialization checks: PASS'); return; }
      // Repeated per-operator comparisons stay in the expandable detailed log.
      if (/: PASS\b/.test(line) || /^(cuMES WebGPU milestone:|adapter\/device ready;|.*stage prolongation)/.test(line)) return;
      if (line) summary.append(line);
    },
    flush() { summary.flush(); flushDetails(); },
    finish(success, detail) {
      if (!terminal) {
        terminal = true;
        summary.append(`Verification: ${success ? 'PASS' : 'FAIL'} — ${detail}`);
      }
      progress.textContent = success ? 'All checks passed.' : 'Verification failed.';
      this.flush();
    },
    text() { return lines.length ? lines.join('\n') + '\n' : ''; }
  };
}

function installCumesBrowser() {
  globalThis.cumesBrowser = {
    result(success, detail, timing) {
      globalThis.cumesResidualPlot?.finish(success);
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
    residual(row) { globalThis.cumesResidualPlot?.append(row); },
    restart(row) { globalThis.cumesResidualPlot?.restart(row); },
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
        case 'residual': globalThis.cumesBrowser.residual(data.value); break;
        case 'restart': globalThis.cumesBrowser.restart(data.value); break;
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
