// Owns the solver Wasm, WebGPU device, CPU references, and virtual files.
// Only logs, status, diagnostics, timing, and final output cross to the UI.
let started = false, finished = false, logs = [], diagnostics = [], timer = null;
function flush() {
  if (timer !== null) clearTimeout(timer);
  timer = null;
  if (logs.length) { postMessage({kind: 'logs', lines: logs}); logs = []; }
  if (diagnostics.length) { postMessage({kind: 'diagnostics', rows: diagnostics}); diagnostics = []; }
}
function schedule() { if (timer === null) timer = setTimeout(flush, 100); }
function append(line) {
  if (finished) return;
  logs.push(line);
  // Deliver setup progress before synchronous Wasm grid generation blocks timers.
  if (line.startsWith('Generating the coil') || line.startsWith('Coil field grid ready')) flush();
  else schedule();
}
function result(success, detail, timing) {
  if (finished) return;
  flush();
  finished = true;
  postMessage({kind: 'result', args: [success, detail, timing]});
}
globalThis.cumesAppendLog = append;
globalThis.cumesBrowser = {
  result,
  output(bytes) {
    // FS.readFile owns a copy, never the Wasm heap. Transfer only that copy.
    const size = bytes.length;
    postMessage({kind: 'output', bytes}, [bytes.buffer]);
    return size;
  },
  diagnostic(row) { diagnostics.push(row); schedule(); },
  residual(value) { postMessage({kind: 'residual', value}); },
  restart(value) { postMessage({kind: 'restart', value}); },
  error(...args) { postMessage({kind: 'error', args}); },
  ready() { postMessage({kind: 'ready'}); },
  equilibrium(value) { postMessage({kind: 'equilibrium', value}); },
  adapter(...args) { postMessage({kind: 'adapter', args}); }
};
globalThis.addEventListener('error', event => result(false, event.message));
globalThis.addEventListener('unhandledrejection', event => result(false, String(event.reason)));
globalThis.onmessage = ({data}) => {
  if (data.kind === 'visibility') { globalThis.cumesVisibility = data.value; return; }
  if (data.kind !== 'start' || started) return;
  started = true;
  globalThis.cumesSearch = data.search;
  globalThis.cumesVisibility = data.visibility;
  const runtime = new URL(data.runtimeUrl);
  globalThis.Module = {
      preRun: [() => {
          for (const file of data.files || []) {
              if (!file.path.startsWith('/inputs/') || file.path.includes('..'))
                  throw Error('Invalid solver input path');
              FS.mkdirTree('/inputs');
              FS.writeFile(file.path, file.bytes);
          }
      }],
      locateFile(path) {
          const url = new URL(path, runtime);
          url.search = runtime.search;
          return url.href;
      },
      print: (...args) => append(args.join(' ')),
      printErr: (...args) => append('ERROR: ' + args.join(' ')),
      onAbort: reason => result(false, 'WebAssembly aborted: ' + reason)
  };
  try { importScripts(runtime.href); }
  catch (error) { result(false, 'Verification runtime failed: ' + error); }
};
