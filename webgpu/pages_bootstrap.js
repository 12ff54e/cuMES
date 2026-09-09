// Included only in the Pages package; ordinary builds use their server headers.
(() => {
  const workerUrl = new URL('coi-serviceworker.js', document.currentScript.src);
  const reloadKey = 'cumes.pages.isolation:' + workerUrl.pathname;
  globalThis.cumesIsolationReady = (async () => {
    if (globalThis.crossOriginIsolated) {
      sessionStorage.removeItem(reloadKey);
      return;
    }
    if (!globalThis.isSecureContext || !navigator.serviceWorker)
      throw Error('This browser cannot prepare the solver. Open this page over HTTPS with service workers enabled.');
    if (sessionStorage.getItem(reloadKey)) {
      sessionStorage.removeItem(reloadKey);
      throw Error('Browser isolation could not be enabled. Allow service workers for this site, then reload.');
    }
    const controlled = () => navigator.serviceWorker.controller?.scriptURL === workerUrl.href;
    let timeout, controllerChanged, stopped = false;
    try {
      await Promise.race([
        (async () => {
          await navigator.serviceWorker.register(workerUrl.href, {updateViaCache: 'none'});
          await navigator.serviceWorker.ready;
          if (!stopped && !controlled()) {
            await new Promise(resolve => {
              controllerChanged = () => { if (controlled()) resolve(); };
              navigator.serviceWorker.addEventListener('controllerchange', controllerChanged);
            });
          }
        })(),
        new Promise((_, reject) => {
          timeout = setTimeout(() => reject(Error('Preparing browser isolation timed out. Reload to try again.')), 30000);
        })
      ]);
    } finally {
      stopped = true;
      clearTimeout(timeout);
      if (controllerChanged) navigator.serviceWorker.removeEventListener('controllerchange', controllerChanged);
    }
    sessionStorage.setItem(reloadKey, '1');
    location.reload();
    // Startup resumes in the isolated document, never in the first document.
    await new Promise(() => {});
  })();
  // The shell and coil reader attach their error handlers later in the document.
  globalThis.cumesIsolationReady.catch(() => {});
})();
