// Emscripten library boundary between the solver and the browser application.
// All DOM, URL, and virtual-filesystem policy stays on the JavaScript side.
mergeInto(LibraryManager.library, {
  publish_browser_result__deps: ['$UTF8ToString'],
  publish_browser_result: function(success, detail) {
    document.body.dataset.cumesWebgpu = success ? 'pass' : 'fail';
    document.body.dataset.cumesDetail = UTF8ToString(detail);
    clearTimeout(window.cumesDeadline);
    clearInterval(window.cumesKeepAlive);
  },

  publish_browser_output__deps: ['$FS', '$UTF8ToString'],
  publish_browser_output: function(path) {
    try {
      const bytes = FS.readFile(UTF8ToString(path));
      const blob = new Blob([bytes], {type: 'application/octet-stream'});
      if (window.cumesOutputUrl) URL.revokeObjectURL(window.cumesOutputUrl);
      window.cumesOutputUrl = URL.createObjectURL(blob);
      const link = document.getElementById('download');
      link.href = window.cumesOutputUrl;
      link.download = 'cumes-webgpu-output.bin';
      link.hidden = false;
      document.body.dataset.cumesOutputBytes = String(bytes.length);
      return bytes.length;
    } catch (error) {
      document.body.dataset.cumesOutputError = String(error);
      return -1;
    }
  },

  requested_w7x_solve: function() {
    return new URLSearchParams(window.location.search).get('solve') === 'w7x';
  },

  requested_app_mode: function() {
    const query = new URLSearchParams(window.location.search);
    return query.get('mode') !== 'test' && query.get('solve') !== 'w7x';
  },

  requested_app_run: function() {
    return new URLSearchParams(window.location.search).get('run') === '1';
  },

  publish_browser_ready: function() {
    document.body.dataset.cumesWebgpu = 'ready';
    if (window.cumesAppReady) window.cumesAppReady();
  },

  publish_browser_equilibrium__deps: ['$UTF8ToString'],
  publish_browser_equilibrium: function(json) {
    try {
      const result = JSON.parse(UTF8ToString(json));
      if (window.cumesPublishEquilibrium) window.cumesPublishEquilibrium(result);
    } catch (error) {
      document.body.dataset.cumesPlotError = String(error);
    }
  },

  publish_browser_adapter__deps: ['$UTF8ToString'],
  publish_browser_adapter: function(device, type, backend) {
    document.body.dataset.cumesAdapter = UTF8ToString(device);
    document.body.dataset.cumesAdapterType = UTF8ToString(type);
    document.body.dataset.cumesAdapterBackend = UTF8ToString(backend);
  }
});
