// Emscripten library boundary between the solver and the browser application.
// No DOM dependencies: the frontend installs a page or worker event sink.
mergeInto(LibraryManager.library, {
  requested_float_solve: function() {
    return new URLSearchParams(globalThis.cumesSearch ?? location.search).get('precision') === 'float' ? 1 : 0;
  },
  requested_double_solve: function() {
    return new URLSearchParams(globalThis.cumesSearch ?? location.search).get('precision') === 'double' ? 1 : 0;
  },
  requested_float_radius_reference: function() {
    return new URLSearchParams(globalThis.cumesSearch ?? location.search).get('radius_reference') === '0' ? 0 : 1;
  },
  requested_compensated_geometry: function() {
    return new URLSearchParams(globalThis.cumesSearch ?? location.search).get('geometry') === 'native' ? 0 : 1;
  },
  publish_browser_result__deps: ['$UTF8ToString'],
  publish_browser_result: function(success, detail) {
    const timing = globalThis.cumesIterationTiming?.finish();
    globalThis.cumesBrowser.result(Boolean(success), UTF8ToString(detail), timing);
  },

  publish_browser_iteration_timing: function(kind, stage) {
    globalThis.cumesIterationTiming?.event(kind, stage);
  },

  publish_browser_output__deps: ['$FS', '$UTF8ToString'],
  publish_browser_output: function(path) {
    try {
      const bytes = FS.readFile(UTF8ToString(path));
      const size = bytes.length;
      globalThis.cumesBrowser.output(bytes);
      return size;
    } catch (error) {
      globalThis.cumesBrowser.error('cumesOutputError', String(error));
      return -1;
    }
  },

  requested_w7x_solve: function() {
    return new URLSearchParams(globalThis.cumesSearch ?? location.search).get('solve') === 'w7x';
  },

  requested_w7x_multigrid: function() {
    return new URLSearchParams(globalThis.cumesSearch ?? location.search).get('grids') === '3';
  },
  requested_reference_transfers: function() {
    return new URLSearchParams(globalThis.cumesSearch ?? location.search).get('resident') === '0';
  },
  requested_direct_dft: function() {
    // At W7-X's 36-point grid the two transforms have equal pass throughput,
    // but direct projection reaches tolerance sooner on the qualified GPU.
    // Keep the reusable FFT available explicitly without slowing the example.
    return new URLSearchParams(globalThis.cumesSearch ?? location.search).get('fft') !== '1';
  },
  requested_generic_fft: function() {
    return new URLSearchParams(globalThis.cumesSearch ?? location.search).get('fft_kernel') === 'generic';
  },
  requested_canonical_zeta: function() {
    return new URLSearchParams(globalThis.cumesSearch ?? location.search).get('basis') === 'canonical';
  },
  requested_solver_trace: function() {
    return new URLSearchParams(globalThis.cumesSearch ?? location.search).get('trace') === '1';
  },
  requested_shadow_norms: function() {
    return new URLSearchParams(globalThis.cumesSearch ?? location.search).get('gpu_norms') === 'shadow';
  },
  requested_device_norms: function() {
    return new URLSearchParams(globalThis.cumesSearch ?? location.search).get('gpu_norms') === '1';
  },
  requested_full_field_readbacks: function() {
    return new URLSearchParams(globalThis.cumesSearch ?? location.search).get('field_readbacks') === 'full';
  },
  requested_geometry_control: function() {
    return new URLSearchParams(globalThis.cumesSearch ?? location.search).get('gpu_control') === 'jacobian';
  },
  requested_compare_fft: function() {
    return new URLSearchParams(globalThis.cumesSearch ?? location.search).get('compare_fft') === '1';
  },
  requested_spectral_fences: function() {
    return new URLSearchParams(globalThis.cumesSearch ?? location.search).get('fences') === '1';
  },
  publish_browser_diagnostic__deps: ['$UTF8ToString'],
  publish_browser_diagnostic: function(json) {
    globalThis.cumesBrowser.diagnostic({milliseconds: performance.now(),
      ...JSON.parse(UTF8ToString(json))});
  },

  requested_app_mode: function() {
    const query = new URLSearchParams(globalThis.cumesSearch ?? location.search);
    return query.get('mode') !== 'test' && query.get('solve') !== 'w7x';
  },

  requested_app_run: function() {
    return new URLSearchParams(globalThis.cumesSearch ?? location.search).get('run') === '1';
  },

  publish_browser_ready: function() {
    globalThis.cumesBrowser.ready();
  },

  publish_browser_equilibrium__deps: ['$UTF8ToString'],
  publish_browser_equilibrium: function(json) {
    try {
      const result = JSON.parse(UTF8ToString(json));
      globalThis.cumesBrowser.equilibrium(result);
    } catch (error) {
      globalThis.cumesBrowser.error('cumesPlotError', String(error));
    }
  },

  publish_browser_adapter__deps: ['$UTF8ToString'],
  publish_browser_adapter: function(device, type, backend) {
    globalThis.cumesBrowser.adapter(UTF8ToString(device), UTF8ToString(type), UTF8ToString(backend));
  }
});
