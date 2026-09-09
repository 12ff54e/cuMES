// Prewarm exactly the workers MAKEGRID can use; fixed-boundary runs need none.
function cumesThreadCount() {
  const query = new URLSearchParams(globalThis.cumesSearch ?? globalThis.location?.search ?? '');
  if (query.get('boundary') !== 'free' || query.get('mode') === 'test') return 1;
  const available = Math.max(1, Math.min(16, globalThis.navigator?.hardwareConcurrency || 1));
  const requested = Number(query.get('makegrid_threads'));
  return Number.isInteger(requested) && requested > 0 ? Math.min(available, requested) : available;
}
Module['preRun'] ??= [];
Module['preRun'].push(() => {
  const threads = cumesThreadCount();
  ENV['VFIELD_MAKEGRID_THREADS'] = String(threads);
  if (threads > 1) Module['print']?.(`MAKEGRID using ${threads} CPU threads.`);
});
