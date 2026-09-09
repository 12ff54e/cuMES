// Evaluate with webgpu_cdp.mjs eval-file in a completed result tab. Hash the
// scientific payload, excluding revision/timing provenance that varies by run.
(async () => {
  if (!window.cumesOutputUrl) throw Error('No published equilibrium output');
  const bytes = await (await fetch(window.cumesOutputUrl)).arrayBuffer();
  const view = new DataView(bytes);
  const version = view.getInt32(8, true);
  const ns = view.getInt32(12, true), mnmax = view.getInt32(16, true);
  if (version !== 8 || ns < 2 || mnmax < 1)
    throw Error(`Unsupported output shape/version: ${version}/${ns}/${mnmax}`);
  const spectralEnd = 20 + 6 * ns * mnmax * 8;
  const ntheta = view.getInt32(spectralEnd, true);
  const nzeta = view.getInt32(spectralEnd + 4, true);
  if (ntheta < 1 || nzeta < 1) throw Error('Missing derived output fields');
  const end = spectralEnd + 8 + (7 * (ns - 1) + 6 * ns) * ntheta * nzeta * 8;
  if (!Number.isSafeInteger(end) || end > bytes.byteLength)
    throw Error('Truncated scientific output');
  const digest = async (begin, end) => Array.from(new Uint8Array(
    await crypto.subtle.digest('SHA-256', bytes.slice(begin, end))))
    .map(value => value.toString(16).padStart(2, '0')).join('');
  return {version, ns, mnmax, ntheta, nzeta, scientificBytes: end,
    state: await digest(20, spectralEnd),
    fields: await digest(spectralEnd + 8, end)};
})()
