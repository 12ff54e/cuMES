// The same vacuum-field parser used by MAKEGRID validates both upload formats.
// Its small converter module is loaded only when coil geometry is requested.
let cumesCoilReader;
async function readCumesCoils(file) {
  if (!cumesCoilReader) {
    const runtime = document.getElementById('cumes-runtime').content.querySelector('script');
    const url = new URL('cumes_coils.mjs', location.href);
    url.search = new URL(runtime.getAttribute('src'), location.href).search;
    cumesCoilReader = import(url.href).then(async ({default: createModule}) => {
      const errors = [];
      const module = await createModule({print() {}, printErr: line => errors.push(line),
        locateFile(path) { const asset = new URL(path, url); asset.search = url.search; return asset.href; }});
      return {module, errors};
    }).catch(error => { cumesCoilReader = null; throw error; });
  }
  const {module, errors} = await cumesCoilReader;
  const input = file.name.endsWith('.json') ? '/source.json' : '/source.coils';
  const output = '/preview.json';
  errors.length = 0;
  try {
    module.FS.writeFile(input, file.bytes);
    if (module.callMain([input, output]) !== 0)
      throw Error(errors.join('\n') || 'Could not read coil geometry.');
    return JSON.parse(module.FS.readFile(output, {encoding: 'utf8'}));
  } finally {
    for (const path of [input, output]) { try { module.FS.unlink(path); } catch (_) {} }
  }
}

function coilLines(configuration, fourier) {
  if (!configuration) return null;
  // This coefficient bound only distinguishes effectively infinite conductors
  // from device-scale coils. The surface's actual framing radius stays on GPU.
  let reference = 1;
  const modes = fourier.mpol * (fourier.ntor + 1);
  for (const surface of fourier.surfaces) {
    let r = 0, z = 0;
    for (let i = 0; i < modes; i++) {
      r += Math.abs(surface.coefficients[i]) + Math.abs(surface.coefficients[3 * modes + i]);
      z += Math.abs(surface.coefficients[modes + i]) + Math.abs(surface.coefficients[4 * modes + i]);
    }
    reference = Math.max(reference, Math.hypot(r, z));
  }
  const filaments = configuration.circuits.flatMap(circuit => circuit.filaments.map(filament => {
    if (filament.type === 'polygon') return filament.vertices;
    return Array.from({length: 160}, (_, i) => {
      const phi = 2 * Math.PI * i / 160;
      return [filament.radius * Math.cos(phi), filament.radius * Math.sin(phi), filament.center_z];
    });
  }));
  let radius = 0, clipped = false;
  for (const filament of filaments) for (const point of filament) {
    const distance = Math.hypot(...point);
    if (distance < reference * 1e4) radius = Math.max(radius, distance);
    else clipped = true;
  }
  // Solovev's central conductor closes at 1e6 m. Clip remote segment ends,
  // preserving the nearby straight conductor without drawing a fake closure.
  const limit = 3 * Math.max(reference, radius), positions = [], indices = [];
  for (const filament of filaments) for (let i = 0; i < filament.length; i++) {
    const a = filament[i], b = filament[(i + 1) % filament.length];
    let first = 0, last = 1;
    for (let axis = 0; axis < 3; axis++) {
      const delta = b[axis] - a[axis];
      if (delta === 0) { if (Math.abs(a[axis]) > limit) last = -1; }
      else {
        const near = (-limit - a[axis]) / delta, far = (limit - a[axis]) / delta;
        first = Math.max(first, Math.min(near, far)); last = Math.min(last, Math.max(near, far));
      }
    }
    if (first >= last) continue;
    const start = positions.length / 4;
    for (const t of [first, last]) {
      for (let axis = 0; axis < 3; axis++) positions.push(a[axis] + t * (b[axis] - a[axis]));
      positions.push(-2);
    }
    indices.push(start, start + 1);
  }
  return {positions: new Float32Array(positions), indices: new Uint32Array(indices), radius,
    clipped, filaments: filaments.length};
}
