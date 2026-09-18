// Preview and solve share the selected coil bytes; MAKEGRID still runs only on Run.
function cumesCoilStore(value) {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('cumes-coils', 1);
    request.onupgradeneeded = () => request.result.createObjectStore('files');
    request.onerror = () => reject(request.error);
    request.onsuccess = () => {
      const db = request.result;
      const transaction = db.transaction('files', value ? 'readwrite' : 'readonly');
      const operation = value ? transaction.objectStore('files').put(value, 'uploaded') :
        transaction.objectStore('files').get('uploaded');
      transaction.oncomplete = () => { db.close(); resolve(operation.result); };
      transaction.onerror = () => { db.close(); reject(transaction.error); };
    };
  });
}

function cumesFreeInputConfig(input, saved) {
  if (!Array.isArray(input.extcur) || !input.extcur.length || !input.extcur.every(Number.isFinite))
    throw Error('Free-boundary input needs an extcur array of finite coil currents.');
  if (!input.makegrid_parameters || typeof input.makegrid_parameters !== 'object' ||
      Array.isArray(input.makegrid_parameters) || input.makegrid_parameters_file || input.mgrid_file)
    throw Error('Free-boundary uploads need inline makegrid_parameters; the browser generates the field grid from coils.');
  const basename = path => typeof path === 'string' ? path.replaceAll('\\', '/').split('/').at(-1) : '';
  const name = basename(input.coils_file);
  const preset = name.match(/^coils\.(solovev|w7x|cth_like)$/)?.[1];
  if (preset) return {preset, input:{...input, coils_file:'/inputs/coils.' + preset}};
  if (name && saved?.preset === 'upload' && [saved.coilName, basename(saved.input?.coils_file)].includes(name))
    return {preset:'upload', coilName:saved.coilName, input:{...input, coils_file:saved.input.coils_file}};
  throw Error('Upload the referenced coil file in Free boundary, then upload this input JSON again.');
}

function installCumesBoundaryMode() {
  const query = new URLSearchParams(location.search);
  const free = query.get('boundary') === 'free';
  document.body.dataset.boundaryMode = free ? 'free' : 'fixed';
  const get = id => document.getElementById(id);
  for (const mode of ['fixed', 'free']) {
    const button = get('boundary-' + mode);
    button.classList.toggle('active', mode === (free ? 'free' : 'fixed'));
    button.setAttribute('aria-pressed', String(mode === (free ? 'free' : 'fixed')));
    button.onclick = () => {
      save();
      const next = new URL(location.href);
      next.searchParams.set('boundary', mode);
      next.searchParams.delete('run');
      next.searchParams.delete('coils');
      if(mode==='free')next.searchParams.delete('preset');
      location.assign(next.href);
    };
  }
  get('free-boundary-controls').hidden = !free;
  let saved;
  try { saved = JSON.parse(localStorage.getItem('cumes.free.v1')); } catch (_) {}
  const requested = query.get('coils');
  const namedPreset = ['solovev', 'w7x', 'cth_like'].includes(requested) ? requested : null;
  let config = namedPreset ? null : saved || null, selection = 0;
  let coilFile = null, coilGeometry = null, coilLoad;
  function publishCoils(file, geometry) {
    coilFile = file; coilGeometry = geometry;
    globalThis.cumesCoilsChanged?.();
  }
  async function loadCoils(data, token) {
    let file;
    if (data.preset === 'upload') {
      file = await cumesCoilStore();
      if (!file) throw Error('The uploaded coil file is no longer stored in this browser. Upload it again.');
    } else {
      const response = await fetch(new URL(`presets/coils.${data.preset}`, location.href));
      if (!response.ok) throw Error(`Could not load ${data.preset} coils (${response.status}).`);
      file = {name: `coils.${data.preset}`, bytes: new Uint8Array(await response.arrayBuffer())};
    }
    const geometry = await readCumesCoils(file);
    if (token === selection) publishCoils(file, geometry);
  }
  function show(data) {
    get('input-upload-status').textContent = data.inputName ? 'Loaded ' + data.inputName : '';
    get('coil-preset').value = data.preset;
    get('coil-description').textContent = data.coilName || `coils.${data.preset}`;
    const {extcur, makegrid_parameters} = data.input;
    get('coil-currents').value = JSON.stringify(extcur);
    get('coil-grid').value = JSON.stringify(makegrid_parameters, null, 2);
    globalThis.cumesBoundaryChanged?.();
  }
  function read() {
    if (!config) throw Error('Choose a coil preset or upload a coil file first.');
    const input = structuredClone(config.input);
    input.lfreeb = true;
    input.extcur = JSON.parse(get('coil-currents').value);
    input.makegrid_parameters = JSON.parse(get('coil-grid').value);
    input.coils_file = config.input.coils_file;
    delete input.mgrid_file;
    delete input.makegrid_parameters_file;
    if (!Array.isArray(input.extcur) || !input.extcur.length || !input.extcur.every(Number.isFinite))
      throw Error('Enter a JSON array of finite coil currents in amperes.');
    return {...config, input};
  }
  function save() {
    if (!free || !config) return;
    try { config = read(); localStorage.setItem('cumes.free.v1', JSON.stringify(config)); }
    catch (_) { /* Keep the last valid setup while an input is being edited. */ }
  }
  function error(message) { get('coil-status').textContent = message; }
  async function preset(name) {
    const token = ++selection;
    publishCoils(null, null);
    error('Loading coil setup…');
    try {
      const response = await fetch(new URL(`presets/${name}.json`, location.href));
      if (!response.ok) throw Error(`Could not load ${name} setup (${response.status}).`);
      const input = await response.json();
      if (token !== selection) return;
      if (query.get('precision') === 'float') input.ftol_array = input.ns_array.map(() => 1e-5);
      config = {preset: name, input};
      show(config); save();
      coilLoad = loadCoils(config, token);
      await coilLoad;
      if (token === selection) error('Coils ready. Field grid will be generated when you run.');
    } catch (failure) { if (token === selection) error(failure.message); throw failure; }
  }
  get('coil-preset').onchange = async event => {
    if (event.target.value === 'upload') { get('coil-upload').click(); return; }
    try { await preset(event.target.value); } catch (_) {}
  };
  get('coil-upload').onchange = async event => {
    const file = event.target.files[0];
    if (!file) { if(config)get('coil-preset').value=config.preset; return; }
    if (!config) { try { await ready; } catch (_) { return; } }
    const token = ++selection;
    try {
      const name = file.name.toLowerCase().endsWith('.json') ? 'coils.json' : 'coils.upload';
      const stored = {name, bytes: new Uint8Array(await file.arrayBuffer())};
      const geometry = await readCumesCoils(stored);
      if (token !== selection) return;
      const updated = read(); updated.preset = 'upload'; updated.coilName = file.name;
      updated.input.coils_file = '/inputs/' + name;
      await cumesCoilStore(stored);
      if (token !== selection) return;
      config = updated;
      show(config); save();
      publishCoils(stored, geometry); coilLoad = Promise.resolve();
      error('Uploaded. Set the circuit currents, grid bounds, field periods, and equilibrium input to match your coils.');
    } catch (failure) { if (token === selection) error(failure.message); }
  };
  get('coil-upload-button').onclick = () => get('coil-upload').click();
  for (const id of ['coil-currents', 'coil-grid']) get(id).addEventListener('change', save);
  if (free && config) show(config);
  const ready = !free ? Promise.resolve() : config ?
    (coilLoad = loadCoils(config, selection)) : preset(namedPreset || 'solovev');
  // Reloading stops an active worker and returns to an editable setup.
  get('stop-run').onclick = () => {
    const next = new URL(location.href); next.searchParams.delete('run'); location.assign(next.href);
  };
  get('stop-run').hidden = query.get('run') !== '1';
  return {
    free, ready, save,
    equilibrium() { return structuredClone(config.input); },
    setEquilibrium(input) { config.input = structuredClone(input); },
    geometry() { return coilGeometry; },
    input() { config = read(); localStorage.setItem('cumes.free.v1', JSON.stringify(config)); return config.input; },
    async files() {
      await ready;
      await coilLoad;
      if (!coilFile) throw Error('Coil geometry has not finished loading.');
      return [{path: '/inputs/' + coilFile.name, bytes: coilFile.bytes}];
    }
  };
}
