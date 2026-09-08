// Coil geometry is loaded on demand; field grids are built by vacuum-field in Wasm.
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
  function show(data) {
    get('coil-preset').value = data.preset;
    get('coil-description').textContent = data.coilName || `coils.${data.preset}`;
    const {extcur, makegrid_parameters, coils_file, lfreeb, ...equilibrium} = data.input;
    get('coil-currents').value = JSON.stringify(extcur);
    get('coil-grid').value = JSON.stringify(makegrid_parameters, null, 2);
    get('coil-equilibrium').value = JSON.stringify(equilibrium, null, 2);
    globalThis.cumesBoundaryChanged?.();
  }
  function read() {
    if (!config) throw Error('Choose a coil preset or upload a coil file first.');
    const input = JSON.parse(get('coil-equilibrium').value);
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
    error('Loading coil setup…');
    try {
      const response = await fetch(new URL(`presets/${name}.json`, location.href));
      if (!response.ok) throw Error(`Could not load ${name} setup (${response.status}).`);
      const input = await response.json();
      if (token !== selection) return;
      config = {preset: name, input};
      show(config); save(); error('Coil geometry will load when you run.');
    } catch (failure) { error(failure.message); throw failure; }
  }
  get('coil-preset').onchange = async event => {
    if (event.target.value === 'upload') { get('coil-upload').click(); return; }
    try { await preset(event.target.value); } catch (_) {}
  };
  get('coil-upload').onchange = async event => {
    const file = event.target.files[0];
    if (!file) { if(config)get('coil-preset').value=config.preset; return; }
    try {
      if (!config) await preset('solovev');
      const name = file.name.toLowerCase().endsWith('.json') ? 'coils.json' : 'coils.upload';
      await cumesCoilStore({name, bytes: new Uint8Array(await file.arrayBuffer())});
      config = read(); config.preset = 'upload'; config.coilName = file.name;
      config.input.coils_file = '/inputs/' + name;
      show(config); save();
      error('Uploaded. Set the circuit currents, grid bounds, field periods, and equilibrium input to match your coils.');
    } catch (failure) { error(failure.message); }
  };
  get('coil-upload-button').onclick = () => get('coil-upload').click();
  for (const id of ['coil-currents', 'coil-grid', 'coil-equilibrium']) get(id).addEventListener('change', save);
  if (free && config) show(config);
  const ready = free && !config ? preset(namedPreset || 'solovev') : Promise.resolve();
  // Reloading stops an active worker and returns to an editable setup.
  get('stop-run').onclick = () => {
    const next = new URL(location.href); next.searchParams.delete('run'); location.assign(next.href);
  };
  get('stop-run').hidden = query.get('run') !== '1';
  return {
    free, ready, save,
    input() { config = read(); localStorage.setItem('cumes.free.v1', JSON.stringify(config)); return config.input; },
    async files() {
      await ready;
      const data = read();
      let coil;
      if (data.preset === 'upload') {
        coil = await cumesCoilStore();
        if (!coil) throw Error('The uploaded coil file is no longer stored in this browser. Upload it again.');
      } else {
        const response = await fetch(new URL(`presets/coils.${data.preset}`, location.href));
        if (!response.ok) throw Error(`Could not load ${data.preset} coils (${response.status}).`);
        coil = {name: `coils.${data.preset}`, bytes: new Uint8Array(await response.arrayBuffer())};
      }
      return [{path: '/inputs/' + coil.name, bytes: coil.bytes}];
    }
  };
}
