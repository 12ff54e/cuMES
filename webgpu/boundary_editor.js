// Boundary controls edit the VMEC input directly; the solver remains its validator.
function migrateCumesEditorUrl(location, history) {
  const url = new URL(location.href);
  if (url.searchParams.get('solve') !== 'w7x') return;
  url.searchParams.delete('solve');
  url.searchParams.delete('mode');
  url.searchParams.delete('run');
  url.searchParams.set('boundary', 'fixed');
  url.searchParams.set('preset', 'w7x');
  history.replaceState(null, '', url.href);
}

function readCumesInputJSON(source, precision) {
  let input;
  try { input = JSON.parse(source.replace(/^\uFEFF/, '')); }
  catch (_) { throw Error('The file is not valid JSON. Choose a cuMES input JSON file.'); }
  if (!input || typeof input !== 'object' || Array.isArray(input))
    throw Error('The input JSON must contain an equilibrium object.');
  for (const key of ['lfreeb', 'lasym'])
    if (input[key] !== undefined && typeof input[key] !== 'boolean')
      throw Error(`${key} must be true or false.`);
  // Reuse the editor checks before replacing a saved setup. The shared solver
  // config API validates profiles, unknown keys and physics when Run is selected.
  boundaryFourier(input);
  const stages = cumesValidateStages(input, 'double');
  if (stages.ftol_array.some(tolerance => tolerance < 1e-6)) precision = 'double';
  return {input, precision};
}

function setBoundaryCoefficient(input, family, m, n, value) {
  const coefficients = input[family] ||= [];
  const existing = coefficients.find(coefficient => coefficient.m === m && coefficient.n === n);
  if (existing) existing.value = value;
  else if (value !== 0) coefficients.push({m, n, value});
}

// The planar contour editor uses the same Fourier projection for both parities.
function fitTokamakContour(pointAt, mpol, lasym, samples = 512) {
  const rbc = Array(mpol).fill(0), zbs = Array(mpol).fill(0);
  const rbs = Array(mpol).fill(0), zbc = Array(mpol).fill(0);
  samples = Math.max(samples, 4 * mpol);
  for (let i = 0; i < samples; i++) {
    const theta = 2 * Math.PI * i / samples, [r, z] = pointAt(theta);
    rbc[0] += r;
    if (lasym) zbc[0] += z;
    for (let m = 1; m < mpol; m++) {
      const c = Math.cos(m * theta), s = Math.sin(m * theta);
      rbc[m] += r * c; zbs[m] += z * s;
      if (lasym) { rbs[m] += r * s; zbc[m] += z * c; }
    }
  }
  const result = {rbc, zbs};
  if (lasym) Object.assign(result, {rbs, zbc});
  for (const coefficients of Object.values(result))
    for (let m = 0; m < mpol; m++) coefficients[m] *= (m ? 2 : 1) / samples;
  return result;
}

function moveContourPoint(points, index, point, lasym) {
  points[index] = [...point];
  if (!lasym) {
    if (index === 0 || index === points.length / 2) points[index][1] = 0;
    points[(points.length - index) % points.length] = [points[index][0], -points[index][1]];
  }
}

function installCumesSurfaceEditor(readInput, writeInput, onChange) {
  const get = id => document.getElementById(id);
  let fourier, phi = 0;
  function rebuildModes(input) {
    const select = get('toroidal-mode'), previous = Number(select.value || 0);
    select.replaceChildren();
    for (let n = -input.ntor; n <= input.ntor; n++) {
      const option = document.createElement('option');
      option.value = n; option.textContent = `n = ${n}`;
      select.append(option);
    }
    select.value = Math.abs(previous) <= input.ntor ? previous : 0;
    table(input);
  }
  function table(input) {
    const n = Number(get('toroidal-mode').value), body = get('boundary-coefficients');
    body.replaceChildren();
    for (const cell of document.querySelectorAll('[data-asymmetric-column]')) cell.hidden = !input.lasym;
    get('allow-asymmetry').checked = !!input.lasym;
    for (let m = 0; m < input.mpol; m++) {
      const row = document.createElement('tr'), label = document.createElement('th');
      label.scope = 'row'; label.textContent = m; row.append(label);
      for (const family of (input.lasym ? ['rbc', 'zbs', 'rbs', 'zbc'] : ['rbc', 'zbs'])) {
        const cell = document.createElement('td'), control = document.createElement('input');
        const original = (input[family] || []).find(coefficient => coefficient.m === m && coefficient.n === n)?.value ?? 0;
        control.type = 'number'; control.step = 'any'; control.value = original;
        control.dataset.family = family; control.dataset.m = m; control.dataset.n = n;
        control.setAttribute('aria-label', `${family.toUpperCase()}(${n},${m}) in meters`);
        control.disabled = document.body.classList.contains('busy') || ((family === 'zbs' || family === 'rbs') && m === 0 && n === 0);
        control.addEventListener('input', () => {
          if (control.value === '' || !Number.isFinite(Number(control.value))) return;
          try {
            const updated = readInput();
            setBoundaryCoefficient(updated, family, m, n, Number(control.value));
            writeInput(updated); fourier = boundaryFourier(updated);
            get('boundary-error').textContent = ''; onChange(true);
          } catch (error) { get('boundary-error').textContent = error.message; }
        });
        cell.append(control); row.append(cell);
      }
      body.append(row);
    }
  }
  function refresh() {
    try {
      const input = readInput(); fourier = boundaryFourier(input);
      get('boundary-dimensions').textContent = `${fourier.nfp} field period${fourier.nfp === 1 ? '' : 's'} · m < ${fourier.mpol} · |n| ≤ ${fourier.ntor}`;
      rebuildModes({...input, ntor: fourier.ntor});
      get('slice-angle').disabled = fourier.ntor === 0;
      updateSlice(false);
      get('boundary-error').textContent = '';
      return true;
    } catch (error) { get('boundary-error').textContent = error.message; return false; }
  }
  function updateSlice(changed) {
    if (!fourier) return;
    phi = Number(get('slice-angle').value) / 100 * 2 * Math.PI / fourier.nfp;
    get('slice-angle-value').textContent = `${(phi * 180 / Math.PI).toFixed(1)}° / ${(360 / fourier.nfp).toFixed(1)}°`;
    onChange(changed);
  }
  get('allow-asymmetry').addEventListener('change', () => {
    const checkbox = get('allow-asymmetry');
    if (checkbox.disabled || document.body.classList.contains('busy')) {
      checkbox.checked = !!readInput().lasym;
      return;
    }
    const input = readInput();
    input.lasym = checkbox.checked;
    if (input.lasym) { input.rbs ||= []; input.zbc ||= []; }
    else for (const key of ['rbs', 'zbc', 'raxis_s', 'zaxis_c']) delete input[key];
    writeInput(input); refresh(); onChange(true);
  });
  get('toroidal-mode').addEventListener('change', () => table(readInput()));
  get('slice-angle').addEventListener('input', () => updateSlice(false));
  return {refresh, get fourier() { return fourier; }, get phi() { return phi; }};
}

// Stage controls edit the shared input arrays. The C++ config API remains the
// final validator, including the precision floor and prolongation requirements.
function cumesStageArrays(input) {
  const keys = ['ns_array', 'niter_array', 'ftol_array'];
  if (keys.every(key => input[key] === undefined))
    return {ns_array:[11], niter_array:[1000], ftol_array:[1e-16]}; // default_stage()
  if (!keys.every(key => Array.isArray(input[key])) || !input.ns_array.length ||
      input.ns_array.length > 510 || !keys.every(key => input[key].length === input.ns_array.length))
    throw Error('Provide 1–510 stages with matching radial grid, step cap and tolerance arrays.');
  return Object.fromEntries(keys.map(key => [key, [...input[key]]]));
}

function cumesValidateStages(input, precision) {
  const stages = cumesStageArrays(input), floor = precision === 'float' ? 1e-6 : 1e-16;
  stages.ns_array.forEach((ns, i) => {
    if (!Number.isInteger(ns) || ns < 3 || ns > 512)
      throw Error(`Stage ${i + 1}: radial grid points must be an integer from 3 to 512.`);
    if (i && ns <= stages.ns_array[i - 1])
      throw Error(`Stage ${i + 1}: the radial grid must be larger than the preceding stage.`);
    if (!Number.isInteger(stages.niter_array[i]) || stages.niter_array[i] < 1 || stages.niter_array[i] > 2147483647)
      throw Error(`Stage ${i + 1}: the step cap must be a positive integer (at most 2147483647).`);
    if (!Number.isFinite(stages.ftol_array[i]) || stages.ftol_array[i] < floor)
      throw Error(`Stage ${i + 1}: tolerance must be finite and at least ${floor} for ${precision === 'float' ? 'single' : 'paired'} precision.`);
  });
  return stages;
}

function cumesResizeStages(stages, count) {
  const last = stages.ns_array.at(-1);
  if (!Number.isInteger(count) || count < 1 || count > last - 2)
    throw Error(`Choose 1–${last - 2} stages, or increase the final radial grid first.`);
  if (count === stages.ns_array.length) return cumesStageArrays(stages);
  const result = {ns_array:[], niter_array:[], ftol_array:[]};
  for (let i = 0; i < count; i++) {
    const source = i === count - 1 ? stages.ns_array.length - 1 : Math.min(i, stages.ns_array.length - 1);
    result.ns_array.push(Math.max(i + 3, Math.round(last * (i + 1) / count)));
    result.niter_array.push(stages.niter_array[source]);
    result.ftol_array.push(stages.ftol_array[source]);
  }
  return result;
}

function cumesStagePrecision(stages, previous, next) {
  const result = cumesStageArrays(stages);
  const defaultTolerance = precision => precision === 'float' ? 1e-5 : 1e-12;
  result.ftol_array = result.ftol_array.map(value => value === defaultTolerance(previous) ? defaultTolerance(next) : value);
  return cumesValidateStages(result, next);
}

function installCumesStageEditor(root, precision, storageKey, onChange, onRefresh, onError = () => {}) {
  const get = id => root.querySelector('#' + id);
  const count = get('stage-count'), sync = get('stage-sync-limits'), limits = get('stage-limits');
  const error = get('stage-error'), keys = ['ns_array', 'niter_array', 'ftol_array'];
  let savedSync = localStorage.getItem(storageKey) === 'true';
  sync.checked = savedSync;
  get('stage-tolerance-floor').textContent = `Minimum tolerance for this precision: ${precision === 'float' ? '1e-6' : '1e-16'}.`;
  function showError(message, reveal = true) {
    error.textContent = message;
    if (message) { limits.open = true; if (reveal) onError(); }
  }
  function fields(key) { return [...root.querySelectorAll(`[data-stage-key="${key}"]`)]; }
  function raw() {
    const result = Object.fromEntries(keys.map(key => [key, fields(key).map(input => input.value === '' ? NaN : Number(input.value))]));
    if (Number(count.value) !== result.ns_array.length) throw Error('Enter a valid stage count before running.');
    return result;
  }
  function read(next = precision) {
    try { return next === precision ? cumesValidateStages(raw(), precision) : cumesStagePrecision(raw(), precision, next); }
    catch (failure) { showError(failure.message); throw failure; }
  }
  function commit() {
    try { const stages = read(); onChange(stages); onRefresh(stages); showError(''); return true; }
    catch (failure) { showError(failure.message); return false; }
  }
  function rememberSync() { savedSync = sync.checked; localStorage.setItem(storageKey, String(savedSync)); }
  function render(stages) {
    count.value = stages.ns_array.length;
    get('stage-grids').replaceChildren(); get('stage-limit-rows').replaceChildren();
    for (let i = 0; i < stages.ns_array.length; i++) {
      for (const [body, columns] of [['stage-grids', ['ns_array']], ['stage-limit-rows', ['niter_array', 'ftol_array']]]) {
        const row = document.createElement('tr'), label = document.createElement('th');
        label.scope = 'row'; label.textContent = i + 1; row.append(label);
        for (const key of columns) {
          const cell = document.createElement('td'), control = document.createElement('input');
          const name = {ns_array:'Radial grid points', niter_array:'Step cap', ftol_array:'Tolerance'}[key];
          control.type = 'number'; control.value = stages[key][i]; control.dataset.stageKey = key;
          control.min = key === 'ns_array' ? 3 : key === 'niter_array' ? 1 : precision === 'float' ? 1e-6 : 1e-16;
          control.step = key === 'ftol_array' ? 'any' : 1;
          if (key !== 'ftol_array') control.max = key === 'ns_array' ? 512 : 2147483647;
          control.setAttribute('aria-label', `Stage ${i + 1}: ${name}`);
          control.setAttribute('aria-describedby', 'stage-error');
          control.addEventListener('input', () => {
            if (root.disabled) return;
            if (sync.checked && key !== 'ns_array')
              for (const peer of fields(key)) if (peer !== control) peer.value = control.value;
            commit();
          });
          cell.append(control); row.append(cell);
        }
        get(body).append(row);
      }
    }
  }
  function refresh(input, revealErrors = true) {
    try {
      const stages = cumesStageArrays(input);
      render(stages);
      if (!['niter_array', 'ftol_array'].every(key => stages[key].every(value => value === stages[key][0]))) {
        sync.checked = false; rememberSync();
      }
      cumesValidateStages(stages, precision); onRefresh(stages); showError(''); return true;
    } catch (failure) { showError(failure.message, revealErrors); return false; }
  }
  count.addEventListener('change', () => {
    if (root.disabled) return;
    try {
      // Read existing rows independently of the new count.
      const stages = Object.fromEntries(keys.map(key => [key, fields(key).map(input => input.value === '' ? NaN : Number(input.value))]));
      const resized = cumesResizeStages(cumesValidateStages(stages, precision), Number(count.value));
      render(resized); commit();
    } catch (failure) { showError(failure.message); }
  });
  sync.addEventListener('change', () => {
    if (root.disabled) { sync.checked = savedSync; return; }
    if (sync.checked) {
      for (const key of ['niter_array', 'ftol_array']) {
        const inputs = fields(key);
        for (const input of inputs.slice(1)) input.value = inputs[0].value;
      }
    }
    rememberSync(); commit();
  });
  return {read, refresh, showError};
}
