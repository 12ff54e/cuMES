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
