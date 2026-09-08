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
  const coefficients = input[family];
  const existing = coefficients.find(coefficient => coefficient.m === m && coefficient.n === n);
  if (existing) existing.value = value;
  else if (value !== 0) coefficients.push({m, n, value});
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
    for (let m = 0; m < input.mpol; m++) {
      const row = document.createElement('tr'), label = document.createElement('th');
      label.scope = 'row'; label.textContent = m; row.append(label);
      for (const family of ['rbc', 'zbs']) {
        const cell = document.createElement('td'), control = document.createElement('input');
        const original = input[family].find(coefficient => coefficient.m === m && coefficient.n === n)?.value ?? 0;
        control.type = 'number'; control.step = 'any'; control.value = original;
        control.dataset.family = family; control.dataset.m = m; control.dataset.n = n;
        control.setAttribute('aria-label', `${family.toUpperCase()}(${n},${m}) in meters`);
        control.disabled = document.body.classList.contains('busy') || (family === 'zbs' && m === 0 && n === 0);
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
  get('toroidal-mode').addEventListener('change', () => table(readInput()));
  get('slice-angle').addEventListener('input', () => updateSlice(false));
  return {refresh, get fourier() { return fourier; }, get phi() { return phi; }};
}
