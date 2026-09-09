// Live convergence history. No plotting dependency, diagnostic trace, or GPU reads.
// Keep every sample; draw a first/min/max/last envelope for each horizontal pixel.
function cumesResidualEnvelope(rows, family, projectX) {
  const points = [];
  let bucket = null, stage = null;
  function flush() {
    if (!bucket) return;
    const ordered = [...new Set([bucket.first, bucket.min, bucket.max, bucket.last])].sort((a, b) => a - b);
    for (const index of ordered) points.push(rows[index]);
    bucket = null;
  }
  for (let i = 0; i < rows.length; ++i) {
    const row = rows[i], value = row.fsq[family];
    if (row.stage !== stage) { flush(); points.push(null); stage = row.stage; }
    if (!(value > 0) || !Number.isFinite(value)) { flush(); points.push(null); continue; }
    const pixel = Math.floor(projectX(row.x));
    if (!bucket || bucket.pixel !== pixel) {
      flush(); bucket = {pixel, first: i, min: i, max: i, last: i};
    } else {
      if (value < rows[bucket.min].fsq[family]) bucket.min = i;
      if (value > rows[bucket.max].fsq[family]) bucket.max = i;
      bucket.last = i;
    }
  }
  flush();
  return points;
}

function createCumesResidualPlot(panel, tolerance) {
  const canvas = panel.querySelector('canvas'), context = canvas.getContext('2d');
  const outputs = [...panel.querySelectorAll('[data-residual-value]')];
  const caption = panel.querySelector('.residual-caption');
  const doc = panel.ownerDocument;
  const rows = [], stages = [], restarts = [];
  const colors = ['#55d6d0', '#7ca4ff', '#ffb65d'], dashes = [[], [6, 3], [2, 3]];
  let offset = 0, extent = 0, timer = null, frame = null, dirty = true, finished = null;
  let minLog = Math.floor(Math.log10(tolerance)) - 1, maxLog = 0;
  let drawCount = 0, drawMilliseconds = 0;
  const positive = value => value > 0 && Number.isFinite(value);
  const validPosition = row => Number.isInteger(row.stage) && row.stage > 0 && Number.isInteger(row.attempt) && row.attempt > 0;
  function position(row) {
    const last = stages.at(-1);
    if (last && row.stage < last.stage) return null;
    if (!last || row.stage !== last.stage) {
      offset = extent; stages.push({stage: row.stage, x: offset + row.attempt});
    }
    const x = offset + row.attempt;
    extent = Math.max(extent, x);
    return x;
  }
  function include(value) {
    if (!positive(value)) return;
    minLog = Math.min(minLog, Math.floor(Math.log10(value)));
    maxLog = Math.max(maxLog, Math.ceil(Math.log10(value)));
  }
  function draw() {
    timer = frame = null;
    if (!dirty || doc.visibilityState === 'hidden') return;
    const rect = canvas.getBoundingClientRect();
    if (rect.width < 2 || rect.height < 2) return;
    const started = performance.now(), width = rect.width, height = rect.height;
    const ratio = Math.min(globalThis.devicePixelRatio || 1, 2);
    const pixelsX = Math.round(width * ratio), pixelsY = Math.round(height * ratio);
    if (canvas.width !== pixelsX || canvas.height !== pixelsY) { canvas.width = pixelsX; canvas.height = pixelsY; }
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);
    const left = 62, right = width - 18, top = 30, bottom = height - 38;
    const last = rows.at(-1), limit = Math.max(10, extent);
    const x = value => left + value / limit * (right - left);
    const y = value => bottom - (Math.log10(value) - minLog) / (maxLog - minLog) * (bottom - top);
    context.font = '11px ui-monospace, monospace';
    context.lineWidth = 1; context.setLineDash([]);
    context.fillStyle = '#9aa9bb'; context.strokeStyle = '#27374b';
    context.textAlign = 'right';
    const step = Math.max(1, Math.ceil((maxLog - minLog) / 6));
    for (let exponent = minLog; exponent <= maxLog; exponent += step) {
      const yy = bottom - (exponent - minLog) / (maxLog - minLog) * (bottom - top);
      context.beginPath(); context.moveTo(left, yy); context.lineTo(right, yy); context.stroke();
      context.fillText('1e' + exponent, left - 9, yy + 4);
    }
    const ticks = width < 450 ? 2 : 4;
    context.textAlign = 'center';
    for (let tick = 0; tick <= ticks; ++tick) {
      const value = Math.round(limit * tick / ticks), xx = x(value);
      context.beginPath(); context.moveTo(xx, top); context.lineTo(xx, bottom); context.stroke();
      context.fillText(String(value), xx, bottom + 17);
    }
    context.fillText('Iteration (all grids, attempted)', (left + right) / 2, height - 5);
    context.textAlign = 'left'; context.fillText('Residual · log scale', left, 15);
    const target = last?.tolerance ?? tolerance;
    if (positive(target)) {
      context.strokeStyle = '#b7c4d6'; context.setLineDash([5, 5]);
      context.beginPath(); context.moveTo(left, y(target)); context.lineTo(right, y(target)); context.stroke();
      context.textAlign = 'right'; context.fillText('target ' + target.toExponential(0), right - 3, y(target) - 5);
    }
    context.save(); context.beginPath(); context.rect(left, top, right - left, bottom - top); context.clip();
    context.strokeStyle = '#718398'; context.setLineDash([3, 5]);
    for (const stage of stages.slice(1)) {
      context.beginPath(); context.moveTo(x(stage.x), top); context.lineTo(x(stage.x), bottom); context.stroke();
    }
    context.strokeStyle = '#ed88b8'; context.setLineDash([]);
    for (const restart of restarts) {
      context.beginPath(); context.moveTo(x(restart.x), top); context.lineTo(x(restart.x), bottom); context.stroke();
    }
    for (let family = 0; family < 3; ++family) {
      context.strokeStyle = colors[family]; context.lineWidth = 1.6; context.setLineDash(dashes[family]);
      context.beginPath(); let connected = false;
      for (const row of cumesResidualEnvelope(rows, family, x)) {
        if (!row) { connected = false; continue; }
        context[connected ? 'lineTo' : 'moveTo'](x(row.x), y(row.fsq[family])); connected = true;
      }
      context.stroke();
      if (last && positive(last.fsq[family])) {
        context.fillStyle = colors[family]; context.beginPath(); context.arc(x(last.x), y(last.fsq[family]), 2.5, 0, 2 * Math.PI); context.fill();
      }
    }
    context.restore(); context.setLineDash([]);
    if (last) {
      for (let i = 0; i < 3; ++i) outputs[i].textContent = last.fsq[i].toExponential(3);
      caption.textContent = `${finished === null ? 'Live' : finished ? 'Converged' : 'Stopped'} · grid ${stages.at(-1).stage} · solver iteration ${last.iteration} · ${restarts.length} restarts. Pink vertical lines: restarts; gray dashed: grid changes. Nonpositive/nonfinite values are omitted.`;
      canvas.setAttribute('aria-label', `Residual history, logarithmic scale. ${extent} attempted iterations; ${restarts.length} restarts. FSQR ${last.fsq[0]}, FSQZ ${last.fsq[1]}, FSQL ${last.fsq[2]}.`);
    } else {
      context.fillStyle = '#9aa9bb'; context.textAlign = 'center';
      context.fillText('Residuals will appear when the solve starts.', (left + right) / 2, (top + bottom) / 2);
    }
    dirty = false; ++drawCount; drawMilliseconds += performance.now() - started;
  }
  function schedule() {
    dirty = true;
    if (timer !== null || frame !== null || doc.visibilityState === 'hidden') return;
    timer = setTimeout(() => { timer = null; frame = requestAnimationFrame(draw); }, 100);
  }
  function flush() {
    if (timer !== null) clearTimeout(timer);
    if (frame !== null) cancelAnimationFrame(frame);
    timer = frame = null; draw();
  }
  if (globalThis.ResizeObserver) new ResizeObserver(schedule).observe(canvas);
  doc.addEventListener('visibilitychange', schedule);
  schedule();
  return {
    append(sample) {
      if (!validPosition(sample) || sample.fsq?.length !== 3 || !sample.fsq.every(value => typeof value === 'number')) return;
      const last = rows.at(-1);
      if (last && (sample.stage < last.stage || sample.stage === last.stage && sample.attempt <= last.attempt)) return;
      const x = position(sample); if (x === null) return;
      rows.push({...sample, fsq: [...sample.fsq], x});
      for (const value of sample.fsq) include(value);
      include(sample.tolerance); schedule();
    },
    restart(event) {
      if (!validPosition(event)) return;
      const last = restarts.at(-1);
      if (last && (event.stage < last.stage || event.stage === last.stage && event.attempt <= last.attempt)) return;
      const x = position(event); if (x === null) return;
      restarts.push({...event, x}); schedule();
    },
    finish(success) { finished = success; dirty = true; flush(); },
    report() { return {samples: rows.map(row => ({...row, fsq: [...row.fsq]})), stages: stages.map(row => ({...row})), restarts: restarts.map(row => ({...row})), minLog, maxLog, drawCount, drawMilliseconds}; }
  };
}
