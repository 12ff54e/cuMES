// Compare captures from webgpu_cdp.mjs eval 'window.cumesDiagnostics'.
// Hashes locate divergence; they are diagnostics, not numerical error bounds.
import {readFile} from 'node:fs/promises';

const paths = process.argv.slice(2);
if (paths.length !== 2) throw Error('Pass two saved CDP trace JSON files');
const traces = await Promise.all(paths.map(async path => {
  const parsed = JSON.parse(await readFile(path, 'utf8'));
  return parsed.result?.value ?? parsed;
}));
const controllers = traces.map(trace => trace.filter(row => row.kind === 'controller')
  .map(row => ({...row, b1_f32: Math.fround(row.b1), fac_f32: Math.fround(row.fac)})));
const fields = ['state_hash', 'state_low_hash', 'preconditioned_hash',
  'fsq', 'preconditioned', 'delta', 'b1', 'fac', 'b1_f32', 'fac_f32',
  'restart', 'anchor', 'refresh', 'checkpoint'];
const firstDifferences = {};
const second = new Map(controllers[1].map(row => [row.attempt, row]));
for (const a of controllers[0]) {
  const b = second.get(a.attempt);
  if (!b) continue;
  for (const field of fields) {
    if (!(field in firstDifferences) && JSON.stringify(a[field]) !== JSON.stringify(b[field]))
      firstDifferences[field] = {attempt: a.attempt, iterations: [a.iter, b.iter],
        values: [a[field], b[field]]};
  }
}
console.log(JSON.stringify({paths, lengths: controllers.map(rows => rows.length),
  firstDifferences, last: controllers.map(rows => rows.at(-1)),
  sameInput: traces.map(trace => trace.filter(row => row.kind === 'transform'))}, null, 2));
