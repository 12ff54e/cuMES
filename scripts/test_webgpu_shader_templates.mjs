import assert from 'node:assert/strict';
import {preprocess} from '../webgpu/shader_template.mjs';

const compile = (source, precision = 'single', options = {}) =>
  preprocess(source, {precision, ...options});
const compact = s => s.replace(/\s+/g, '');

// Precedence, nesting, and argument separation (including WGSL type arguments).
assert.equal(compact(compile('mul(a + b, sub(c, d / e))').source), '((a+b)*(((c)-(d/e))))');
assert.equal(compact(compile('neg(a + b)').source), '(-(a+b))');
assert.equal(compact(compile('reciprocal(a + b)').source), '(1.0/(a+b))');
assert.equal(compile('add(a, mul(b, c))', 'paired').source,
  'compensate_add(a, compensate_mul(b, c, slot), slot)');
assert.equal(compile('add(a, b, lane)', 'paired').source, 'compensate_add(a, b, lane)');
assert.match(compile('add(array<vec2<f32>, 2>(vec2f(1, 2), vec2f(3, 4))[0].x, b)').source, /\+ \(b\)/);
assert.match(compile('add(select(a, b, a < b), c)').source, /select\(a, b, a < b\)/);

// Explicit paired islands retain both words and compensation in scalar kernels.
const mixed = 'var x: Real = real(a); var sum: Pair = pair_words(0.0, 0.0); sum = pair_add(sum, pair_real(x), lane);';
const scalar = compile(mixed);
assert.equal(scalar.needs_pair, true);
assert.match(scalar.source, /var x: f32 = \(a\)/);
assert.match(scalar.source, /var sum: FF = FF\(0.0, 0.0\)/);
assert.match(scalar.source, /compensate_add\(sum, FF\(x, 0.0\), lane\)/);
assert.equal(compile('var x: Real = add(a, b);').needs_pair, false);
assert.match(compile('var x: array<Real, 4>;', 'paired').source, /array<FF, 4>/);
assert.equal(compile('words(buffer_hi[i], buffer_lo[i])').source, '(buffer_hi[i])');

// Decimal constants are split once at build time, without rounding their low
// word away. Test known mathematical constants against independently given words.
assert.equal(compile('literal(0.7071067811865476)', 'paired').source,
  'FF(0.7071067690849304, 1.2101617485882343e-8)');
assert.equal(compile('literal(-0.0)', 'paired').source, 'FF(-0.0, 0.0)');
assert.equal(compile('literal(0.1)').source, '(0.1)');

const source = `// add(x, y) stays a comment.
/* nested /* real(2) */
#if PAIRED
*/
#if PAIRED
let x: Real = literal(1.0);
#else
let x: Real = real(2.0);
#if !PAIRED
let y = add(x, x);
#endif
#endif
`;
assert.match(compile(source).source, /let x: f32 = \(2.0\)/);
assert.match(compile(source, 'paired').source, /let x: FF = FF\(1.0, 0.0\)/);
assert.match(compile(source).source, /\/\/ add\(x, y\) stays a comment/);
assert.equal(compile('#include "part"\n', 'single', {filename: 'main', include: () =>
  ({filename: 'part', source: 'add(a, b)'})}).source, '((a) + (b))');

for (const invalid of ['add(a)', 'add(a,)', 'mul(a, add(b,c)', '#if PAIRED\n', '#else\n',
  '#if PAIRED\n#else\n#else\n#endif', '#if UNKNOWN\n#endif', '#unknown',
  'literal(variable)', 'literal(1e50)', '/* unclosed', 'fn add(a: Real, b: Real) {}']) {
  assert.throws(() => compile(invalid), undefined, invalid);
}
assert.throws(() => compile('#include "main"\n', 'single', {filename: 'main', include: () =>
  ({filename: 'main', source: ''})}), /Include cycle/);
assert.throws(() => compile('', 'double'), /Unknown precision/);
console.log('WGSL precision templates: nesting, constants, types, mixed precision, and diagnostics passed');
