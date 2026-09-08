// Build-time WGSL precision substitution. No eval, runtime branches, or shader
// wrappers: scalar arithmetic expands directly at the original expression.
const BINARY = {add: '+', sub: '-', mul: '*', div: '/', greater: '>'};
const ARITY = {
  add: 2, sub: 2, mul: 2, div: 2, scale: 2, greater: 2,
  neg: 1, reciprocal: 1, normalize: 1, round: 1, square: 1,
  real: 1, words: 2, literal: 1, hi: 1, lo: 1, scalar: 1,
};

// Keep comments opaque, including WGSL's nested block comments. The same lexer
// is used for argument balancing, so commas inside calls/types are not split.
function tokenize(source) {
  const tokens = [];
  for (let i = 0; i < source.length;) {
    const start = i;
    if (source.startsWith('//', i)) {
      i = source.indexOf('\n', i);
      if (i < 0) i = source.length;
    } else if (source.startsWith('/*', i)) {
      i += 2;
      let depth = 1;
      while (i < source.length && depth) {
        if (source.startsWith('/*', i)) { depth++; i += 2; }
        else if (source.startsWith('*/', i)) { depth--; i += 2; }
        else i++;
      }
      if (depth) throw new Error('Unterminated block comment');
    } else {
      const match = /^(?:\s+|[A-Za-z_][A-Za-z_0-9]*|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?[fhiu]?)/.exec(source.slice(i));
      i += match ? match[0].length : 1;
    }
    const text = source.slice(start, i);
    tokens.push({text, start, trivia: /^\s|^\/\//.test(text) || text.startsWith('/*')});
  }
  return tokens;
}

function float_literal(value) {
  if (!Number.isFinite(value)) throw new Error('Constant is outside the f32 range');
  if (Object.is(value, -0)) return '-0.0';
  const text = String(value);
  return /[.e]/i.test(text) ? text : `${text}.0`;
}

function expand(name, args, paired) {
  const count = ARITY[name];
  const has_slot = ['add', 'sub', 'mul', 'div', 'scale', 'reciprocal', 'normalize', 'round', 'square'].includes(name);
  if (args.length !== count && !(has_slot && args.length === count + 1)) {
    throw new Error(`${name} expects ${count}${has_slot ? ` or ${count + 1}` : ''} arguments`);
  }
  const [a, b] = args;
  const slot = args[count] ?? 'slot';
  if (name === 'literal') {
    if (!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(a)) {
      throw new Error('literal requires a decimal constant');
    }
    const value = Number(a);
    const high = Math.fround(value);
    const low = Math.fround(value - high);
    const words = `FF(${float_literal(high)}, ${float_literal(low)})`;
    return paired ? words : `(${a})`;
  }
  if (name === 'real') return paired ? `FF(${a}, 0.0)` : `(${a})`;
  if (name === 'words') return paired ? `FF(${a}, ${b})` : `(${a})`;
  if (name === 'hi') return paired ? `(${a}).hi` : `(${a})`;
  if (name === 'lo') return paired ? `(${a}).lo` : '0.0';
  if (name === 'scalar') return paired ? `compensate_scalar(${a})` : `(${a})`;
  if (!paired) {
    if (BINARY[name]) return `((${a}) ${BINARY[name]} (${b}))`;
    if (name === 'scale') return `((${a}) * (${b}))`;
    if (name === 'neg') return `(-(${a}))`;
    if (name === 'reciprocal') return `(1.0 / (${a}))`;
    if (name === 'square') return `((${a}) * (${a}))`;
    return `(${a})`; // normalize/round are identities for scalar arithmetic.
  }
  if (name === 'neg' || name === 'greater') {
    return `compensate_${name}(${args.join(', ')})`;
  }
  return `compensate_${name}(${args.slice(0, count).join(', ')}, ${slot})`;
}

export function preprocess(source, {precision, filename = '<shader>', include} = {}) {
  if (!['single', 'paired'].includes(precision)) throw new Error(`Unknown precision: ${precision}`);
  const paired = precision === 'paired';
  function directives(text, file, chain = []) {
    if (chain.includes(file)) throw new Error(`Include cycle: ${[...chain, file].join(' -> ')}`);
    // Consult tokens so a #if in a block comment is ordinary documentation.
    const comments = tokenize(text).filter(t => t.text.startsWith('/*'));
    const stack = [];
    let active = true;
    let offset = 0;
    const result = text.split(/(?<=\n)/).map((line, index) => {
      const start = offset; offset += line.length;
      const hash = start + line.search(/\S/);
      const match = /^\s*#(\w+)\s*(.*?)\s*$/.exec(line);
      if (!match || comments.some(t => hash >= t.start && hash < t.start + t.text.length)) return active ? line : '\n';
      const [, op, argument] = match;
      const fail = message => { throw new Error(`${file}:${index + 1}: ${message}`); };
      if (op === 'if') {
        if (!['PAIRED', '!PAIRED'].includes(argument)) fail('Expected #if PAIRED or #if !PAIRED');
        stack.push({parent: active, condition: argument === 'PAIRED' ? paired : !paired, alternate: false});
        active = active && stack.at(-1).condition;
      } else if (op === 'else') {
        if (argument || !stack.length || stack.at(-1).alternate) fail('Unexpected #else');
        stack.at(-1).alternate = true;
        active = stack.at(-1).parent && !stack.at(-1).condition;
      } else if (op === 'endif') {
        if (argument || !stack.length) fail('Unexpected #endif');
        active = stack.pop().parent;
      } else if (op === 'include') {
        if (!/^"[^"\n]+"$/.test(argument)) fail('Expected #include "file"');
        if (active) {
          if (!include) fail('No include resolver');
          const included = include(argument.slice(1, -1), file);
          return directives(included.source, included.filename, [...chain, file]);
        }
      } else fail(`Unknown directive #${op}`);
      return '\n';
    }).join('');
    if (stack.length) throw new Error(`${file}: Missing #endif`);
    return result;
  }
  const tokens = tokenize(directives(source, filename));
  let needs_pair = false;
  const next = i => { while (tokens[i]?.trivia) i++; return i; };
  const previous = [];
  let last = '';
  for (const token of tokens) {
    previous.push(last);
    if (!token.trivia) last = token.text;
  }
  function substitute(begin, end) {
    let result = '';
    for (let i = begin; i < end; i++) {
      const token = tokens[i];
      if (token.trivia) { result += token.text; continue; }
      if (token.text === 'Real' || token.text === 'Pair') {
        const is_pair = paired || token.text === 'Pair';
        needs_pair ||= is_pair;
        result += is_pair ? 'FF' : 'f32';
        continue;
      }
      const forced = token.text.startsWith('pair_');
      const name = forced ? token.text.slice(5) : token.text;
      if (!Object.hasOwn(ARITY, name)) { result += token.text; continue; }
      const open = next(i + 1);
      if (tokens[open]?.text !== '(') { result += token.text; continue; }
      if (previous[i] === 'fn') {
        throw new Error(`${filename}: ${token.text} is a reserved template intrinsic`);
      }
      let arg_start = open + 1;
      const args = [];
      const stack = [')'];
      let j = open + 1;
      for (; j < end; j++) {
        if (tokens[j].trivia) continue;
        const t = tokens[j].text;
        if (t === '(') stack.push(')');
        else if (t === '[') stack.push(']');
        else if (t === '{') stack.push('}');
        else if (t === '<' && /^(?:array|vec[234]|mat[234]x[234]|bitcast|ptr|atomic)$/.test(previous[j])) stack.push('>');
        else if (t === stack.at(-1)) {
          stack.pop();
          if (!stack.length) { args.push(substitute(arg_start, j).trim()); break; }
        } else if (t === ',' && stack.length === 1) {
          args.push(substitute(arg_start, j).trim()); arg_start = j + 1;
        }
      }
      if (stack.length) throw new Error(`${filename}: Unclosed ${token.text} call`);
      if (args.some(a => !a)) throw new Error(`${filename}: Empty ${token.text} argument`);
      const is_pair = paired || forced;
      needs_pair ||= is_pair;
      try { result += expand(name, args, is_pair); }
      catch (error) { throw new Error(`${filename}: ${error.message}`); }
      i = j;
    }
    return result;
  }
  return {source: substitute(0, tokens.length), needs_pair};
}
