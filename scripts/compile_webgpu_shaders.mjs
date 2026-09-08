import {readFile, readdir, mkdir, writeFile} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {preprocess} from '../webgpu/shader_template.mjs';

const source_dir = fileURLToPath(new URL('../src/webgpu/shaders/', import.meta.url));
const manifest = JSON.parse(await readFile(path.join(source_dir, 'templates.json'), 'utf8'));
const outputs = new Map((await readdir(source_dir)).filter(name => name.endsWith('.wgsl')).sort()
  .map(name => [name, {file: name}]));
for (const {template, single, paired} of manifest) {
  for (const [precision, name] of Object.entries({single, paired})) {
    if (!name) continue;
    if (outputs.has(name)) throw new Error(`Duplicate shader output: ${name}`);
    outputs.set(name, {file: template, precision});
  }
}
if (process.argv[2] === '--list') {
  process.stdout.write([...outputs.keys()].join(';'));
} else {
  const destination = process.argv[2];
  if (!destination) throw new Error('Usage: node compile_webgpu_shaders.mjs OUTPUT_DIR | --list');
  const library = await readFile(path.join(source_dir, 'templates/compensated.wgsl'), 'utf8');
  await mkdir(destination, {recursive: true});
  // Resolve includes once before the synchronous substitution pass.
  const includes = new Map();
  async function read_source(file) {
    if (includes.has(file)) return;
    const source = await readFile(file, 'utf8');
    includes.set(file, source);
    for (const match of source.matchAll(/^\s*#include\s+"([^"]+)"/gm)) {
      await read_source(path.resolve(path.dirname(file), match[1]));
    }
  }
  for (const [name, {file, precision}] of outputs) {
    const filename = path.join(source_dir, file);
    await read_source(filename);
    let source = includes.get(filename);
    if (precision) {
      const result = preprocess(source, {precision, filename, include(name, from) {
        const filename = path.resolve(path.dirname(from), name);
        return {filename, source: includes.get(filename)};
      }});
      const slots = Math.max(...[...result.source.matchAll(/@workgroup_size\((\d+)\)/g)].map(m => Number(m[1])));
      if (result.needs_pair && !Number.isFinite(slots)) throw new Error(`${file}: Missing workgroup size`);
      source = (result.needs_pair ? library.replaceAll('ROUNDING_SLOTS', String(slots)) + '\n' : '') + result.source;
    }
    await writeFile(path.join(destination, name), source);
  }
}
