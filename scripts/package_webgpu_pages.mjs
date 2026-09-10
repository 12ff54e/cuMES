import {copyFile, mkdir, readFile, readdir, stat, writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {join, relative, resolve, sep} from 'node:path';
import {fileURLToPath} from 'node:url';

const frontend = fileURLToPath(new URL('../webgpu/', import.meta.url));
const artifacts = [
  'cumes_webgpu.html', 'cumes_webgpu.js', 'cumes_webgpu.wasm',
  'cumes_coils.mjs', 'cumes_coils.wasm', 'browser_ui.js', 'verification_worker.js',
  'residual_plot.js', 'free_boundary.js', 'coil_geometry.js', 'equilibrium_view.js',
  'orbit_renderer.js', 'boundary_editor.js'
];
const presets = [
  'solovev.json', 'w7x.json', 'cth_like.json', 'fixed-w7x.json',
  'coils.solovev', 'coils.w7x', 'coils.cth_like', 'LICENSE.vmecpp', 'README.md'
];

export async function packageWebgpuPages(build, output, inputsFile) {
  build = resolve(build); output = resolve(output);
  const inputs = new Set([fileURLToPath(import.meta.url)]);
  const read = (path, ...options) => { inputs.add(resolve(path)); return readFile(path, ...options); };
  const copy = (source, destination) => { inputs.add(resolve(source)); return copyFile(source, destination); };
  const destination = relative(build, output), source = relative(output, build);
  const outside = path => path === '..' || path.startsWith('..' + sep);
  if (!outside(destination) || !outside(source))
    throw Error('Build and output directories must be separate.');
  for (const asset of [...artifacts, ...presets.map(name => join('presets', name))]) {
    const info = await stat(join(build, asset));
    if (!info.isFile() || !info.size)
      throw Error('Missing or empty WebGPU artifact: ' + asset);
  }
  let html = await read(join(build, 'cumes_webgpu.html'), 'utf8');
  const coils = await read(join(build, 'coil_geometry.js'), 'utf8');
  if (!html.includes('await globalThis.cumesIsolationReady') || !coils.includes('await globalThis.cumesIsolationReady'))
    throw Error('Rebuild WebGPU before packaging: isolation startup support is missing.');
  if (!html.includes('<head>') || !/cumes_webgpu\.js\?v=[0-9a-f]+/.test(html))
    throw Error('Expected generated WebGPU HTML with a versioned runtime.');
  const bootstrap = await read(join(frontend, 'pages_bootstrap.js'));
  const bootstrapVersion = createHash('sha256').update(bootstrap).digest('hex');
  html = html.replace('<head>', `<head>\n  <script src="pages_bootstrap.js?v=${bootstrapVersion}"></script>`);
  await mkdir(output, {recursive: true});
  if ((await readdir(output)).length) throw Error('Output directory must be empty: ' + output);
  for (const asset of artifacts.filter(name => !name.endsWith('.html')))
    await copy(join(build, asset), join(output, asset));
  await mkdir(join(output, 'presets'));
  for (const name of presets)
    await copy(join(build, 'presets', name), join(output, 'presets', name));
  await copy(join(frontend, 'pages_bootstrap.js'), join(output, 'pages_bootstrap.js'));
  await copy(join(frontend, 'vendor/coi-serviceworker/coi-serviceworker.js'), join(output, 'coi-serviceworker.js'));
  const licenseDir = join(output, 'licenses/coi-serviceworker');
  await mkdir(licenseDir, {recursive: true});
  for (const name of ['LICENSE', 'README.md'])
    await copy(join(frontend, 'vendor/coi-serviceworker', name), join(licenseDir, name));
  for (const [name, path] of [['cuMES', '../LICENSE'], ['webgpu-fft', '../deps/webgpu-fft/LICENSE'],
    ['vacuum-field', '../deps/vacuum-field/LICENSE']])
    await copy(join(frontend, path), join(output, 'licenses', name + '.txt'));
  await writeFile(join(output, 'index.html'), html);
  await writeFile(join(output, 'cumes_webgpu.html'), html);
  await writeFile(join(output, '.nojekyll'), '');
  if (inputsFile) await writeFile(inputsFile, JSON.stringify([...inputs].sort()) + '\n');
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv.length < 4 || process.argv.length > 5) {
    console.error('Usage: node scripts/package_webgpu_pages.mjs BUILD_WEBGPU_DIR OUTPUT_DIR [INPUTS_JSON]');
    process.exitCode = 1;
  } else {
    try {
      await packageWebgpuPages(process.argv[2], process.argv[3], process.argv[4]);
      console.log('Packaged WebGPU for GitHub Pages: ' + resolve(process.argv[3]));
    } catch (error) { console.error(error.message); process.exitCode = 1; }
  }
}
