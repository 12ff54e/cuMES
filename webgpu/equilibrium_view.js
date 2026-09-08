// Fold the VMEC signed-n input into the same six families as solver output.
function boundaryFourier(input) {
  const {mpol, ntor = 0, nfp = 1} = input;
  if (!Number.isInteger(mpol) || mpol < 1 || !Number.isInteger(ntor) || ntor < 0 ||
      !Number.isInteger(nfp) || nfp < 1 || mpol * (ntor + 1) > 4096)
    throw Error('Boundary preview needs valid mpol, ntor and field periods (up to 4096 modes).');
  const modes = mpol * (ntor + 1), coefficients = new Float64Array(6 * modes);
  for (const family of ['rbc', 'zbs']) {
    if (!Array.isArray(input[family])) throw Error(`Boundary input needs a ${family} array.`);
    for (const {m, n, value} of input[family]) {
      if (!Number.isInteger(m) || m < 0 || !Number.isInteger(n) || !Number.isFinite(value))
        throw Error(`Invalid ${family} boundary coefficient.`);
      // The config validator likewise excludes harmonics outside the solver basis.
      if (m >= mpol || Math.abs(n) > ntor) continue;
      const mode = m * (ntor + 1) + Math.abs(n);
      if (family === 'rbc') {
        coefficients[mode] += value;
        if (m > 0) coefficients[3 * modes + mode] += Math.sign(n) * value;
      } else {
        if (m > 0) coefficients[modes + mode] += value;
        coefficients[4 * modes + mode] -= Math.sign(n) * value;
      }
    }
  }
  return {mpol, ntor, nfp, ns: 2, surfaces: [{index: 1, coefficients}]};
}

function fourierPoint(fourier, coefficients, theta, phi) {
  const {mpol, ntor, nfp} = fourier, modes = mpol * (ntor + 1);
  let r = 0, z = 0;
  for (let m = 0; m < mpol; m++) {
    const cm = Math.cos(m * theta), sm = Math.sin(m * theta);
    for (let n = 0; n <= ntor; n++) {
      const mode = m * (ntor + 1) + n, cn = Math.cos(n * nfp * phi), sn = Math.sin(n * nfp * phi);
      r += coefficients[mode] * cm * cn + coefficients[3 * modes + mode] * sm * sn;
      z += coefficients[modes + mode] * sm * cn + coefficients[4 * modes + mode] * cm * sn;
    }
  }
  return [r, z];
}

function fourierSections(fourier, phi, segments = 160) {
  return fourier.surfaces.map(surface => Array.from({length: segments + 1}, (_, i) =>
    fourierPoint(fourier, surface.coefficients, 2 * Math.PI * i / segments, phi)));
}

function equilibriumMesh(fourier){const{ntor,nfp,ns}=fourier,thetaSegments=40,
  phiSegments=ntor?Math.min(256,Math.max(64,nfp*16)):64,surfaces=[];
  for(const source of fourier.surfaces){const points=new Float32Array((phiSegments+1)*(thetaSegments+1)*3),c=source.coefficients;
      for(let p=0;p<=phiSegments;p++){const phi=2*Math.PI*p/phiSegments,cp=Math.cos(phi),sp=Math.sin(phi);
        for(let t=0;t<=thetaSegments;t++){const [r,z]=fourierPoint(fourier,c,2*Math.PI*t/thetaSegments,phi);
        const i=(p*(thetaSegments+1)+t)*3;points[i]=r*cp;points[i+1]=r*sp;points[i+2]=z}}
    surfaces.push({radial:source.index/(ns-1),points})}return{surfaces,thetaSegments,phiSegments}}
