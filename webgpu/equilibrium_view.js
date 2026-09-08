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
  phiSegments=ntor?Math.max(64,nfp*16):64,surfaces=[];
  for(const source of fourier.surfaces){const points=new Float32Array((phiSegments+1)*(thetaSegments+1)*3),c=source.coefficients;
      for(let p=0;p<=phiSegments;p++){const phi=2*Math.PI*p/phiSegments,cp=Math.cos(phi),sp=Math.sin(phi);
        for(let t=0;t<=thetaSegments;t++){const [r,z]=fourierPoint(fourier,c,2*Math.PI*t/thetaSegments,phi);
        const i=(p*(thetaSegments+1)+t)*3;points[i]=r*cp;points[i+1]=r*sp;points[i+2]=z}}
    surfaces.push({radial:source.index/(ns-1),points})}return{surfaces,thetaSegments,phiSegments}}
function installOrbitRenderer(canvas,fourier){if(!canvas||!fourier?.surfaces?.length)return;const mesh=equilibriumMesh(fourier),
  state={mesh,yaw:-.55,pitch:.55,zoom:1,drag:null,frame:0};canvas.cumesOrbit=state;
  let radius=1;for(const surface of mesh.surfaces)for(let i=0;i<surface.points.length;i+=3)radius=Math.max(radius,
    Math.hypot(surface.points[i],surface.points[i+1],surface.points[i+2]));state.radius=radius;
  if(!canvas.dataset.orbitReady){canvas.dataset.orbitReady='1';canvas.addEventListener('pointerdown',event=>{const s=canvas.cumesOrbit;
      s.drag={id:event.pointerId,x:event.clientX,y:event.clientY,yaw:s.yaw,pitch:s.pitch};canvas.setPointerCapture(event.pointerId)});
    canvas.addEventListener('pointermove',event=>{const s=canvas.cumesOrbit;if(!s?.drag||s.drag.id!==event.pointerId)return;
      s.yaw=s.drag.yaw+(event.clientX-s.drag.x)*.008;s.pitch=Math.max(-1.35,Math.min(1.35,s.drag.pitch+(event.clientY-s.drag.y)*.008));queueOrbitDraw(canvas)});
    const release=event=>{const s=canvas.cumesOrbit;if(s?.drag?.id===event.pointerId)s.drag=null};canvas.addEventListener('pointerup',release);
    canvas.addEventListener('pointercancel',release);canvas.addEventListener('wheel',event=>{const s=canvas.cumesOrbit;if(!s)return;
      s.zoom=Math.max(.45,Math.min(3,s.zoom*Math.exp(-event.deltaY*.001)));queueOrbitDraw(canvas);event.preventDefault()},{passive:false});
    new ResizeObserver(()=>queueOrbitDraw(canvas)).observe(canvas)}queueOrbitDraw(canvas)}
function queueOrbitDraw(canvas){const state=canvas?.cumesOrbit;if(!state||state.frame||canvas.hidden)return;
  state.frame=requestAnimationFrame(()=>{state.frame=0;drawOrbit(canvas,state)})}
function drawOrbit(canvas,state){const rect=canvas.getBoundingClientRect();if(rect.width<2||rect.height<2)return;
  const dpr=Math.min(2,devicePixelRatio||1),width=rect.width,height=rect.height;if(canvas.width!==Math.round(width*dpr)||canvas.height!==Math.round(height*dpr)){
    canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr)}const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);
  const background=ctx.createRadialGradient(width*.52,height*.44,0,width*.52,height*.44,Math.max(width,height)*.68);
  background.addColorStop(0,'#13263a');background.addColorStop(1,'#07101a');ctx.fillStyle=background;ctx.fillRect(0,0,width,height);
  const cy=Math.cos(state.yaw),sy=Math.sin(state.yaw),cp=Math.cos(state.pitch),sp=Math.sin(state.pitch),camera=4.5*state.radius,
    base=Math.min(width,height)*.43/state.radius*state.zoom,project=(points,index)=>{const x=points[index],y=points[index+1],z=points[index+2],
      rx=cy*x-sy*y,ry=sy*x+cy*y,depth=cp*ry-sp*z,vertical=sp*ry+cp*z,k=camera/(camera+depth);
      return[width/2+rx*base*k,height/2-vertical*base*k,depth]};
  const curves=[],surfaceStep=Math.max(1,Math.floor(state.mesh.surfaces.length/10)),selected=[];
  for(let s=0;s<state.mesh.surfaces.length;s+=surfaceStep)selected.push(s);if(selected.at(-1)!==state.mesh.surfaces.length-1)selected.push(state.mesh.surfaces.length-1);
  for(const s of selected){const surface=state.mesh.surfaces[s],row=state.mesh.thetaSegments+1,add=indices=>{const path=[];let depth=0;
      for(const index of indices){const point=project(surface.points,index*3);path.push(point);depth+=point[2]}curves.push({path,depth:depth/path.length,radial:surface.radial,outer:s===state.mesh.surfaces.length-1})};
    for(let p=0;p<state.mesh.phiSegments;p+=4)add(Array.from({length:row},(_,t)=>p*row+t));
    for(let t=0;t<state.mesh.thetaSegments;t+=4)add(Array.from({length:state.mesh.phiSegments+1},(_,p)=>p*row+t))}
  curves.sort((a,b)=>b.depth-a.depth);ctx.lineJoin='round';for(const curve of curves){ctx.beginPath();curve.path.forEach((point,i)=>ctx[i?'lineTo':'moveTo'](point[0],point[1]));
    const light=55+curve.radial*18,alpha=curve.outer?.78:.12+.35*curve.radial;ctx.strokeStyle=`hsla(${186+35*curve.radial} 82% ${light}% / ${alpha})`;
    ctx.lineWidth=curve.outer?1.35:.6+.45*curve.radial;ctx.stroke()}
  ctx.fillStyle='#9aa9bb';ctx.font='12px system-ui,sans-serif';ctx.fillText('Drag to orbit · wheel to zoom',18,height-18)}
