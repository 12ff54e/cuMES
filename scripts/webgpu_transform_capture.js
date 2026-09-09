// Inject before the app through webgpu_route_experiment.mjs. Diagnostic only:
// copies selected resident inputs/intermediates immediately after projection,
// in the existing command encoder, before any later shadow can overwrite them.
// Uses the existing compare_fft=1 route: primary, generic FFT, direct,
// canonical direct, first unconstrained then constrained projection.
(() => {
  const query=new URLSearchParams(location.search);
  const iterations=(query.get('audit_iterations')||'1,2800').split(',').map(Number);
  const shadows=query.get('compare_fft')==='1';
  const surfaces=[1,50,97],ns=99,mpol=12,ntor=12,ntheta=30,nzeta=36;
  const tr=ntheta/2+1,modes=mpol*(ntor+1),angular=ntheta*nzeta;
  const captures=window.cumesTransformCaptures=[];
  const bindings=new WeakMap(),passes=new WeakMap(),ordinals=new Map();
  let inputs;
  const createBuffer=GPUDevice.prototype.createBuffer;
  GPUDevice.prototype.createBuffer=function(desc){
    // Add read-only diagnostic copy access; do not change data or arithmetic.
    if(desc.usage&(GPUBufferUsage.STORAGE|GPUBufferUsage.UNIFORM))
      desc={...desc,usage:desc.usage|GPUBufferUsage.COPY_SRC};
    return createBuffer.call(this,desc);
  };
  const createGroup=GPUDevice.prototype.createBindGroup;
  GPUDevice.prototype.createBindGroup=function(desc){
    const group=createGroup.call(this,desc),entries=new Map(desc.entries.map(e=>[e.binding,e.resource]));
    if(desc.label==='cuMES toroidal forward first bindings')inputs=entries;
    if(desc.label==='cuMES toroidal forward second bindings')
      bindings.set(group,{device:this,entries,inputs});
    return group;
  };
  const begin=GPUCommandEncoder.prototype.beginComputePass;
  GPUCommandEncoder.prototype.beginComputePass=function(...args){
    const pass=begin.apply(this,args);passes.set(pass,{encoder:this});return pass;
  };
  const bind=GPUComputePassEncoder.prototype.setBindGroup;
  GPUComputePassEncoder.prototype.setBindGroup=function(index,group,...args){
    if(index===0&&passes.has(this))passes.get(this).info=bindings.get(group);
    return bind.call(this,index,group,...args);
  };
  const end=GPUComputePassEncoder.prototype.end;
  GPUComputePassEncoder.prototype.end=function(){
    end.call(this);
    const pass=passes.get(this),info=pass?.info;if(!info)return;
    const last=window.cumesDiagnostics?.findLast(row=>row.kind==='controller');
    const iteration=(last?.iter||0)+1;if(!iterations.includes(iteration))return;
    if(info.inputs.get(0).size!==20*ns*angular*4)return;
    const ordinal=ordinals.get(iteration)||0;ordinals.set(iteration,ordinal+1);
    const variants=shadows&&(iteration<=3||iteration%100===0)?4:1;
    const capture={iteration,ordinal,phase:ordinal<variants?'force':'constraint',
      variant:['primary','generic-fft','direct','canonical-direct'][ordinal%variants],
      ns,mpol,ntor,ntheta,nzeta,surfaces,sections:[]};
    const segments=[];let size=0;
    const section=(name,resource,pieces)=>{
      const start=size;
      for(const [offset,bytes] of pieces){segments.push({resource,offset,bytes,destination:size});size+=bytes;}
      capture.sections.push({name,offset:start,bytes:size-start});
    };
    const entire=(name,r)=>section(name,r,[[0,r.size||r.buffer.size]]);
    if(ordinal%variants===0){
      for(const [name,binding] of [['fields_hi',0],['fields_lo',5]]){
        const pieces=[];
        for(let field=0;field<20;field++)for(const surface of surfaces)
          pieces.push([(field*ns+surface)*angular*4,angular*4]);
        section(name,info.inputs.get(binding),pieces);
      }
      entire('basis_hi',info.entries.get(1));entire('basis_lo',info.entries.get(6));
      entire('params',info.entries.get(3));
    }
    entire('residual',info.entries.get(2));
    const projected=[];
    for(let word=0;word<2;word++)for(let family=0;family<40;family++)for(const surface of surfaces)
      projected.push([((word*40+family)*ns+surface)*tr*(ntor+1)*4,tr*(ntor+1)*4]);
    section('projected',info.entries.get(4),projected);
    const read=info.device.createBuffer({label:'transform audit readback',size,
      usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    for(const s of segments)pass.encoder.copyBufferToBuffer(s.resource.buffer,
      (s.resource.offset||0)+s.offset,read,s.destination,s.bytes);
    capture.ready=Promise.resolve().then(async()=>{
      await read.mapAsync(GPUMapMode.READ);capture.bytes=new Uint8Array(read.getMappedRange()).slice();
      read.unmap();read.destroy();
    });
    captures.push(capture);
  };
})();
