struct Params {
    ns:u32,mpol:u32,ntor:u32,ntheta:u32,nzeta:u32,nfp:u32,n_z_n_t:u32,include_lcfs:u32,
    norm_hi:f32,norm_lo:f32,sqrt2_hi:f32,sqrt2_lo:f32,
};
@group(0) @binding(0) var<storage,read> hi:array<f32>;
@group(0) @binding(1) var<storage,read> lo:array<f32>;
@group(0) @binding(2) var<storage,read_write> complex_values:array<vec4f>;
@group(0) @binding(3) var<uniform> params:Params;
@group(0) @binding(4) var<storage,read_write> projected:array<f32>;

@compute @workgroup_size(128)
fn pack(@builtin(global_invocation_id) id:vec3u){
    let tr=params.ntheta/2u+1u;let count=20u*params.ns*tr*params.nzeta;
    let i=id.x;if(i>=count){return;}
    let zeta=i%params.nzeta;let q=i/params.nzeta;let theta=q%tr;let sf=q/tr;
    let source=sf*params.n_z_n_t+zeta*params.ntheta+theta;
    complex_values[i]=vec4f(hi[source],0.0,lo[source],0.0);
}

@compute @workgroup_size(128)
fn unpack(@builtin(global_invocation_id) id:vec3u){
    let tr=params.ntheta/2u+1u;let plane=20u*params.ns*tr*(params.ntor+1u);
    let i=id.x;if(i>=plane){return;}
    let n=i%(params.ntor+1u);let q=i/(params.ntor+1u);
    let v=complex_values[q*params.nzeta+n];
    projected[i]=v.x;projected[plane+i]=-v.y;
    projected[2u*plane+i]=v.z;projected[3u*plane+i]=-v.w;
}
