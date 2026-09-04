struct Params {
    ns: u32,
    ntheta: u32,
    full_points: u32,
    half_points: u32,
    delta_s: f32,
    lamscale: f32,
    _padding0: u32,
    _padding1: u32,
};

struct Values { data: array<f32>, };

@group(0) @binding(0) var<storage, read> geometry: Values;
@group(0) @binding(1) var<storage, read> base: Values;
@group(0) @binding(2) var<storage, read> magnetic: Values;
// sqrt_s_f[ns], sqrt_s_h[ns-1], phip_f[ns].
@group(0) @binding(3) var<storage, read> radial: Values;
@group(0) @binding(4) var<storage, read_write> force: Values;
@group(0) @binding(5) var<uniform> params: Params;

fn full(field_index: u32, point: u32) -> f32 {
    return geometry.data[field_index * params.full_points + point];
}
fn half(field_index: u32, point: u32) -> f32 {
    return base.data[field_index * params.half_points + point];
}
fn bfield(field_index: u32, point: u32) -> f32 {
    return magnetic.data[field_index * params.half_points + point];
}
fn store(field_index: u32, point: u32, value: f32) {
    force.data[field_index * params.full_points + point] = value;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let point = invocation.x;
    if (point >= params.full_points) { return; }
    let surface = point / params.ntheta;
    let theta = point % params.ntheta;
    let half_count = params.ns - 1u;
    let sqrt_f = radial.data[surface];
    let s_full = sqrt_f * sqrt_f;

    var r12_i = 0.0; var ru12_i = 0.0; var zu12_i = 0.0;
    var rs_i = 0.0; var zs_i = 0.0; var tau_i = 0.0;
    var gsqrt_i = 0.0; var gvv_i = 0.0;
    var bsupu_i = 0.0; var bsupv_i = 0.0; var bsubv_i = 0.0;
    var total_p_i = 0.0; var sqrt_h_i = 0.0;
    var r12_o = 0.0; var ru12_o = 0.0; var zu12_o = 0.0;
    var rs_o = 0.0; var zs_o = 0.0; var tau_o = 0.0;
    var gsqrt_o = 0.0; var gvv_o = 0.0;
    var bsupu_o = 0.0; var bsupv_o = 0.0; var bsubv_o = 0.0;
    var total_p_o = 0.0; var sqrt_h_o = 0.0;

    if (surface > 0u) {
        let h = (surface - 1u) * params.ntheta + theta;
        r12_i = half(0u, h); ru12_i = half(1u, h); zu12_i = half(2u, h);
        rs_i = half(3u, h); zs_i = half(4u, h); tau_i = half(5u, h);
        gsqrt_i = half(6u, h); gvv_i = half(9u, h);
        bsupu_i = bfield(0u, h); bsupv_i = bfield(1u, h);
        bsubv_i = bfield(3u, h); total_p_i = bfield(4u, h);
        sqrt_h_i = radial.data[params.ns + surface - 1u];
    }
    if (surface + 1u < params.ns) {
        let h = surface * params.ntheta + theta;
        r12_o = half(0u, h); ru12_o = half(1u, h); zu12_o = half(2u, h);
        rs_o = half(3u, h); zs_o = half(4u, h); tau_o = half(5u, h);
        gsqrt_o = half(6u, h); gvv_o = half(9u, h);
        bsupu_o = bfield(0u, h); bsupv_o = bfield(1u, h);
        bsubv_o = bfield(3u, h); total_p_o = bfield(4u, h);
        sqrt_h_o = radial.data[params.ns + surface];
    }

    let p_i = r12_i * total_p_i; let p_o = r12_o * total_p_o;
    let zup_i = zu12_i * p_i; let zup_o = zu12_o * p_o;
    let rup_i = ru12_i * p_i; let rup_o = ru12_o * p_o;
    let rsp_i = rs_i * p_i; let rsp_o = rs_o * p_o;
    let zsp_i = zs_i * p_i; let zsp_o = zs_o * p_o;
    let taup_i = tau_i * total_p_i; let taup_o = tau_o * total_p_o;
    let gbubu_i = gsqrt_i * bsupu_i * bsupu_i;
    let gbubu_o = gsqrt_o * bsupu_o * bsupu_o;
    let gbvbv_i = gsqrt_i * bsupv_i * bsupv_i;
    let gbvbv_o = gsqrt_o * bsupv_o * bsupv_o;
    let inv_ds = 1.0 / params.delta_s;
    let inv_s_i = select(0.0, 1.0 / sqrt_h_i, surface > 0u);
    let inv_s_o = select(0.0, 1.0 / sqrt_h_o, surface + 1u < params.ns);
    let p_avg = 0.5 * (p_o + p_i);
    let p_wavg = 0.5 * (p_o * inv_s_o + p_i * inv_s_i);
    let gbubu_avg = 0.5 * (gbubu_o + gbubu_i);
    let gbubu_wavg = 0.5 * (gbubu_o * sqrt_h_o + gbubu_i * sqrt_h_i);
    let gbvbv_avg = 0.5 * (gbvbv_o + gbvbv_i);
    let gbvbv_wavg = 0.5 * (gbvbv_o * sqrt_h_o + gbvbv_i * sqrt_h_i);

    let r_e = full(0u, point); let r_o = full(6u, point);
    let z_o = full(7u, point);
    let ru_e = full(3u, point); let ru_o = full(9u, point);
    let zu_e = full(4u, point); let zu_o = full(10u, point);
    let armn_e = (zup_o - zup_i) * inv_ds + 0.5 * (taup_o + taup_i) -
                 gbvbv_avg * r_e - gbvbv_wavg * r_o;
    let armn_o = (zup_o * sqrt_h_o - zup_i * sqrt_h_i) * inv_ds -
                 0.5 * p_wavg * zu_e - 0.5 * p_avg * zu_o +
                 0.5 * (taup_o * sqrt_h_o + taup_i * sqrt_h_i) -
                 gbvbv_wavg * r_e - gbvbv_avg * r_o * s_full;
    let azmn_e = -(rup_o - rup_i) * inv_ds;
    let azmn_o = -(rup_o * sqrt_h_o - rup_i * sqrt_h_i) * inv_ds +
                 0.5 * p_wavg * ru_e + 0.5 * p_avg * ru_o;
    let brmn_e = 0.5 * (zsp_o + zsp_i) + 0.5 * p_wavg * z_o -
                 gbubu_avg * ru_e - gbubu_wavg * ru_o;
    let brmn_o = 0.5 * (zsp_o * sqrt_h_o + zsp_i * sqrt_h_i) +
                 0.5 * p_avg * z_o - gbubu_wavg * ru_e -
                 gbubu_avg * ru_o * s_full;
    let bzmn_e = -0.5 * (rsp_o + rsp_i) - 0.5 * p_wavg * r_o -
                 gbubu_avg * zu_e - gbubu_wavg * zu_o;
    let bzmn_o = -0.5 * (rsp_o * sqrt_h_o + rsp_i * sqrt_h_i) -
                 0.5 * p_avg * r_o - gbubu_wavg * zu_e -
                 gbubu_avg * zu_o * s_full;

    let gvv_gsqrt_i = select(0.0, gvv_i / gsqrt_i, surface > 0u);
    let gvv_gsqrt_o = select(0.0, gvv_o / gsqrt_o, surface + 1u < params.ns);
    let phip = radial.data[params.ns + half_count + surface];
    let lu_e_norm = params.lamscale * full(5u, point) + phip;
    let lu_o_norm = params.lamscale * full(11u, point);
    let bsubv_alt = 0.5 * (gvv_gsqrt_i + gvv_gsqrt_o) * lu_e_norm +
        0.5 * (gvv_gsqrt_i * sqrt_h_i + gvv_gsqrt_o * sqrt_h_o) * lu_o_norm;
    let blending = 0.1 * (1.0 - s_full);
    var lambda_force = 0.5 * (bsubv_o + bsubv_i) * (1.0 - blending) +
                       bsubv_alt * blending;
    if (surface > 0u) { lambda_force *= -params.lamscale; }

    store(0u, point, armn_e); store(1u, point, armn_o);
    store(2u, point, azmn_e); store(3u, point, azmn_o);
    store(4u, point, brmn_e); store(5u, point, brmn_o);
    store(6u, point, bzmn_e); store(7u, point, bzmn_o);
    store(8u, point, lambda_force);
    store(9u, point, lambda_force * sqrt_f);
}
