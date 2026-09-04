struct Params {
    ns: u32,
    n_z_n_t: u32,
    full_points: u32,
    half_points: u32,
    lamscale: f32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
};

struct Values {
    data: array<f32>,
};

@group(0) @binding(0) var<storage, read> geometry: Values;
@group(0) @binding(1) var<storage, read> base: Values;
// sqrt_s_h[ns-1], phip_f[ns], chip_h[ns-1], pres_h[ns-1].
@group(0) @binding(2) var<storage, read> profiles: Values;
@group(0) @binding(3) var<storage, read_write> field: Values;
@group(0) @binding(4) var<uniform> params: Params;

fn full(field_index: u32, point: u32) -> f32 {
    return geometry.data[field_index * params.full_points + point];
}

fn half(field_index: u32, point: u32) -> f32 {
    return base.data[field_index * params.half_points + point];
}

fn store(field_index: u32, point: u32, value: f32) {
    field.data[field_index * params.half_points + point] = value;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let point = invocation.x;
    if (point >= params.half_points) {
        return;
    }
    let surface = point / params.n_z_n_t;
    let angular = point % params.n_z_n_t;
    let inside = surface * params.n_z_n_t + angular;
    let outside = inside + params.n_z_n_t;
    let half_count = params.ns - 1u;
    let sqrt_h = profiles.data[surface];
    let lu_h = 0.5 * ((full(5u, inside) + full(5u, outside)) +
                      sqrt_h * (full(11u, inside) + full(11u, outside)));
    let lv_h = 0.5 * ((full(14u, inside) + full(14u, outside)) +
                      sqrt_h * (full(17u, inside) + full(17u, outside)));
    let phip_offset = half_count;
    let chip_offset = phip_offset + params.ns;
    let pres_offset = chip_offset + half_count;
    let phip_average =
        0.5 * (profiles.data[phip_offset + surface] +
               profiles.data[phip_offset + surface + 1u]);
    let gsqrt = half(6u, point);
    var bsupu = 0.0;
    var bsupv = 0.0;
    if (abs(gsqrt) > 1.0e-30 && abs(gsqrt) <= 3.402823e38) {
        bsupv = (params.lamscale * lu_h + phip_average) / gsqrt;
        bsupu = (params.lamscale * lv_h +
                 profiles.data[chip_offset + surface]) /
                gsqrt;
    }
    let bsubu = half(7u, point) * bsupu + half(8u, point) * bsupv;
    let bsubv = half(8u, point) * bsupu + half(9u, point) * bsupv;
    let magnetic_pressure = 0.5 * (bsupu * bsubu + bsupv * bsubv);
    store(0u, point, bsupu);
    store(1u, point, bsupv);
    store(2u, point, bsubu);
    store(3u, point, bsubv);
    store(4u, point, magnetic_pressure + profiles.data[pres_offset + surface]);
}
