struct Params {
    ns: u32,
    mpol: u32,
    ntheta: u32,
    nzeta: u32,
    n_z_n_t: u32,
    points: u32,
    reset_reference: u32,
    refresh_preconditioner: u32,
    delta_s: f32,
    tcon_multiplier: f32,
    _padding0: u32,
    _padding1: u32,
};
struct Values { data: array<f32>, };
@group(0) @binding(0) var<storage, read> geometry: Values;
// rCon, zCon, rCon0, zCon0; each has params.points values.
@group(0) @binding(1) var<storage, read> constraint: Values;
// sqrtS_F[ns], tcon[ns], ard[2*ns], azd[2*ns].
@group(0) @binding(2) var<storage, read> radial: Values;
// gConEff[points], rCon0[points], zCon0[points], tcon[ns].
@group(0) @binding(3) var<storage, read_write> output: Values;
@group(0) @binding(4) var<uniform> params: Params;

fn geom(field: u32, point: u32) -> f32 {
    return geometry.data[field * params.points + point];
}
fn con(field: u32, point: u32) -> f32 {
    return constraint.data[field * params.points + point];
}
fn reference(field: u32, surface: u32, angular: u32) -> f32 {
    let point = surface * params.n_z_n_t + angular;
    if (params.reset_reference != 0u && surface != 0u) {
        let lcfs = (params.ns - 1u) * params.n_z_n_t + angular;
        let sqrt_s = radial.data[surface];
        return con(field, lcfs) * sqrt_s * sqrt_s;
    }
    return con(field + 2u, point);
}
fn tcon_base(surface: u32) -> f32 {
    let ntheta_red = params.ntheta / 2u + 1u;
    let norm = 1.0 / f32(params.nzeta * (ntheta_red - 1u));
    let sqrt_s = radial.data[surface];
    var ar_n = 0.0;
    var az_n = 0.0;
    for (var zeta = 0u; zeta < params.nzeta; zeta++) {
        for (var theta = 0u; theta < ntheta_red; theta++) {
            var weight = norm;
            if (theta == 0u || theta + 1u == ntheta_red) { weight *= 0.5; }
            let point = surface * params.n_z_n_t + zeta * params.ntheta + theta;
            let ru = geom(3u, point) + sqrt_s * geom(9u, point);
            let zu = geom(4u, point) + sqrt_s * geom(10u, point);
            ar_n += ru * ru * weight;
            az_n += zu * zu * weight;
        }
    }
    if (ar_n == 0.0) { ar_n = 1.0e-10; }
    if (az_n == 0.0) { az_n = 1.0e-10; }
    let ard_offset = 2u * params.ns;
    let azd_offset = 4u * params.ns;
    let ard_even = abs(radial.data[ard_offset + 2u * surface]);
    let azd_even = abs(radial.data[azd_offset + 2u * surface]);
    let base = min(ard_even / ar_n, azd_even / az_n);
    return base * params.tcon_multiplier * 32.0 * params.delta_s *
           32.0 * params.delta_s;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let point = invocation.x;
    if (point >= params.points) { return; }
    let surface = point / params.n_z_n_t;
    let angular = point % params.n_z_n_t;
    let r0 = reference(0u, surface, angular);
    let z0 = reference(1u, surface, angular);
    output.data[params.points + point] = r0;
    output.data[2u * params.points + point] = z0;
    if (surface == 0u) {
        output.data[point] = 0.0;
    } else {
        let sqrt_s = radial.data[surface];
        let ru = geom(3u, point) + sqrt_s * geom(9u, point);
        let zu = geom(4u, point) + sqrt_s * geom(10u, point);
        output.data[point] =
            (con(0u, point) - r0) * ru + (con(1u, point) - z0) * zu;
    }
    if (angular == 0u) {
        let tcon_offset = 3u * params.points;
        if (surface == 0u) {
            output.data[tcon_offset] = 0.0;
        } else if (params.refresh_preconditioner == 0u) {
            output.data[tcon_offset + surface] = radial.data[params.ns + surface];
        } else if (surface + 1u == params.ns) {
            output.data[tcon_offset + surface] = 0.5 * tcon_base(surface - 1u);
        } else {
            output.data[tcon_offset + surface] = tcon_base(surface);
        }
    }
}
