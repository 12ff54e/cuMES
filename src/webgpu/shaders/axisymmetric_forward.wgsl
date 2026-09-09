// Same parameter, field layout, and binding contract as the 3-D projector.
struct Params {
    ns: u32,
    mpol: u32,
    ntor: u32,
    ntheta: u32,
    nzeta: u32,
    nfp: u32,
    n_z_n_t: u32,
    include_lcfs: u32,
    norm: f32,
    norm_lo: f32,
    sqrt2_hi: f32,
    sqrt2_lo: f32,
};

struct Values {
    data: array<f32>,
};

// Input field order: armn e/o, azmn e/o, brmn e/o, bzmn e/o, blmn e/o,
// crmn e/o, czmn e/o, clmn e/o, frcon e/o, fzcon e/o.
@group(0) @binding(0) var<storage, read> fields: Values;
// cache_basis reads the shared unweighted tables; poloidal_stage reads the
// compact weighted cache produced at setup.
@group(0) @binding(1) var<storage, read> basis: Values;
@group(0) @binding(2) var<storage, read_write> residual: Values;
@group(0) @binding(3) var<uniform> params: Params;

fn field_value(field: u32, surface: u32, theta: u32) -> f32 {
    return fields.data[field * params.ns * params.ntheta + surface * params.ntheta + theta];
}

fn basis_value(kind: u32, mode: u32, theta: u32) -> f32 {
    let reduced_theta = params.ntheta / 2u + 1u;
    return basis.data[(kind * params.mpol + mode) * reduced_theta + theta];
}

@compute @workgroup_size(128)
fn cache_basis(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    let reduced_theta = params.ntheta / 2u + 1u;
    if (index >= 4u * params.mpol * reduced_theta) { return; }
    let theta = index % reduced_theta;
    let kind_mode = index / reduced_theta;
    let kind = kind_mode / params.mpol;
    // The two n=0 zeta entries separate the cosine/sine and derivative tables.
    let offset = select(0u, 2u, kind >= 2u);
    let source = kind_mode * params.ntheta + theta + offset;
    let weight = select(params.norm, params.norm * 0.5,
                        theta == 0u || theta == reduced_theta - 1u);
    residual.data[index] = weight * basis.data[source];
}

fn store(component: u32, mode: u32, surface: u32, value: f32) {
    residual.data[(component * params.mpol + mode) * params.ns + surface] = value;
}

@compute @workgroup_size(128)
fn poloidal_stage(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let projection = invocation.x;
    let projection_count = params.ns * params.mpol;
    if (projection >= projection_count) {
        return;
    }

    let surface = projection / params.mpol;
    let mode = projection % params.mpol;
    let odd = (mode & 1u) == 1u;
    let parity = select(0u, 1u, odd);
    let xmpq = f32(mode) * f32(i32(mode) - 1);
    let reduced_theta = params.ntheta / 2u + 1u;

    var r_value = 0.0;
    var z_value = 0.0;
    var l_value = 0.0;
    for (var theta = 0u; theta < reduced_theta; theta++) {
        let cosine = basis_value(0u, mode, theta);
        let sine = basis_value(1u, mode, theta);
        let mcosine = basis_value(2u, mode, theta);
        let msine = basis_value(3u, mode, theta);
        let force_r = field_value(parity, surface, theta) +
                      xmpq * field_value(16u + parity, surface, theta);
        let force_z = field_value(2u + parity, surface, theta) +
                      xmpq * field_value(18u + parity, surface, theta);
        r_value += force_r * cosine +
                   field_value(4u + parity, surface, theta) * msine;
        z_value += force_z * sine +
                   field_value(6u + parity, surface, theta) * mcosine;
        l_value += field_value(8u + parity, surface, theta) * mcosine;
    }

    let mode_scale = select(sqrt(2.0), 1.0, mode == 0u);
    var r_output = mode_scale * r_value;
    var z_output = mode_scale * z_value;
    var l_output = mode_scale * l_value;
    if (surface == 0u) {
        r_output = select(0.0, r_output, mode == 0u);
        z_output = 0.0;
        l_output = 0.0;
    } else if (surface == params.ns - 1u && params.include_lcfs == 0u) {
        r_output = 0.0;
        z_output = 0.0;
    }

    store(0u, mode, surface, r_output);
    store(1u, mode, surface, z_output);
    store(2u, mode, surface, l_output);
    store(3u, mode, surface, 0.0);
    store(4u, mode, surface, 0.0);
    store(5u, mode, surface, 0.0);
}
