struct Params {
    ns: u32,
    mpol: u32,
    ntheta: u32,
    points: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
    _padding3: u32,
};

struct Values {
    data: array<f32>,
};

@group(0) @binding(0) var<storage, read> state: Values;
// Host-generated Fourier tables match the CUDA operator's table contract and
// avoid implementation-dependent WGSL transcendental approximations.
@group(0) @binding(1) var<storage, read> basis: Values;
// 18 geometry parity fields followed by rCon and zCon.
@group(0) @binding(2) var<storage, read_write> result: Values;
@group(0) @binding(3) var<uniform> params: Params;

fn coefficient(component: u32, mode: u32, surface: u32) -> f32 {
    return state.data[(component * params.mpol + mode) * params.ns + surface];
}

fn store(field: u32, point: u32, value: f32) {
    result.data[field * params.points + point] = value;
}

fn basis_value(kind: u32, mode: u32, theta_index: u32) -> f32 {
    return basis.data[(kind * params.mpol + mode) * params.ntheta + theta_index];
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let point = invocation.x;
    if (point >= params.points) {
        return;
    }

    let surface = point / params.ntheta;
    let theta_index = point % params.ntheta;
    let maxsc = max(sqrt(f32(surface) / f32(params.ns - 1u)),
                    sqrt(1.0 / f32(params.ns - 1u)));
    let odd_scale = 1.0 / maxsc;

    var r_e = 0.0;
    var z_e = 0.0;
    var l_e = 0.0;
    var ru_e = 0.0;
    var zu_e = 0.0;
    var lu_e = 0.0;
    var r_o = 0.0;
    var z_o = 0.0;
    var l_o = 0.0;
    var ru_o = 0.0;
    var zu_o = 0.0;
    var lu_o = 0.0;
    var r_con = 0.0;
    var z_con = 0.0;

    for (var mode = 0u; mode < params.mpol; mode++) {
        let mf = f32(mode);
        let cosine = basis_value(0u, mode, theta_index);
        let sine = basis_value(1u, mode, theta_index);
        let rc = coefficient(0u, mode, surface);
        let zs = coefficient(1u, mode, surface);
        let ls = coefficient(2u, mode, surface);
        let odd = (mode & 1u) == 1u;
        let scale = select(1.0, odd_scale, odd);

        let r_value = scale * rc * cosine;
        let z_value = scale * zs * sine;
        let l_value = scale * ls * sine;
        let ru_value = scale * rc * (-mf * sine);
        let zu_value = scale * zs * (mf * cosine);
        let lu_value = scale * ls * (mf * cosine);
        if (odd) {
            r_o += r_value;
            z_o += z_value;
            l_o += l_value;
            ru_o += ru_value;
            zu_o += zu_value;
            lu_o += lu_value;
        } else {
            r_e += r_value;
            z_e += z_value;
            l_e += l_value;
            ru_e += ru_value;
            zu_e += zu_value;
            lu_e += lu_value;
        }

        let xmpq = mf * (mf - 1.0);
        r_con += xmpq * rc * cosine;
        z_con += xmpq * zs * sine;
    }

    store(0u, point, r_e);
    store(1u, point, z_e);
    store(2u, point, l_e);
    store(3u, point, ru_e);
    store(4u, point, zu_e);
    store(5u, point, lu_e);
    store(6u, point, r_o);
    store(7u, point, z_o);
    store(8u, point, l_o);
    store(9u, point, ru_o);
    store(10u, point, zu_o);
    store(11u, point, lu_o);
    for (var field = 12u; field < 18u; field++) {
        store(field, point, 0.0);
    }
    store(18u, point, r_con);
    store(19u, point, z_con);
}
