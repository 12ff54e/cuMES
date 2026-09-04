struct Params {
    ns: u32,
    mpol: u32,
    ntor: u32,
    ntheta: u32,
    nzeta: u32,
    nfp: u32,
    n_z_n_t: u32,
    total_points: u32,
};

struct Values {
    data: array<f32>,
};

@group(0) @binding(0) var<storage, read> state: Values;
// Four mode-major basis tables: cc, ss, sc, cs.
@group(0) @binding(1) var<storage, read> basis: Values;
// 18 geometry parity fields followed by rCon and zCon.
@group(0) @binding(2) var<storage, read_write> output: Values;
@group(0) @binding(3) var<uniform> params: Params;

fn coefficient(component: u32, mode: u32, surface: u32) -> f32 {
    let mnmax = params.mpol * (params.ntor + 1u);
    return state.data[(component * mnmax + mode) * params.ns + surface];
}

fn basis_value(field: u32, mode: u32, angular: u32) -> f32 {
    let mnmax = params.mpol * (params.ntor + 1u);
    return basis.data[(field * mnmax + mode) * params.n_z_n_t + angular];
}

fn store(field: u32, point: u32, value: f32) {
    output.data[field * params.total_points + point] = value;
}

fn compensated_add(sum: ptr<function, f32>, correction: ptr<function, f32>,
                   term: f32) {
    let adjusted = term - *correction;
    let next = *sum + adjusted;
    *correction = (next - *sum) - adjusted;
    *sum = next;
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let point = invocation.x;
    if (point >= params.total_points) {
        return;
    }
    let surface = point / params.n_z_n_t;
    let angular = point % params.n_z_n_t;
    let maxsc = max(sqrt(f32(surface) / f32(params.ns - 1u)),
                    sqrt(1.0 / f32(params.ns - 1u)));
    let odd_scale = 1.0 / maxsc;
    let mnmax = params.mpol * (params.ntor + 1u);

    var values: array<f32, 18>;
    var corrections: array<f32, 18>;
    for (var field = 0u; field < 18u; field++) {
        values[field] = 0.0;
        corrections[field] = 0.0;
    }
    var r_con = 0.0;
    var z_con = 0.0;
    var r_con_correction = 0.0;
    var z_con_correction = 0.0;
    for (var mode = 0u; mode < mnmax; mode++) {
        let m = mode / (params.ntor + 1u);
        let n = mode % (params.ntor + 1u);
        let mf = f32(m);
        let nf = f32(n * params.nfp);
        let cc = basis_value(0u, mode, angular);
        let ss = basis_value(1u, mode, angular);
        let sc = basis_value(2u, mode, angular);
        let cs = basis_value(3u, mode, angular);
        let rc = coefficient(0u, mode, surface);
        let zs = coefficient(1u, mode, surface);
        let ls = coefficient(2u, mode, surface);
        let rs = coefficient(3u, mode, surface);
        let zc = coefficient(4u, mode, surface);
        let lc = coefficient(5u, mode, surface);
        let odd = (m & 1u) == 1u;
        let scale = select(1.0, odd_scale, odd);
        let parity = select(0u, 6u, odd);

        compensated_add(&values[parity + 0u], &corrections[parity + 0u],
                        scale * (rc * cc + rs * ss));
        compensated_add(&values[parity + 1u], &corrections[parity + 1u],
                        scale * (zs * sc + zc * cs));
        compensated_add(&values[parity + 2u], &corrections[parity + 2u],
                        scale * (ls * sc + lc * cs));
        compensated_add(&values[parity + 3u], &corrections[parity + 3u],
                        scale * (-mf * rc * sc + mf * rs * cs));
        compensated_add(&values[parity + 4u], &corrections[parity + 4u],
                        scale * (mf * zs * cc - mf * zc * ss));
        compensated_add(&values[parity + 5u], &corrections[parity + 5u],
                        scale * (mf * ls * cc - mf * lc * ss));
        let toroidal_parity = select(0u, 3u, odd);
        compensated_add(&values[12u + toroidal_parity],
                        &corrections[12u + toroidal_parity],
                        scale * (-nf * rc * cs + nf * rs * sc));
        compensated_add(&values[13u + toroidal_parity],
                        &corrections[13u + toroidal_parity],
                        scale * (-nf * zs * ss + nf * zc * cc));
        // VMEC stores lambda's toroidal slot as -d(lambda)/d(zeta).
        compensated_add(&values[14u + toroidal_parity],
                        &corrections[14u + toroidal_parity],
                        scale * (nf * ls * ss - nf * lc * cc));

        let xmpq = mf * (mf - 1.0);
        compensated_add(&r_con, &r_con_correction,
                        xmpq * (rc * cc + rs * ss));
        compensated_add(&z_con, &z_con_correction,
                        xmpq * (zs * sc + zc * cs));
    }

    for (var field = 0u; field < 18u; field++) {
        store(field, point, values[field]);
    }
    store(18u, point, r_con);
    store(19u, point, z_con);
}
