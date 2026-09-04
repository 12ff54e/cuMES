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

struct Values { data: array<f32>, };

@group(0) @binding(0) var<storage, read> state: Values;
// Separable tables: cos(m theta), sin(m theta), cos(n zeta), sin(n zeta).
@group(0) @binding(1) var<storage, read> basis: Values;
// 18 geometry parity fields followed by rCon and zCon.
@group(0) @binding(2) var<storage, read_write> output: Values;
@group(0) @binding(3) var<uniform> params: Params;
// Twelve toroidally synthesized coefficient/derivative planes.
@group(0) @binding(4) var<storage, read_write> intermediate: Values;

fn coefficient(component: u32, m: u32, n: u32, surface: u32) -> f32 {
    let mnmax = params.mpol * (params.ntor + 1u);
    let mode = m * (params.ntor + 1u) + n;
    return state.data[(component * mnmax + mode) * params.ns + surface];
}

fn theta_basis(sine: bool, m: u32, theta: u32) -> f32 {
    let plane = select(0u, 1u, sine);
    return basis.data[plane * params.mpol * params.ntheta +
                      m * params.ntheta + theta];
}

fn zeta_basis(sine: bool, n: u32, zeta: u32) -> f32 {
    let theta_values = 2u * params.mpol * params.ntheta;
    let plane = select(0u, 1u, sine);
    return basis.data[theta_values +
                      plane * (params.ntor + 1u) * params.nzeta +
                      n * params.nzeta + zeta];
}

fn intermediate_index(field: u32, surface: u32, m: u32, zeta: u32) -> u32 {
    let plane_size = params.ns * params.mpol * params.nzeta;
    return field * plane_size +
           (surface * params.mpol + m) * params.nzeta + zeta;
}

fn intermediate_at(field: u32, surface: u32, m: u32, zeta: u32) -> f32 {
    return intermediate.data[intermediate_index(field, surface, m, zeta)];
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

// First synthesize every positive toroidal mode for a fixed (surface,m,zeta).
// This is the same separability used by the CUDA cuFFT+poloidal implementation,
// but a short compensated sum is faster than padding an FFT for ntor <= 12.
@compute @workgroup_size(128)
fn toroidal_stage(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let count = params.ns * params.mpol * params.nzeta;
    let index = invocation.x;
    if (index >= count) { return; }
    let zeta = index % params.nzeta;
    let surface_m = index / params.nzeta;
    let m = surface_m % params.mpol;
    let surface = surface_m / params.mpol;
    var sums: array<f32, 12>;
    var corrections: array<f32, 12>;
    for (var field = 0u; field < 12u; field++) {
        sums[field] = 0.0;
        corrections[field] = 0.0;
    }
    for (var n = 0u; n <= params.ntor; n++) {
        let cn = zeta_basis(false, n, zeta);
        let sn = zeta_basis(true, n, zeta);
        let nf = f32(n * params.nfp);
        let rc = coefficient(0u, m, n, surface);
        let zs = coefficient(1u, m, n, surface);
        let ls = coefficient(2u, m, n, surface);
        let rs = coefficient(3u, m, n, surface);
        let zc = coefficient(4u, m, n, surface);
        let lc = coefficient(5u, m, n, surface);
        compensated_add(&sums[0], &corrections[0], rc * cn);
        compensated_add(&sums[1], &corrections[1], rs * sn);
        compensated_add(&sums[2], &corrections[2], zs * cn);
        compensated_add(&sums[3], &corrections[3], zc * sn);
        compensated_add(&sums[4], &corrections[4], ls * cn);
        compensated_add(&sums[5], &corrections[5], lc * sn);
        compensated_add(&sums[6], &corrections[6], -nf * rc * sn);
        compensated_add(&sums[7], &corrections[7], nf * rs * cn);
        compensated_add(&sums[8], &corrections[8], -nf * zs * sn);
        compensated_add(&sums[9], &corrections[9], nf * zc * cn);
        compensated_add(&sums[10], &corrections[10], nf * ls * sn);
        compensated_add(&sums[11], &corrections[11], -nf * lc * cn);
    }
    for (var field = 0u; field < 12u; field++) {
        intermediate.data[intermediate_index(field, surface, m, zeta)] =
            sums[field];
    }
}

// Then synthesize the short poloidal series into the real-space fields.
@compute @workgroup_size(128)
fn poloidal_stage(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let point = invocation.x;
    if (point >= params.total_points) { return; }
    let surface = point / params.n_z_n_t;
    let angular = point % params.n_z_n_t;
    let theta = angular % params.ntheta;
    let zeta = angular / params.ntheta;
    let maxsc = max(sqrt(f32(surface) / f32(params.ns - 1u)),
                    sqrt(1.0 / f32(params.ns - 1u)));
    let odd_scale = 1.0 / maxsc;
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
    for (var m = 0u; m < params.mpol; m++) {
        let cm = theta_basis(false, m, theta);
        let sm = theta_basis(true, m, theta);
        let mf = f32(m);
        let odd = (m & 1u) == 1u;
        let scale = select(1.0, odd_scale, odd);
        let parity = select(0u, 6u, odd);
        let a0 = intermediate_at(0u, surface, m, zeta);
        let a1 = intermediate_at(1u, surface, m, zeta);
        let a2 = intermediate_at(2u, surface, m, zeta);
        let a3 = intermediate_at(3u, surface, m, zeta);
        let a4 = intermediate_at(4u, surface, m, zeta);
        let a5 = intermediate_at(5u, surface, m, zeta);
        let r = a0 * cm + a1 * sm;
        let z = a2 * sm + a3 * cm;
        let lambda = a4 * sm + a5 * cm;
        compensated_add(&values[parity], &corrections[parity], scale * r);
        compensated_add(&values[parity + 1u], &corrections[parity + 1u],
                        scale * z);
        compensated_add(&values[parity + 2u], &corrections[parity + 2u],
                        scale * lambda);
        compensated_add(&values[parity + 3u], &corrections[parity + 3u],
                        scale * mf * (-a0 * sm + a1 * cm));
        compensated_add(&values[parity + 4u], &corrections[parity + 4u],
                        scale * mf * (a2 * cm - a3 * sm));
        compensated_add(&values[parity + 5u], &corrections[parity + 5u],
                        scale * mf * (a4 * cm - a5 * sm));
        let toroidal_parity = select(0u, 3u, odd);
        compensated_add(&values[12u + toroidal_parity],
                        &corrections[12u + toroidal_parity],
                        scale * (intermediate_at(6u, surface, m, zeta) * cm +
                                 intermediate_at(7u, surface, m, zeta) * sm));
        compensated_add(&values[13u + toroidal_parity],
                        &corrections[13u + toroidal_parity],
                        scale * (intermediate_at(8u, surface, m, zeta) * sm +
                                 intermediate_at(9u, surface, m, zeta) * cm));
        compensated_add(&values[14u + toroidal_parity],
                        &corrections[14u + toroidal_parity],
                        scale * (intermediate_at(10u, surface, m, zeta) * sm +
                                 intermediate_at(11u, surface, m, zeta) * cm));
        let xmpq = mf * (mf - 1.0);
        compensated_add(&r_con, &r_con_correction, xmpq * r);
        compensated_add(&z_con, &z_con_correction, xmpq * z);
    }
    for (var field = 0u; field < 18u; field++) {
        store(field, point, values[field]);
    }
    store(18u, point, r_con);
    store(19u, point, z_con);
}
