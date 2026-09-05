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

// Return the updated (sum, correction) pair instead of passing pointers to
// dynamically indexed function-local arrays.  Both forms are valid WGSL, but
// the value-returning form avoids a known weak spot in some browser backend
// compilers when lowering non-constant access-chain pointers.
fn compensated_add(accumulator: vec2<f32>, term: f32) -> vec2<f32> {
    let adjusted = term - accumulator.y;
    let next = accumulator.x + adjusted;
    return vec2<f32>(next, (next - accumulator.x) - adjusted);
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
    // Keep these accumulators named. Chrome/Dawn's D3D12 DXC path can
    // miscompile a dynamically indexed function-local array here to zeros.
    var sum0 = vec2<f32>(0.0, 0.0);
    var sum1 = vec2<f32>(0.0, 0.0);
    var sum2 = vec2<f32>(0.0, 0.0);
    var sum3 = vec2<f32>(0.0, 0.0);
    var sum4 = vec2<f32>(0.0, 0.0);
    var sum5 = vec2<f32>(0.0, 0.0);
    var sum6 = vec2<f32>(0.0, 0.0);
    var sum7 = vec2<f32>(0.0, 0.0);
    var sum8 = vec2<f32>(0.0, 0.0);
    var sum9 = vec2<f32>(0.0, 0.0);
    var sum10 = vec2<f32>(0.0, 0.0);
    var sum11 = vec2<f32>(0.0, 0.0);
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
        sum0 = compensated_add(sum0, rc * cn);
        sum1 = compensated_add(sum1, rs * sn);
        sum2 = compensated_add(sum2, zs * cn);
        sum3 = compensated_add(sum3, zc * sn);
        sum4 = compensated_add(sum4, ls * cn);
        sum5 = compensated_add(sum5, lc * sn);
        sum6 = compensated_add(sum6, -nf * rc * sn);
        sum7 = compensated_add(sum7, nf * rs * cn);
        sum8 = compensated_add(sum8, -nf * zs * sn);
        sum9 = compensated_add(sum9, nf * zc * cn);
        sum10 = compensated_add(sum10, nf * ls * sn);
        sum11 = compensated_add(sum11, -nf * lc * cn);
    }
    intermediate.data[intermediate_index(0u, surface, m, zeta)] = sum0.x;
    intermediate.data[intermediate_index(1u, surface, m, zeta)] = sum1.x;
    intermediate.data[intermediate_index(2u, surface, m, zeta)] = sum2.x;
    intermediate.data[intermediate_index(3u, surface, m, zeta)] = sum3.x;
    intermediate.data[intermediate_index(4u, surface, m, zeta)] = sum4.x;
    intermediate.data[intermediate_index(5u, surface, m, zeta)] = sum5.x;
    intermediate.data[intermediate_index(6u, surface, m, zeta)] = sum6.x;
    intermediate.data[intermediate_index(7u, surface, m, zeta)] = sum7.x;
    intermediate.data[intermediate_index(8u, surface, m, zeta)] = sum8.x;
    intermediate.data[intermediate_index(9u, surface, m, zeta)] = sum9.x;
    intermediate.data[intermediate_index(10u, surface, m, zeta)] = sum10.x;
    intermediate.data[intermediate_index(11u, surface, m, zeta)] = sum11.x;
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
    // See the D3D12 portability note in toroidal_stage.
    var value0 = vec2<f32>(0.0, 0.0);
    var value1 = vec2<f32>(0.0, 0.0);
    var value2 = vec2<f32>(0.0, 0.0);
    var value3 = vec2<f32>(0.0, 0.0);
    var value4 = vec2<f32>(0.0, 0.0);
    var value5 = vec2<f32>(0.0, 0.0);
    var value6 = vec2<f32>(0.0, 0.0);
    var value7 = vec2<f32>(0.0, 0.0);
    var value8 = vec2<f32>(0.0, 0.0);
    var value9 = vec2<f32>(0.0, 0.0);
    var value10 = vec2<f32>(0.0, 0.0);
    var value11 = vec2<f32>(0.0, 0.0);
    var value12 = vec2<f32>(0.0, 0.0);
    var value13 = vec2<f32>(0.0, 0.0);
    var value14 = vec2<f32>(0.0, 0.0);
    var value15 = vec2<f32>(0.0, 0.0);
    var value16 = vec2<f32>(0.0, 0.0);
    var value17 = vec2<f32>(0.0, 0.0);
    var r_con = vec2<f32>(0.0, 0.0);
    var z_con = vec2<f32>(0.0, 0.0);
    for (var m = 0u; m < params.mpol; m++) {
        let cm = theta_basis(false, m, theta);
        let sm = theta_basis(true, m, theta);
        let mf = f32(m);
        let odd = (m & 1u) == 1u;
        let scale = select(1.0, odd_scale, odd);
        let a0 = intermediate_at(0u, surface, m, zeta);
        let a1 = intermediate_at(1u, surface, m, zeta);
        let a2 = intermediate_at(2u, surface, m, zeta);
        let a3 = intermediate_at(3u, surface, m, zeta);
        let a4 = intermediate_at(4u, surface, m, zeta);
        let a5 = intermediate_at(5u, surface, m, zeta);
        let r = a0 * cm + a1 * sm;
        let z = a2 * sm + a3 * cm;
        let lambda = a4 * sm + a5 * cm;
        let ru = scale * mf * (-a0 * sm + a1 * cm);
        let zu = scale * mf * (a2 * cm - a3 * sm);
        let lu = scale * mf * (a4 * cm - a5 * sm);
        let rv = scale * (intermediate_at(6u, surface, m, zeta) * cm +
                          intermediate_at(7u, surface, m, zeta) * sm);
        let zv = scale * (intermediate_at(8u, surface, m, zeta) * sm +
                          intermediate_at(9u, surface, m, zeta) * cm);
        let lv = scale * (intermediate_at(10u, surface, m, zeta) * sm +
                          intermediate_at(11u, surface, m, zeta) * cm);
        if (odd) {
            value6 = compensated_add(value6, scale * r);
            value7 = compensated_add(value7, scale * z);
            value8 = compensated_add(value8, scale * lambda);
            value9 = compensated_add(value9, ru);
            value10 = compensated_add(value10, zu);
            value11 = compensated_add(value11, lu);
            value15 = compensated_add(value15, rv);
            value16 = compensated_add(value16, zv);
            value17 = compensated_add(value17, lv);
        } else {
            value0 = compensated_add(value0, scale * r);
            value1 = compensated_add(value1, scale * z);
            value2 = compensated_add(value2, scale * lambda);
            value3 = compensated_add(value3, ru);
            value4 = compensated_add(value4, zu);
            value5 = compensated_add(value5, lu);
            value12 = compensated_add(value12, rv);
            value13 = compensated_add(value13, zv);
            value14 = compensated_add(value14, lv);
        }
        let xmpq = mf * (mf - 1.0);
        r_con = compensated_add(r_con, xmpq * r);
        z_con = compensated_add(z_con, xmpq * z);
    }
    store(0u, point, value0.x);
    store(1u, point, value1.x);
    store(2u, point, value2.x);
    store(3u, point, value3.x);
    store(4u, point, value4.x);
    store(5u, point, value5.x);
    store(6u, point, value6.x);
    store(7u, point, value7.x);
    store(8u, point, value8.x);
    store(9u, point, value9.x);
    store(10u, point, value10.x);
    store(11u, point, value11.x);
    store(12u, point, value12.x);
    store(13u, point, value13.x);
    store(14u, point, value14.x);
    store(15u, point, value15.x);
    store(16u, point, value16.x);
    store(17u, point, value17.x);
    store(18u, point, r_con.x);
    store(19u, point, z_con.x);
}
