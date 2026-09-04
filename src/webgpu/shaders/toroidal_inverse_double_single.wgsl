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
struct AtomicValues { data: array<atomic<u32>>, };

@group(0) @binding(0) var<storage, read> state_hi: Values;
@group(0) @binding(1) var<storage, read> basis: Values;
// Twenty high planes followed by twenty low planes.
@group(0) @binding(2) var<storage, read_write> output: Values;
@group(0) @binding(3) var<uniform> params: Params;
// Twelve high planes followed by twelve low planes.
@group(0) @binding(4) var<storage, read_write> intermediate: Values;
@group(0) @binding(5) var<storage, read> state_lo: Values;
@group(0) @binding(6) var<storage, read_write> rounding: AtomicValues;
@group(0) @binding(7) var<storage, read> radial_scale_hi: Values;
@group(0) @binding(8) var<storage, read> radial_scale_lo: Values;
@group(0) @binding(9) var<storage, read> basis_lo: Values;

fn ff_strict_round(value: f32, slot: u32) -> f32 {
    atomicStore(&rounding.data[slot], bitcast<u32>(value));
    return bitcast<f32>(atomicLoad(&rounding.data[slot]));
}

fn ff_strict_quick_two_sum(a: f32, b: f32, slot: u32) -> FF {
    let sum = ff_strict_round(a + b, slot);
    let sum_minus_a = ff_strict_round(sum - a, slot);
    return FF(sum, ff_strict_round(b - sum_minus_a, slot));
}

fn ff_strict_two_sum(a: f32, b: f32, slot: u32) -> FF {
    let sum = ff_strict_round(a + b, slot);
    let b_virtual = ff_strict_round(sum - a, slot);
    let a_virtual = ff_strict_round(sum - b_virtual, slot);
    let a_error = ff_strict_round(a - a_virtual, slot);
    let b_error = ff_strict_round(b - b_virtual, slot);
    return FF(sum, ff_strict_round(a_error + b_error, slot));
}

fn ff_strict_normalize(value: FF, slot: u32) -> FF {
    return ff_strict_quick_two_sum(value.hi, value.lo, slot);
}

fn ff_strict_add(a: FF, b: FF, slot: u32) -> FF {
    let leading = ff_strict_two_sum(a.hi, b.hi, slot);
    let low = ff_strict_round(leading.lo + a.lo, slot);
    return ff_strict_normalize(
        FF(leading.hi, ff_strict_round(low + b.lo, slot)), slot);
}

fn ff_strict_mul_f32(a: FF, b: f32, slot: u32) -> FF {
    let product = ff_strict_round(a.hi * b, slot);
    let leading_error = ff_strict_round(fma(a.hi, b, -product), slot);
    let low_product = ff_strict_round(a.lo * b, slot);
    return ff_strict_normalize(
        FF(product, ff_strict_round(leading_error + low_product, slot)), slot);
}

fn ff_strict_mul(a: FF, b: FF, slot: u32) -> FF {
    let product = ff_strict_round(a.hi * b.hi, slot);
    let leading_error = ff_strict_round(fma(a.hi, b.hi, -product), slot);
    let cross0 = ff_strict_round(a.hi * b.lo, slot);
    let cross1 = ff_strict_round(a.lo * b.hi, slot);
    let low0 = ff_strict_round(leading_error + cross0, slot);
    let low1 = ff_strict_round(low0 + cross1, slot);
    return ff_strict_normalize(FF(product, low1), slot);
}

fn ff_strict_sub(a: FF, b: FF, slot: u32) -> FF {
    return ff_strict_add(a, FF(-b.hi, -b.lo), slot);
}

fn coefficient(component: u32, m: u32, n: u32, surface: u32) -> FF {
    let mnmax = params.mpol * (params.ntor + 1u);
    let mode = m * (params.ntor + 1u) + n;
    let index = (component * mnmax + mode) * params.ns + surface;
    return FF(state_hi.data[index], state_lo.data[index]);
}

fn theta_basis(sine: bool, m: u32, theta: u32, slot: u32) -> FF {
    let plane = select(0u, 1u, sine);
    let index = plane * params.mpol * params.ntheta +
                m * params.ntheta + theta;
    return ff_strict_normalize(FF(basis.data[index], basis_lo.data[index]),
                               slot);
}

fn zeta_basis(sine: bool, n: u32, zeta: u32, slot: u32) -> FF {
    let theta_values = 2u * params.mpol * params.ntheta;
    let plane = select(0u, 1u, sine);
    let index = theta_values +
                plane * (params.ntor + 1u) * params.nzeta +
                n * params.nzeta + zeta;
    return ff_strict_normalize(FF(basis.data[index], basis_lo.data[index]),
                               slot);
}

fn intermediate_index(field: u32, surface: u32, m: u32, zeta: u32) -> u32 {
    let plane_size = params.ns * params.mpol * params.nzeta;
    return field * plane_size +
           (surface * params.mpol + m) * params.nzeta + zeta;
}

fn intermediate_at(field: u32, surface: u32, m: u32, zeta: u32) -> FF {
    let plane_size = params.ns * params.mpol * params.nzeta;
    let index = intermediate_index(field, surface, m, zeta);
    return FF(intermediate.data[index],
              intermediate.data[12u * plane_size + index]);
}

fn put_intermediate(field: u32, surface: u32, m: u32, zeta: u32, value: FF) {
    let plane_size = params.ns * params.mpol * params.nzeta;
    let index = intermediate_index(field, surface, m, zeta);
    intermediate.data[index] = value.hi;
    intermediate.data[12u * plane_size + index] = value.lo;
}

fn put_output(field: u32, point: u32, value: FF) {
    let index = field * params.total_points + point;
    output.data[index] = value.hi;
    output.data[20u * params.total_points + index] = value.lo;
}

@compute @workgroup_size(128)
fn toroidal_stage(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let count = params.ns * params.mpol * params.nzeta;
    let index = invocation.x;
    if (index >= count) { return; }
    let zeta = index % params.nzeta;
    let surface_m = index / params.nzeta;
    let m = surface_m % params.mpol;
    let surface = surface_m / params.mpol;
    var sums: array<FF, 12>;
    for (var field = 0u; field < 12u; field++) {
        sums[field] = FF(0.0, 0.0);
    }
    for (var n = 0u; n <= params.ntor; n++) {
        let cn = zeta_basis(false, n, zeta, index);
        let sn = zeta_basis(true, n, zeta, index);
        let nf = f32(n * params.nfp);
        let rc = coefficient(0u, m, n, surface);
        let zs = coefficient(1u, m, n, surface);
        let ls = coefficient(2u, m, n, surface);
        let rs = coefficient(3u, m, n, surface);
        let zc = coefficient(4u, m, n, surface);
        let lc = coefficient(5u, m, n, surface);
        sums[0] = ff_strict_add(sums[0], ff_strict_mul(rc, cn, index), index);
        sums[1] = ff_strict_add(sums[1], ff_strict_mul(rs, sn, index), index);
        sums[2] = ff_strict_add(sums[2], ff_strict_mul(zs, cn, index), index);
        sums[3] = ff_strict_add(sums[3], ff_strict_mul(zc, sn, index), index);
        sums[4] = ff_strict_add(sums[4], ff_strict_mul(ls, cn, index), index);
        sums[5] = ff_strict_add(sums[5], ff_strict_mul(lc, sn, index), index);
        sums[6] = ff_strict_add(sums[6], ff_strict_mul(
            ff_strict_mul_f32(rc, -nf, index), sn, index), index);
        sums[7] = ff_strict_add(sums[7], ff_strict_mul(
            ff_strict_mul_f32(rs, nf, index), cn, index), index);
        sums[8] = ff_strict_add(sums[8], ff_strict_mul(
            ff_strict_mul_f32(zs, -nf, index), sn, index), index);
        sums[9] = ff_strict_add(sums[9], ff_strict_mul(
            ff_strict_mul_f32(zc, nf, index), cn, index), index);
        sums[10] = ff_strict_add(sums[10], ff_strict_mul(
            ff_strict_mul_f32(ls, nf, index), sn, index), index);
        sums[11] = ff_strict_add(sums[11], ff_strict_mul(
            ff_strict_mul_f32(lc, -nf, index), cn, index), index);
    }
    for (var field = 0u; field < 12u; field++) {
        put_intermediate(field, surface, m, zeta, sums[field]);
    }
}

@compute @workgroup_size(128)
fn poloidal_stage(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let point = invocation.x;
    if (point >= params.total_points) { return; }
    let surface = point / params.n_z_n_t;
    let angular = point % params.n_z_n_t;
    let theta = angular % params.ntheta;
    let zeta = angular / params.ntheta;
    let odd_scale = FF(radial_scale_hi.data[surface],
                       radial_scale_lo.data[surface]);
    var values: array<FF, 18>;
    for (var field = 0u; field < 18u; field++) {
        values[field] = FF(0.0, 0.0);
    }
    var r_con = FF(0.0, 0.0);
    var z_con = FF(0.0, 0.0);
    for (var m = 0u; m < params.mpol; m++) {
        let cm = theta_basis(false, m, theta, point);
        let sm = theta_basis(true, m, theta, point);
        let mf = f32(m);
        let odd = (m & 1u) == 1u;
        var scale = FF(1.0, 0.0);
        if (odd) { scale = odd_scale; }
        let parity = select(0u, 6u, odd);
        let a0 = intermediate_at(0u, surface, m, zeta);
        let a1 = intermediate_at(1u, surface, m, zeta);
        let a2 = intermediate_at(2u, surface, m, zeta);
        let a3 = intermediate_at(3u, surface, m, zeta);
        let a4 = intermediate_at(4u, surface, m, zeta);
        let a5 = intermediate_at(5u, surface, m, zeta);
        let r = ff_strict_add(ff_strict_mul(a0, cm, point),
                              ff_strict_mul(a1, sm, point), point);
        let z = ff_strict_add(ff_strict_mul(a2, sm, point),
                              ff_strict_mul(a3, cm, point), point);
        let lambda = ff_strict_add(ff_strict_mul(a4, sm, point),
                                   ff_strict_mul(a5, cm, point), point);
        values[parity] = ff_strict_add(
            values[parity], ff_strict_mul(r, scale, point), point);
        values[parity + 1u] = ff_strict_add(
            values[parity + 1u], ff_strict_mul(z, scale, point), point);
        values[parity + 2u] = ff_strict_add(
            values[parity + 2u], ff_strict_mul(lambda, scale, point), point);
        let ru = ff_strict_add(ff_strict_mul(a0, FF(-sm.hi, -sm.lo), point),
                               ff_strict_mul(a1, cm, point), point);
        let zu = ff_strict_sub(ff_strict_mul(a2, cm, point),
                               ff_strict_mul(a3, sm, point), point);
        let lu = ff_strict_sub(ff_strict_mul(a4, cm, point),
                               ff_strict_mul(a5, sm, point), point);
        let derivative_scale = ff_strict_mul_f32(scale, mf, point);
        values[parity + 3u] = ff_strict_add(
            values[parity + 3u],
            ff_strict_mul(ru, derivative_scale, point), point);
        values[parity + 4u] = ff_strict_add(
            values[parity + 4u],
            ff_strict_mul(zu, derivative_scale, point), point);
        values[parity + 5u] = ff_strict_add(
            values[parity + 5u],
            ff_strict_mul(lu, derivative_scale, point), point);
        let toroidal_parity = select(0u, 3u, odd);
        for (var family = 0u; family < 3u; family++) {
            let left = intermediate_at(6u + 2u * family, surface, m, zeta);
            let right = intermediate_at(7u + 2u * family, surface, m, zeta);
            var left_basis = cm;
            var right_basis = sm;
            if (family != 0u) {
                left_basis = sm;
                right_basis = cm;
            }
            let term = ff_strict_add(
                ff_strict_mul(left, left_basis, point),
                ff_strict_mul(right, right_basis, point), point);
            let field = 12u + family + toroidal_parity;
            values[field] = ff_strict_add(
                values[field], ff_strict_mul(term, scale, point), point);
        }
        let xmpq = ff_strict_round(mf * ff_strict_round(mf - 1.0, point), point);
        r_con = ff_strict_add(r_con, ff_strict_mul_f32(r, xmpq, point), point);
        z_con = ff_strict_add(z_con, ff_strict_mul_f32(z, xmpq, point), point);
    }
    for (var field = 0u; field < 18u; field++) {
        put_output(field, point, values[field]);
    }
    put_output(18u, point, r_con);
    put_output(19u, point, z_con);
}
