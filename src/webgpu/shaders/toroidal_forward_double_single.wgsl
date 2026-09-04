struct Params {
    ns: u32, mpol: u32, ntor: u32, ntheta: u32,
    nzeta: u32, nfp: u32, n_z_n_t: u32, include_lcfs: u32,
    norm_hi: f32, norm_lo: f32, sqrt2_hi: f32, sqrt2_lo: f32,
};
struct FF { hi: f32, lo: f32, };
struct Values { data: array<f32>, };
struct AtomicValues { data: array<atomic<u32>>, };
@group(0) @binding(0) var<storage, read> fields_hi: Values;
@group(0) @binding(1) var<storage, read> basis_hi: Values;
// Six high residual planes followed by six low planes.
@group(0) @binding(2) var<storage, read_write> residual: Values;
@group(0) @binding(3) var<uniform> params: Params;
// Two families x twenty fields, with all high values followed by all lows.
@group(0) @binding(4) var<storage, read_write> intermediate: Values;
@group(0) @binding(5) var<storage, read> fields_lo: Values;
@group(0) @binding(6) var<storage, read> basis_lo: Values;
@group(0) @binding(7) var<storage, read_write> rounding: AtomicValues;

fn rnd(value: f32, slot: u32) -> f32 {
    atomicStore(&rounding.data[slot], bitcast<u32>(value));
    return bitcast<f32>(atomicLoad(&rounding.data[slot]));
}
fn qts(a: f32, b: f32, slot: u32) -> FF {
    let s = rnd(a + b, slot); let v = rnd(s - a, slot);
    return FF(s, rnd(b - v, slot));
}
fn ts(a: f32, b: f32, slot: u32) -> FF {
    let s = rnd(a + b, slot); let bv = rnd(s - a, slot);
    let av = rnd(s - bv, slot); let ae = rnd(a - av, slot);
    let be = rnd(b - bv, slot); return FF(s, rnd(ae + be, slot));
}
fn norm(a: FF, slot: u32) -> FF { return qts(a.hi, a.lo, slot); }
fn add(a: FF, b: FF, slot: u32) -> FF {
    let h = ts(a.hi, b.hi, slot);
    return norm(FF(h.hi, rnd(rnd(h.lo + a.lo, slot) + b.lo, slot)), slot);
}
fn neg(a: FF) -> FF { return FF(-a.hi, -a.lo); }
fn sub(a: FF, b: FF, slot: u32) -> FF { return add(a, neg(b), slot); }
fn mulf(a: FF, b: f32, slot: u32) -> FF {
    let p = rnd(a.hi * b, slot); let e = rnd(fma(a.hi, b, -p), slot);
    return norm(FF(p, rnd(e + rnd(a.lo * b, slot), slot)), slot);
}
fn mul(a: FF, b: FF, slot: u32) -> FF {
    let p = rnd(a.hi * b.hi, slot); let e = rnd(fma(a.hi, b.hi, -p), slot);
    return norm(FF(p, rnd(rnd(e + rnd(a.hi * b.lo, slot), slot) +
                               rnd(a.lo * b.hi, slot), slot)), slot);
}
fn field_value(field: u32, surface: u32, theta: u32, zeta: u32,
               slot: u32) -> FF {
    let angular = zeta * params.ntheta + theta;
    let i = (field * params.ns + surface) * params.n_z_n_t + angular;
    return norm(FF(fields_hi.data[i], fields_lo.data[i]), slot);
}
fn theta_basis(sine: bool, m: u32, theta: u32, slot: u32) -> FF {
    let plane = select(0u, 1u, sine);
    let i = plane * params.mpol * params.ntheta + m * params.ntheta + theta;
    return norm(FF(basis_hi.data[i], basis_lo.data[i]), slot);
}
fn zeta_basis(sine: bool, n: u32, zeta: u32, slot: u32) -> FF {
    let start = 2u * params.mpol * params.ntheta;
    let plane = select(0u, 1u, sine);
    let i = start + plane * (params.ntor + 1u) * params.nzeta +
            n * params.nzeta + zeta;
    return norm(FF(basis_hi.data[i], basis_lo.data[i]), slot);
}
fn intermediate_index(family: u32, field: u32, surface: u32, theta: u32,
                      n: u32) -> u32 {
    let tr = params.ntheta / 2u + 1u;
    let plane = params.ns * tr * (params.ntor + 1u);
    return (family * 20u + field) * plane +
           ((surface * tr + theta) * (params.ntor + 1u) + n);
}
fn put_intermediate(family: u32, field: u32, surface: u32, theta: u32,
                    n: u32, value: FF) {
    let i = intermediate_index(family, field, surface, theta, n);
    let count = 40u * params.ns * (params.ntheta / 2u + 1u) *
                (params.ntor + 1u);
    intermediate.data[i] = value.hi; intermediate.data[count + i] = value.lo;
}
fn projected(family: u32, field: u32, surface: u32, theta: u32, n: u32,
             slot: u32) -> FF {
    let i = intermediate_index(family, field, surface, theta, n);
    let count = 40u * params.ns * (params.ntheta / 2u + 1u) *
                (params.ntor + 1u);
    return norm(FF(intermediate.data[i], intermediate.data[count + i]), slot);
}
fn put_result(component: u32, mode: u32, surface: u32, value: FF) {
    let count = 6u * params.mpol * (params.ntor + 1u) * params.ns;
    let i = (component * params.mpol * (params.ntor + 1u) + mode) * params.ns + surface;
    residual.data[i] = value.hi; residual.data[count + i] = value.lo;
}

@compute @workgroup_size(128)
fn toroidal_stage(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let tr = params.ntheta / 2u + 1u;
    let count = 20u * params.ns * tr * (params.ntor + 1u);
    let index = invocation.x; if (index >= count) { return; }
    let n = index % (params.ntor + 1u);
    let q = index / (params.ntor + 1u); let theta = q % tr;
    let sf = q / tr; let surface = sf % params.ns; let field = sf / params.ns;
    var cosine = FF(0.0, 0.0); var sine = FF(0.0, 0.0);
    for (var zeta = 0u; zeta < params.nzeta; zeta++) {
        let v = field_value(field, surface, theta, zeta, index);
        cosine = add(cosine, mul(v, zeta_basis(false, n, zeta, index), index), index);
        sine = add(sine, mul(v, zeta_basis(true, n, zeta, index), index), index);
    }
    put_intermediate(0u, field, surface, theta, n, cosine);
    put_intermediate(1u, field, surface, theta, n, sine);
}

@compute @workgroup_size(128)
fn poloidal_stage(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let mnmax = params.mpol * (params.ntor + 1u);
    let index = invocation.x; if (index >= params.ns * mnmax) { return; }
    let surface = index % params.ns; let mode = index / params.ns;
    let m = mode / (params.ntor + 1u); let n = mode % (params.ntor + 1u);
    let mf = f32(m); let nf = f32(n * params.nfp);
    let parity = select(0u, 1u, (m & 1u) == 1u);
    let xmpq = mf * (mf - 1.0); let tr = params.ntheta / 2u + 1u;
    var sums: array<FF, 6>;
    for (var c = 0u; c < 6u; c++) { sums[c] = FF(0.0, 0.0); }
    for (var theta = 0u; theta < tr; theta++) {
        var weight = FF(params.norm_hi, params.norm_lo);
        if (theta == 0u || theta + 1u == tr) { weight = mulf(weight, 0.5, index); }
        let cm = theta_basis(false, m, theta, index);
        let sm = theta_basis(true, m, theta, index);
        let trc = add(projected(0u, parity, surface, theta, n, index),
                      mulf(projected(0u, 16u + parity, surface, theta, n, index), xmpq, index), index);
        let trs = add(projected(1u, parity, surface, theta, n, index),
                      mulf(projected(1u, 16u + parity, surface, theta, n, index), xmpq, index), index);
        let tzc = add(projected(0u, 2u + parity, surface, theta, n, index),
                      mulf(projected(0u, 18u + parity, surface, theta, n, index), xmpq, index), index);
        let tzs = add(projected(1u, 2u + parity, surface, theta, n, index),
                      mulf(projected(1u, 18u + parity, surface, theta, n, index), xmpq, index), index);
        let brc = projected(0u, 4u + parity, surface, theta, n, index);
        let brs = projected(1u, 4u + parity, surface, theta, n, index);
        let bzc = projected(0u, 6u + parity, surface, theta, n, index);
        let bzs = projected(1u, 6u + parity, surface, theta, n, index);
        let blc = projected(0u, 8u + parity, surface, theta, n, index);
        let bls = projected(1u, 8u + parity, surface, theta, n, index);
        let crc = projected(0u, 10u + parity, surface, theta, n, index);
        let crs = projected(1u, 10u + parity, surface, theta, n, index);
        let czc = projected(0u, 12u + parity, surface, theta, n, index);
        let czs = projected(1u, 12u + parity, surface, theta, n, index);
        let clc = projected(0u, 14u + parity, surface, theta, n, index);
        let cls = projected(1u, 14u + parity, surface, theta, n, index);
        let r0 = add(sub(mul(trc, cm, index), mulf(mul(brc, sm, index), mf, index), index), mulf(mul(crs, cm, index), nf, index), index);
        let r3 = sub(add(mul(trs, sm, index), mulf(mul(brs, cm, index), mf, index), index), mulf(mul(crc, sm, index), nf, index), index);
        let r1 = add(add(mul(tzc, sm, index), mulf(mul(bzc, cm, index), mf, index), index), mulf(mul(czs, sm, index), nf, index), index);
        let r4 = sub(sub(mul(tzs, cm, index), mulf(mul(bzs, sm, index), mf, index), index), mulf(mul(czc, cm, index), nf, index), index);
        let r2 = add(mulf(mul(blc, cm, index), mf, index), mulf(mul(cls, sm, index), nf, index), index);
        let r5 = neg(add(mulf(mul(bls, sm, index), mf, index), mulf(mul(clc, cm, index), nf, index), index));
        sums[0] = add(sums[0], mul(weight, r0, index), index);
        sums[1] = add(sums[1], mul(weight, r1, index), index);
        sums[2] = add(sums[2], mul(weight, r2, index), index);
        sums[3] = add(sums[3], mul(weight, r3, index), index);
        sums[4] = add(sums[4], mul(weight, r4, index), index);
        sums[5] = add(sums[5], mul(weight, r5, index), index);
    }
    var scale = FF(1.0, 0.0); let sqrt2 = FF(params.sqrt2_hi, params.sqrt2_lo);
    if (m != 0u) { scale = mul(scale, sqrt2, index); }
    if (n != 0u) { scale = mul(scale, sqrt2, index); }
    for (var c = 0u; c < 6u; c++) {
        var value = mul(scale, sums[c], index);
        if (surface == 0u) {
            if (!(m == 0u && (c == 0u || c == 4u))) { value = FF(0.0, 0.0); }
        } else if (surface + 1u == params.ns && params.include_lcfs == 0u) {
            if (!(c == 2u || c == 5u)) { value = FF(0.0, 0.0); }
        }
        put_result(c, mode, surface, value);
    }
}
