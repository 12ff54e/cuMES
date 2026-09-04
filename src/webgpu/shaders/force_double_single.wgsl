struct Params {
    ns: u32,
    n_z_n_t: u32,
    full_points: u32,
    half_points: u32,
    delta_s: f32,
    delta_s_lo: f32,
    lamscale: f32,
    lamscale_lo: f32,
};
struct FF { hi: f32, lo: f32, };
struct Values { data: array<f32>, };
struct AtomicValues { data: array<atomic<u32>>, };
@group(0) @binding(0) var<storage, read> geometry_hi: Values;
@group(0) @binding(1) var<storage, read> base_hi: Values;
@group(0) @binding(2) var<storage, read> magnetic_hi: Values;
// sqrt_s_f[ns], sqrt_s_h[ns-1], phip_f[ns], inverse_sqrt_s_h[ns-1].
@group(0) @binding(3) var<storage, read> radial_hi: Values;
// Sixteen high planes followed by sixteen low planes.
@group(0) @binding(4) var<storage, read_write> force: Values;
@group(0) @binding(5) var<uniform> params: Params;
@group(0) @binding(6) var<storage, read> geometry_lo: Values;
@group(0) @binding(7) var<storage, read> base_lo: Values;
@group(0) @binding(8) var<storage, read> magnetic_lo: Values;
@group(0) @binding(9) var<storage, read> radial_lo: Values;
@group(0) @binding(10) var<storage, read_write> rounding: AtomicValues;

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
    let h = ts(a.hi, b.hi, slot); let l = rnd(h.lo + a.lo, slot);
    return norm(FF(h.hi, rnd(l + b.lo, slot)), slot);
}
fn sub(a: FF, b: FF, slot: u32) -> FF {
    return add(a, FF(-b.hi, -b.lo), slot);
}
fn mulf(a: FF, b: f32, slot: u32) -> FF {
    let p = rnd(a.hi * b, slot); let e = rnd(fma(a.hi, b, -p), slot);
    return norm(FF(p, rnd(e + rnd(a.lo * b, slot), slot)), slot);
}
fn mul(a: FF, b: FF, slot: u32) -> FF {
    let p = rnd(a.hi * b.hi, slot); let e = rnd(fma(a.hi, b.hi, -p), slot);
    let c0 = rnd(a.hi * b.lo, slot); let c1 = rnd(a.lo * b.hi, slot);
    return norm(FF(p, rnd(rnd(e + c0, slot) + c1, slot)), slot);
}
fn reciprocal(a: FF, slot: u32) -> FF {
    let e = rnd(1.0 / a.hi, slot); var x = FF(e, 0.0);
    var r = sub(FF(1.0, 0.0), mul(a, x, slot), slot);
    x = add(x, mulf(r, e, slot), slot);
    r = sub(FF(1.0, 0.0), mul(a, x, slot), slot);
    return add(x, mul(x, r, slot), slot);
}
fn div(a: FF, b: FF, slot: u32) -> FF { return mul(a, reciprocal(b, slot), slot); }
fn avg(a: FF, b: FF, slot: u32) -> FF { return mulf(add(a, b, slot), 0.5, slot); }
fn neg(a: FF) -> FF { return FF(-a.hi, -a.lo); }
fn full(f: u32, p: u32, slot: u32) -> FF {
    let i = f * params.full_points + p;
    return norm(FF(geometry_hi.data[i], geometry_lo.data[i]), slot);
}
fn half(f: u32, p: u32, slot: u32) -> FF {
    let i = f * params.half_points + p;
    return norm(FF(base_hi.data[i], base_lo.data[i]), slot);
}
fn bfield(f: u32, p: u32, slot: u32) -> FF {
    let i = f * params.half_points + p;
    return norm(FF(magnetic_hi.data[i], magnetic_lo.data[i]), slot);
}
fn radial(i: u32, slot: u32) -> FF {
    return norm(FF(radial_hi.data[i], radial_lo.data[i]), slot);
}
fn put(f: u32, p: u32, value: FF) {
    let i = f * params.full_points + p;
    force.data[i] = value.hi;
    force.data[16u * params.full_points + i] = value.lo;
}
fn sum4(a: FF, b: FF, c: FF, d: FF, slot: u32) -> FF {
    return add(add(a, b, slot), add(c, d, slot), slot);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let point = invocation.x; if (point >= params.full_points) { return; }
    let surface = point / params.n_z_n_t;
    let angular = point % params.n_z_n_t;
    let half_count = params.ns - 1u;
    let zero = FF(0.0, 0.0);
    var i: array<FF, 15>; var o: array<FF, 15>;
    for (var x = 0u; x < 15u; x++) { i[x] = zero; o[x] = zero; }
    if (surface > 0u) {
        let h = (surface - 1u) * params.n_z_n_t + angular;
        for (var x = 0u; x < 7u; x++) { i[x] = half(x, h, point); }
        i[7] = half(8u, h, point); i[8] = half(9u, h, point);
        for (var x = 0u; x < 5u; x++) { i[9u + x] = bfield(x, h, point); }
        i[14] = radial(params.ns + surface - 1u, point);
    }
    if (surface + 1u < params.ns) {
        let h = surface * params.n_z_n_t + angular;
        for (var x = 0u; x < 7u; x++) { o[x] = half(x, h, point); }
        o[7] = half(8u, h, point); o[8] = half(9u, h, point);
        for (var x = 0u; x < 5u; x++) { o[9u + x] = bfield(x, h, point); }
        o[14] = radial(params.ns + surface, point);
    }
    let sqrt_f = radial(surface, point); let s_full = mul(sqrt_f, sqrt_f, point);
    let pi = mul(i[0], i[13], point); let po = mul(o[0], o[13], point);
    let zupi = mul(i[2], pi, point); let zupo = mul(o[2], po, point);
    let rupi = mul(i[1], pi, point); let rupo = mul(o[1], po, point);
    let rspi = mul(i[3], pi, point); let rspo = mul(o[3], po, point);
    let zspi = mul(i[4], pi, point); let zspo = mul(o[4], po, point);
    let taupi = mul(i[5], i[13], point); let taupo = mul(o[5], o[13], point);
    let gbuui = mul(mul(i[6], i[9], point), i[9], point);
    let gbuuo = mul(mul(o[6], o[9], point), o[9], point);
    let gbvvi = mul(mul(i[6], i[10], point), i[10], point);
    let gbvvo = mul(mul(o[6], o[10], point), o[10], point);
    let gbuvi = mul(mul(i[6], i[9], point), i[10], point);
    let gbuvo = mul(mul(o[6], o[9], point), o[10], point);
    let pav = avg(po, pi, point);
    var pinv = zero; var ponv = zero;
    if (surface > 0u) { pinv = mul(pi, radial(3u * params.ns - 1u + surface - 1u, point), point); }
    if (surface + 1u < params.ns) { ponv = mul(po, radial(3u * params.ns - 1u + surface, point), point); }
    let pw = avg(ponv, pinv, point);
    let guav = avg(gbuuo, gbuui, point);
    let guw = avg(mul(gbuuo, o[14], point), mul(gbuui, i[14], point), point);
    let gvav = avg(gbvvo, gbvvi, point);
    let gvw = avg(mul(gbvvo, o[14], point), mul(gbvvi, i[14], point), point);
    let guvav = avg(gbuvo, gbuvi, point);
    let guvw = avg(mul(gbuvo, o[14], point), mul(gbuvi, i[14], point), point);
    let re = full(0u, point, point); let ro = full(6u, point, point);
    let zo = full(7u, point, point); let rue = full(3u, point, point);
    let ruo = full(9u, point, point); let zue = full(4u, point, point);
    let zuo = full(10u, point, point); let rve = full(12u, point, point);
    let rvo = full(15u, point, point); let zve = full(13u, point, point);
    let zvo = full(16u, point, point);
    let inv_ds = reciprocal(FF(params.delta_s, params.delta_s_lo), point);

    var armne = mul(sub(zupo, zupi, point), inv_ds, point);
    armne = add(armne, avg(taupo, taupi, point), point);
    armne = sub(armne, mul(gvav, re, point), point);
    armne = sub(armne, mul(gvw, ro, point), point);
    var armno = mul(sub(mul(zupo, o[14], point), mul(zupi, i[14], point), point), inv_ds, point);
    armno = sub(armno, mulf(mul(pw, zue, point), 0.5, point), point);
    armno = sub(armno, mulf(mul(pav, zuo, point), 0.5, point), point);
    armno = add(armno, avg(mul(taupo, o[14], point), mul(taupi, i[14], point), point), point);
    armno = sub(armno, mul(gvw, re, point), point);
    armno = sub(armno, mul(mul(gvav, ro, point), s_full, point), point);
    let azmne = neg(mul(sub(rupo, rupi, point), inv_ds, point));
    var azmno = neg(mul(sub(mul(rupo, o[14], point), mul(rupi, i[14], point), point), inv_ds, point));
    azmno = add(azmno, mulf(mul(pw, rue, point), 0.5, point), point);
    azmno = add(azmno, mulf(mul(pav, ruo, point), 0.5, point), point);
    var brmne = add(avg(zspo, zspi, point), mulf(mul(pw, zo, point), 0.5, point), point);
    brmne = sub(brmne, sum4(mul(guav, rue, point), mul(guw, ruo, point), mul(guvav, rve, point), mul(guvw, rvo, point), point), point);
    var brmno = add(avg(mul(zspo, o[14], point), mul(zspi, i[14], point), point), mulf(mul(pav, zo, point), 0.5, point), point);
    brmno = sub(brmno, sum4(mul(guw, rue, point), mul(mul(guav, ruo, point), s_full, point), mul(guvw, rve, point), mul(mul(guvav, rvo, point), s_full, point), point), point);
    var bzmne = neg(add(avg(rspo, rspi, point), mulf(mul(pw, ro, point), 0.5, point), point));
    bzmne = sub(bzmne, sum4(mul(guav, zue, point), mul(guw, zuo, point), mul(guvav, zve, point), mul(guvw, zvo, point), point), point);
    var bzmno = neg(add(avg(mul(rspo, o[14], point), mul(rspi, i[14], point), point), mulf(mul(pav, ro, point), 0.5, point), point));
    bzmno = sub(bzmno, sum4(mul(guw, zue, point), mul(mul(guav, zuo, point), s_full, point), mul(guvw, zve, point), mul(mul(guvav, zvo, point), s_full, point), point), point);
    let crmne = sum4(mul(guvav, rue, point), mul(guvw, ruo, point), mul(gvav, rve, point), mul(gvw, rvo, point), point);
    let crmno = sum4(mul(guvw, rue, point), mul(mul(guvav, ruo, point), s_full, point), mul(gvw, rve, point), mul(mul(gvav, rvo, point), s_full, point), point);
    let czmne = sum4(mul(guvav, zue, point), mul(guvw, zuo, point), mul(gvav, zve, point), mul(gvw, zvo, point), point);
    let czmno = sum4(mul(guvw, zue, point), mul(mul(guvav, zuo, point), s_full, point), mul(gvw, zve, point), mul(mul(gvav, zvo, point), s_full, point), point);

    var gvi = zero; var gvo = zero; var guv_bu_i = zero; var guv_bu_o = zero;
    if (surface > 0u) { gvi = div(i[8], i[6], point); guv_bu_i = mul(i[7], i[9], point); }
    if (surface + 1u < params.ns) { gvo = div(o[8], o[6], point); guv_bu_o = mul(o[7], o[9], point); }
    let lamscale = FF(params.lamscale, params.lamscale_lo);
    let phip = radial(params.ns + half_count + surface, point);
    let lue = add(mul(lamscale, full(5u, point, point), point), phip, point);
    let luo = mul(lamscale, full(11u, point, point), point);
    var alt = mul(avg(gvi, gvo, point), lue, point);
    alt = add(alt, mul(avg(mul(gvi, i[14], point), mul(gvo, o[14], point), point), luo, point), point);
    alt = add(alt, avg(guv_bu_i, guv_bu_o, point), point);
    let blend = mulf(sub(FF(1.0, 0.0), s_full, point), 0.1, point);
    var lambda = add(mul(avg(i[12], o[12], point), sub(FF(1.0, 0.0), blend, point), point), mul(alt, blend, point), point);
    if (surface > 0u) { lambda = neg(mul(lambda, lamscale, point)); }
    var lambda_toroidal = avg(i[11], o[11], point);
    if (surface > 0u) { lambda_toroidal = neg(mul(lambda_toroidal, lamscale, point)); }

    put(0u, point, armne); put(1u, point, armno); put(2u, point, azmne); put(3u, point, azmno);
    put(4u, point, brmne); put(5u, point, brmno); put(6u, point, bzmne); put(7u, point, bzmno);
    put(8u, point, lambda); put(9u, point, mul(lambda, sqrt_f, point));
    put(10u, point, crmne); put(11u, point, crmno); put(12u, point, czmne); put(13u, point, czmno);
    put(14u, point, lambda_toroidal); put(15u, point, mul(lambda_toroidal, sqrt_f, point));
}
