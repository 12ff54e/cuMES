struct Params {
    ns: u32, mpol: u32, ntheta: u32, nzeta: u32,
    n_z_n_t: u32, points: u32, reset_reference: u32,
    refresh_preconditioner: u32, delta_s: f32, tcon_multiplier: f32,
    _padding0: u32, _padding1: u32,
};
struct FF { hi: f32, lo: f32, };
struct Values { data: array<f32>, };
struct AtomicValues { data: array<atomic<u32>>, };
@group(0) @binding(0) var<storage, read> geometry_hi: Values;
@group(0) @binding(1) var<storage, read> constraint_hi: Values;
@group(0) @binding(2) var<storage, read> radial_hi: Values;
// gConEff, rCon0, zCon0 high; the same three low planes; then tcon[ns].
@group(0) @binding(3) var<storage, read_write> output: Values;
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var<storage, read> geometry_lo: Values;
@group(0) @binding(6) var<storage, read> constraint_lo: Values;
@group(0) @binding(7) var<storage, read> radial_lo: Values;
@group(0) @binding(8) var<storage, read_write> rounding: AtomicValues;

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
fn sub(a: FF, b: FF, slot: u32) -> FF { return add(a, FF(-b.hi, -b.lo), slot); }
fn mul(a: FF, b: FF, slot: u32) -> FF {
    let p = rnd(a.hi * b.hi, slot); let e = rnd(fma(a.hi, b.hi, -p), slot);
    return norm(FF(p, rnd(rnd(e + rnd(a.hi * b.lo, slot), slot) +
                               rnd(a.lo * b.hi, slot), slot)), slot);
}
fn geom(field: u32, point: u32, slot: u32) -> FF {
    let i = field * params.points + point;
    return norm(FF(geometry_hi.data[i], geometry_lo.data[i]), slot);
}
fn con(field: u32, point: u32, slot: u32) -> FF {
    let i = field * params.points + point;
    return norm(FF(constraint_hi.data[i], constraint_lo.data[i]), slot);
}
fn radial(i: u32, slot: u32) -> FF {
    return norm(FF(radial_hi.data[i], radial_lo.data[i]), slot);
}
fn put(field: u32, point: u32, value: FF) {
    output.data[field * params.points + point] = value.hi;
    output.data[(3u + field) * params.points + point] = value.lo;
}
fn reference(field: u32, surface: u32, angular: u32, slot: u32) -> FF {
    let point = surface * params.n_z_n_t + angular;
    if (params.reset_reference != 0u && surface != 0u) {
        let lcfs = (params.ns - 1u) * params.n_z_n_t + angular;
        let sqrt_s = radial(surface, slot);
        return mul(mul(con(field, lcfs, slot), sqrt_s, slot), sqrt_s, slot);
    }
    return con(field + 2u, point, slot);
}
fn geom_high(field: u32, point: u32) -> f32 {
    return geometry_hi.data[field * params.points + point];
}
fn tcon_base(surface: u32) -> f32 {
    let tr = params.ntheta / 2u + 1u;
    let quadrature = 1.0 / f32(params.nzeta * (tr - 1u));
    let sqrt_s = radial_hi.data[surface];
    var ar_n = 0.0; var az_n = 0.0;
    for (var zeta = 0u; zeta < params.nzeta; zeta++) {
        for (var theta = 0u; theta < tr; theta++) {
            var weight = quadrature;
            if (theta == 0u || theta + 1u == tr) { weight *= 0.5; }
            let point = surface * params.n_z_n_t + zeta * params.ntheta + theta;
            let ru = geom_high(3u, point) + sqrt_s * geom_high(9u, point);
            let zu = geom_high(4u, point) + sqrt_s * geom_high(10u, point);
            ar_n += ru * ru * weight; az_n += zu * zu * weight;
        }
    }
    if (ar_n == 0.0) { ar_n = 1.0e-10; }
    if (az_n == 0.0) { az_n = 1.0e-10; }
    let ard = abs(radial_hi.data[2u * params.ns + 2u * surface]);
    let azd = abs(radial_hi.data[4u * params.ns + 2u * surface]);
    return min(ard / ar_n, azd / az_n) * params.tcon_multiplier *
           32.0 * params.delta_s * 32.0 * params.delta_s;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let point = invocation.x; if (point >= params.points) { return; }
    let surface = point / params.n_z_n_t; let angular = point % params.n_z_n_t;
    let r0 = reference(0u, surface, angular, point);
    let z0 = reference(1u, surface, angular, point);
    put(1u, point, r0); put(2u, point, z0);
    if (surface == 0u) {
        put(0u, point, FF(0.0, 0.0));
    } else {
        let sqrt_s = radial(surface, point);
        let ru = add(geom(3u, point, point), mul(sqrt_s, geom(9u, point, point), point), point);
        let zu = add(geom(4u, point, point), mul(sqrt_s, geom(10u, point, point), point), point);
        put(0u, point, add(mul(sub(con(0u, point, point), r0, point), ru, point),
                            mul(sub(con(1u, point, point), z0, point), zu, point), point));
    }
    if (angular == 0u) {
        let tcon_offset = 6u * params.points;
        if (surface == 0u) { output.data[tcon_offset] = 0.0; }
        else if (params.refresh_preconditioner == 0u) {
            output.data[tcon_offset + surface] = radial_hi.data[params.ns + surface];
        } else if (surface + 1u == params.ns) {
            output.data[tcon_offset + surface] = 0.5 * tcon_base(surface - 1u);
        } else { output.data[tcon_offset + surface] = tcon_base(surface); }
    }
}
