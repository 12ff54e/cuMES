struct Params {
    ns: u32, n_z_n_t: u32, points: u32, force_fields: u32,
    output_fields: u32, _padding0: u32, _padding1: u32, _padding2: u32,
};
struct FF { hi: f32, lo: f32, };
struct Values { data: array<f32>, };
struct AtomicValues { data: array<atomic<u32>>, };
@group(0) @binding(0) var<storage, read> force_hi: Values;
@group(0) @binding(1) var<storage, read> geometry: Values;
@group(0) @binding(2) var<storage, read> constraint: Values;
@group(0) @binding(3) var<storage, read> sqrt_s_f: Values;
// All output high planes followed by all low planes.
@group(0) @binding(4) var<storage, read_write> output: Values;
@group(0) @binding(5) var<uniform> params: Params;
@group(0) @binding(6) var<storage, read> force_lo: Values;
@group(0) @binding(7) var<storage, read> geometry_lo: Values;
@group(0) @binding(8) var<storage, read> constraint_lo: Values;
@group(0) @binding(9) var<storage, read> sqrt_s_f_lo: Values;
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
    let h = ts(a.hi, b.hi, slot);
    return norm(FF(h.hi, rnd(rnd(h.lo + a.lo, slot) + b.lo, slot)), slot);
}
fn mul(a: FF, b: FF, slot: u32) -> FF {
    let p = rnd(a.hi * b.hi, slot); let e = rnd(fma(a.hi, b.hi, -p), slot);
    return norm(FF(p, rnd(rnd(e + rnd(a.hi * b.lo, slot), slot) +
                               rnd(a.lo * b.hi, slot), slot)), slot);
}
fn sub(a: FF, b: FF, slot: u32) -> FF { return add(a, FF(-b.hi, -b.lo), slot); }
fn con(field: u32, point: u32, slot: u32) -> FF {
    let i = field * params.points + point;
    return norm(FF(constraint.data[i], constraint_lo.data[i]), slot);
}
fn geom(field: u32, point: u32, slot: u32) -> FF {
    let i = field * params.points + point;
    return norm(FF(geometry.data[i], geometry_lo.data[i]), slot);
}
fn force(field: u32, point: u32, slot: u32) -> FF {
    let i = field * params.points + point;
    return norm(FF(force_hi.data[i], force_lo.data[i]), slot);
}
fn put(field: u32, point: u32, value: FF) {
    let i = field * params.points + point;
    let count = params.output_fields * params.points;
    output.data[i] = value.hi; output.data[count + i] = value.lo;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let point = invocation.x; if (point >= params.points) { return; }
    for (var field = 0u; field < params.force_fields; field++) {
        put(field, point, force(field, point, point));
    }
    let surface = point / params.n_z_n_t;
    let offset = params.output_fields - 4u;
    if (surface == 0u) {
        for (var field = offset; field < params.output_fields; field++) {
            put(field, point, FF(0.0, 0.0));
        }
        return;
    }
    let sqrt_s = norm(FF(sqrt_s_f.data[surface], sqrt_s_f_lo.data[surface]), point);
    let dr = sub(con(0u, point, point), con(2u, point, point), point);
    let dz = sub(con(1u, point, point), con(3u, point, point), point);
    let gc = con(4u, point, point);
    let brcon = mul(dr, gc, point); let bzcon = mul(dz, gc, point);
    put(4u, point, add(force(4u, point, point), brcon, point));
    put(5u, point, add(force(5u, point, point), mul(brcon, sqrt_s, point), point));
    put(6u, point, add(force(6u, point, point), bzcon, point));
    put(7u, point, add(force(7u, point, point), mul(bzcon, sqrt_s, point), point));
    let ru = add(geom(3u, point, point), mul(sqrt_s, geom(9u, point, point), point), point);
    let zu = add(geom(4u, point, point), mul(sqrt_s, geom(10u, point, point), point), point);
    let rug = mul(ru, gc, point); let zug = mul(zu, gc, point);
    put(offset + 0u, point, rug);
    put(offset + 1u, point, mul(rug, sqrt_s, point));
    put(offset + 2u, point, zug);
    put(offset + 3u, point, mul(zug, sqrt_s, point));
}
