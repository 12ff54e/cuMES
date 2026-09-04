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
fn con(field: u32, point: u32) -> f32 {
    return constraint.data[field * params.points + point];
}
fn geom(field: u32, point: u32) -> f32 {
    return geometry.data[field * params.points + point];
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
    let sqrt_s = sqrt_s_f.data[surface];
    let dr = con(0u, point) - con(2u, point);
    let dz = con(1u, point) - con(3u, point);
    let gc = con(4u, point);
    let brcon = dr * gc; let bzcon = dz * gc;
    put(4u, point, add(force(4u, point, point), FF(brcon, 0.0), point));
    put(5u, point, add(force(5u, point, point), FF(brcon * sqrt_s, 0.0), point));
    put(6u, point, add(force(6u, point, point), FF(bzcon, 0.0), point));
    put(7u, point, add(force(7u, point, point), FF(bzcon * sqrt_s, 0.0), point));
    let ru = geom(3u, point) + sqrt_s * geom(9u, point);
    let zu = geom(4u, point) + sqrt_s * geom(10u, point);
    put(offset + 0u, point, FF(ru * gc, 0.0));
    put(offset + 1u, point, FF(ru * gc * sqrt_s, 0.0));
    put(offset + 2u, point, FF(zu * gc, 0.0));
    put(offset + 3u, point, FF(zu * gc * sqrt_s, 0.0));
}
