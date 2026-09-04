struct Params {
    ns: u32, mode_count: u32, ntor_plus_one: u32, zero_m1_z: u32,
    _padding0: u32, _padding1: u32, _padding2: u32, _padding3: u32,
};
struct FF { hi: f32, lo: f32, };
struct Values { data: array<f32>, };
struct AtomicValues { data: array<atomic<u32>>, };
@group(0) @binding(0) var<storage, read> input_hi: Values;
@group(0) @binding(1) var<storage, read> sqrt_hi: Values;
// Six high planes followed by six low planes.
@group(0) @binding(2) var<storage, read_write> output: Values;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read> input_lo: Values;
@group(0) @binding(5) var<storage, read> sqrt_lo: Values;
@group(0) @binding(6) var<storage, read_write> rounding: AtomicValues;

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
fn mulf(a: FF, b: f32, slot: u32) -> FF {
    let p = rnd(a.hi * b, slot); let e = rnd(fma(a.hi, b, -p), slot);
    return norm(FF(p, rnd(e + rnd(a.lo * b, slot), slot)), slot);
}
fn mul(a: FF, b: FF, slot: u32) -> FF {
    let p = rnd(a.hi * b.hi, slot); let e = rnd(fma(a.hi, b.hi, -p), slot);
    return norm(FF(p, rnd(rnd(e + rnd(a.hi * b.lo, slot), slot) +
                               rnd(a.lo * b.hi, slot), slot)), slot);
}
fn reciprocal(a: FF, slot: u32) -> FF {
    let e = rnd(1.0 / a.hi, slot); var x = FF(e, 0.0);
    var r = sub(FF(1.0, 0.0), mul(a, x, slot), slot);
    x = add(x, mulf(r, e, slot), slot);
    r = sub(FF(1.0, 0.0), mul(a, x, slot), slot);
    return add(x, mul(x, r, slot), slot);
}
fn at(component: u32, mode: u32, surface: u32, slot: u32) -> FF {
    let i = component * params.ns * params.mode_count + mode * params.ns + surface;
    return norm(FF(input_hi.data[i], input_lo.data[i]), slot);
}
fn put(component: u32, mode: u32, surface: u32, value: FF) {
    let count = 6u * params.ns * params.mode_count;
    let i = component * params.ns * params.mode_count + mode * params.ns + surface;
    output.data[i] = value.hi; output.data[count + i] = value.lo;
}
fn greater(a: FF, b: FF) -> bool {
    return a.hi > b.hi || (a.hi == b.hi && a.lo > b.lo);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    if (index >= params.ns * params.mode_count) { return; }
    let mode = index / params.ns; let surface = index % params.ns;
    let m = mode / params.ntor_plus_one;
    var scale = FF(1.0, 0.0);
    if ((m & 1u) == 1u) {
        let current = norm(FF(sqrt_hi.data[surface], sqrt_lo.data[surface]), index);
        let first = norm(FF(sqrt_hi.data[1u], sqrt_lo.data[1u]), index);
        var chosen = first;
        if (greater(current, first)) { chosen = current; }
        scale = reciprocal(chosen, index);
    }
    for (var component = 0u; component < 6u; component++) {
        put(component, mode, surface,
            mul(at(component, mode, surface, index), scale, index));
    }
    if (m == 1u) {
        let old_r = mul(at(3u, mode, surface, index), scale, index);
        let old_z = mul(at(4u, mode, surface, index), scale, index);
        let inv_sqrt_two = FF(0.7071067690849304, 1.2101617152815436e-8);
        put(3u, mode, surface, mul(add(old_r, old_z, index), inv_sqrt_two, index));
        var mixed_z = mul(sub(old_r, old_z, index), inv_sqrt_two, index);
        if (params.zero_m1_z != 0u) { mixed_z = FF(0.0, 0.0); }
        put(4u, mode, surface, mixed_z);
    }
}
