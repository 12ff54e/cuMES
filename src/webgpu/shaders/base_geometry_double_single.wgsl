struct Params {
    ns: u32,
    n_z_n_t: u32,
    full_points: u32,
    half_points: u32,
    delta_s: f32,
    delta_s_lo: f32,
    _padding1: u32,
    _padding2: u32,
};

struct Values { data: array<f32>, };
struct AtomicValues { data: array<atomic<u32>>, };

@group(0) @binding(0) var<storage, read> geometry_hi: Values;
// sqrt_s_f[ns], sqrt_s_h[ns-1], inverse_sqrt_s_h[ns-1].
@group(0) @binding(1) var<storage, read> radial_hi: Values;
// Ten high planes followed by ten low planes.
@group(0) @binding(2) var<storage, read_write> half: Values;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read> geometry_lo: Values;
@group(0) @binding(5) var<storage, read> radial_lo: Values;
@group(0) @binding(6) var<storage, read_write> rounding: AtomicValues;

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

fn ff_strict_sub(a: FF, b: FF, slot: u32) -> FF {
    return ff_strict_add(a, FF(-b.hi, -b.lo), slot);
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

fn full(field: u32, point: u32, slot: u32) -> FF {
    let index = field * params.full_points + point;
    return ff_strict_normalize(
        FF(geometry_hi.data[index], geometry_lo.data[index]), slot);
}

fn radial(index: u32, slot: u32) -> FF {
    return ff_strict_normalize(
        FF(radial_hi.data[index], radial_lo.data[index]), slot);
}

fn put(field: u32, point: u32, value: FF) {
    let index = field * params.half_points + point;
    half.data[index] = value.hi;
    half.data[10u * params.half_points + index] = value.lo;
}

fn product_sum4(a0: FF, b0: FF, a1: FF, b1: FF,
                a2: FF, b2: FF, a3: FF, b3: FF, slot: u32) -> FF {
    var sum = ff_strict_mul(a0, b0, slot);
    sum = ff_strict_add(sum, ff_strict_mul(a1, b1, slot), slot);
    sum = ff_strict_add(sum, ff_strict_mul(a2, b2, slot), slot);
    return ff_strict_add(sum, ff_strict_mul(a3, b3, slot), slot);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let point = invocation.x;
    if (point >= params.half_points) { return; }
    let surface = point / params.n_z_n_t;
    let angular = point % params.n_z_n_t;
    let inside = surface * params.n_z_n_t + angular;
    let outside = inside + params.n_z_n_t;
    let sqrt_i = radial(surface, point);
    let sqrt_o = radial(surface + 1u, point);
    let sqrt_h = radial(params.ns + surface, point);
    let inverse_sqrt_h = radial(2u * params.ns - 1u + surface, point);
    let sqrt_i_squared = ff_strict_mul(sqrt_i, sqrt_i, point);
    let sqrt_o_squared = ff_strict_mul(sqrt_o, sqrt_o, point);

    let r12 = ff_strict_mul_f32(ff_strict_add(
        ff_strict_add(full(0u, inside, point), full(0u, outside, point), point),
        ff_strict_mul(sqrt_h, ff_strict_add(
            full(6u, inside, point), full(6u, outside, point), point), point),
        point), 0.5, point);
    let ru12 = ff_strict_mul_f32(ff_strict_add(
        ff_strict_add(full(3u, inside, point), full(3u, outside, point), point),
        ff_strict_mul(sqrt_h, ff_strict_add(
            full(9u, inside, point), full(9u, outside, point), point), point),
        point), 0.5, point);
    let zu12 = ff_strict_mul_f32(ff_strict_add(
        ff_strict_add(full(4u, inside, point), full(4u, outside, point), point),
        ff_strict_mul(sqrt_h, ff_strict_add(
            full(10u, inside, point), full(10u, outside, point), point), point),
        point), 0.5, point);
    let inverse_delta_s = f32(params.ns - 1u);
    let rs = ff_strict_mul_f32(ff_strict_add(
        ff_strict_sub(full(0u, outside, point), full(0u, inside, point), point),
        ff_strict_mul(sqrt_h, ff_strict_sub(
            full(6u, outside, point), full(6u, inside, point), point), point),
        point), inverse_delta_s, point);
    let zs = ff_strict_mul_f32(ff_strict_add(
        ff_strict_sub(full(1u, outside, point), full(1u, inside, point), point),
        ff_strict_mul(sqrt_h, ff_strict_sub(
            full(7u, outside, point), full(7u, inside, point), point), point),
        point), inverse_delta_s, point);

    let tau1 = ff_strict_sub(ff_strict_mul(ru12, zs, point),
                             ff_strict_mul(rs, zu12, point), point);
    var tau2 = product_sum4(
        full(9u, outside, point), full(7u, outside, point),
        full(9u, inside, point), full(7u, inside, point),
        FF(-full(10u, outside, point).hi, -full(10u, outside, point).lo),
        full(6u, outside, point),
        FF(-full(10u, inside, point).hi, -full(10u, inside, point).lo),
        full(6u, inside, point), point);
    var tau2_fraction = product_sum4(
        full(3u, outside, point), full(7u, outside, point),
        full(3u, inside, point), full(7u, inside, point),
        FF(-full(4u, outside, point).hi, -full(4u, outside, point).lo),
        full(6u, outside, point),
        FF(-full(4u, inside, point).hi, -full(4u, inside, point).lo),
        full(6u, inside, point), point);
    tau2 = ff_strict_add(
        tau2, ff_strict_mul(tau2_fraction, inverse_sqrt_h, point), point);
    let tau = ff_strict_add(
        tau1, ff_strict_mul_f32(tau2, 0.25, point), point);
    let gsqrt = ff_strict_mul(tau, r12, point);

    var guu_base = product_sum4(
        full(3u, inside, point), full(3u, inside, point),
        full(4u, inside, point), full(4u, inside, point),
        full(3u, outside, point), full(3u, outside, point),
        full(4u, outside, point), full(4u, outside, point), point);
    let guu_odd_i = ff_strict_mul(sqrt_i_squared, product_sum4(
        full(9u, inside, point), full(9u, inside, point),
        full(10u, inside, point), full(10u, inside, point),
        FF(0.0, 0.0), FF(0.0, 0.0), FF(0.0, 0.0), FF(0.0, 0.0), point), point);
    let guu_odd_o = ff_strict_mul(sqrt_o_squared, product_sum4(
        full(9u, outside, point), full(9u, outside, point),
        full(10u, outside, point), full(10u, outside, point),
        FF(0.0, 0.0), FF(0.0, 0.0), FF(0.0, 0.0), FF(0.0, 0.0), point), point);
    guu_base = ff_strict_add(guu_base, ff_strict_add(guu_odd_i, guu_odd_o, point), point);
    let guu_cross = product_sum4(
        full(3u, inside, point), full(9u, inside, point),
        full(4u, inside, point), full(10u, inside, point),
        full(3u, outside, point), full(9u, outside, point),
        full(4u, outside, point), full(10u, outside, point), point);
    let guu = ff_strict_add(ff_strict_mul_f32(guu_base, 0.5, point),
                            ff_strict_mul(sqrt_h, guu_cross, point), point);

    var gvv_base = ff_strict_add(
        ff_strict_mul(full(0u, inside, point), full(0u, inside, point), point),
        ff_strict_mul(full(0u, outside, point), full(0u, outside, point), point), point);
    gvv_base = ff_strict_add(gvv_base, ff_strict_mul(sqrt_i_squared,
        ff_strict_mul(full(6u, inside, point), full(6u, inside, point), point), point), point);
    gvv_base = ff_strict_add(gvv_base, ff_strict_mul(sqrt_o_squared,
        ff_strict_mul(full(6u, outside, point), full(6u, outside, point), point), point), point);
    let gvv_cross0 = ff_strict_add(
        ff_strict_mul(full(0u, inside, point), full(6u, inside, point), point),
        ff_strict_mul(full(0u, outside, point), full(6u, outside, point), point), point);
    var gvv = ff_strict_add(ff_strict_mul_f32(gvv_base, 0.5, point),
                            ff_strict_mul(sqrt_h, gvv_cross0, point), point);

    var guv_base = product_sum4(
        full(3u, inside, point), full(12u, inside, point),
        full(4u, inside, point), full(13u, inside, point),
        full(3u, outside, point), full(12u, outside, point),
        full(4u, outside, point), full(13u, outside, point), point);
    let guv_odd_i = ff_strict_mul(sqrt_i_squared, ff_strict_add(
        ff_strict_mul(full(9u, inside, point), full(15u, inside, point), point),
        ff_strict_mul(full(10u, inside, point), full(16u, inside, point), point), point), point);
    let guv_odd_o = ff_strict_mul(sqrt_o_squared, ff_strict_add(
        ff_strict_mul(full(9u, outside, point), full(15u, outside, point), point),
        ff_strict_mul(full(10u, outside, point), full(16u, outside, point), point), point), point);
    guv_base = ff_strict_add(guv_base, ff_strict_add(guv_odd_i, guv_odd_o, point), point);
    var guv_cross = product_sum4(
        full(3u, inside, point), full(15u, inside, point),
        full(4u, inside, point), full(16u, inside, point),
        full(3u, outside, point), full(15u, outside, point),
        full(4u, outside, point), full(16u, outside, point), point);
    guv_cross = ff_strict_add(guv_cross, product_sum4(
        full(12u, inside, point), full(9u, inside, point),
        full(13u, inside, point), full(10u, inside, point),
        full(12u, outside, point), full(9u, outside, point),
        full(13u, outside, point), full(10u, outside, point), point), point);
    let guv = ff_strict_mul_f32(ff_strict_add(
        guv_base, ff_strict_mul(sqrt_h, guv_cross, point), point), 0.5, point);

    var gvv_toroidal = product_sum4(
        full(12u, inside, point), full(12u, inside, point),
        full(13u, inside, point), full(13u, inside, point),
        full(12u, outside, point), full(12u, outside, point),
        full(13u, outside, point), full(13u, outside, point), point);
    let gvv_odd_i = ff_strict_mul(sqrt_i_squared, ff_strict_add(
        ff_strict_mul(full(15u, inside, point), full(15u, inside, point), point),
        ff_strict_mul(full(16u, inside, point), full(16u, inside, point), point), point), point);
    let gvv_odd_o = ff_strict_mul(sqrt_o_squared, ff_strict_add(
        ff_strict_mul(full(15u, outside, point), full(15u, outside, point), point),
        ff_strict_mul(full(16u, outside, point), full(16u, outside, point), point), point), point);
    gvv_toroidal = ff_strict_add(gvv_toroidal,
        ff_strict_add(gvv_odd_i, gvv_odd_o, point), point);
    let gvv_cross1 = product_sum4(
        full(12u, inside, point), full(15u, inside, point),
        full(13u, inside, point), full(16u, inside, point),
        full(12u, outside, point), full(15u, outside, point),
        full(13u, outside, point), full(16u, outside, point), point);
    gvv = ff_strict_add(gvv, ff_strict_add(
        ff_strict_mul_f32(gvv_toroidal, 0.5, point),
        ff_strict_mul(sqrt_h, gvv_cross1, point), point), point);

    put(0u, point, r12); put(1u, point, ru12); put(2u, point, zu12);
    put(3u, point, rs); put(4u, point, zs); put(5u, point, tau);
    put(6u, point, gsqrt); put(7u, point, guu); put(8u, point, guv);
    put(9u, point, gvv);
}
