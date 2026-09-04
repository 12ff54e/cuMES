struct Params {
    ns: u32,
    n_z_n_t: u32,
    ntheta: u32,
    nzeta: u32,
    full_points: u32,
    half_points: u32,
    prescribed_current: u32,
    _padding0: u32,
    lamscale: f32,
    lamscale_lo: f32,
    _padding2: u32,
    _padding3: u32,
};

struct FF { hi: f32, lo: f32, };
struct Values { data: array<f32>, };
struct AtomicValues { data: array<atomic<u32>>, };
@group(0) @binding(0) var<storage, read> geometry_hi: Values;
@group(0) @binding(1) var<storage, read> base_hi: Values;
@group(0) @binding(2) var<storage, read> profiles_hi: Values;
// High fields/chip/iota followed by the corresponding low values.
@group(0) @binding(3) var<storage, read_write> field: Values;
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var<storage, read> geometry_lo: Values;
@group(0) @binding(6) var<storage, read> base_lo: Values;
@group(0) @binding(7) var<storage, read> profiles_lo: Values;
@group(0) @binding(8) var<storage, read_write> rounding: AtomicValues;

fn ff_strict_round(value: f32, slot: u32) -> f32 {
    atomicStore(&rounding.data[slot], bitcast<u32>(value));
    return bitcast<f32>(atomicLoad(&rounding.data[slot]));
}
fn ff_qts(a: f32, b: f32, slot: u32) -> FF {
    let sum = ff_strict_round(a + b, slot);
    let sum_minus_a = ff_strict_round(sum - a, slot);
    return FF(sum, ff_strict_round(b - sum_minus_a, slot));
}
fn ff_ts(a: f32, b: f32, slot: u32) -> FF {
    let sum = ff_strict_round(a + b, slot);
    let bv = ff_strict_round(sum - a, slot);
    let av = ff_strict_round(sum - bv, slot);
    let ae = ff_strict_round(a - av, slot);
    let be = ff_strict_round(b - bv, slot);
    return FF(sum, ff_strict_round(ae + be, slot));
}
fn ff_norm(a: FF, slot: u32) -> FF { return ff_qts(a.hi, a.lo, slot); }
fn ff_add(a: FF, b: FF, slot: u32) -> FF {
    let lead = ff_ts(a.hi, b.hi, slot);
    let low0 = ff_strict_round(lead.lo + a.lo, slot);
    return ff_norm(FF(lead.hi, ff_strict_round(low0 + b.lo, slot)), slot);
}
fn ff_sub(a: FF, b: FF, slot: u32) -> FF {
    return ff_add(a, FF(-b.hi, -b.lo), slot);
}
fn ff_mul_f32(a: FF, b: f32, slot: u32) -> FF {
    let product = ff_strict_round(a.hi * b, slot);
    let error = ff_strict_round(fma(a.hi, b, -product), slot);
    let low = ff_strict_round(a.lo * b, slot);
    return ff_norm(FF(product, ff_strict_round(error + low, slot)), slot);
}
fn ff_mul(a: FF, b: FF, slot: u32) -> FF {
    let product = ff_strict_round(a.hi * b.hi, slot);
    let error = ff_strict_round(fma(a.hi, b.hi, -product), slot);
    let cross0 = ff_strict_round(a.hi * b.lo, slot);
    let cross1 = ff_strict_round(a.lo * b.hi, slot);
    let low0 = ff_strict_round(error + cross0, slot);
    return ff_norm(FF(product, ff_strict_round(low0 + cross1, slot)), slot);
}
fn ff_reciprocal(a: FF, slot: u32) -> FF {
    let estimate = ff_strict_round(1.0 / a.hi, slot);
    var inverse = FF(estimate, 0.0);
    var error = ff_sub(FF(1.0, 0.0), ff_mul(a, inverse, slot), slot);
    inverse = ff_add(inverse, ff_mul_f32(error, estimate, slot), slot);
    error = ff_sub(FF(1.0, 0.0), ff_mul(a, inverse, slot), slot);
    return ff_add(inverse, ff_mul(inverse, error, slot), slot);
}
fn ff_div(a: FF, b: FF, slot: u32) -> FF {
    return ff_mul(a, ff_reciprocal(b, slot), slot);
}

fn result_values() -> u32 {
    return 5u * params.half_points + 2u * (params.ns - 1u);
}
fn full(field_index: u32, point: u32, slot: u32) -> FF {
    let index = field_index * params.full_points + point;
    return ff_norm(FF(geometry_hi.data[index], geometry_lo.data[index]), slot);
}
fn half(field_index: u32, point: u32, slot: u32) -> FF {
    let index = field_index * params.half_points + point;
    return ff_norm(FF(base_hi.data[index], base_lo.data[index]), slot);
}
fn profile(index: u32, slot: u32) -> FF {
    return ff_norm(FF(profiles_hi.data[index], profiles_lo.data[index]), slot);
}
fn field_at(field_index: u32, point: u32, slot: u32) -> FF {
    let index = field_index * params.half_points + point;
    return ff_norm(FF(field.data[index], field.data[result_values() + index]),
                   slot);
}
fn store_index(index: u32, value: FF) {
    field.data[index] = value.hi;
    field.data[result_values() + index] = value.lo;
}
fn store(field_index: u32, point: u32, value: FF) {
    store_index(field_index * params.half_points + point, value);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let point = invocation.x;
    if (point >= params.half_points) { return; }
    let surface = point / params.n_z_n_t;
    let angular = point % params.n_z_n_t;
    let inside = surface * params.n_z_n_t + angular;
    let outside = inside + params.n_z_n_t;
    let half_count = params.ns - 1u;
    let sqrt_h = profile(surface, point);
    let lu_h = ff_mul_f32(ff_add(
        ff_add(full(5u, inside, point), full(5u, outside, point), point),
        ff_mul(sqrt_h, ff_add(full(11u, inside, point),
                              full(11u, outside, point), point), point), point),
        0.5, point);
    let lv_h = ff_mul_f32(ff_add(
        ff_add(full(14u, inside, point), full(14u, outside, point), point),
        ff_mul(sqrt_h, ff_add(full(17u, inside, point),
                              full(17u, outside, point), point), point), point),
        0.5, point);
    let phip_offset = half_count;
    let chip_offset = phip_offset + params.ns;
    let pres_offset = chip_offset + half_count;
    let phip_average = ff_mul_f32(ff_add(
        profile(phip_offset + surface, point),
        profile(phip_offset + surface + 1u, point), point), 0.5, point);
    let gsqrt = half(6u, point, point);
    var bsupu = FF(0.0, 0.0);
    var bsupv = FF(0.0, 0.0);
    if (abs(gsqrt.hi) > 1.0e-30 && abs(gsqrt.hi) <= 3.402823e38) {
        let lamscale = FF(params.lamscale, params.lamscale_lo);
        bsupv = ff_div(ff_add(ff_mul(lamscale, lu_h, point), phip_average, point),
                       gsqrt, point);
        bsupu = ff_div(ff_mul(lamscale, lv_h, point), gsqrt, point);
        if (params.prescribed_current == 0u) {
            bsupu = ff_add(bsupu,
                ff_div(profile(chip_offset + surface, point), gsqrt, point), point);
        }
    }
    store(0u, point, bsupu); store(1u, point, bsupv);
    if (params.prescribed_current != 0u) { return; }
    let bsubu = ff_add(ff_mul(half(7u, point, point), bsupu, point),
                       ff_mul(half(8u, point, point), bsupv, point), point);
    let bsubv = ff_add(ff_mul(half(8u, point, point), bsupu, point),
                       ff_mul(half(9u, point, point), bsupv, point), point);
    let pressure = ff_add(ff_mul_f32(ff_add(ff_mul(bsupu, bsubu, point),
        ff_mul(bsupv, bsubv, point), point), 0.5, point),
        profile(pres_offset + surface, point), point);
    store(2u, point, bsubu); store(3u, point, bsubv);
    store(4u, point, pressure);
}

@compute @workgroup_size(64)
fn finalize_current(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let surface = invocation.x;
    let half_count = params.ns - 1u;
    if (surface >= half_count) { return; }
    let field_values = 5u * params.half_points;
    let chip_result = field_values + surface;
    let iota_result = field_values + half_count + surface;
    let phip_offset = half_count;
    let chip_offset = phip_offset + params.ns;
    let pres_offset = chip_offset + half_count;
    let curr_offset = pres_offset + half_count;
    let phip_h_offset = curr_offset + half_count;
    let iota_offset = phip_h_offset + half_count;
    if (params.prescribed_current == 0u) {
        store_index(chip_result, profile(chip_offset + surface, surface));
        store_index(iota_result, profile(iota_offset + surface, surface));
        return;
    }
    let reduced_ntheta = params.ntheta / 2u + 1u;
    let normalization = ff_reciprocal(
        FF(f32(params.nzeta * (reduced_ntheta - 1u)), 0.0), surface);
    let base_point = surface * params.n_z_n_t;
    var jv = FF(0.0, 0.0);
    var average = FF(0.0, 0.0);
    for (var izeta = 0u; izeta < params.nzeta; izeta++) {
        for (var itheta = 0u; itheta < reduced_ntheta; itheta++) {
            let point = base_point + izeta * params.ntheta + itheta;
            var weight = normalization;
            if (itheta == 0u || itheta == reduced_ntheta - 1u) {
                weight = ff_mul_f32(weight, 0.5, surface);
            }
            let gsqrt = half(6u, point, surface);
            let integrand = ff_add(
                ff_mul(half(7u, point, surface), field_at(0u, point, surface), surface),
                ff_mul(half(8u, point, surface), field_at(1u, point, surface), surface), surface);
            jv = ff_add(jv, ff_mul(integrand, weight, surface), surface);
            if (abs(gsqrt.hi) > 1.0e-30 && abs(gsqrt.hi) <= 3.402823e38) {
                average = ff_add(average, ff_mul(
                    ff_div(half(7u, point, surface), gsqrt, surface), weight, surface), surface);
            }
        }
    }
    var chip = FF(0.0, 0.0);
    if (average.hi != 0.0) {
        chip = ff_div(ff_sub(profile(curr_offset + surface, surface), jv, surface),
                      average, surface);
    }
    var iota = profile(iota_offset + surface, surface);
    let phip_h = profile(phip_h_offset + surface, surface);
    if (phip_h.hi != 0.0) { iota = ff_div(chip, phip_h, surface); }
    store_index(chip_result, chip); store_index(iota_result, iota);
    for (var angular = 0u; angular < params.n_z_n_t; angular++) {
        let point = base_point + angular;
        let gsqrt = half(6u, point, surface);
        var bsupu = field_at(0u, point, surface);
        let bsupv = field_at(1u, point, surface);
        if (abs(gsqrt.hi) > 1.0e-30 && abs(gsqrt.hi) <= 3.402823e38) {
            bsupu = ff_add(bsupu, ff_div(chip, gsqrt, surface), surface);
        }
        let bsubu = ff_add(ff_mul(half(7u, point, surface), bsupu, surface),
                           ff_mul(half(8u, point, surface), bsupv, surface), surface);
        let bsubv = ff_add(ff_mul(half(8u, point, surface), bsupu, surface),
                           ff_mul(half(9u, point, surface), bsupv, surface), surface);
        let pressure = ff_add(ff_mul_f32(ff_add(
            ff_mul(bsupu, bsubu, surface), ff_mul(bsupv, bsubv, surface), surface),
            0.5, surface), profile(pres_offset + surface, surface), surface);
        store(0u, point, bsupu); store(2u, point, bsubu);
        store(3u, point, bsubv); store(4u, point, pressure);
    }
}
