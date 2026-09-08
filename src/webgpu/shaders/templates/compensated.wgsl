// Shared pair-single arithmetic. Preserve every rounding fence and operation order.
struct FF { hi: f32, lo: f32, };
var<workgroup> rounding: array<atomic<u32>, ROUNDING_SLOTS>;

fn compensate_round(value: f32, slot: u32) -> f32 {
    let local_slot = slot % ROUNDING_SLOTSu;
    // RMW on both sides preserves rounding and ordering on Firefox/SPIR-V,
    // including single-invocation workgroups used by the current integral.
    atomicExchange(&rounding[local_slot], bitcast<u32>(value));
    return bitcast<f32>(atomicAdd(&rounding[local_slot], 0u));
}
fn compensate_quick_sum(a: f32, b: f32, slot: u32) -> FF {
    let sum = compensate_round(a + b, slot);
    let sum_minus_a = compensate_round(sum - a, slot);
    return FF(sum, compensate_round(b - sum_minus_a, slot));
}
fn compensate_sum(a: f32, b: f32, slot: u32) -> FF {
    let sum = compensate_round(a + b, slot);
    let bv = compensate_round(sum - a, slot);
    let av = compensate_round(sum - bv, slot);
    let ae = compensate_round(a - av, slot);
    let be = compensate_round(b - bv, slot);
    return FF(sum, compensate_round(ae + be, slot));
}
fn compensate_normalize(a: FF, slot: u32) -> FF { return compensate_quick_sum(a.hi, a.lo, slot); }
fn compensate_add(a: FF, b: FF, slot: u32) -> FF {
    let lead = compensate_sum(a.hi, b.hi, slot);
    let low0 = compensate_round(lead.lo + a.lo, slot);
    return compensate_normalize(FF(lead.hi, compensate_round(low0 + b.lo, slot)), slot);
}
fn compensate_sub(a: FF, b: FF, slot: u32) -> FF {
    return compensate_add(a, FF(-b.hi, -b.lo), slot);
}
fn compensate_scale(a: FF, b: f32, slot: u32) -> FF {
    let product = compensate_round(a.hi * b, slot);
    let error = compensate_round(fma(a.hi, b, -product), slot);
    let low = compensate_round(a.lo * b, slot);
    return compensate_normalize(FF(product, compensate_round(error + low, slot)), slot);
}
fn compensate_mul(a: FF, b: FF, slot: u32) -> FF {
    let product = compensate_round(a.hi * b.hi, slot);
    let error = compensate_round(fma(a.hi, b.hi, -product), slot);
    let cross0 = compensate_round(a.hi * b.lo, slot);
    let cross1 = compensate_round(a.lo * b.hi, slot);
    let low0 = compensate_round(error + cross0, slot);
    return compensate_normalize(FF(product, compensate_round(low0 + cross1, slot)), slot);
}
fn compensate_reciprocal(a: FF, slot: u32) -> FF {
    let estimate = compensate_round(1.0 / a.hi, slot);
    var inverse = FF(estimate, 0.0);
    var error = compensate_sub(FF(1.0, 0.0), compensate_mul(a, inverse, slot), slot);
    inverse = compensate_add(inverse, compensate_scale(error, estimate, slot), slot);
    error = compensate_sub(FF(1.0, 0.0), compensate_mul(a, inverse, slot), slot);
    return compensate_add(inverse, compensate_mul(inverse, error, slot), slot);
}
fn compensate_div(a: FF, b: FF, slot: u32) -> FF {
    return compensate_mul(a, compensate_reciprocal(b, slot), slot);
}

fn compensate_neg(a: FF) -> FF { return FF(-a.hi, -a.lo); }
fn compensate_scalar(a: FF) -> f32 { return a.hi + a.lo; }
// Norms retain the low-low product as well as the two cross products.
fn compensate_square(a: FF, slot: u32) -> FF {
    let hi = compensate_round(a.hi * a.hi, slot);
    let e = compensate_round(fma(a.hi, a.hi, -hi), slot);
    let cross = compensate_round(compensate_round(a.hi * a.lo, slot) * 2.0, slot);
    let lo = compensate_round(compensate_round(e + cross, slot) +
                              compensate_round(a.lo * a.lo, slot), slot);
    return compensate_normalize(FF(hi, lo), slot);
}
fn compensate_greater(a: FF, b: FF) -> bool {
    return a.hi > b.hi || (a.hi == b.hi && a.lo > b.lo);
}
