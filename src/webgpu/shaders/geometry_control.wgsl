// GPU geometry guards and oriented-Jacobian restart classification.
// Ambiguous double-rounding comparisons request the original host path.
struct Params { points: u32, high: u32, low: u32, paired: u32,
                axisymmetric: u32, axis_points: u32, threshold: f32, pad: u32, }
struct Stat { minimum: vec2<f32>, maximum: vec2<f32>,
              index: u32, bad: u32, flags: u32, ambiguous_min: u32, }
@group(0) @binding(0) var<storage, read> words: array<u32>;
@group(0) @binding(1) var<storage, read_write> parts: array<Stat>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;
var<workgroup> stats: array<Stat, 64>;
var<workgroup> rounding: array<atomic<u32>, 64>;
fn round32(x: f32, lane: u32) -> f32 {
    // RMW on both sides preserves rounding and ordering on Firefox/SPIR-V,
    // including single-invocation workgroups used by the current integral.
    atomicExchange(&rounding[lane], bitcast<u32>(x));
    return bitcast<f32>(atomicAdd(&rounding[lane], 0u));
}
fn finite(x: u32) -> bool { return (x & 0x7f800000u) != 0x7f800000u; }
fn subnormal(x: u32) -> bool {
    return (x & 0x7f800000u) == 0u && (x & 0x7fffffu) != 0u;
}
fn normalize(a: f32, b: f32, lane: u32) -> vec2<f32> {
    let hi = round32(a + b, lane);
    let bv = round32(hi - a, lane);
    let av = round32(hi - bv, lane);
    let lo = round32(round32(a - av, lane) + round32(b - bv, lane), lane);
    return vec2(hi, lo);
}
fn less(a: vec2<f32>, b: vec2<f32>) -> bool {
    return a.x < b.x || (a.x == b.x && a.y < b.y);
}
fn ambiguous(a: vec2<f32>, b: vec2<f32>) -> bool {
    return a.x == b.x && a.y != b.y &&
           abs(a.y - b.y) <= abs(a.x) * 1.0e-15;
}
fn empty() -> Stat { return Stat(vec2(0.0), vec2(0.0), 0xffffffffu, 0u, 0u, 0u); }
fn merge(a: Stat, b: Stat) -> Stat {
    var r = a;
    r.flags |= b.flags;
    r.bad += b.bad;
    if (b.index == 0xffffffffu) { return r; }
    if (a.index == 0xffffffffu) {
        r.minimum = b.minimum; r.maximum = b.maximum; r.index = b.index;
        r.ambiguous_min = b.ambiguous_min;
        return r;
    }
    let close = ambiguous(a.minimum, b.minimum);
    let tied = all(a.minimum == b.minimum);
    if (less(b.minimum, a.minimum) ||
        (tied && b.index < a.index)) {
        r.minimum = b.minimum; r.index = b.index; r.ambiguous_min = b.ambiguous_min;
    }
    // Only ambiguity in the surviving minimum can change its earliest index.
    // Rounding is monotone, so maximum-value selection needs no tie fallback.
    if (close || tied) {
        r.ambiguous_min |= a.ambiguous_min | b.ambiguous_min | select(0u, 1u, close);
    }
    if (less(a.maximum, b.maximum)) { r.maximum = b.maximum; }
    return r;
}
fn reduce(lane: u32) {
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride /= 2u) {
        if (lane < stride) { stats[lane] = merge(stats[lane], stats[lane + stride]); }
        workgroupBarrier();
    }
}
@compute @workgroup_size(64)
fn partial(@builtin(local_invocation_index) lane: u32,
           @builtin(workgroup_id) group: vec3<u32>) {
    var s = empty();
    for (var j = 0u; j < 4u; j++) {
        let i = group.x * 256u + lane + j * 64u;
        if (i >= p.points) { continue; }
        let hi = words[p.high + 6u * p.points + i];
        var lo = 0u;
        var v = empty();
        if (!finite(hi) || (hi & 0x7fffffffu) == 0u) { v.flags |= 1u; }
        if (p.axisymmetric != 0u &&
            (words[p.high + 8u * p.points + i] & 0x7fffffffu) != 0u) {
            v.flags |= 4u;
        }
        if (p.paired != 0u) {
            lo = words[p.low + 6u * p.points + i];
            for (var f = 0u; f < 10u; f++) {
                if (!finite(words[p.low + f * p.points + i])) { v.flags |= 2u; }
            }
        }
        if (!finite(hi) || !finite(lo)) {
            v.bad = 1u;
        } else {
            if (subnormal(hi) || subnormal(lo)) { v.flags |= 8u; }
            let value = normalize(bitcast<f32>(hi), bitcast<f32>(lo), lane);
            if (!finite(bitcast<u32>(value.x)) || !finite(bitcast<u32>(value.y)) ||
                subnormal(bitcast<u32>(value.x)) || subnormal(bitcast<u32>(value.y))) {
                v.flags |= 8u;
            }
            v.minimum = -value;
            v.maximum = select(value, -value, value.x < 0.0);
            v.index = i;
        }
        s = merge(s, v);
    }
    stats[lane] = s;
    reduce(lane);
    if (lane == 0u) { parts[group.x] = stats[0]; }
}
@compute @workgroup_size(64)
fn finalize(@builtin(local_invocation_index) lane: u32) {
    let blocks = (p.points + 255u) / 256u;
    var s = empty();
    for (var i = lane; i < blocks; i += 64u) { s = merge(s, parts[i]); }
    stats[lane] = s;
    reduce(lane);
    if (lane == 0u) {
        s = stats[0];
        if (s.ambiguous_min != 0u) { s.flags |= 8u; }
        var invalid = s.bad > 0u || s.maximum.x <= 0.0 || s.minimum.x <= 0.0;
        if (!invalid && s.index >= p.axis_points) {
            let ratio = s.minimum.x / s.maximum.x;
            if (ratio >= 0.99999 * p.threshold && ratio <= 1.00001 * p.threshold) {
                s.flags |= 8u;
            }
            invalid = ratio < p.threshold;
        }
        output[0] = s.minimum.x; output[1] = s.minimum.y;
        output[2] = s.maximum.x; output[3] = s.maximum.y;
        output[4] = select(f32(s.index), -1.0, s.index == 0xffffffffu);
        output[5] = f32(s.bad); output[6] = f32(s.flags);
        output[7] = select(0.0, 1.0, invalid);
    }
}
