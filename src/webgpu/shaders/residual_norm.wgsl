// Deterministic paired-f32 sum of squares. R/Z omit the LCFS when requested;
// lambda always includes it. Finite checks include every high/low input word.
struct FF { hi: f32, lo: f32, };
struct Params { points: u32, ns: u32, paired: u32, edge: u32, };
struct Values { data: array<f32>, };
struct Part { sum: FF, bad: u32, nonzero: u32, };
struct Partials { data: array<Part>, };
@group(0) @binding(0) var<storage, read> source_hi: Values;
@group(0) @binding(1) var<storage, read> source_lo: Values;
@group(0) @binding(2) var<storage, read_write> partials: Partials;
@group(0) @binding(3) var<storage, read_write> output: Values;
@group(0) @binding(4) var<uniform> params: Params;
var<workgroup> rounding: array<atomic<u32>, 64>;
var<workgroup> sums: array<FF, 64>;
var<workgroup> bad: array<u32, 64>;
var<workgroup> nonzero: array<u32, 64>;

fn round32(x: f32, lane: u32) -> f32 {
    // RMW on both sides preserves rounding and ordering on Firefox/SPIR-V,
    // including single-invocation workgroups used by the current integral.
    atomicExchange(&rounding[lane], bitcast<u32>(x));
    return bitcast<f32>(atomicAdd(&rounding[lane], 0u));
}
fn normalize(a: f32, b: f32, lane: u32) -> FF {
    let hi = round32(a + b, lane);
    return FF(hi, round32(b - round32(hi - a, lane), lane));
}
fn add(a: FF, b: FF, lane: u32) -> FF {
    let hi = round32(a.hi + b.hi, lane);
    let bv = round32(hi - a.hi, lane);
    let av = round32(hi - bv, lane);
    let e = round32(round32(a.hi - av, lane) + round32(b.hi - bv, lane), lane);
    let lo = round32(round32(e + a.lo, lane) + b.lo, lane);
    return normalize(hi, lo, lane);
}
fn square(a: FF, lane: u32) -> FF {
    let hi = round32(a.hi * a.hi, lane);
    let e = round32(fma(a.hi, a.hi, -hi), lane);
    let cross = round32(round32(a.hi * a.lo, lane) * 2.0, lane);
    let lo = round32(round32(e + cross, lane) + round32(a.lo * a.lo, lane), lane);
    return normalize(hi, lo, lane);
}
fn finite(a: f32) -> bool {
    return (bitcast<u32>(a) & 0x7f800000u) != 0x7f800000u;
}
fn value(i: u32) -> FF {
    return FF(source_hi.data[i], select(0.0, source_lo.data[i], params.paired != 0u));
}
fn reduce(lane: u32) {
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride /= 2u) {
        if (lane < stride) {
            sums[lane] = add(sums[lane], sums[lane + stride], lane);
            bad[lane] |= bad[lane + stride];
            nonzero[lane] |= nonzero[lane + stride];
        }
        workgroupBarrier();
    }
}
@compute @workgroup_size(64)
fn partial(@builtin(local_invocation_index) lane: u32,
           @builtin(workgroup_id) group: vec3<u32>) {
    let blocks = (params.points + 255u) / 256u;
    var sum = FF(0.0, 0.0);
    var invalid = 0u;
    var populated = 0u;
    for (var offset = lane; offset < 256u; offset += 64u) {
        let i = group.x * 256u + offset;
        if (i >= params.points) { continue; }
        let a = value(group.y * params.points + i);
        let b = value((group.y + 3u) * params.points + i);
        let ok = finite(a.hi) && finite(a.lo) && finite(b.hi) && finite(b.lo);
        invalid |= select(1u, 0u, ok);
        if (ok && (group.y == 2u || params.edge != 0u || i % params.ns != params.ns - 1u)) {
            populated |= select(0u, 1u, a.hi != -a.lo || b.hi != -b.lo);
            sum = add(sum, add(square(a, lane), square(b, lane), lane), lane);
        }
    }
    sums[lane] = sum;
    bad[lane] = invalid;
    nonzero[lane] = populated;
    reduce(lane);
    if (lane == 0u) {
        partials.data[group.y * blocks + group.x] = Part(sums[0], bad[0], nonzero[0]);
    }
}
@compute @workgroup_size(64)
fn finalize(@builtin(local_invocation_index) lane: u32,
            @builtin(workgroup_id) group: vec3<u32>) {
    let blocks = (params.points + 255u) / 256u;
    var sum = FF(0.0, 0.0);
    var invalid = 0u;
    var populated = 0u;
    for (var i = lane; i < blocks; i += 64u) {
        let p = partials.data[group.x * blocks + i];
        sum = add(sum, p.sum, lane);
        invalid |= p.bad;
        populated |= p.nonzero;
    }
    sums[lane] = sum;
    bad[lane] = invalid;
    nonzero[lane] = populated;
    reduce(lane);
    if (lane == 0u) {
        let divisor = f32(params.points);
        let hi = round32(sums[0].hi / divisor, lane);
        let remainder = round32(fma(-hi, divisor, sums[0].hi), lane);
        let lo = round32(round32(remainder + sums[0].lo, lane) / divisor, lane);
        let mean = normalize(hi, lo, lane);
        output.data[group.x] = mean.hi;
        output.data[group.x + 3u] = mean.lo;
        // Do not mistake paired-f32 underflow for a zero residual.
        let underflow = nonzero[0] != 0u && mean.hi == 0.0 && mean.lo == 0.0;
        output.data[group.x + 6u] = f32(bad[0] | select(1u, 0u,
            finite(mean.hi) && finite(mean.lo) && !underflow));
    }
}
