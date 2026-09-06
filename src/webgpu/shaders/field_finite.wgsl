struct Params {
    offset: u32,
    count: u32,
    pad0: u32,
    pad1: u32,
}
@group(0) @binding(0) var<storage, read> words: array<u32>;
@group(0) @binding(1) var<storage, read_write> flags: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
var<workgroup> bad: array<u32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) lane: u32,
        @builtin(workgroup_id) group: vec3<u32>) {
    var invalid = 0u;
    for (var j = 0u; j < 4u; j++) {
        let index = group.x * 256u + lane + j * 64u;
        if (index < params.count) {
            let word = words[params.offset + index];
            let exponent = word & 0x7f800000u;
            invalid |= select(0u, 1u, exponent == 0x7f800000u);
            // Integer magnitude preserves subnormals on flush-to-zero GPUs;
            // both signs of zero remain zero.
            invalid |= select(0u, 2u, (word & 0x7fffffffu) != 0u);
        }
    }
    bad[lane] = invalid;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride /= 2u) {
        if (lane < stride) { bad[lane] |= bad[lane + stride]; }
        workgroupBarrier();
    }
    if (lane == 0u) { flags[group.x] = f32(bad[0]); }
}
