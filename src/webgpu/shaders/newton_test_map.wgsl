// Analytic test operators; no equilibrium/solver policy enters this kernel.
struct Params { count: u32, offset: u32, kind: u32, pad: u32, }
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= p.count) { return; }
    var value = source[p.offset + i];
    if (p.kind == 1u) {
        value = (1.0 + 0.125 * f32(i % 7u)) * value +
            0.08 * source[p.offset + (i + 1u) % p.count];
    }
    if (p.kind == 2u) { value = 0.0; }
    if (p.kind == 3u) { value = bitcast<f32>(0x7fc00000u | p.pad); }
    output[i] = value;
}
