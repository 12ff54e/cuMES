struct Params {
    ns: u32,
    mode_count: u32,
    ntor_plus_one: u32,
    zero_m1_z: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
    _padding3: u32,
};
struct Values { data: array<f32>, };
@group(0) @binding(0) var<storage, read> input_residual: Values;
@group(0) @binding(1) var<storage, read> sqrt_s_f: Values;
@group(0) @binding(2) var<storage, read_write> output_residual: Values;
@group(0) @binding(3) var<uniform> params: Params;

fn at(component: u32, mode: u32, surface: u32) -> f32 {
    return input_residual.data[component * params.ns * params.mode_count +
                               mode * params.ns + surface];
}
fn put(component: u32, mode: u32, surface: u32, value: f32) {
    output_residual.data[component * params.ns * params.mode_count +
                         mode * params.ns + surface] = value;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    let values_per_component = params.ns * params.mode_count;
    if (index >= values_per_component) { return; }
    let mode = index / params.ns;
    let surface = index % params.ns;
    let m = mode / params.ntor_plus_one;
    let scale = select(1.0, 1.0 / max(sqrt_s_f.data[surface],
                                      sqrt_s_f.data[1u]),
                       (m & 1u) == 1u);
    for (var component = 0u; component < 6u; component++) {
        put(component, mode, surface, at(component, mode, surface) * scale);
    }
    if (m == 1u) {
        let old_rss = at(3u, mode, surface) * scale;
        let old_zcs = at(4u, mode, surface) * scale;
        put(3u, mode, surface, (old_rss + old_zcs) * 0.7071067811865476);
        let mixed_z = (old_rss - old_zcs) * 0.7071067811865476;
        put(4u, mode, surface, select(mixed_z, 0.0,
                                     params.zero_m1_z != 0u));
    }
}
