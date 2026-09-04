struct Params {
    ns: u32,
    mpol: u32,
    points: u32,
    move_lcfs: u32,
    delta_t: f32,
    damping_b1: f32,
    damping_fac: f32,
    _padding0: f32,
};
struct Values { data: array<f32>, };
@group(0) @binding(0) var<storage, read> state: Values;
@group(0) @binding(1) var<storage, read> velocity: Values;
@group(0) @binding(2) var<storage, read> residual: Values;
// Updated state[6*points], then velocity[6*points].
@group(0) @binding(3) var<storage, read_write> output: Values;
@group(0) @binding(4) var<uniform> params: Params;

fn index(component: u32, point: u32) -> u32 {
    return component * params.points + point;
}
fn update_velocity(component: u32, point: u32) -> f32 {
    let i = index(component, point);
    return params.damping_fac *
           (params.damping_b1 * velocity.data[i] +
            params.delta_t * residual.data[i]);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let point = invocation.x;
    if (point >= params.points) { return; }
    for (var component = 0u; component < 6u; component++) {
        let i = index(component, point);
        output.data[i] = state.data[i];
        output.data[6u * params.points + i] = velocity.data[i];
    }
    let mode = point / params.ns;
    let surface = point % params.ns;
    if (surface == 0u && mode > 0u) { return; }
    let basis_scale = select(1.4142135623730951, 1.0, mode == 0u);
    let j_max = select(params.ns - 1u, params.ns, params.move_lcfs != 0u);
    if (surface < j_max) {
        let vr = update_velocity(0u, point);
        let vz = update_velocity(1u, point);
        let vrs = update_velocity(3u, point);
        let vzc = update_velocity(4u, point);
        output.data[6u * params.points + index(0u, point)] = vr;
        output.data[6u * params.points + index(1u, point)] = vz;
        output.data[6u * params.points + index(3u, point)] = vrs;
        output.data[6u * params.points + index(4u, point)] = vzc;
        output.data[index(0u, point)] += params.delta_t * vr * basis_scale;
        output.data[index(1u, point)] += params.delta_t * vz * basis_scale;
        if (mode == 1u) {
            output.data[index(3u, point)] +=
                params.delta_t * (vrs + vzc) * basis_scale;
            output.data[index(4u, point)] +=
                params.delta_t * (vrs - vzc) * basis_scale;
        } else {
            output.data[index(3u, point)] +=
                params.delta_t * vrs * basis_scale;
            output.data[index(4u, point)] +=
                params.delta_t * vzc * basis_scale;
        }
    }
    let vl = update_velocity(2u, point);
    let vlc = update_velocity(5u, point);
    output.data[6u * params.points + index(2u, point)] = vl;
    output.data[6u * params.points + index(5u, point)] = vlc;
    output.data[index(2u, point)] += params.delta_t * vl * basis_scale;
    output.data[index(5u, point)] += params.delta_t * vlc * basis_scale;
}
