struct Params {
    ns_old: u32,
    ns_new: u32,
    mnmax: u32,
    ntorp1: u32,
    interpolation: u32,
    total: u32,
    _padding0: u32,
    _padding1: u32,
};

struct Values {
    data: array<f32>,
};

@group(0) @binding(0) var<storage, read> input_state: Values;
// First `total` elements are state; the second `total` are zero velocity.
@group(0) @binding(1) var<storage, read_write> output: Values;
@group(0) @binding(2) var<uniform> params: Params;

fn scalxc(j: u32, ns: u32) -> f32 {
    let s = f32(j) / f32(ns - 1u);
    let sqrt_s1 = sqrt(1.0 / f32(ns - 1u));
    return 1.0 / max(sqrt(s), sqrt_s1);
}

fn sample(profile: u32, j: u32, odd: bool) -> f32 {
    let offset = profile * params.ns_old + j;
    let value = input_state.data[offset];
    if (odd) {
        return value * scalxc(j, params.ns_old);
    }
    return value;
}

fn regular_sample(profile: u32, j: u32, odd: bool) -> f32 {
    if (odd && j == 0u) {
        return 2.0 * sample(profile, 1u, odd) - sample(profile, 2u, odd);
    }
    return sample(profile, j, odd);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    if (index >= params.total) {
        return;
    }

    let profile = index / params.ns_new;
    let mode = profile % params.mnmax;
    let j_new = index % params.ns_new;
    let m = mode / params.ntorp1;
    let odd = (m & 1u) == 1u;

    let s = f32(j_new) / f32(params.ns_new - 1u);
    let numerator = j_new * (params.ns_old - 1u);
    let j0 = numerator / (params.ns_new - 1u);
    let j1 = min(j0 + 1u, params.ns_old - 1u);
    let t = clamp(s * f32(params.ns_old - 1u) - f32(j0), 0.0, 1.0);

    let y0 = regular_sample(profile, j0, odd);
    let y1 = regular_sample(profile, j1, odd);
    var interpolated = (1.0 - t) * y0 + t * y1;

    if (params.interpolation == 1u && j0 != j1) {
        var ym1 = 2.0 * y0 - y1;
        if (j0 > 0u) {
            ym1 = regular_sample(profile, j0 - 1u, odd);
        }
        var yp2 = 2.0 * y1 - y0;
        if (j1 + 1u < params.ns_old) {
            yp2 = regular_sample(profile, j1 + 1u, odd);
        }
        interpolated = y0 + 0.5 * t *
            (y1 - ym1 + t * (2.0 * ym1 - 5.0 * y0 + 4.0 * y1 - yp2 +
             t * (3.0 * (y0 - y1) + yp2 - ym1)));
    }

    let sqrt_s1_new = sqrt(1.0 / f32(params.ns_new - 1u));
    var value = interpolated;
    if (odd) {
        value *= max(sqrt(s), sqrt_s1_new);
        if (j_new == 0u) {
            value = 0.0;
        }
    }
    output.data[index] = value;
    output.data[params.total + index] = 0.0;
}
