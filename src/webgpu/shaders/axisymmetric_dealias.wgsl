struct Params {
    ns: u32,
    mpol: u32,
    ntheta: u32,
    points: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
    _padding3: u32,
};

struct Values {
    data: array<f32>,
};

@group(0) @binding(0) var<storage, read> g_con_eff: Values;
// tcon[ns] followed by faccon[mpol].
@group(0) @binding(1) var<storage, read> profiles: Values;
// Host-generated sin(m*theta), [mode][theta].
@group(0) @binding(2) var<storage, read> sine_basis: Values;
@group(0) @binding(3) var<storage, read_write> g_con: Values;
@group(0) @binding(4) var<uniform> params: Params;

fn sine(mode: u32, theta: u32) -> f32 {
    return sine_basis.data[mode * params.ntheta + theta];
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let point = invocation.x;
    if (point >= params.points) {
        return;
    }
    let surface = point / params.ntheta;
    let theta = point % params.ntheta;
    if (surface == 0u) {
        g_con.data[point] = 0.0;
        return;
    }

    let norm = 2.0 / f32(params.ntheta);
    var value = 0.0;
    for (var mode = 1u; mode + 1u < params.mpol; mode++) {
        var coefficient = 0.0;
        for (var source_theta = 0u; source_theta < params.ntheta;
             source_theta++) {
            coefficient +=
                g_con_eff.data[surface * params.ntheta + source_theta] *
                sine(mode, source_theta);
        }
        value += norm * profiles.data[surface] *
                 profiles.data[params.ns + mode] * coefficient *
                 sine(mode, theta);
    }
    g_con.data[point] = value;
}
