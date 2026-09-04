struct Params {
    ns: u32,
    n_z_n_t: u32,
    full_points: u32,
    half_points: u32,
    delta_s: f32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
};

struct Values {
    data: array<f32>,
};

@group(0) @binding(0) var<storage, read> geometry: Values;
// sqrt_s_f[ns] followed by sqrt_s_h[ns-1].
@group(0) @binding(1) var<storage, read> radial: Values;
@group(0) @binding(2) var<storage, read_write> half: Values;
@group(0) @binding(3) var<uniform> params: Params;

fn full(field: u32, point: u32) -> f32 {
    return geometry.data[field * params.full_points + point];
}

fn store(field: u32, point: u32, value: f32) {
    half.data[field * params.half_points + point] = value;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let point = invocation.x;
    if (point >= params.half_points) {
        return;
    }
    let surface = point / params.n_z_n_t;
    let angular = point % params.n_z_n_t;
    let inside = surface * params.n_z_n_t + angular;
    let outside = inside + params.n_z_n_t;
    let sqrt_h = radial.data[params.ns + surface];
    let sqrt_i = radial.data[surface];
    let sqrt_o = radial.data[surface + 1u];

    let r12 = 0.5 * ((full(0u, inside) + full(0u, outside)) +
                     sqrt_h * (full(6u, inside) + full(6u, outside)));
    let ru12 = 0.5 * ((full(3u, inside) + full(3u, outside)) +
                      sqrt_h * (full(9u, inside) + full(9u, outside)));
    let zu12 = 0.5 * ((full(4u, inside) + full(4u, outside)) +
                      sqrt_h * (full(10u, inside) + full(10u, outside)));
    let rs = ((full(0u, outside) - full(0u, inside)) +
              sqrt_h * (full(6u, outside) - full(6u, inside))) /
             params.delta_s;
    let zs = ((full(1u, outside) - full(1u, inside)) +
              sqrt_h * (full(7u, outside) - full(7u, inside))) /
             params.delta_s;

    let tau1 = ru12 * zs - rs * zu12;
    let tau2 = full(9u, outside) * full(7u, outside) +
               full(9u, inside) * full(7u, inside) -
               full(10u, outside) * full(6u, outside) -
               full(10u, inside) * full(6u, inside) +
               (full(3u, outside) * full(7u, outside) +
                full(3u, inside) * full(7u, inside) -
                full(4u, outside) * full(6u, outside) -
                full(4u, inside) * full(6u, inside)) /
                   sqrt_h;
    let tau = tau1 + 0.25 * tau2;
    let gsqrt = tau * r12;
    let sqrt_i_squared = sqrt_i * sqrt_i;
    let sqrt_o_squared = sqrt_o * sqrt_o;

    let guu =
        0.5 *
            (full(3u, inside) * full(3u, inside) +
             full(4u, inside) * full(4u, inside) +
             full(3u, outside) * full(3u, outside) +
             full(4u, outside) * full(4u, outside) +
             sqrt_i_squared *
                 (full(9u, inside) * full(9u, inside) +
                  full(10u, inside) * full(10u, inside)) +
             sqrt_o_squared *
                 (full(9u, outside) * full(9u, outside) +
                  full(10u, outside) * full(10u, outside))) +
        sqrt_h *
            (full(3u, inside) * full(9u, inside) +
             full(4u, inside) * full(10u, inside) +
             full(3u, outside) * full(9u, outside) +
             full(4u, outside) * full(10u, outside));

    var gvv =
        0.5 *
            (full(0u, inside) * full(0u, inside) +
             full(0u, outside) * full(0u, outside) +
             sqrt_i_squared * full(6u, inside) * full(6u, inside) +
             sqrt_o_squared * full(6u, outside) * full(6u, outside)) +
        sqrt_h *
            (full(0u, inside) * full(6u, inside) +
             full(0u, outside) * full(6u, outside));

    let guv =
        0.5 *
        (full(3u, inside) * full(12u, inside) +
         full(4u, inside) * full(13u, inside) +
         full(3u, outside) * full(12u, outside) +
         full(4u, outside) * full(13u, outside) +
         sqrt_i_squared *
             (full(9u, inside) * full(15u, inside) +
              full(10u, inside) * full(16u, inside)) +
         sqrt_o_squared *
             (full(9u, outside) * full(15u, outside) +
              full(10u, outside) * full(16u, outside)) +
         sqrt_h *
             (full(3u, inside) * full(15u, inside) +
              full(4u, inside) * full(16u, inside) +
              full(3u, outside) * full(15u, outside) +
              full(4u, outside) * full(16u, outside) +
              full(12u, inside) * full(9u, inside) +
              full(13u, inside) * full(10u, inside) +
              full(12u, outside) * full(9u, outside) +
              full(13u, outside) * full(10u, outside)));
    gvv +=
        0.5 *
            (full(12u, inside) * full(12u, inside) +
             full(13u, inside) * full(13u, inside) +
             full(12u, outside) * full(12u, outside) +
             full(13u, outside) * full(13u, outside) +
             sqrt_i_squared *
                 (full(15u, inside) * full(15u, inside) +
                  full(16u, inside) * full(16u, inside)) +
             sqrt_o_squared *
                 (full(15u, outside) * full(15u, outside) +
                  full(16u, outside) * full(16u, outside))) +
        sqrt_h *
            (full(12u, inside) * full(15u, inside) +
             full(13u, inside) * full(16u, inside) +
             full(12u, outside) * full(15u, outside) +
             full(13u, outside) * full(16u, outside));

    store(0u, point, r12);
    store(1u, point, ru12);
    store(2u, point, zu12);
    store(3u, point, rs);
    store(4u, point, zs);
    store(5u, point, tau);
    store(6u, point, gsqrt);
    store(7u, point, guu);
    store(8u, point, guv);
    store(9u, point, gvv);
}
