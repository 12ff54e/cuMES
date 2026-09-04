struct Params {
    ns: u32,
    mpol: u32,
    ntor: u32,
    ntheta: u32,
    nzeta: u32,
    n_z_n_t: u32,
    band_modes: u32,
    points: u32,
};

struct Values { data: array<f32>, };

@group(0) @binding(0) var<storage, read> g_con_eff: Values;
// tcon[ns], followed by faccon[mpol].
@group(0) @binding(1) var<storage, read> profiles: Values;
// cc, ss, sc, cs for every folded (m,n) mode and angular point.
@group(0) @binding(2) var<storage, read> basis: Values;
// sc and cs coefficients, each [surface][band-m][n].
@group(0) @binding(3) var<storage, read_write> coefficients: Values;
@group(0) @binding(4) var<storage, read_write> g_con: Values;
@group(0) @binding(5) var<uniform> params: Params;

fn basis_value(field: u32, mode: u32, angular: u32) -> f32 {
    let mnmax = params.mpol * (params.ntor + 1u);
    return basis.data[(field * mnmax + mode) * params.n_z_n_t + angular];
}

fn coefficient_index(family: u32, surface: u32, m1: u32, n: u32) -> u32 {
    let family_stride = params.ns * params.band_modes * (params.ntor + 1u);
    return family * family_stride +
           (surface * params.band_modes + m1) * (params.ntor + 1u) + n;
}

fn compensated_add(sum: ptr<function, f32>, correction: ptr<function, f32>,
                   term: f32) {
    let adjusted = term - *correction;
    let next = *sum + adjusted;
    *correction = (next - *sum) - adjusted;
    *sum = next;
}

@compute @workgroup_size(128)
fn analyze(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    let values = params.ns * params.band_modes * (params.ntor + 1u);
    if (index >= values) { return; }
    let n = index % (params.ntor + 1u);
    let radial_mode = index / (params.ntor + 1u);
    let m1 = radial_mode % params.band_modes;
    let surface = radial_mode / params.band_modes;
    let m = m1 + 1u;
    let mode = m * (params.ntor + 1u) + n;
    var sum_sc = 0.0;
    var sum_cs = 0.0;
    var correction_sc = 0.0;
    var correction_cs = 0.0;
    if (surface != 0u) {
        for (var angular = 0u; angular < params.n_z_n_t; angular++) {
            let value = g_con_eff.data[surface * params.n_z_n_t + angular];
            compensated_add(&sum_sc, &correction_sc,
                            value * basis_value(2u, mode, angular));
            compensated_add(&sum_cs, &correction_cs,
                            value * basis_value(3u, mode, angular));
        }
    }
    let norm = select(4.0 / f32(params.n_z_n_t),
                      2.0 / f32(params.n_z_n_t), n == 0u);
    let scale = norm * profiles.data[surface] *
                profiles.data[params.ns + m];
    coefficients.data[coefficient_index(0u, surface, m1, n)] = scale * sum_sc;
    coefficients.data[coefficient_index(1u, surface, m1, n)] = scale * sum_cs;
}

@compute @workgroup_size(128)
fn synthesize(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let point = invocation.x;
    if (point >= params.points) { return; }
    let surface = point / params.n_z_n_t;
    let angular = point % params.n_z_n_t;
    var value = 0.0;
    var correction = 0.0;
    if (surface != 0u) {
        for (var m1 = 0u; m1 < params.band_modes; m1++) {
            let m = m1 + 1u;
            for (var n = 0u; n <= params.ntor; n++) {
                let mode = m * (params.ntor + 1u) + n;
                compensated_add(
                    &value, &correction,
                    coefficients.data[coefficient_index(0u, surface, m1, n)] *
                        basis_value(2u, mode, angular) +
                    coefficients.data[coefficient_index(1u, surface, m1, n)] *
                        basis_value(3u, mode, angular));
            }
        }
    }
    g_con.data[point] = value;
}
