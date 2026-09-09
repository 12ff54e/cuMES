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
// Separable cos/sin tables in the same layout as the main transforms.
@group(0) @binding(2) var<storage, read> basis: Values;
// sc and cs coefficients, each [surface][band-m][n].
@group(0) @binding(3) var<storage, read_write> coefficients: Values;
@group(0) @binding(4) var<storage, read_write> g_con: Values;
@group(0) @binding(5) var<uniform> params: Params;
// Reused between analysis [family][surface][theta][n] and synthesis
// [family][surface][band-m][zeta].
@group(0) @binding(6) var<storage, read_write> scratch: Values;

fn theta_basis(sine: bool, m: u32, theta: u32) -> f32 {
    let plane = select(0u, 1u, sine);
    return basis.data[plane * params.mpol * params.ntheta +
                      m * params.ntheta + theta];
}

fn zeta_basis(sine: bool, n: u32, zeta: u32) -> f32 {
    let theta_values = 2u * params.mpol * params.ntheta;
    let plane = select(0u, 1u, sine);
    return basis.data[theta_values +
                      plane * (params.ntor + 1u) * params.nzeta +
                      n * params.nzeta + zeta];
}

fn coefficient_index(family: u32, surface: u32, m1: u32, n: u32) -> u32 {
    let family_stride = params.ns * params.band_modes * (params.ntor + 1u);
    return family * family_stride +
           (surface * params.band_modes + m1) * (params.ntor + 1u) + n;
}

fn analysis_index(family: u32, surface: u32, theta: u32, n: u32) -> u32 {
    let family_stride = params.ns * params.ntheta * (params.ntor + 1u);
    return family * family_stride +
           (surface * params.ntheta + theta) * (params.ntor + 1u) + n;
}

fn synthesis_index(family: u32, surface: u32, m1: u32, zeta: u32) -> u32 {
    let family_stride = params.ns * params.band_modes * params.nzeta;
    return family * family_stride +
           (surface * params.band_modes + m1) * params.nzeta + zeta;
}

fn compensated_add(sum: ptr<function, f32>, correction: ptr<function, f32>,
                   term: f32) {
    let adjusted = term - *correction;
    let next = *sum + adjusted;
    *correction = (next - *sum) - adjusted;
    *sum = next;
}

@compute @workgroup_size(128)
fn toroidal_analyze(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let sequence_count = params.ns * params.ntheta * (params.ntor + 1u);
    let index = invocation.x;
    if (index >= sequence_count) { return; }
    let n = index % (params.ntor + 1u);
    let theta_surface = index / (params.ntor + 1u);
    let theta = theta_surface % params.ntheta;
    let surface = theta_surface / params.ntheta;
    var cosine_sum = 0.0;
    var sine_sum = 0.0;
    var cosine_correction = 0.0;
    var sine_correction = 0.0;
    if (surface != 0u) {
        for (var zeta = 0u; zeta < params.nzeta; zeta++) {
            let angular = zeta * params.ntheta + theta;
            let value = g_con_eff.data[surface * params.n_z_n_t + angular];
            compensated_add(&cosine_sum, &cosine_correction,
                            value * zeta_basis(false, n, zeta));
            compensated_add(&sine_sum, &sine_correction,
                            value * zeta_basis(true, n, zeta));
        }
    }
    scratch.data[analysis_index(0u, surface, theta, n)] = cosine_sum;
    scratch.data[analysis_index(1u, surface, theta, n)] = sine_sum;
}

@compute @workgroup_size(128)
fn poloidal_analyze(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let values = params.ns * params.band_modes * (params.ntor + 1u);
    let index = invocation.x;
    if (index >= values) { return; }
    let n = index % (params.ntor + 1u);
    let radial_mode = index / (params.ntor + 1u);
    let m1 = radial_mode % params.band_modes;
    let surface = radial_mode / params.band_modes;
    let m = m1 + 1u;
    var sum_sc = 0.0;
    var sum_cs = 0.0;
    var correction_sc = 0.0;
    var correction_cs = 0.0;
    if (surface != 0u) {
        for (var theta = 0u; theta < params.ntheta; theta++) {
            compensated_add(
                &sum_sc, &correction_sc,
                scratch.data[analysis_index(0u, surface, theta, n)] *
                    theta_basis(true, m, theta));
            compensated_add(
                &sum_cs, &correction_cs,
                scratch.data[analysis_index(1u, surface, theta, n)] *
                    theta_basis(false, m, theta));
        }
    }
    let norm = select(4.0 / f32(params.n_z_n_t),
                      2.0 / f32(params.n_z_n_t), n == 0u);
    let scale = norm * profiles.data[surface] * profiles.data[params.ns + m];
    coefficients.data[coefficient_index(0u, surface, m1, n)] = scale * sum_sc;
    coefficients.data[coefficient_index(1u, surface, m1, n)] = scale * sum_cs;
}

@compute @workgroup_size(128)
fn toroidal_synthesize(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let sequence_count = params.ns * params.band_modes * params.nzeta;
    let index = invocation.x;
    if (index >= sequence_count) { return; }
    let zeta = index % params.nzeta;
    let radial_mode = index / params.nzeta;
    let m1 = radial_mode % params.band_modes;
    let surface = radial_mode / params.band_modes;
    var sc_sum = 0.0;
    var cs_sum = 0.0;
    var sc_correction = 0.0;
    var cs_correction = 0.0;
    if (surface != 0u) {
        for (var n = 0u; n <= params.ntor; n++) {
            compensated_add(
                &sc_sum, &sc_correction,
                coefficients.data[coefficient_index(0u, surface, m1, n)] *
                    zeta_basis(false, n, zeta));
            compensated_add(
                &cs_sum, &cs_correction,
                coefficients.data[coefficient_index(1u, surface, m1, n)] *
                    zeta_basis(true, n, zeta));
        }
    }
    scratch.data[synthesis_index(0u, surface, m1, zeta)] = sc_sum;
    scratch.data[synthesis_index(1u, surface, m1, zeta)] = cs_sum;
}

@compute @workgroup_size(128)
fn poloidal_synthesize(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let point = invocation.x;
    if (point >= params.points) { return; }
    let surface = point / params.n_z_n_t;
    let angular = point % params.n_z_n_t;
    let theta = angular % params.ntheta;
    let zeta = angular / params.ntheta;
    var value = 0.0;
    var correction = 0.0;
    if (surface != 0u) {
        for (var m1 = 0u; m1 < params.band_modes; m1++) {
            let m = m1 + 1u;
            compensated_add(
                &value, &correction,
                scratch.data[synthesis_index(0u, surface, m1, zeta)] *
                    theta_basis(true, m, theta) +
                scratch.data[synthesis_index(1u, surface, m1, zeta)] *
                    theta_basis(false, m, theta));
        }
    }
    g_con.data[point] = value;
}
