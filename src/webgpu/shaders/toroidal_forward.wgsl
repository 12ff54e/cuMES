struct Params {
    ns: u32,
    mpol: u32,
    ntor: u32,
    ntheta: u32,
    nzeta: u32,
    nfp: u32,
    n_z_n_t: u32,
    include_lcfs: u32,
};

struct Values { data: array<f32>, };

@group(0) @binding(0) var<storage, read> fields: Values;
@group(0) @binding(1) var<storage, read> basis: Values;
@group(0) @binding(2) var<storage, read_write> residual: Values;
@group(0) @binding(3) var<uniform> params: Params;
// Cosine/sine toroidal projections for all 20 real-space fields.
@group(0) @binding(4) var<storage, read_write> intermediate: Values;

fn field_value(field: u32, surface: u32, theta: u32, zeta: u32) -> f32 {
    let angular = zeta * params.ntheta + theta;
    return fields.data[(field * params.ns + surface) * params.n_z_n_t + angular];
}

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

fn intermediate_index(family: u32, field: u32, surface: u32, theta: u32,
                      n: u32) -> u32 {
    let theta_reduced = params.ntheta / 2u + 1u;
    let plane_size = params.ns * theta_reduced * (params.ntor + 1u);
    return (family * 20u + field) * plane_size +
           ((surface * theta_reduced + theta) * (params.ntor + 1u) + n);
}

fn projected(family: u32, field: u32, surface: u32, theta: u32, n: u32)
    -> f32 {
    return intermediate.data[
        intermediate_index(family, field, surface, theta, n)];
}

fn store(component: u32, mode: u32, surface: u32, value: f32) {
    let mnmax = params.mpol * (params.ntor + 1u);
    residual.data[(component * mnmax + mode) * params.ns + surface] = value;
}

fn compensated_add(sum: ptr<function, f32>, correction: ptr<function, f32>,
                   term: f32) {
    let adjusted = term - *correction;
    let next = *sum + adjusted;
    *correction = (next - *sum) - adjusted;
    *sum = next;
}

@compute @workgroup_size(128)
fn toroidal_stage(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let theta_reduced = params.ntheta / 2u + 1u;
    let sequence_count = 20u * params.ns * theta_reduced * (params.ntor + 1u);
    let index = invocation.x;
    if (index >= sequence_count) { return; }
    let n = index % (params.ntor + 1u);
    let theta_surface_field = index / (params.ntor + 1u);
    let theta = theta_surface_field % theta_reduced;
    let surface_field = theta_surface_field / theta_reduced;
    let surface = surface_field % params.ns;
    let field = surface_field / params.ns;
    var cosine_sum = 0.0;
    var sine_sum = 0.0;
    var cosine_correction = 0.0;
    var sine_correction = 0.0;
    for (var zeta = 0u; zeta < params.nzeta; zeta++) {
        let value = field_value(field, surface, theta, zeta);
        compensated_add(&cosine_sum, &cosine_correction,
                        value * zeta_basis(false, n, zeta));
        compensated_add(&sine_sum, &sine_correction,
                        value * zeta_basis(true, n, zeta));
    }
    intermediate.data[intermediate_index(0u, field, surface, theta, n)] =
        cosine_sum;
    intermediate.data[intermediate_index(1u, field, surface, theta, n)] =
        sine_sum;
}

@compute @workgroup_size(128)
fn poloidal_stage(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let mnmax = params.mpol * (params.ntor + 1u);
    let index = invocation.x;
    if (index >= params.ns * mnmax) { return; }
    let surface = index % params.ns;
    let mode = index / params.ns;
    let m = mode / (params.ntor + 1u);
    let n = mode % (params.ntor + 1u);
    let mf = f32(m);
    let nf = f32(n * params.nfp);
    let parity = select(0u, 1u, (m & 1u) == 1u);
    let xmpq = mf * (mf - 1.0);
    let theta_reduced = params.ntheta / 2u + 1u;
    let norm = 1.0 / (f32(params.nzeta) * f32(theta_reduced - 1u));
    var sums: array<f32, 6>;
    var corrections: array<f32, 6>;
    for (var component = 0u; component < 6u; component++) {
        sums[component] = 0.0;
        corrections[component] = 0.0;
    }
    for (var theta = 0u; theta < theta_reduced; theta++) {
        var weight = norm;
        if (theta == 0u || theta + 1u == theta_reduced) { weight *= 0.5; }
        let cm = theta_basis(false, m, theta);
        let sm = theta_basis(true, m, theta);
        let temp_r_cos = projected(0u, parity, surface, theta, n) +
                         xmpq * projected(0u, 16u + parity, surface, theta, n);
        let temp_r_sin = projected(1u, parity, surface, theta, n) +
                         xmpq * projected(1u, 16u + parity, surface, theta, n);
        let temp_z_cos = projected(0u, 2u + parity, surface, theta, n) +
                         xmpq * projected(0u, 18u + parity, surface, theta, n);
        let temp_z_sin = projected(1u, 2u + parity, surface, theta, n) +
                         xmpq * projected(1u, 18u + parity, surface, theta, n);
        let br_cos = projected(0u, 4u + parity, surface, theta, n);
        let br_sin = projected(1u, 4u + parity, surface, theta, n);
        let bz_cos = projected(0u, 6u + parity, surface, theta, n);
        let bz_sin = projected(1u, 6u + parity, surface, theta, n);
        let bl_cos = projected(0u, 8u + parity, surface, theta, n);
        let bl_sin = projected(1u, 8u + parity, surface, theta, n);
        let cr_cos = projected(0u, 10u + parity, surface, theta, n);
        let cr_sin = projected(1u, 10u + parity, surface, theta, n);
        let cz_cos = projected(0u, 12u + parity, surface, theta, n);
        let cz_sin = projected(1u, 12u + parity, surface, theta, n);
        let cl_cos = projected(0u, 14u + parity, surface, theta, n);
        let cl_sin = projected(1u, 14u + parity, surface, theta, n);
        compensated_add(&sums[0], &corrections[0], weight *
            (temp_r_cos * cm - mf * br_cos * sm + nf * cr_sin * cm));
        compensated_add(&sums[3], &corrections[3], weight *
            (temp_r_sin * sm + mf * br_sin * cm - nf * cr_cos * sm));
        compensated_add(&sums[1], &corrections[1], weight *
            (temp_z_cos * sm + mf * bz_cos * cm + nf * cz_sin * sm));
        compensated_add(&sums[4], &corrections[4], weight *
            (temp_z_sin * cm - mf * bz_sin * sm - nf * cz_cos * cm));
        compensated_add(&sums[2], &corrections[2], weight *
            (mf * bl_cos * cm + nf * cl_sin * sm));
        compensated_add(&sums[5], &corrections[5], weight *
            (-mf * bl_sin * sm - nf * cl_cos * cm));
    }
    let mscale = select(sqrt(2.0), 1.0, m == 0u);
    let nscale = select(sqrt(2.0), 1.0, n == 0u);
    let scale = mscale * nscale;
    for (var component = 0u; component < 6u; component++) {
        var value = scale * sums[component];
        if (surface == 0u) {
            let keep = m == 0u && (component == 0u || component == 4u);
            value = select(0.0, value, keep);
        } else if (surface + 1u == params.ns && params.include_lcfs == 0u) {
            let keep = component == 2u || component == 5u;
            value = select(0.0, value, keep);
        }
        store(component, mode, surface, value);
    }
}
