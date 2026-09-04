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

struct Values {
    data: array<f32>,
};

@group(0) @binding(0) var<storage, read> fields: Values;
@group(0) @binding(1) var<storage, read> basis: Values;
@group(0) @binding(2) var<storage, read_write> residual: Values;
@group(0) @binding(3) var<uniform> params: Params;

fn field_value(field: u32, surface: u32, angular: u32) -> f32 {
    return fields.data[(field * params.ns + surface) * params.n_z_n_t + angular];
}

fn basis_value(field: u32, mode: u32, angular: u32) -> f32 {
    let mnmax = params.mpol * (params.ntor + 1u);
    return basis.data[(field * mnmax + mode) * params.n_z_n_t + angular];
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
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let mnmax = params.mpol * (params.ntor + 1u);
    let index = invocation.x;
    if (index >= params.ns * mnmax) {
        return;
    }
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
    for (var zeta = 0u; zeta < params.nzeta; zeta++) {
        for (var theta = 0u; theta < theta_reduced; theta++) {
            let angular = zeta * params.ntheta + theta;
            var weight = norm;
            if (theta == 0u || theta + 1u == theta_reduced) {
                weight *= 0.5;
            }
            let cc = weight * basis_value(0u, mode, angular);
            let ss = weight * basis_value(1u, mode, angular);
            let sc = weight * basis_value(2u, mode, angular);
            let cs = weight * basis_value(3u, mode, angular);
            let temp_r = field_value(0u + parity, surface, angular) +
                         xmpq * field_value(16u + parity, surface, angular);
            let temp_z = field_value(2u + parity, surface, angular) +
                         xmpq * field_value(18u + parity, surface, angular);
            let br = field_value(4u + parity, surface, angular);
            let bz = field_value(6u + parity, surface, angular);
            let bl = field_value(8u + parity, surface, angular);
            let cr = field_value(10u + parity, surface, angular);
            let cz = field_value(12u + parity, surface, angular);
            let cl = field_value(14u + parity, surface, angular);
            compensated_add(&sums[0], &corrections[0],
                            temp_r * cc - mf * br * sc + nf * cr * cs);
            compensated_add(&sums[3], &corrections[3],
                            temp_r * ss + mf * br * cs - nf * cr * sc);
            compensated_add(&sums[1], &corrections[1],
                            temp_z * sc + mf * bz * cc + nf * cz * ss);
            compensated_add(&sums[4], &corrections[4],
                            temp_z * cs - mf * bz * ss - nf * cz * cc);
            compensated_add(&sums[2], &corrections[2],
                            mf * bl * cc + nf * cl * ss);
            compensated_add(&sums[5], &corrections[5],
                            -mf * bl * ss - nf * cl * cc);
        }
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
