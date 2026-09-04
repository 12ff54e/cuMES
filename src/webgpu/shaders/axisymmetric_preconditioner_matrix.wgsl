struct Params {
    ns: u32,
    mpol: u32,
    ntor: u32,
    ntheta: u32,
    nzeta: u32,
    n_z_n_t: u32,
    points: u32,
    half_points: u32,
    nfp: u32,
    free_boundary: u32,
    delta_s: f32,
    _padding0: u32,
};
struct Values { data: array<f32>, };
// Element cache emitted by axisymmetric_preconditioner_elements.wgsl.
@group(0) @binding(0) var<storage, read> elements: Values;
@group(0) @binding(1) var<storage, read> base_geometry: Values;
// sqrtS_F[ns], phip_H[ns-1], rmsPhiP scalar.
@group(0) @binding(2) var<storage, read> radial: Values;
// upper/diagonal/lower R, upper/diagonal/lower Z, lambda; then scale[mpol].
@group(0) @binding(3) var<storage, read_write> output: Values;
@group(0) @binding(4) var<uniform> params: Params;

struct MatrixValues {
    upper_r: f32, diagonal_r: f32, lower_r: f32,
    upper_z: f32, diagonal_z: f32, lower_z: f32,
};

fn pair_count() -> u32 { return 2u * params.ns; }
fn half_count() -> u32 { return 2u * (params.ns - 1u); }
fn half_base() -> u32 { return 9u * params.ns; }
fn ard(surface: u32, parity: u32) -> f32 {
    return elements.data[2u * surface + parity];
}
fn brd(surface: u32, parity: u32) -> f32 {
    return elements.data[pair_count() + 2u * surface + parity];
}
fn azd(surface: u32, parity: u32) -> f32 {
    return elements.data[2u * pair_count() + 2u * surface + parity];
}
fn bzd(surface: u32, parity: u32) -> f32 {
    return elements.data[3u * pair_count() + 2u * surface + parity];
}
fn cxd(surface: u32) -> f32 {
    return elements.data[4u * pair_count() + surface];
}
fn arm(surface: u32, parity: u32) -> f32 {
    return elements.data[half_base() + 2u * surface + parity];
}
fn brm(surface: u32, parity: u32) -> f32 {
    return elements.data[half_base() + half_count() + 2u * surface + parity];
}
fn azm(surface: u32, parity: u32) -> f32 {
    return elements.data[half_base() + 2u * half_count() +
                         2u * surface + parity];
}
fn bzm(surface: u32, parity: u32) -> f32 {
    return elements.data[half_base() + 3u * half_count() +
                         2u * surface + parity];
}
fn matrix_values(mode: u32, surface: u32) -> MatrixValues {
    var out: MatrixValues;
    out.upper_r = 0.0; out.lower_r = 0.0;
    out.upper_z = 0.0; out.lower_z = 0.0;
    let m = mode / (params.ntor + 1u);
    let n = mode % (params.ntor + 1u);
    let parity = m & 1u;
    let m2 = f32(m * m);
    let nf = n * params.nfp;
    let n2 = f32(nf * nf);
    if (surface + 1u < params.ns) {
        out.upper_r = -(arm(surface, parity) + brm(surface, parity) * m2);
        out.upper_z = -(azm(surface, parity) + bzm(surface, parity) * m2);
    }
    out.diagonal_r = -(ard(surface, parity) + brd(surface, parity) * m2 +
                       cxd(surface) * n2);
    out.diagonal_z = -(azd(surface, parity) + bzd(surface, parity) * m2 +
                       cxd(surface) * n2);
    if (params.free_boundary != 0u && surface + 1u == params.ns) {
        let pedestal = select(0.1, 0.05, m <= 1u);
        out.diagonal_r *= 1.0 + pedestal;
        out.diagonal_z *= 1.0 + pedestal;
        if (m == 0u && n == 0u) {
            let mult_fact = min(0.25, 0.25 * params.delta_s * 15.0);
            out.diagonal_z *= (1.0 - mult_fact) / 1.05;
        }
    }
    if (surface > 0u) {
        out.lower_r = -(arm(surface - 1u, parity) +
                        brm(surface - 1u, parity) * m2);
        out.lower_z = -(azm(surface - 1u, parity) +
                        bzm(surface - 1u, parity) * m2);
    }
    if (surface == 1u && m == 1u) {
        out.diagonal_r += out.lower_r;
        out.diagonal_z += out.lower_z;
    }
    return out;
}
fn half_lambda(shifted: u32, metric_field: u32) -> f32 {
    if (shifted == 0u || shifted == params.ns) { return 0.0; }
    let half_surface = shifted - 1u;
    let ntheta_red = params.ntheta / 2u + 1u;
    let norm = 1.0 / f32(params.nzeta * (ntheta_red - 1u));
    var sum = 0.0;
    for (var zeta = 0u; zeta < params.nzeta; zeta++) {
        for (var theta = 0u; theta < ntheta_red; theta++) {
            var weight = norm;
            if (theta == 0u || theta + 1u == ntheta_red) { weight *= 0.5; }
            let point = half_surface * params.n_z_n_t +
                        zeta * params.ntheta + theta;
            let gsqrt = base_geometry.data[6u * params.half_points + point];
            let metric =
                base_geometry.data[metric_field * params.half_points + point];
            sum += metric / gsqrt * weight;
        }
    }
    return sum;
}
fn lambda_value(mode: u32, surface: u32) -> f32 {
    let m = mode / (params.ntor + 1u);
    let n = mode % (params.ntor + 1u);
    if (surface == 0u || (m == 0u && n == 0u)) { return 0.0; }
    let rms = radial.data[2u * params.ns - 1u];
    let lamscale = sqrt(max(rms * params.delta_s, 1.0e-30));
    let p_factor = 2.0 / (4.0 * lamscale * lamscale);
    let pwr = min(f32(m * m) / (16.0 * 16.0), 8.0);
    let b_full = 0.5 * (half_lambda(surface + 1u, 7u) +
                        half_lambda(surface, 7u));
    let d_full = 0.5 * (half_lambda(surface + 1u, 8u) +
                        half_lambda(surface, 8u));
    let c_full = 0.5 * (half_lambda(surface + 1u, 9u) +
                        half_lambda(surface, 9u));
    let toroidal = f32(n * params.nfp);
    var faclam = toroidal * toroidal * b_full +
                 2.0 * f32(m) * toroidal * abs(d_full) *
                     select(-1.0, 1.0, b_full >= 0.0) +
                 f32(m * m) * c_full;
    if (faclam == 0.0) { faclam = -1.0e-10; }
    return p_factor / faclam * pow(radial.data[surface], pwr);
}
fn mode_scale(mode: u32) -> f32 {
    var scale = 0.0;
    for (var surface = 0u; surface < params.ns; surface++) {
        let value = matrix_values(mode, surface);
        scale = max(scale, abs(value.upper_r));
        scale = max(scale, abs(value.diagonal_r));
        scale = max(scale, abs(value.lower_r));
        scale = max(scale, abs(value.upper_z));
        scale = max(scale, abs(value.diagonal_z));
        scale = max(scale, abs(value.lower_z));
    }
    return scale;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let index = invocation.x;
    if (index >= params.points) { return; }
    let mode = index / params.ns;
    let surface = index % params.ns;
    let value = matrix_values(mode, surface);
    output.data[index] = value.upper_r;
    output.data[params.points + index] = value.diagonal_r;
    output.data[2u * params.points + index] = value.lower_r;
    output.data[3u * params.points + index] = value.upper_z;
    output.data[4u * params.points + index] = value.diagonal_z;
    output.data[5u * params.points + index] = value.lower_z;
    output.data[6u * params.points + index] = lambda_value(mode, surface);
    if (surface == 0u) {
        output.data[7u * params.points + mode] = mode_scale(mode);
    }
}
