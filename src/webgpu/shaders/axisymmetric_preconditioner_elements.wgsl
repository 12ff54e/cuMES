struct Params {
    ns: u32,
    n_z_n_t: u32,
    full_points: u32,
    half_points: u32,
    delta_s: f32,
    free_boundary: u32,
    _padding0: u32,
    _padding1: u32,
};
struct Values { data: array<f32>, };
@group(0) @binding(0) var<storage, read> geometry: Values;
@group(0) @binding(1) var<storage, read> base_geometry: Values;
@group(0) @binding(2) var<storage, read> magnetic_field: Values;
// sqrtS_F[ns], then sqrtS_H[ns-1].
@group(0) @binding(3) var<storage, read> radial: Values;
// ard[2*ns], brd[2*ns], azd[2*ns], bzd[2*ns], cxd[ns], then
// arm/brm/azm/bzm, each [2*(ns-1)].
@group(0) @binding(4) var<storage, read_write> output: Values;
@group(0) @binding(5) var<uniform> params: Params;

struct HalfTerms {
    ar0: f32, ar1: f32, ar2: f32, ar3: f32,
    az0: f32, az1: f32, az2: f32, az3: f32,
    br0: f32, br1: f32, br2: f32,
    bz0: f32, bz1: f32, bz2: f32,
    cx: f32,
};
struct Diagonal {
    ard: f32, brd: f32, azd: f32, bzd: f32, cxd: f32,
};

fn fg(field: u32, point: u32) -> f32 {
    return geometry.data[field * params.full_points + point];
}
fn hg(field: u32, point: u32) -> f32 {
    return base_geometry.data[field * params.half_points + point];
}
fn bf(field: u32, point: u32) -> f32 {
    return magnetic_field.data[field * params.half_points + point];
}
fn half_terms(surface: u32) -> HalfTerms {
    var out: HalfTerms;
    out.ar0 = 0.0; out.ar1 = 0.0; out.ar2 = 0.0; out.ar3 = 0.0;
    out.az0 = 0.0; out.az1 = 0.0; out.az2 = 0.0; out.az3 = 0.0;
    out.br0 = 0.0; out.br1 = 0.0; out.br2 = 0.0;
    out.bz0 = 0.0; out.bz1 = 0.0; out.bz2 = 0.0; out.cx = 0.0;
    let sqrt_h = radial.data[params.ns + surface];
    let inv_sqrt_h = 1.0 / sqrt_h;
    let weight = 1.0 / f32(params.n_z_n_t);
    for (var angular = 0u; angular < params.n_z_n_t; angular++) {
        let half_point = surface * params.n_z_n_t + angular;
        let inner = half_point;
        let outer = half_point + params.n_z_n_t;
        let p_tau = -4.0 * hg(0u, half_point) * bf(4u, half_point) /
                    hg(5u, half_point) * weight;
        let r_t1a = hg(2u, half_point) / params.delta_s;
        let r_t2a = 0.25 * (fg(4u, outer) / sqrt_h + fg(10u, outer)) /
                    sqrt_h;
        let r_t3a = 0.25 * (fg(4u, inner) / sqrt_h + fg(10u, inner)) /
                    sqrt_h;
        out.ar0 += p_tau * r_t1a * r_t1a;
        out.ar1 += p_tau * (r_t1a + r_t2a) * (-r_t1a + r_t3a);
        out.ar2 += p_tau * (r_t1a + r_t2a) * (r_t1a + r_t2a);
        out.ar3 += p_tau * (-r_t1a + r_t3a) * (-r_t1a + r_t3a);
        let r_t1b = 0.5 *
            (hg(4u, half_point) + 0.5 * inv_sqrt_h * fg(7u, outer));
        let r_t2b = 0.5 *
            (hg(4u, half_point) + 0.5 * inv_sqrt_h * fg(7u, inner));
        out.br0 += p_tau * r_t1b * r_t2b;
        out.br1 += p_tau * r_t1b * r_t1b;
        out.br2 += p_tau * r_t2b * r_t2b;

        let z_t1a = hg(1u, half_point) / params.delta_s;
        let z_t2a = 0.25 * (fg(3u, outer) / sqrt_h + fg(9u, outer)) /
                    sqrt_h;
        let z_t3a = 0.25 * (fg(3u, inner) / sqrt_h + fg(9u, inner)) /
                    sqrt_h;
        out.az0 += p_tau * z_t1a * z_t1a;
        out.az1 += p_tau * (z_t1a + z_t2a) * (-z_t1a + z_t3a);
        out.az2 += p_tau * (z_t1a + z_t2a) * (z_t1a + z_t2a);
        out.az3 += p_tau * (-z_t1a + z_t3a) * (-z_t1a + z_t3a);
        let z_t1b = 0.5 *
            (hg(3u, half_point) + 0.5 * inv_sqrt_h * fg(6u, outer));
        let z_t2b = 0.5 *
            (hg(3u, half_point) + 0.5 * inv_sqrt_h * fg(6u, inner));
        out.bz0 += p_tau * z_t1b * z_t2b;
        out.bz1 += p_tau * z_t1b * z_t1b;
        out.bz2 += p_tau * z_t2b * z_t2b;
        let bsupv = bf(1u, half_point);
        out.cx += -bsupv * bsupv * hg(6u, half_point) * weight;
    }
    return out;
}
fn sm(surface: u32) -> f32 {
    return radial.data[params.ns + surface] / radial.data[surface + 1u];
}
fn sp(surface: u32) -> f32 {
    if (surface == 0u) { return sm(surface); }
    return radial.data[params.ns + surface] / radial.data[surface];
}
fn diagonal(surface: u32, parity: u32) -> Diagonal {
    var out: Diagonal;
    out.ard = 0.0; out.brd = 0.0; out.azd = 0.0;
    out.bzd = 0.0; out.cxd = 0.0;
    let has_inner = surface > 0u;
    let has_outer = surface + 1u < params.ns;
    if (has_inner && has_outer) {
        let inner = half_terms(surface - 1u);
        let outer = half_terms(surface);
        if (parity == 0u) {
            out.ard = inner.ar0 + outer.ar0;
            out.brd = inner.br1 + outer.br2;
            out.azd = inner.az0 + outer.az0;
            out.bzd = inner.bz1 + outer.bz2;
        } else {
            out.ard = inner.ar2 * sm(surface - 1u) * sm(surface - 1u) +
                      outer.ar3 * sp(surface) * sp(surface);
            out.brd = inner.br1 * sm(surface - 1u) * sm(surface - 1u) +
                      outer.br2 * sp(surface) * sp(surface);
            out.azd = inner.az2 * sm(surface - 1u) * sm(surface - 1u) +
                      outer.az3 * sp(surface) * sp(surface);
            out.bzd = inner.bz1 * sm(surface - 1u) * sm(surface - 1u) +
                      outer.bz2 * sp(surface) * sp(surface);
        }
        out.cxd = inner.cx + outer.cx;
    } else if (has_outer) {
        let outer = half_terms(surface);
        if (parity == 0u) {
            out.ard = outer.ar0; out.brd = outer.br2;
            out.azd = outer.az0; out.bzd = outer.bz2;
        } else {
            out.ard = outer.ar3 * sp(surface) * sp(surface);
            out.brd = outer.br2 * sp(surface) * sp(surface);
            out.azd = outer.az3 * sp(surface) * sp(surface);
            out.bzd = outer.bz2 * sp(surface) * sp(surface);
        }
        out.cxd = outer.cx;
    } else {
        let inner = half_terms(surface - 1u);
        if (parity == 0u) {
            out.ard = inner.ar0; out.brd = inner.br1;
            out.azd = inner.az0; out.bzd = inner.bz1;
        } else {
            out.ard = inner.ar2 * sm(surface - 1u) * sm(surface - 1u);
            out.brd = inner.br1 * sm(surface - 1u) * sm(surface - 1u);
            out.azd = inner.az2 * sm(surface - 1u) * sm(surface - 1u);
            out.bzd = inner.bz1 * sm(surface - 1u) * sm(surface - 1u);
        }
        out.cxd = inner.cx;
    }
    if (surface + 1u == params.ns && params.free_boundary == 0u) {
        out.ard *= 1.05; out.brd *= 1.05;
        out.azd *= 1.05; out.bzd *= 1.05; out.cxd *= 1.05;
    }
    return out;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let surface = invocation.x;
    if (surface >= params.ns) { return; }
    let even = diagonal(surface, 0u);
    let odd = diagonal(surface, 1u);
    let pair = 2u * surface;
    let pair_count = 2u * params.ns;
    output.data[pair] = even.ard;
    output.data[pair + 1u] = odd.ard;
    output.data[pair_count + pair] = even.brd;
    output.data[pair_count + pair + 1u] = odd.brd;
    output.data[2u * pair_count + pair] = even.azd;
    output.data[2u * pair_count + pair + 1u] = odd.azd;
    output.data[3u * pair_count + pair] = even.bzd;
    output.data[3u * pair_count + pair + 1u] = odd.bzd;
    output.data[4u * pair_count + surface] = even.cxd;
    if (surface + 1u < params.ns) {
        let half = half_terms(surface);
        let smsp = sm(surface) * sp(surface);
        let half_count = 2u * (params.ns - 1u);
        let half_base = 4u * pair_count + params.ns;
        output.data[half_base + 2u * surface] = -half.ar0;
        output.data[half_base + 2u * surface + 1u] = half.ar1 * smsp;
        output.data[half_base + half_count + 2u * surface] = half.br0;
        output.data[half_base + half_count + 2u * surface + 1u] =
            half.br0 * smsp;
        output.data[half_base + 2u * half_count + 2u * surface] = -half.az0;
        output.data[half_base + 2u * half_count + 2u * surface + 1u] =
            half.az1 * smsp;
        output.data[half_base + 3u * half_count + 2u * surface] = half.bz0;
        output.data[half_base + 3u * half_count + 2u * surface + 1u] =
            half.bz0 * smsp;
    }
}
