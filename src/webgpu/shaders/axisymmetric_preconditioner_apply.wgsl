const MAX_SURFACES: u32 = 512u;
struct Params {
    ns: u32,
    mpol: u32,
    points: u32,
    last_surface: u32,
};
struct Values { data: array<f32>, };
// Seven matrix planes followed by scale[mpol].
@group(0) @binding(0) var<storage, read> matrix: Values;
// ard, brd, azd, bzd; only these first 8*ns values are consumed.
@group(0) @binding(1) var<storage, read> elements: Values;
@group(0) @binding(2) var<storage, read> input_residual: Values;
// Six residual planes followed by one breakdown flag per mode.
@group(0) @binding(3) var<storage, read_write> output: Values;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> cprime: array<f32, 512>;
var<workgroup> dprime0: array<f32, 512>;
var<workgroup> dprime1: array<f32, 512>;

fn matrix_at(field: u32, mode: u32, surface: u32) -> f32 {
    return matrix.data[field * params.points + mode * params.ns + surface];
}
fn residual_at(component: u32, mode: u32, surface: u32) -> f32 {
    return output.data[component * params.points + mode * params.ns + surface];
}
fn put_residual(component: u32, mode: u32, surface: u32, value: f32) {
    output.data[component * params.points + mode * params.ns + surface] = value;
}
fn guarded_pivot(value: f32, floor: f32) -> f32 {
    if (abs(value) >= floor) { return value; }
    return select(-floor, floor, value >= 0.0);
}
fn solve_pair(mode: u32, z_system: bool, floor: f32) -> bool {
    let first = select(1u, 0u, mode == 0u);
    let count = params.last_surface - first;
    if (count == 0u) { return false; }
    let matrix_offset = select(0u, 3u, z_system);
    let component0 = select(0u, 1u, z_system);
    let component1 = component0 + 3u;
    var broke = false;
    var denominator = matrix_at(matrix_offset + 1u, mode, first);
    if (abs(denominator) < floor) { broke = true; }
    denominator = guarded_pivot(denominator, floor);
    cprime[0] = matrix_at(matrix_offset, mode, first) / denominator;
    dprime0[0] = residual_at(component0, mode, first) / denominator;
    dprime1[0] = residual_at(component1, mode, first) / denominator;
    for (var i = 1u; i < count; i++) {
        let surface = first + i;
        let lower = matrix_at(matrix_offset + 2u, mode, surface);
        var pivot = matrix_at(matrix_offset + 1u, mode, surface) -
                    lower * cprime[i - 1u];
        if (abs(pivot) < floor) { broke = true; }
        pivot = guarded_pivot(pivot, floor);
        cprime[i] = matrix_at(matrix_offset, mode, surface) / pivot;
        dprime0[i] = (residual_at(component0, mode, surface) -
                      lower * dprime0[i - 1u]) / pivot;
        dprime1[i] = (residual_at(component1, mode, surface) -
                      lower * dprime1[i - 1u]) / pivot;
    }
    put_residual(component0, mode, params.last_surface - 1u,
                 dprime0[count - 1u]);
    put_residual(component1, mode, params.last_surface - 1u,
                 dprime1[count - 1u]);
    for (var reverse = count - 1u; reverse > 0u; reverse--) {
        let i = reverse - 1u;
        let surface = first + i;
        put_residual(component0, mode, surface,
                     dprime0[i] - cprime[i] *
                         residual_at(component0, mode, surface + 1u));
        put_residual(component1, mode, surface,
                     dprime1[i] - cprime[i] *
                         residual_at(component1, mode, surface + 1u));
    }
    return broke;
}

@compute @workgroup_size(1)
fn main(@builtin(workgroup_id) workgroup: vec3<u32>) {
    let mode = workgroup.x;
    if (mode >= params.mpol || params.ns > MAX_SURFACES) { return; }
    for (var component = 0u; component < 6u; component++) {
        for (var surface = 0u; surface < params.ns; surface++) {
            let index = component * params.points + mode * params.ns + surface;
            output.data[index] = input_residual.data[index];
        }
    }
    if (mode == 1u) {
        let pair_count = 2u * params.ns;
        for (var surface = 0u; surface < params.ns; surface++) {
            let odd = 2u * surface + 1u;
            let rsum = elements.data[odd] + elements.data[pair_count + odd];
            let zsum = elements.data[2u * pair_count + odd] +
                       elements.data[3u * pair_count + odd];
            let denominator = rsum + zsum;
            if (abs(denominator) >= 1.0e-30) {
                put_residual(3u, mode, surface,
                             residual_at(3u, mode, surface) *
                                 rsum / denominator);
                put_residual(4u, mode, surface,
                             residual_at(4u, mode, surface) *
                                 zsum / denominator);
            }
        }
    }
    let scale = matrix.data[7u * params.points + mode];
    let relative_floor = 1.1920928955078125e-7;
    let floor = select(relative_floor, relative_floor * scale, scale > 0.0);
    var broke = solve_pair(mode, false, floor);
    broke = solve_pair(mode, true, floor) || broke;
    let first = select(1u, 0u, mode == 0u);
    for (var surface = 0u; surface < first; surface++) {
        for (var component = 0u; component < 5u; component++) {
            put_residual(component, mode, surface, 0.0);
        }
    }
    for (var surface = 0u; surface < params.ns; surface++) {
        let lambda = matrix_at(6u, mode, surface);
        put_residual(2u, mode, surface,
                     residual_at(2u, mode, surface) * lambda);
        put_residual(5u, mode, surface,
                     residual_at(5u, mode, surface) * lambda);
    }
    output.data[6u * params.points + mode] = select(0.0, 1.0, broke);
}
