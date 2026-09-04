const MAX_SURFACES: u32 = 512u;
struct Params {
    ns: u32,
    mode_count: u32,
    ntor: u32,
    points: u32,
    last_surface: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
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
fn scratch_index(bank: u32, field: u32, mode: u32, row: u32) -> u32 {
    return 6u * params.points + params.mode_count +
           (bank * 5u + field) * params.points + mode * params.ns + row;
}
fn scratch_at(bank: u32, field: u32, mode: u32, row: u32) -> f32 {
    return output.data[scratch_index(bank, field, mode, row)];
}
fn put_scratch(bank: u32, field: u32, mode: u32, row: u32, value: f32) {
    output.data[scratch_index(bank, field, mode, row)] = value;
}
fn solve_pair(mode: u32, z_system: bool, floor: f32) -> bool {
    let m = mode / (params.ntor + 1u);
    let first = select(1u, 0u, m == 0u);
    let count = params.last_surface - first;
    if (count == 0u) { return false; }
    let matrix_offset = select(0u, 3u, z_system);
    let component0 = select(0u, 1u, z_system);
    let component1 = component0 + 3u;
    var broke = false;
    for (var row = 0u; row < count; row++) {
        let surface = first + row;
        put_scratch(0u, 0u, mode, row,
                    matrix_at(matrix_offset + 2u, mode, surface));
        put_scratch(0u, 1u, mode, row,
                    matrix_at(matrix_offset + 1u, mode, surface));
        put_scratch(0u, 2u, mode, row,
                    matrix_at(matrix_offset, mode, surface));
        put_scratch(0u, 3u, mode, row,
                    residual_at(component0, mode, surface));
        put_scratch(0u, 4u, mode, row,
                    residual_at(component1, mode, surface));
    }
    var current_bank = 0u;
    for (var stride = 1u; stride <= count; stride *= 2u) {
        let next_bank = 1u - current_bank;
        for (var row = 0u; row < count; row++) {
            let has_lower = row >= stride;
            let has_upper = row + stride < count;
            var lower_diagonal = 0.0;
            var upper_diagonal = 0.0;
            var neighbor_lower = 0.0;
            var neighbor_lower_upper = 0.0;
            var neighbor_upper_lower = 0.0;
            var neighbor_upper = 0.0;
            var lower_rhs0 = 0.0;
            var upper_rhs0 = 0.0;
            var lower_rhs1 = 0.0;
            var upper_rhs1 = 0.0;
            if (has_lower) {
                lower_diagonal = scratch_at(
                    current_bank, 1u, mode, row - stride);
                neighbor_lower = scratch_at(
                    current_bank, 0u, mode, row - stride);
                neighbor_lower_upper = scratch_at(
                    current_bank, 2u, mode, row - stride);
                lower_rhs0 = scratch_at(
                    current_bank, 3u, mode, row - stride);
                lower_rhs1 = scratch_at(
                    current_bank, 4u, mode, row - stride);
            }
            if (has_upper) {
                upper_diagonal = scratch_at(
                    current_bank, 1u, mode, row + stride);
                neighbor_upper_lower = scratch_at(
                    current_bank, 0u, mode, row + stride);
                neighbor_upper = scratch_at(
                    current_bank, 2u, mode, row + stride);
                upper_rhs0 = scratch_at(
                    current_bank, 3u, mode, row + stride);
                upper_rhs1 = scratch_at(
                    current_bank, 4u, mode, row + stride);
            }
            if (has_lower && abs(lower_diagonal) < floor) { broke = true; }
            if (has_upper && abs(upper_diagonal) < floor) { broke = true; }
            lower_diagonal = select(
                1.0, guarded_pivot(lower_diagonal, floor), has_lower);
            upper_diagonal = select(
                1.0, guarded_pivot(upper_diagonal, floor), has_upper);
            let inverse_lower = select(0.0, 1.0 / lower_diagonal, has_lower);
            let inverse_upper = select(0.0, 1.0 / upper_diagonal, has_upper);
            let row_lower = scratch_at(current_bank, 0u, mode, row);
            let row_diagonal = scratch_at(current_bank, 1u, mode, row);
            let row_upper = scratch_at(current_bank, 2u, mode, row);
            put_scratch(next_bank, 0u, mode, row,
                        -row_lower * neighbor_lower * inverse_lower);
            put_scratch(next_bank, 2u, mode, row,
                        -row_upper * neighbor_upper * inverse_upper);
            var reduced_diagonal =
                row_diagonal - row_lower * neighbor_lower_upper * inverse_lower -
                row_upper * neighbor_upper_lower * inverse_upper;
            if (abs(reduced_diagonal) < floor) { broke = true; }
            reduced_diagonal = guarded_pivot(reduced_diagonal, floor);
            put_scratch(next_bank, 1u, mode, row, reduced_diagonal);
            put_scratch(next_bank, 3u, mode, row,
                        scratch_at(current_bank, 3u, mode, row) -
                        row_lower * lower_rhs0 * inverse_lower -
                        row_upper * upper_rhs0 * inverse_upper);
            put_scratch(next_bank, 4u, mode, row,
                        scratch_at(current_bank, 4u, mode, row) -
                        row_lower * lower_rhs1 * inverse_lower -
                        row_upper * upper_rhs1 * inverse_upper);
        }
        current_bank = next_bank;
    }
    for (var row = 0u; row < count; row++) {
        var pivot = scratch_at(current_bank, 1u, mode, row);
        if (abs(pivot) < floor) { broke = true; }
        pivot = guarded_pivot(pivot, floor);
        let surface = first + row;
        put_residual(component0, mode, surface,
                     scratch_at(current_bank, 3u, mode, row) / pivot);
        put_residual(component1, mode, surface,
                     scratch_at(current_bank, 4u, mode, row) / pivot);
    }
    return broke;
}

@compute @workgroup_size(1)
fn main(@builtin(workgroup_id) workgroup: vec3<u32>) {
    let mode = workgroup.x;
    if (mode >= params.mode_count || params.ns > MAX_SURFACES) { return; }
    let m = mode / (params.ntor + 1u);
    for (var component = 0u; component < 6u; component++) {
        for (var surface = 0u; surface < params.ns; surface++) {
            let index = component * params.points + mode * params.ns + surface;
            output.data[index] = input_residual.data[index];
        }
    }
    if (m == 1u) {
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
    let first = select(1u, 0u, m == 0u);
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
