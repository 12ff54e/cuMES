// Restarted f32 GMRES, following numerics/DeviceGmres's CGS2/Givens policy.
// All vectors and recurrence arrays occupy disjoint regions of one workspace.
struct Params {
    size: u32, capacity: u32, column: u32, pad: u32,
    basis: u32, work: u32, residual: u32, x: u32,
    rhs: u32, h: u32, projection: u32, cosine: u32,
    sine: u32, g: u32, y: u32, control: u32,
    tolerance: f32, pad1: u32, pad2: u32, pad3: u32,
}
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> p: Params;
var<workgroup> sums: array<f32, 256>;
// Control: rhs norm, true residual, target, steps, cycles, active,
// converged, breakdown, cycle_active, cycle_steps, cycle_started, update_ready.
fn finite(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u;
}
fn reduce(value: f32, maximum: bool, lane: u32) -> f32 {
    sums[lane] = value;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride /= 2u) {
        if (lane < stride) {
            if (maximum) { sums[lane] = max(sums[lane], sums[lane + stride]); }
            else { sums[lane] += sums[lane + stride]; }
        }
        workgroupBarrier();
    }
    let result = sums[0];
    workgroupBarrier();
    return result;
}
fn norm(offset: u32, lane: u32) -> f32 {
    var largest = 0.0;
    for (var i = lane; i < p.size; i += 256u) {
        let value = data[offset + i];
        largest = max(largest, select(bitcast<f32>(0x7f800000u | p.pad), abs(value), finite(value)));
    }
    let scale = reduce(largest, true, lane);
    var total = 0.0;
    // Do not branch around a barrier, even when the norm is zero/nonfinite.
    let divisor = select(1.0, scale, scale > 0.0 && finite(scale));
    for (var i = lane; i < p.size; i += 256u) {
        let value = data[offset + i] / divisor;
        total += value * value;
    }
    let sum = reduce(total, false, lane);
    return select(scale, scale * sqrt(sum), scale > 0.0 && finite(scale));
}
@compute @workgroup_size(256)
fn initialize(@builtin(local_invocation_index) lane: u32) {
    for (var i = lane; i < p.size; i += 256u) {
        data[p.x + i] = 0.0;
        data[p.residual + i] = data[p.rhs + i];
    }
    let rhs_norm = norm(p.rhs, lane);
    if (lane == 0u) {
        for (var i = 0u; i < 16u; i++) { data[p.control + i] = 0.0; }
        data[p.control] = rhs_norm;
        data[p.control + 1u] = rhs_norm;
        data[p.control + 2u] = rhs_norm * p.tolerance;
        data[p.control + 5u] = select(0.0, 1.0, finite(rhs_norm) && rhs_norm > 0.0);
        data[p.control + 6u] = select(0.0, 1.0, rhs_norm == 0.0);
        data[p.control + 7u] = select(2.0, 0.0, finite(rhs_norm));
    }
}
@compute @workgroup_size(256)
fn begin_cycle(@builtin(local_invocation_index) lane: u32) {
    let enabled = data[p.control + 5u];
    let beta = data[p.control + 1u];
    for (var i = lane; i < (p.capacity + 1u) * p.size; i += 256u) {
        var value = 0.0;
        if (i < p.size && enabled != 0.0) { value = data[p.residual + i] / beta; }
        data[p.basis + i] = value;
    }
    for (var i = lane; i < (p.capacity + 1u) * p.capacity; i += 256u) { data[p.h + i] = 0.0; }
    for (var i = lane; i <= p.capacity; i += 256u) { data[p.g + i] = 0.0; }
    for (var i = lane; i < p.capacity; i += 256u) { data[p.y + i] = 0.0; }
    storageBarrier();
    if (lane == 0u) {
        data[p.control + 8u] = enabled;
        data[p.control + 9u] = 0.0;
        data[p.control + 10u] = enabled;
        data[p.control + 11u] = 0.0;
        data[p.g] = select(0.0, beta, enabled != 0.0);
    }
}
@compute @workgroup_size(256)
fn project(@builtin(local_invocation_index) lane: u32,
           @builtin(workgroup_id) group: vec3<u32>) {
    let enabled = data[p.control + 8u] != 0.0;
    var dot = 0.0;
    if (enabled) {
        for (var i = lane; i < p.size; i += 256u) {
            dot += data[p.basis + group.x * p.size + i] * data[p.work + i];
        }
    }
    let value = reduce(dot, false, lane);
    if (lane == 0u && enabled) {
        data[p.projection + group.x] = value;
        data[p.h + p.column * (p.capacity + 1u) + group.x] += value;
    }
}
@compute @workgroup_size(256)
fn subtract_projection(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= p.size || data[p.control + 8u] == 0.0) { return; }
    var value = 0.0;
    for (var k = 0u; k <= p.column; k++) {
        value += data[p.basis + k * p.size + i] * data[p.projection + k];
    }
    data[p.work + i] -= value;
}
@compute @workgroup_size(256)
fn arnoldi(@builtin(local_invocation_index) lane: u32) {
    let enabled = data[p.control + 8u] != 0.0;
    let next_norm = norm(p.work, lane);
    if (!enabled) { return; }
    if (!finite(next_norm)) {
        if (lane == 0u) { data[p.control + 7u] = 2.0; data[p.control + 8u] = 0.0; }
        return;
    }
    if (next_norm > 0.0) {
        for (var i = lane; i < p.size; i += 256u) {
            data[p.basis + (p.column + 1u) * p.size + i] = data[p.work + i] / next_norm;
        }
    }
    if (lane != 0u) { return; }
    let h = p.h + p.column * (p.capacity + 1u);
    var reference = next_norm;
    for (var i = 0u; i <= p.column; i++) {
        if (!finite(data[h + i])) { data[p.control + 7u] = 2.0; data[p.control + 8u] = 0.0; return; }
        reference = max(reference, abs(data[h + i]));
    }
    let happy = next_norm <= 1.1920928955078125e-7 * reference;
    data[h + p.column + 1u] = next_norm;
    for (var i = 0u; i < p.column; i++) {
        let a = data[h + i]; let b = data[h + i + 1u];
        data[h + i] = data[p.cosine + i] * a + data[p.sine + i] * b;
        data[h + i + 1u] = -data[p.sine + i] * a + data[p.cosine + i] * b;
    }
    let a = data[h + p.column]; let b = data[h + p.column + 1u];
    let scale = max(abs(a), abs(b));
    var radius = 0.0;
    if (scale > 0.0) { radius = scale * sqrt((a / scale) * (a / scale) + (b / scale) * (b / scale)); }
    if (!finite(radius) || radius == 0.0) {
        data[p.control + 7u] = select(2.0, 1.0, finite(radius));
        data[p.control + 8u] = 0.0;
        return;
    }
    let cosine = a / radius; let sine = b / radius;
    data[p.cosine + p.column] = cosine; data[p.sine + p.column] = sine;
    data[h + p.column] = radius; data[h + p.column + 1u] = 0.0;
    data[p.g + p.column + 1u] = -sine * data[p.g + p.column];
    data[p.g + p.column] *= cosine;
    data[p.control + 3u] += 1.0;
    data[p.control + 9u] = f32(p.column + 1u);
    if (happy || abs(data[p.g + p.column + 1u]) <= data[p.control + 2u]) { data[p.control + 8u] = 0.0; }
}
@compute @workgroup_size(1)
fn backsolve() {
    let count = u32(data[p.control + 9u]);
    if (data[p.control + 10u] == 0.0 || count == 0u) { return; }
    for (var row = count; row > 0u; row--) {
        let i = row - 1u;
        var value = data[p.g + i];
        for (var j = i + 1u; j < count; j++) { value -= data[p.h + j * (p.capacity + 1u) + i] * data[p.y + j]; }
        let diagonal = data[p.h + i * (p.capacity + 1u) + i];
        if (diagonal == 0.0) { data[p.control + 7u] = 1.0; return; }
        data[p.y + i] = value / diagonal;
        if (!finite(data[p.y + i])) { data[p.control + 7u] = 2.0; return; }
    }
    data[p.control + 11u] = 1.0;
}
@compute @workgroup_size(256)
fn update(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= p.size || data[p.control + 11u] == 0.0) { return; }
    var value = 0.0;
    for (var j = 0u; j < u32(data[p.control + 9u]); j++) { value += data[p.basis + j * p.size + i] * data[p.y + j]; }
    data[p.x + i] += value;
}
@compute @workgroup_size(256)
fn residual(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < p.size) { data[p.residual + id.x] = data[p.rhs + id.x] - data[p.work + id.x]; }
}
@compute @workgroup_size(256)
fn finish_cycle(@builtin(local_invocation_index) lane: u32) {
    let value = norm(p.residual, lane);
    if (lane == 0u) {
        data[p.control + 1u] = value;
        data[p.control + 4u] += data[p.control + 10u];
        if (!finite(value)) { data[p.control + 7u] = 2.0; }
        let converged = data[p.control + 7u] == 0.0 && value <= data[p.control + 2u];
        data[p.control + 6u] = select(0.0, 1.0, converged);
        data[p.control + 5u] = select(0.0, 1.0, data[p.control + 7u] == 0.0 && !converged);
    }
}
