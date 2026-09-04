struct Params {
    ns: u32,
    ntor_plus_one: u32,
    points: u32,
    move_lcfs: u32,
    delta_t: f32,
    damping_b1: f32,
    damping_fac: f32,
    _padding0: f32,
};
struct Values { data: array<f32>, };
struct AtomicValues { data: array<atomic<u32>>, };
@group(0) @binding(0) var<storage, read> state_hi: Values;
@group(0) @binding(1) var<storage, read> state_lo: Values;
@group(0) @binding(2) var<storage, read> velocity_hi: Values;
@group(0) @binding(3) var<storage, read> velocity_lo: Values;
@group(0) @binding(4) var<storage, read> residual: Values;
// state_hi, state_lo, velocity_hi, velocity_lo, each with 6*points values.
@group(0) @binding(5) var<storage, read_write> output: Values;
@group(0) @binding(6) var<uniform> params: Params;
// One temporary per spectral point forces binary32 rounding at the boundaries
// of error-free transforms. The u32 atomic round trip prevents WebGPU shader
// compilers from reassociating the cancellation expressions.
@group(0) @binding(7) var<storage, read_write> rounding: AtomicValues;
@group(0) @binding(8) var<storage, read> residual_lo: Values;

fn ff_strict_round(value: f32, slot: u32) -> f32 {
    atomicStore(&rounding.data[slot], bitcast<u32>(value));
    return bitcast<f32>(atomicLoad(&rounding.data[slot]));
}

fn ff_strict_quick_two_sum(a: f32, b: f32, slot: u32) -> FF {
    let sum = ff_strict_round(a + b, slot);
    let sum_minus_a = ff_strict_round(sum - a, slot);
    return FF(sum, ff_strict_round(b - sum_minus_a, slot));
}

fn ff_strict_two_sum(a: f32, b: f32, slot: u32) -> FF {
    let sum = ff_strict_round(a + b, slot);
    let b_virtual = ff_strict_round(sum - a, slot);
    let a_virtual = ff_strict_round(sum - b_virtual, slot);
    let a_error = ff_strict_round(a - a_virtual, slot);
    let b_error = ff_strict_round(b - b_virtual, slot);
    return FF(sum, ff_strict_round(a_error + b_error, slot));
}

fn ff_strict_normalize(value: FF, slot: u32) -> FF {
    return ff_strict_quick_two_sum(value.hi, value.lo, slot);
}

fn ff_strict_add(a: FF, b: FF, slot: u32) -> FF {
    let leading = ff_strict_two_sum(a.hi, b.hi, slot);
    let low = ff_strict_round(leading.lo + a.lo, slot);
    return ff_strict_normalize(
        FF(leading.hi, ff_strict_round(low + b.lo, slot)), slot);
}

fn ff_strict_mul_f32(a: FF, b: f32, slot: u32) -> FF {
    let product = ff_strict_round(a.hi * b, slot);
    let leading_error = ff_strict_round(fma(a.hi, b, -product), slot);
    let low_product = ff_strict_round(a.lo * b, slot);
    return ff_strict_normalize(
        FF(product, ff_strict_round(leading_error + low_product, slot)), slot);
}

fn index(component: u32, point: u32) -> u32 {
    return component * params.points + point;
}

fn state_at(i: u32) -> FF {
    return FF(state_hi.data[i], state_lo.data[i]);
}

fn velocity_at(i: u32) -> FF {
    return FF(velocity_hi.data[i], velocity_lo.data[i]);
}

fn residual_at(i: u32) -> FF {
    return FF(residual.data[i], residual_lo.data[i]);
}

fn put_state(i: u32, value: FF) {
    output.data[i] = value.hi;
    output.data[6u * params.points + i] = value.lo;
}

fn put_velocity(i: u32, value: FF) {
    output.data[12u * params.points + i] = value.hi;
    output.data[18u * params.points + i] = value.lo;
}

fn update_velocity(component: u32, point: u32) -> FF {
    let i = index(component, point);
    let damped = ff_strict_mul_f32(velocity_at(i), params.damping_b1, point);
    let forced = ff_strict_mul_f32(residual_at(i), params.delta_t, point);
    return ff_strict_mul_f32(ff_strict_add(damped, forced, point),
                             params.damping_fac, point);
}

fn advance_state(i: u32, velocity: FF, scale: f32) {
    let point = i % params.points;
    put_state(i, ff_strict_add(
        state_at(i), ff_strict_mul_f32(velocity, scale, point), point));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let point = invocation.x;
    if (point >= params.points) { return; }
    for (var component = 0u; component < 6u; component++) {
        let i = index(component, point);
        put_state(i, state_at(i));
        put_velocity(i, velocity_at(i));
    }
    let mode = point / params.ns;
    let surface = point % params.ns;
    let m = mode / params.ntor_plus_one;
    let n = mode % params.ntor_plus_one;
    if (surface == 0u && m > 0u) { return; }
    let m_scale = select(1.4142135623730951, 1.0, m == 0u);
    let n_scale = select(1.4142135623730951, 1.0, n == 0u);
    let basis_scale = ff_strict_round(m_scale * n_scale, point);
    let step_scale = ff_strict_round(params.delta_t * basis_scale, point);
    let j_max = select(params.ns - 1u, params.ns, params.move_lcfs != 0u);
    if (surface < j_max) {
        let vr = update_velocity(0u, point);
        let vz = update_velocity(1u, point);
        let vrs = update_velocity(3u, point);
        let vzc = update_velocity(4u, point);
        put_velocity(index(0u, point), vr);
        put_velocity(index(1u, point), vz);
        put_velocity(index(3u, point), vrs);
        put_velocity(index(4u, point), vzc);
        advance_state(index(0u, point), vr, step_scale);
        advance_state(index(1u, point), vz, step_scale);
        if (m == 1u) {
            advance_state(index(3u, point),
                          ff_strict_add(vrs, vzc, point), step_scale);
            advance_state(index(4u, point),
                          ff_strict_add(
                              vrs, ff_strict_mul_f32(vzc, -1.0, point), point),
                          step_scale);
        } else {
            advance_state(index(3u, point), vrs, step_scale);
            advance_state(index(4u, point), vzc, step_scale);
        }
    }
    let vl = update_velocity(2u, point);
    let vlc = update_velocity(5u, point);
    put_velocity(index(2u, point), vl);
    put_velocity(index(5u, point), vlc);
    advance_state(index(2u, point), vl, step_scale);
    advance_state(index(5u, point), vlc, step_scale);
}
