// Double-single arithmetic shared by precision-critical WebGPU kernels.
// An FF value has about 44-48 significant bits while using only WGSL f32.
struct FF {
    hi: f32,
    lo: f32,
};

fn ff_quick_two_sum(a: f32, b: f32) -> FF {
    let sum = a + b;
    let sum_minus_a = fma(-1.0, a, sum);
    return FF(sum, fma(-1.0, sum_minus_a, b));
}

fn ff_two_sum(a: f32, b: f32) -> FF {
    let sum = a + b;
    let b_virtual = fma(-1.0, a, sum);
    let a_virtual = fma(-1.0, b_virtual, sum);
    let a_error = fma(-1.0, a_virtual, a);
    let b_error = fma(-1.0, b_virtual, b);
    let error = a_error + b_error;
    return FF(sum, error);
}

fn ff_normalize(value: FF) -> FF {
    return ff_quick_two_sum(value.hi, value.lo);
}

fn ff_add(a: FF, b: FF) -> FF {
    let leading = ff_two_sum(a.hi, b.hi);
    return ff_normalize(FF(leading.hi, leading.lo + a.lo + b.lo));
}

fn ff_add_f32(a: FF, b: f32) -> FF {
    let leading = ff_two_sum(a.hi, b);
    return ff_normalize(FF(leading.hi, leading.lo + a.lo));
}

fn ff_mul_f32(a: FF, b: f32) -> FF {
    let product = a.hi * b;
    let error = fma(a.hi, b, -product) + a.lo * b;
    return ff_normalize(FF(product, error));
}

fn ff_from_f32(value: f32) -> FF {
    return FF(value, 0.0);
}

fn ff_value(value: FF) -> f32 {
    return value.hi + value.lo;
}
