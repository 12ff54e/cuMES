#ifndef CUMES_INCLUDE_CUMES_WEBGPU_FLOAT_FLOAT_HPP_
#define CUMES_INCLUDE_CUMES_WEBGPU_FLOAT_FLOAT_HPP_

#include <cmath>

namespace cumes::webgpu {

// Unevaluated sum of two binary32 values. The pair is normalized after every
// public operation so |lo| is at most half an ulp of hi. This mirrors the WGSL
// implementation used by precision-critical WebGPU kernels.
struct FloatFloat {
    using val_type = float;

    float hi = 0.0F;
    float lo = 0.0F;
};

inline FloatFloat quick_two_sum(float a, float b) {
    const float sum = a + b;
    const float sum_minus_a = std::fma(-1.0F, a, sum);
    return {sum, std::fma(-1.0F, sum_minus_a, b)};
}

inline FloatFloat two_sum(float a, float b) {
    const float sum = a + b;
    const float b_virtual = std::fma(-1.0F, a, sum);
    const float a_virtual = std::fma(-1.0F, b_virtual, sum);
    const float a_error = std::fma(-1.0F, a_virtual, a);
    const float b_error = std::fma(-1.0F, b_virtual, b);
    const float error = a_error + b_error;
    return {sum, error};
}

inline FloatFloat normalize(FloatFloat value) {
    return quick_two_sum(value.hi, value.lo);
}

inline FloatFloat add(FloatFloat a, FloatFloat b) {
    const FloatFloat leading = two_sum(a.hi, b.hi);
    return normalize({leading.hi, leading.lo + a.lo + b.lo});
}

inline FloatFloat add(FloatFloat a, float b) {
    const FloatFloat leading = two_sum(a.hi, b);
    return normalize({leading.hi, leading.lo + a.lo});
}

inline FloatFloat multiply(FloatFloat a, float b) {
    const float product = a.hi * b;
    const float error = std::fma(a.hi, b, -product) + a.lo * b;
    return normalize({product, error});
}

inline double value(FloatFloat input) {
    return static_cast<double>(input.hi) + static_cast<double>(input.lo);
}

}  // namespace cumes::webgpu

#endif  // CUMES_INCLUDE_CUMES_WEBGPU_FLOAT_FLOAT_HPP_
