#ifndef CUMES_INCLUDE_CUMES_NUMERICS_FLOAT_FLOAT_CUH_
#define CUMES_INCLUDE_CUMES_NUMERICS_FLOAT_FLOAT_CUH_

#include <cuda_runtime.h>

namespace cumes {
// Unevaluated sum hi+lo. Explicit round-to-nearest intrinsics protect the
// error terms from contraction/reassociation, including fast-math builds.
struct FloatFloat {
    float hi = 0.0F;
    float lo = 0.0F;

    __host__ __device__ FloatFloat() = default;
    __host__ __device__ explicit FloatFloat(float value) : hi(value) {}
    __host__ __device__ FloatFloat(float high, float low) : hi(high), lo(low) {}
    __host__ static FloatFloat from_double(double value) {
        float high = float(value);
        return {high, float(value - double(high))};
    }
    __device__ explicit operator float() const { return __fadd_rn(hi, lo); }
};

__device__ inline FloatFloat operator+(FloatFloat a, FloatFloat b) {
    float s = __fadd_rn(a.hi, b.hi);
    float v = __fsub_rn(s, a.hi);
    float e = __fadd_rn(__fsub_rn(a.hi, __fsub_rn(s, v)), __fsub_rn(b.hi, v));
    e = __fadd_rn(e, __fadd_rn(a.lo, b.lo));
    float h = __fadd_rn(s, e);
    return {h, __fsub_rn(e, __fsub_rn(h, s))};
}

__device__ inline FloatFloat operator*(FloatFloat a, FloatFloat b) {
    float p = __fmul_rn(a.hi, b.hi);
    float e = __fmaf_rn(a.hi, b.hi, -p);
    e = __fmaf_rn(a.hi, b.lo, e);
    e = __fmaf_rn(a.lo, b.hi, e);
    e = __fmaf_rn(a.lo, b.lo, e);
    float h = __fadd_rn(p, e);
    return {h, __fsub_rn(e, __fsub_rn(h, p))};
}
}  // namespace cumes
#endif
