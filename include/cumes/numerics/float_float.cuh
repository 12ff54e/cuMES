#ifndef CUMES_INCLUDE_CUMES_NUMERICS_FLOAT_FLOAT_CUH_
#define CUMES_INCLUDE_CUMES_NUMERICS_FLOAT_FLOAT_CUH_

#include <cuda_runtime.h>

#include <type_traits>

namespace cumes {
// Unevaluated sum hi+lo. Explicit round-to-nearest intrinsics protect the
// error terms from contraction/reassociation, including fast-math builds.
template <class T>
struct Compensated {
    using val_type = T;
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);

    T hi = T(0);
    T lo = T(0);

    __host__ __device__ Compensated() = default;
    __host__ __device__ explicit Compensated(T value) : hi(value) {}
    __host__ __device__ Compensated(T high, T low) : hi(high), lo(low) {}
    __host__ static Compensated from_double(double value) {
        T high = T(value);
        return {high, T(value - double(high))};
    }
    __device__ explicit operator T() const { return add(hi, lo); }

    __device__ static T add(T a, T b) {
        if constexpr (std::is_same_v<T, float>)
            return __fadd_rn(a, b);
        else
            return __dadd_rn(a, b);
    }
    __device__ static T sub(T a, T b) {
        if constexpr (std::is_same_v<T, float>)
            return __fsub_rn(a, b);
        else
            return __dsub_rn(a, b);
    }
    __device__ static T mul(T a, T b) {
        if constexpr (std::is_same_v<T, float>)
            return __fmul_rn(a, b);
        else
            return __dmul_rn(a, b);
    }
    __device__ static T fma(T a, T b, T c) {
        if constexpr (std::is_same_v<T, float>)
            return __fmaf_rn(a, b, c);
        else
            return __fma_rn(a, b, c);
    }

    __device__ friend Compensated operator+(Compensated a, Compensated b) {
        T s = add(a.hi, b.hi);
        T v = sub(s, a.hi);
        T e = add(sub(a.hi, sub(s, v)), sub(b.hi, v));
        e = add(e, add(a.lo, b.lo));
        T h = add(s, e);
        return {h, sub(e, sub(h, s))};
    }

    __device__ friend Compensated operator*(Compensated a, Compensated b) {
        T p = mul(a.hi, b.hi);
        T e = fma(a.hi, b.hi, -p);
        e = fma(a.hi, b.lo, e);
        e = fma(a.lo, b.hi, e);
        e = fma(a.lo, b.lo, e);
        T h = add(p, e);
        return {h, sub(e, sub(h, p))};
    }
};

using FloatFloat = Compensated<float>;
}  // namespace cumes
#endif
