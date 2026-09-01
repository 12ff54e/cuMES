// forward_dual.cuh — one-direction forward-mode scalar for CUDA operator
// linearization. This is an implementation scalar, never an on-disk or public
// equilibrium precision.
#ifndef CUMES_INCLUDE_CUMES_NUMERICS_FORWARD_DUAL_CUH_
#define CUMES_INCLUDE_CUMES_NUMERICS_FORWARD_DUAL_CUH_

#include <cuda_runtime.h>

#include <cmath>
#include <type_traits>

namespace cumes {

template <class T>
struct ForwardDual {
    using val_type = T;

    T value{};
    T tangent{};

    __host__ __device__ constexpr ForwardDual() = default;
    __host__ __device__ constexpr ForwardDual(T primal)
        : value(primal), tangent(T(0)) {}
    __host__ __device__ constexpr ForwardDual(T primal, T direction)
        : value(primal), tangent(direction) {}

    __host__ __device__ explicit constexpr operator T() const { return value; }

    __host__ __device__ constexpr ForwardDual& operator+=(ForwardDual rhs) {
        value += rhs.value;
        tangent += rhs.tangent;
        return *this;
    }
    __host__ __device__ constexpr ForwardDual& operator-=(ForwardDual rhs) {
        value -= rhs.value;
        tangent -= rhs.tangent;
        return *this;
    }
    __host__ __device__ constexpr ForwardDual& operator*=(ForwardDual rhs) {
        tangent = tangent * rhs.value + value * rhs.tangent;
        value *= rhs.value;
        return *this;
    }
    __host__ __device__ constexpr ForwardDual& operator/=(ForwardDual rhs) {
        tangent = (tangent * rhs.value - value * rhs.tangent) /
                  (rhs.value * rhs.value);
        value /= rhs.value;
        return *this;
    }
};

template <class T>
__host__ __device__ constexpr ForwardDual<T> operator+(ForwardDual<T> lhs,
                                                       ForwardDual<T> rhs) {
    lhs += rhs;
    return lhs;
}

template <class T>
__host__ __device__ constexpr ForwardDual<T> operator-(ForwardDual<T> lhs,
                                                       ForwardDual<T> rhs) {
    lhs -= rhs;
    return lhs;
}

template <class T>
__host__ __device__ constexpr ForwardDual<T> operator*(ForwardDual<T> lhs,
                                                       ForwardDual<T> rhs) {
    lhs *= rhs;
    return lhs;
}

template <class T>
__host__ __device__ constexpr ForwardDual<T> operator/(ForwardDual<T> lhs,
                                                       ForwardDual<T> rhs) {
    lhs /= rhs;
    return lhs;
}

template <class T>
__host__ __device__ constexpr ForwardDual<T> operator-(ForwardDual<T> value) {
    return {-value.value, -value.tangent};
}

template <class T>
__host__ __device__ constexpr bool operator==(ForwardDual<T> lhs,
                                              ForwardDual<T> rhs) {
    return lhs.value == rhs.value;
}

template <class T>
__host__ __device__ constexpr bool operator!=(ForwardDual<T> lhs,
                                              ForwardDual<T> rhs) {
    return !(lhs == rhs);
}

template <class T>
__host__ __device__ constexpr bool operator<(ForwardDual<T> lhs,
                                             ForwardDual<T> rhs) {
    return lhs.value < rhs.value;
}

template <class T>
__host__ __device__ constexpr bool operator>(ForwardDual<T> lhs,
                                             ForwardDual<T> rhs) {
    return rhs < lhs;
}

template <class T>
__host__ __device__ constexpr bool operator<=(ForwardDual<T> lhs,
                                              ForwardDual<T> rhs) {
    return !(rhs < lhs);
}

template <class T>
__host__ __device__ constexpr bool operator>=(ForwardDual<T> lhs,
                                              ForwardDual<T> rhs) {
    return !(lhs < rhs);
}

template <class T>
__host__ __device__ inline ForwardDual<T> sqrt(ForwardDual<T> x) {
    const T root = ::sqrt(x.value);
    return {root, x.tangent / (T(2) * root)};
}

template <class T>
__host__ __device__ inline ForwardDual<T> fabs(ForwardDual<T> x) {
    return x.value < T(0) ? ForwardDual<T>{-x.value, -x.tangent} : x;
}

template <class T>
__host__ __device__ inline ForwardDual<T> fmin(ForwardDual<T> lhs,
                                               ForwardDual<T> rhs) {
    return lhs.value <= rhs.value ? lhs : rhs;
}

template <class T>
__host__ __device__ inline ForwardDual<T> fmax(ForwardDual<T> lhs,
                                               ForwardDual<T> rhs) {
    return lhs.value >= rhs.value ? lhs : rhs;
}

template <class T>
__host__ __device__ inline ForwardDual<T> pow(ForwardDual<T> base,
                                              ForwardDual<T> exponent) {
    const T primal = ::pow(base.value, exponent.value);
    return {primal, primal * (exponent.tangent * ::log(base.value) +
                              exponent.value * base.tangent / base.value)};
}

template <class T, class I>
    requires std::is_integral_v<I>
__host__ __device__ inline ForwardDual<T> pow(ForwardDual<T> base, I exponent) {
    const T primal = ::pow(base.value, exponent);
    if (exponent == 0) return {primal, T(0)};
    return {primal,
            T(exponent) * ::pow(base.value, exponent - 1) * base.tangent};
}

template <class T>
__host__ __device__ inline bool isfinite(ForwardDual<T> x) {
    return ::isfinite(x.value) && ::isfinite(x.tangent);
}

}  // namespace cumes

// A few production kernels qualify these elementary functions with std::.
// ForwardDual is a cuMES implementation type, and these overloads preserve
// the ordinary scalar call spelling while propagating its derivative lane.
namespace std {

template <class T>
__host__ __device__ inline cumes::ForwardDual<T> sqrt(cumes::ForwardDual<T> x) {
    return cumes::sqrt(x);
}

template <class T>
__host__ __device__ inline bool isfinite(cumes::ForwardDual<T> x) {
    return cumes::isfinite(x);
}

}  // namespace std

#endif  // CUMES_INCLUDE_CUMES_NUMERICS_FORWARD_DUAL_CUH_
