// accumulation.hpp — the norm/reduction accumulator type (blueprint §8.8, §8.12).
//
// The mixed-float policy accumulates residual/norm reductions in double even
// when the state/geometry is float, so a ~15k-term summation (mnmax*ns) does not
// lose precision to per-add float rounding. NormAccum<T> encodes that default:
//
//   float  state -> double accumulation  (mixed-float, Class B for the float
//                                         build only)
//   double state -> double (== T, so the verified double build is bit-identical)
//
// The *stored* output scalar stays T — the benefit is the summation width, not a
// wider residual value — so this is a change to the float build alone. A full
// double ControlRecord (double invariants threaded through the host controller)
// is a separate, larger follow-up.
#pragma once

namespace cumes {

template <class T>
struct NormAccum {
    using type = T;  // double state: accumulate in T (double) — bit-identical
};

template <>
struct NormAccum<float> {
    using type = double;  // mixed-float: accumulate norms in double
};

}  // namespace cumes
