// Norm reductions keep the kernel's scalar type. Float uses a compensated
// pair to retain small summands; the stored control values remain float.
#ifndef CUMES_INCLUDE_CUMES_NUMERICS_ACCUMULATION_HPP_
#define CUMES_INCLUDE_CUMES_NUMERICS_ACCUMULATION_HPP_

#include "cumes/numerics/float_float.cuh"

namespace cumes {
template <class T>
struct NormAccum {
    using val_type = T;
    using type = T;
};

template <>
struct NormAccum<float> {
    using val_type = float;
    using type = FloatFloat;
};
}  // namespace cumes
#endif
