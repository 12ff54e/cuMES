#ifndef CUMES_INCLUDE_CUMES_PHYSICS_CONSTRAINT_FILTER_HPP_
#define CUMES_INCLUDE_CUMES_PHYSICS_CONSTRAINT_FILTER_HPP_

#include <cstddef>
#include <span>

namespace cumes {

// Spectral-condensation weights: faccon[0] = 0 and
// faccon[m] = 0.25 / (m*m*(m+1)*(m+1)) for m > 0.
// Retain the integer product before conversion and the scalar square/divide.
template <typename T>
void fill_constraint_filter(std::span<T> faccon) {
    for (std::size_t index = 0; index < faccon.size(); ++index) {
        const int m = static_cast<int>(index);
        const T xmpq = T((m + 1) * m);
        faccon[index] = m > 0 ? T(0.25) / (xmpq * xmpq) : T(0.0);
    }
}

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_PHYSICS_CONSTRAINT_FILTER_HPP_
