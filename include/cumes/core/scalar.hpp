// scalar.hpp — the computation scalar type tags shared by host model and
// device code. The on-disk state and config model are always double; only the
// GPU computation templating uses T (double or float). A ScalarType lets the
// host model reason about precision without carrying a device type.
#ifndef CUMES_INCLUDE_CUMES_CORE_SCALAR_HPP_
#define CUMES_INCLUDE_CUMES_CORE_SCALAR_HPP_

#include <cstdint>
#include <type_traits>

namespace cumes {

enum class ScalarType : std::uint8_t { FLOAT, DOUBLE };

template <class T>
inline constexpr ScalarType SCALAR_TYPE_OF =
    std::is_same_v<std::remove_cv_t<T>, float> ? ScalarType::FLOAT
                                               : ScalarType::DOUBLE;

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_CORE_SCALAR_HPP_
