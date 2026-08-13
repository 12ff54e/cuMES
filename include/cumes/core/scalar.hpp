// scalar.hpp — the computation scalar type tags shared by host model and
// device code. The on-disk state and config model are always double; only the
// GPU computation templating uses T (double or float). A ScalarType lets the
// host model reason about precision without carrying a device type.
#pragma once

#include <cstdint>
#include <type_traits>

namespace cumes {

enum class ScalarType : std::uint8_t { kFloat, kDouble };

template <class T>
inline constexpr ScalarType kScalarTypeOf =
    std::is_same_v<std::remove_cv_t<T>, float> ? ScalarType::kFloat
                                               : ScalarType::kDouble;

}  // namespace cumes
