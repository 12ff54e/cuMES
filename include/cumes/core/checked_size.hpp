// checked_size.hpp — overflow-checked size_t arithmetic.
//
// Every derived element-count product (full_points, half_points, modes, the
// zeta-scratch and force-buffer extents) must go through these helpers, not a
// bare `a * b`. Blueprint §4.1: "All element-count products must use checked
// size_t arithmetic." Returns std::nullopt on overflow; callers fold that into
// a validation error before any allocation.
#ifndef CUMES_INCLUDE_CUMES_CORE_CHECKED_SIZE_HPP_
#define CUMES_INCLUDE_CUMES_CORE_CHECKED_SIZE_HPP_

#include <cstddef>
#include <limits>
#include <optional>

namespace cumes {

inline std::optional<std::size_t> checked_mul(std::size_t a, std::size_t b) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        return std::nullopt;
    }
    return a * b;
}

inline std::optional<std::size_t> checked_add(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a) {
        return std::nullopt;
    }
    return a + b;
}

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_CORE_CHECKED_SIZE_HPP_
