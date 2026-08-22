// grid_shape.cpp — GridShape derived extents and resolved-shape validation.
#include "cumes/core/grid_shape.hpp"

#include "cumes/core/checked_size.hpp"

#include <string>

namespace cumes {

std::size_t GridShape::full_points() const {
    auto a = checked_mul(static_cast<std::size_t>(ns),
                         static_cast<std::size_t>(ntheta));
    if (!a) return 0;
    auto b = checked_mul(*a, static_cast<std::size_t>(nzeta));
    return b.value_or(0);
}

std::size_t GridShape::half_points() const {
    if (ns < 1) return 0;
    auto a = checked_mul(static_cast<std::size_t>(ns - 1),
                         static_cast<std::size_t>(ntheta));
    if (!a) return 0;
    auto b = checked_mul(*a, static_cast<std::size_t>(nzeta));
    return b.value_or(0);
}

std::size_t GridShape::modes() const {
    // (ntor + 1) is evaluated in size_t to avoid signed overflow if a caller
    // passes an unchecked ntor; the config bounds ntor before building tables.
    auto a = checked_mul(static_cast<std::size_t>(mpol),
                         static_cast<std::size_t>(ntor) + 1);
    return a.value_or(0);
}

int GridShape::ntheta_reduced() const {
    return ntheta / 2 + 1;
}

Status GridShape::validate() const {
    if (ns < 3) return Status("ns must be >= 3");
    if (mpol < 2) return Status("mpol must be >= 2");
    if (ntor < 0) return Status("ntor must be >= 0");
    if (nfp < 1) return Status("nfp must be >= 1");
    if (ntheta < 2 || ntheta % 2 != 0) {
        return Status("ntheta must be a positive even number");
    }
    if (nzeta < 1) return Status("nzeta must be >= 1");
    if (modes() == 0)
        return Status("mode count (mpol*(ntor+1)) overflows size_t");
    if (full_points() == 0) {
        return Status("full-grid point count overflows size_t");
    }
    return Status();
}

}  // namespace cumes
