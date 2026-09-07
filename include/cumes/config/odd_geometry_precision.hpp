#ifndef CUMES_INCLUDE_CUMES_CONFIG_ODD_GEOMETRY_PRECISION_HPP_
#define CUMES_INCLUDE_CUMES_CONFIG_ODD_GEOMETRY_PRECISION_HPP_

#include <stdexcept>
#include <string_view>

namespace cumes {
// Experimental accuracy scopes for the odd R/Z position reconstruction.
// POLOIDAL compensates in the state scalar type (float-float or double-double).
// Other non-native scopes are float-only diagnostic controls.
enum class OddGeometryPrecision {
    NATIVE,
    FLOAT_ORDER,
    SUM,
    POLOIDAL,
    POLOIDAL_SCALE,
    FLOAT_FLOAT,
};

inline OddGeometryPrecision parse_odd_geometry_precision(
    std::string_view name) {
    if (name == "native") return OddGeometryPrecision::NATIVE;
    if (name == "float-order") return OddGeometryPrecision::FLOAT_ORDER;
    if (name == "sum") return OddGeometryPrecision::SUM;
    if (name == "poloidal") return OddGeometryPrecision::POLOIDAL;
    if (name == "poloidal-scale") return OddGeometryPrecision::POLOIDAL_SCALE;
    if (name == "float-float") return OddGeometryPrecision::FLOAT_FLOAT;
    throw std::invalid_argument(
        "odd geometry precision: expected native, float-order, sum, poloidal, "
        "poloidal-scale or float-float");
}
}  // namespace cumes
#endif
