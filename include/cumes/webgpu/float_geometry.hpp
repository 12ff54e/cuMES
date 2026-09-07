#ifndef CUMES_INCLUDE_CUMES_WEBGPU_FLOAT_GEOMETRY_HPP_
#define CUMES_INCLUDE_CUMES_WEBGPU_FLOAT_GEOMETRY_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

namespace cumes::webgpu {

// Immutable m=0 Rcc reference. Evolved coefficients and even R geometry
// contain displacements; absolute-radius consumers restore this reference.
struct FloatRadiusReference {
    std::vector<double> coefficients;
    // n>0 synthesis, repeated over theta; the large mean is kept separate.
    std::vector<float> angular;

    bool valid(std::size_t angular_points) const {
        return !coefficients.empty() && angular.size() == angular_points &&
               std::all_of(coefficients.begin(), coefficients.end(),
                           [](double value) { return std::isfinite(value); }) &&
               std::all_of(angular.begin(), angular.end(),
                           [](float value) { return std::isfinite(value); });
    }

    float restore(float displacement, std::size_t point) const {
        return (displacement + static_cast<float>(coefficients.front())) +
               angular[point % angular.size()];
    }
};

using FloatRadiusReferencePtr = std::shared_ptr<const FloatRadiusReference>;

}  // namespace cumes::webgpu
#endif
