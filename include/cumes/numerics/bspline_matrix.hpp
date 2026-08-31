// bspline_matrix.hpp — host construction of a reusable radial transfer map.
#ifndef CUMES_INCLUDE_CUMES_NUMERICS_BSPLINE_MATRIX_HPP_
#define CUMES_INCLUDE_CUMES_NUMERICS_BSPLINE_MATRIX_HPP_

#include <vector>

namespace cumes {

// Return the row-major [ns_new][ns_old] linear map produced by the cubic
// BSplineInterpolation template on the uniform normalized-flux interval.
// The map is independent of the spectral family and mode, so it is built once
// per multigrid boundary and applied as a batched matrix-vector product on the
// device.
std::vector<double> cubic_bspline_interpolation_matrix(int ns_old, int ns_new);

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_NUMERICS_BSPLINE_MATRIX_HPP_
