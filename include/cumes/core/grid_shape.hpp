// grid_shape.hpp — extents-only grid description (blueprint §6.2).
//
// GridShape owns no pointers and no physical values; it is the shared shape
// contract between the config model, the mode/quadrature tables, and every
// operator. The extents are the RESOLVED values (resolution defaults already
// applied by validate()), so validate() checks the resolved invariants rather
// than re-deriving them.
#ifndef CUMES_INCLUDE_CUMES_CORE_GRID_SHAPE_HPP_
#define CUMES_INCLUDE_CUMES_CORE_GRID_SHAPE_HPP_

#include "cumes/core/result.hpp"

#include <cstddef>

namespace cumes {

struct GridShape {
    int ns = 0;      // radial surfaces (full grid)
    int ntheta = 0;  // poloidal points (even, resolved)
    int nzeta = 0;   // toroidal points (resolved)
    int mpol = 0;    // poloidal mode count
    int ntor = 0;    // toroidal mode count (folded basis: n = 0..ntor)
    int nfp = 0;     // field periods

    // ns * ntheta * nzeta, or 0 on overflow.
    std::size_t full_points() const;
    // (ns-1) * ntheta * nzeta (half-grid surfaces), or 0 on overflow.
    std::size_t half_points() const;
    // mpol * (ntor + 1) folded mode count (mnmax).
    std::size_t modes() const;
    // ntheta / 2 + 1 reduced-theta quadrature points.
    int ntheta_reduced() const;

    // Resolved-shape invariants. Returns an error message on the first
    // violation; callers that want every issue use ValidatedProblem::validate
    // (which also drives this check but aggregates all findings).
    Status validate() const;
};

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_CORE_GRID_SHAPE_HPP_
