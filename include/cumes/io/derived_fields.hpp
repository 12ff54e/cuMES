// derived_fields.hpp — build scientific real-space output from solver fields.
#ifndef CUMES_INCLUDE_CUMES_IO_DERIVED_FIELDS_HPP_
#define CUMES_INCLUDE_CUMES_IO_DERIVED_FIELDS_HPP_

#include "cumes/core/result.hpp"
#include "cumes/io/equilibrium_snapshot.hpp"

#include <vector>

namespace cumes {

// Host mirror of the final geometry pass. Full-grid parity arrays retain the
// solver's even/odd representation; the remaining arrays are native half-grid
// quantities. Keeping this boundary explicit makes the current-density
// post-processing independently unit-testable without CUDA.
struct DerivedFieldInputs {
    int ns = 0;
    int ntheta = 0;
    int nzeta = 0;
    int nfp = 0;
    double delta_s = 0.0;
    double mu0 = 0.0;

    std::vector<double> sqrt_s_full;
    std::vector<double> sqrt_s_half;

    std::vector<double> r_e, r_o, z_e, z_o;
    std::vector<double> ru_e, ru_o, zu_e, zu_o;
    std::vector<double> rv_e, rv_o, zv_e, zv_o;

    std::vector<double> rs, zs, ru12, zu12;
    std::vector<double> sqrtg;
    std::vector<double> bsupu, bsupv, bsubu, bsubv;
};

// Populate snapshot.{ntheta,nzeta,half_fields,full_fields}. Existing spectral
// families and state dimensions are left untouched except that snapshot.ns is
// initialized from input when it is zero. Returns a diagnostic on malformed
// shapes; ordinary converged solver fields produce a complete snapshot.
Status populate_derived_fields(const DerivedFieldInputs& input,
                               EquilibriumSnapshot& snapshot);

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_IO_DERIVED_FIELDS_HPP_
