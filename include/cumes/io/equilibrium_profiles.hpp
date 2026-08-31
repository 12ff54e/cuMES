// equilibrium_profiles.hpp — host radial profiles associated with a solved
// equilibrium. These are in-memory solver outputs, separate from optimizer
// observables and target/reduction policy.
#ifndef CUMES_INCLUDE_CUMES_IO_EQUILIBRIUM_PROFILES_HPP_
#define CUMES_INCLUDE_CUMES_IO_EQUILIBRIUM_PROFILES_HPP_

#include <cstddef>
#include <vector>

namespace cumes {

struct EquilibriumProfiles {
    // Half-grid profiles at s=(j+1/2)/(ns-1). Flux derivatives use the public
    // VMEC wout convention in webers per unit s: the internal oriented
    // phip/chip values are multiplied by sign(J)*2*pi during capture.
    std::vector<double> toroidal_flux_derivative;
    std::vector<double> poloidal_flux_derivative;
    std::vector<double> rotational_transform;

    // VMEC-compatible covariant flux functions on the half grid:
    //   I(s) = <B_theta>  (wout buco)
    //   G(s) = <B_zeta>   (wout bvco).
    // The angle brackets are the normalized VMEC angular quadrature, not a
    // physical-area-weighted average.
    std::vector<double> poloidal_covariant_field;
    std::vector<double> toroidal_covariant_field;

    bool has_half_grid_profiles(int ns) const {
        if (ns < 2) return false;
        const std::size_t expected = static_cast<std::size_t>(ns - 1);
        return toroidal_flux_derivative.size() == expected &&
               poloidal_flux_derivative.size() == expected &&
               rotational_transform.size() == expected &&
               poloidal_covariant_field.size() == expected &&
               toroidal_covariant_field.size() == expected;
    }
};

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_IO_EQUILIBRIUM_PROFILES_HPP_
