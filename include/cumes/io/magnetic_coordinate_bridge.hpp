// magnetic_coordinate_bridge.hpp — zero-copy adapter from a converged cuMES
// host snapshot to the public magnetic-coordinate transform input.
#ifndef CUMES_INCLUDE_CUMES_IO_MAGNETIC_COORDINATE_BRIDGE_HPP_
#define CUMES_INCLUDE_CUMES_IO_MAGNETIC_COORDINATE_BRIDGE_HPP_

#include "cumes/io/equilibrium_snapshot.hpp"
#include "cumes/io/input_params.hpp"

#include <cstddef>
#include <stdexcept>

#include <magnetic_coordinate/cumes_binary.hpp>

namespace cumes {

inline magnetic_coordinate::CumesEquilibriumView make_magnetic_coordinate_view(
    const EquilibriumSnapshot& snapshot,
    const InputParams& input) {
    const long long expected_mnmax =
        static_cast<long long>(input.mpol) * (input.ntor + 1LL);
    if (snapshot.mnmax != expected_mnmax || snapshot.ntheta != input.ntheta ||
        snapshot.nzeta != input.nzeta) {
        throw std::invalid_argument(
            "cuMES snapshot metadata is inconsistent with its input record");
    }
    if (!snapshot.has_derived_fields()) {
        throw std::invalid_argument(
            "cuMES snapshot lacks the fields required by magnetic-coordinate");
    }
    for (const auto& family : snapshot.families) {
        if (family.size() != snapshot.family_size()) {
            throw std::invalid_argument(
                "cuMES snapshot has an incomplete spectral family");
        }
    }

    magnetic_coordinate::CumesEquilibriumView view;
    view.format_version = 8;
    view.ns = snapshot.ns;
    view.mnmax = snapshot.mnmax;
    view.mpol = input.mpol;
    view.ntor = input.ntor;
    view.nfp = input.nfp;
    view.ntheta = snapshot.ntheta;
    view.nzeta = snapshot.nzeta;
    view.ncurr = input.ncurr;
    view.phiedge = input.phiedge;
    view.aphi = input.aphi;

    view.families[magnetic_coordinate::CumesEquilibrium::RMNCC] =
        snapshot.families[EquilibriumSnapshot::RMNCC];
    view.families[magnetic_coordinate::CumesEquilibrium::ZMNSC] =
        snapshot.families[EquilibriumSnapshot::ZMNSC];
    view.families[magnetic_coordinate::CumesEquilibrium::LMNSC] =
        snapshot.families[EquilibriumSnapshot::LMNSC];
    view.families[magnetic_coordinate::CumesEquilibrium::RMNSS] =
        snapshot.families[EquilibriumSnapshot::RMNSS];
    view.families[magnetic_coordinate::CumesEquilibrium::ZMNCS] =
        snapshot.families[EquilibriumSnapshot::ZMNCS];
    view.families[magnetic_coordinate::CumesEquilibrium::LMNCS] =
        snapshot.families[EquilibriumSnapshot::LMNCS];

    view.half_fields[magnetic_coordinate::CumesEquilibrium::SQRTG] =
        snapshot.half_fields[EquilibriumSnapshot::SQRTG];
    view.half_fields[magnetic_coordinate::CumesEquilibrium::BSUPS] =
        snapshot.half_fields[EquilibriumSnapshot::BSUPS];
    view.half_fields[magnetic_coordinate::CumesEquilibrium::BSUPU] =
        snapshot.half_fields[EquilibriumSnapshot::BSUPU];
    view.half_fields[magnetic_coordinate::CumesEquilibrium::BSUPV] =
        snapshot.half_fields[EquilibriumSnapshot::BSUPV];
    view.half_fields[magnetic_coordinate::CumesEquilibrium::BSUBS] =
        snapshot.half_fields[EquilibriumSnapshot::BSUBS];
    view.half_fields[magnetic_coordinate::CumesEquilibrium::BSUBU] =
        snapshot.half_fields[EquilibriumSnapshot::BSUBU];
    view.half_fields[magnetic_coordinate::CumesEquilibrium::BSUBV] =
        snapshot.half_fields[EquilibriumSnapshot::BSUBV];
    return view;
}

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_IO_MAGNETIC_COORDINATE_BRIDGE_HPP_
