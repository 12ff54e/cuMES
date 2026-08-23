// equilibrium_snapshot.hpp — host-side component-major state snapshot
// (blueprint §6.13).
//
// The single host representation every writer consumes. The six coefficient
// families are double on the host regardless of the computation scalar type
// (the device->host copy converts T -> double). Each family is laid out
// mode-major, surface-contiguous: index = surface + mode * ns, matching the
// device column-major layout.
#ifndef CUMES_INCLUDE_CUMES_IO_EQUILIBRIUM_SNAPSHOT_HPP_
#define CUMES_INCLUDE_CUMES_IO_EQUILIBRIUM_SNAPSHOT_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace cumes {

struct EquilibriumSnapshot {
    enum Component : std::uint8_t {
        kRmncc = 0,  // R: cos(mθ)cos(nζ)
        kZmnsc = 1,  // Z: sin(mθ)cos(nζ)
        kLmnsc = 2,  // λ: sin(mθ)cos(nζ)
        kRmnss = 3,  // R: sin(mθ)sin(nζ)
        kZmncs = 4,  // Z: cos(mθ)sin(nζ)
        kLmncs = 5,  // λ: cos(mθ)sin(nζ)
        kCount = 6,
    };

    int ns = 0;
    int mnmax = 0;
    // families[c] has ns * mnmax doubles, mode-major (surface contiguous).
    std::array<std::vector<double>, kCount> families;

    const std::vector<double>& component(Component c) const {
        return families[static_cast<int>(c)];
    }
    std::vector<double>& component(Component c) {
        return families[static_cast<int>(c)];
    }

    // Number of doubles per family (ns * mnmax).
    std::size_t family_size() const {
        return static_cast<std::size_t>(ns) * static_cast<std::size_t>(mnmax);
    }
};

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_IO_EQUILIBRIUM_SNAPSHOT_HPP_
