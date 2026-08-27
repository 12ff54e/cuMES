// equilibrium_snapshot.hpp — host-side state and derived-field snapshot
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
        RMNCC = 0,  // R: cos(mθ)cos(nζ)
        ZMNSC = 1,  // Z: sin(mθ)cos(nζ)
        LMNSC = 2,  // λ: sin(mθ)cos(nζ)
        RMNSS = 3,  // R: sin(mθ)sin(nζ)
        ZMNCS = 4,  // Z: cos(mθ)sin(nζ)
        LMNCS = 5,  // λ: cos(mθ)sin(nζ)
        COUNT = 6,
    };

    int ns = 0;
    int mnmax = 0;
    // families[c] has ns * mnmax doubles, mode-major (surface contiguous).
    std::array<std::vector<double>, COUNT> families;

    // Scientific result fields. The solver state above remains mode-major;
    // these real-space arrays use the native point-major layout
    //   point + surface * (ntheta * nzeta),
    // where point = theta + zeta * ntheta. Magnetic quantities and sqrt(g)
    // live on the ns-1 half grid. Current-density quantities are derived from
    // curl(B)/mu0 on the ns-point full grid, with explicit linear endpoint
    // extrapolation.
    enum HalfField : std::uint8_t {
        SQRTG = 0,
        BSUPS = 1,
        BSUPU = 2,
        BSUPV = 3,
        BSUBS = 4,
        BSUBU = 5,
        BSUBV = 6,
        HALF_FIELD_COUNT = 7,
    };

    enum FullField : std::uint8_t {
        JSUPS = 0,
        JSUPU = 1,
        JSUPV = 2,
        JSUBS = 3,
        JSUBU = 4,
        JSUBV = 5,
        FULL_FIELD_COUNT = 6,
    };

    int ntheta = 0;
    int nzeta = 0;
    std::array<std::vector<double>, HALF_FIELD_COUNT> half_fields;
    std::array<std::vector<double>, FULL_FIELD_COUNT> full_fields;

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

    std::size_t points_per_surface() const {
        return static_cast<std::size_t>(ntheta) *
               static_cast<std::size_t>(nzeta);
    }

    std::size_t half_field_size() const {
        return ns > 0 ? static_cast<std::size_t>(ns - 1) * points_per_surface()
                      : 0;
    }

    std::size_t full_field_size() const {
        return static_cast<std::size_t>(ns) * points_per_surface();
    }

    bool has_derived_fields() const {
        if (ntheta <= 0 || nzeta <= 0 || ns <= 1) return false;
        const std::size_t half_size = half_field_size();
        const std::size_t full_size = full_field_size();
        for (const auto& field : half_fields)
            if (field.size() != half_size) return false;
        for (const auto& field : full_fields)
            if (field.size() != full_size) return false;
        return true;
    }

    bool has_any_derived_fields() const {
        if (ntheta != 0 || nzeta != 0) return true;
        for (const auto& field : half_fields)
            if (!field.empty()) return true;
        for (const auto& field : full_fields)
            if (!field.empty()) return true;
        return false;
    }
};

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_IO_EQUILIBRIUM_SNAPSHOT_HPP_
