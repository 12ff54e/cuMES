// mode_table.hpp — per-mode metadata table (blueprint §6.2).
//
// Built once per resolution and read by transform, constraint, residual,
// preconditioner, and descent kernels. It removes the repeated division/modulo,
// square roots, parity branches, and normalization rules that the legacy code
// recomputes inline. The table is surface-agnostic (mode metadata only); the
// folded boundary amplitudes live in ValidatedProblem::FoldedBoundary.
#ifndef CUMES_INCLUDE_CUMES_CORE_MODE_TABLE_HPP_
#define CUMES_INCLUDE_CUMES_CORE_MODE_TABLE_HPP_

#include "cumes/core/grid_shape.hpp"

#include <cstdint>
#include <vector>

namespace cumes {

enum class ModeParity : std::uint8_t { EVEN, ODD };

template <class T>
struct ModeEntry {
    using val_type = T;

    int m = 0;
    int n = 0;
    int physical_n = 0;  // n * nfp (physical toroidal mode N, blueprint §4.2)
    int first_surface = 0;  // 0 for m==0, 1 for m>0 (blueprint §4.9 j_min)
    T mn_scale = T(1);      // mscale * nscale (blueprint §4.2 S_mn)
    T xmpq = T(0);          // m*(m-1) (constraint weight, blueprint §4.8)
    ModeParity parity = ModeParity::EVEN;
};

template <class T>
struct ModeTable {
    using val_type = T;

    GridShape shape;
    // entries[mode] with mode = m * (ntor + 1) + n.
    std::vector<ModeEntry<T>> entries;

    // Build the full folded-mode table for a resolved shape.
    static ModeTable<T> build(const GridShape& shape);

    const ModeEntry<T>& operator[](std::size_t mode) const {
        return entries[mode];
    }
    std::size_t size() const { return entries.size(); }
};

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_CORE_MODE_TABLE_HPP_
