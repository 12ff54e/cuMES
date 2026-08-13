// mode_table.cpp — ModeTable<T> construction (folded-mode metadata).
#include "cumes/core/mode_table.hpp"

#include <cmath>

namespace cumes {

template <class T>
ModeTable<T> ModeTable<T>::build(const GridShape& shape) {
    ModeTable<T> table;
    table.shape = shape;
    table.entries.reserve(shape.modes());
    for (int m = 0; m < shape.mpol; ++m) {
        for (int n = 0; n <= shape.ntor; ++n) {
            ModeEntry<T> e;
            e.m = m;
            e.n = n;
            e.physical_n = n * shape.nfp;
            e.first_surface = (m == 0) ? 0 : 1;
            const T mscale = (m == 0) ? T(1) : T(std::sqrt(2.0));
            const T nscale = (n == 0) ? T(1) : T(std::sqrt(2.0));
            e.mn_scale = mscale * nscale;
            e.xmpq = T(m * (m - 1));
            e.parity = (m % 2 == 0) ? ModeParity::kEven : ModeParity::kOdd;
            table.entries.push_back(e);
        }
    }
    return table;
}

// The host model builds the table at double precision (the config/normalized
// view); a float instantiation exists for completeness but is not used by the
// host-only path yet.
template struct ModeTable<double>;
template struct ModeTable<float>;

}  // namespace cumes
