// mode_table.cuh — folded-mode device table (blueprint §6.2).
//
// The two int arrays (d_xm/d_xn) that the legacy FourierBasis held: for folded
// mode index `mode = m*(ntor+1)+n`, d_xm[mode] = m and d_xn[mode] = n. This is
// resolution-scoped MODE METADATA — read by the transform, preconditioner, and
// descent kernels — not transform scratch, so it does not belong inside the
// ToroidalFftOperator (blueprint §5.1: transforms own only tables/plans/scratch). It is
// owned by the stage (blueprint §6.5 "Resolution" lifetime) and shared by every
// operator that needs it; the richer ModeEntry<T> host table (cumes/core/
// mode_table.hpp) is the eventual device-side replacement.
#pragma once

#include <cuda_runtime.h>

#include "vmec_types.h"

namespace cumes {

class DeviceArena;

struct DeviceModeTable {
    int* d_xm = nullptr;  // [mnmax] poloidal mode m per folded mode index
    int* d_xn = nullptr;  // [mnmax] toroidal mode n per folded mode index
    bool arena_backed = false;

    DeviceModeTable() = default;

    // Raw owning pointers: an implicit copy would leave two tables holding
    // the same allocations, and modeTableFree frees them unconditionally
    // (unless arena_backed) — a guaranteed double-free. Copies are deleted;
    // the modeTableCreate call sites use copy-init from a prvalue, which is
    // guaranteed copy elision in C++17/20, so they need no change.
    DeviceModeTable(const DeviceModeTable&) = delete;
    DeviceModeTable& operator=(const DeviceModeTable&) = delete;

    // The move ctor exists so `return mt;` in modeTableCreate keeps working
    // (a deleted copy ctor would otherwise suppress the implicit move). It
    // nulls the source, keeping a moved-from table inert and safe to pass
    // to modeTableFree.
    DeviceModeTable(DeviceModeTable&& o) noexcept
        : d_xm(o.d_xm), d_xn(o.d_xn), arena_backed(o.arena_backed) {
        o.d_xm = nullptr;
        o.d_xn = nullptr;
        o.arena_backed = false;
    }
    DeviceModeTable& operator=(DeviceModeTable&& o) noexcept {
        if (this != &o) {
            d_xm = o.d_xm;
            d_xn = o.d_xn;
            arena_backed = o.arena_backed;
            o.d_xm = nullptr;
            o.d_xn = nullptr;
            o.arena_backed = false;
        }
        return *this;
    }
};

// Build + upload the folded-mode table (same values/layout the legacy
// fourierCreate produced; now built by the transform module's
// modeTableCreate). With `arena == nullptr` each array is its own
// cudaMalloc; with an arena they are aligned named subspans of the stage
// allocation.
template <typename T>
DeviceModeTable modeTableCreate(const DeviceParams<T>& p, DeviceArena* arena = nullptr);

inline void modeTableFree(DeviceModeTable& mt) {
    if (!mt.arena_backed) {
        cudaFree(mt.d_xm);
        cudaFree(mt.d_xn);
    }
    mt = DeviceModeTable{};
}

}  // namespace cumes
