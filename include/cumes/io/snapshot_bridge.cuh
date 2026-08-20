// snapshot_bridge.cuh — device spectral state -> host EquilibriumSnapshot
// (the "Phase 3 snapshot bridge" the writers consume).
//
// SpectralStorage<T> already co-locates the six coefficient families into one
// contiguous slab (`state_slab()`, 6*mnmax*ns T values) in exactly
// EquilibriumSnapshot::Component order (Rcc Zsc Lsc Rss Zcs Lcs), each family
// mode-major / surface-contiguous. So the bridge is a single D2H copy followed
// by a per-element T -> double conversion. The host snapshot is always double
// regardless of T (the on-disk state container stays double).
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

#include "cumes/core/checked_size.hpp"
#include "cumes/io/equilibrium_snapshot.hpp"
#include "cumes/runtime/cuda_status.hpp"
#include "cumes/state/spectral_storage.hpp"

namespace cumes {

template <class T>
EquilibriumSnapshot snapshot_from_device(const SpectralStorage<T>& storage) {
    // Element counts go through checked_mul per the checked_size.hpp mandate
    // (no bare a * b). The storage comes from a validated problem, so an
    // overflow here is unreachable in-tree; report it like a CUDA failure
    // (the application boundary catches CumesError) rather than silently
    // truncating.
    auto one = checked_mul(static_cast<std::size_t>(storage.ns()),
                           static_cast<std::size_t>(storage.mnmax()));
    if (!one) {
        throw CumesError("snapshot_from_device: element count overflows size_t");
    }
    auto count = checked_mul(static_cast<std::size_t>(EquilibriumSnapshot::kCount),
                             *one);
    if (!count) {
        throw CumesError("snapshot_from_device: element count overflows size_t");
    }

    std::vector<T> buf(*count);
    if (*count != 0) {
        auto bytes = checked_mul(*count, sizeof(T));
        if (!bytes) {
            throw CumesError("snapshot_from_device: byte count overflows size_t");
        }
        check_cuda(cudaMemcpy(buf.data(), storage.state_slab(), *bytes,
                              cudaMemcpyDeviceToHost),
                   "snapshot_from_device");
    }

    EquilibriumSnapshot snap;
    snap.ns = storage.ns();
    snap.mnmax = storage.mnmax();
    for (int c = 0; c < EquilibriumSnapshot::kCount; ++c) {
        snap.families[c].resize(*one);
        const T* src = buf.data() + static_cast<std::size_t>(c) * *one;
        for (std::size_t i = 0; i < *one; ++i) {
            snap.families[c][i] = static_cast<double>(src[i]);
        }
    }
    return snap;
}

}  // namespace cumes
