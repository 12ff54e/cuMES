// snapshot_bridge.cuh — device spectral state -> host EquilibriumSnapshot
// (the "Phase 3 snapshot bridge" the writers consume).
//
// SpectralStorage<T> already co-locates the six coefficient families into one
// contiguous slab (`state_slab()`, 6*mnmax*ns T values) in exactly
// EquilibriumSnapshot::Component order (Rcc Zsc Lsc Rss Zcs Lcs), each family
// mode-major / surface-contiguous. So the bridge is a single D2H copy followed
// by a per-element T -> double conversion — the same conversion
// outputSaveBinary performs (src/output.cpp), which is what makes the two paths
// byte-identical. The host snapshot is always double regardless of T (the
// on-disk state container stays double).
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

#include "cumes/io/equilibrium_snapshot.hpp"
#include "cumes/runtime/cuda_status.hpp"
#include "cumes/state/spectral_storage.hpp"

namespace cumes {

template <class T>
EquilibriumSnapshot snapshot_from_device(const SpectralStorage<T>& storage) {
    const std::size_t one =
        static_cast<std::size_t>(storage.ns()) * storage.mnmax();
    const std::size_t count = EquilibriumSnapshot::kCount * one;

    std::vector<T> buf(count);
    if (count != 0) {
        check_cuda(cudaMemcpy(buf.data(), storage.state_slab(),
                              count * sizeof(T), cudaMemcpyDeviceToHost),
                   "snapshot_from_device");
    }

    EquilibriumSnapshot snap;
    snap.ns = storage.ns();
    snap.mnmax = storage.mnmax();
    for (int c = 0; c < EquilibriumSnapshot::kCount; ++c) {
        snap.families[c].resize(one);
        const T* src = buf.data() + static_cast<std::size_t>(c) * one;
        for (std::size_t i = 0; i < one; ++i) {
            snap.families[c][i] = static_cast<double>(src[i]);
        }
    }
    return snap;
}

}  // namespace cumes
