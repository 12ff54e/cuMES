// prolongation.hpp — coarse-to-fine grid-state interpolation boundary
// (blueprint §4.11, §6.11).
//
// Interpolates a converged state onto a finer radial grid: odd modes in the
// scalxc-decomposed coordinate, old-axis 2*x1-x2 extrapolation, odd-m zeroed at
// the new axis, LCFS copied exactly. The legacy interpolateState (refine.cu) is
// the reference implementation.
//
// Strangler-fig form: a stateless thin wrapper over interpolateState (verbatim
// — migration step 11), which allocates the new state's slabs and returns them.
#pragma once

#include <cuda_runtime.h>

#include "cumes/state/spectral_storage.hpp"
#include "refine.cuh"

namespace cumes {

template <class T>
class Prolongation {
 public:
  // state_old (coarse) -> state_new (fine), both physical coefficients. The new
  // state's slabs are allocated here (velocities zeroed) and returned.
  SpectralStorage<T> enqueue(const GridParams<T>& p_new,
                             const SpectralStorage<T>& state_old,
                             const GridParams<T>& p_old,
                             cudaStream_t stream) const {
    return interpolateState(p_new, state_old, p_old, stream);
  }
};

}  // namespace cumes
