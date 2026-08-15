// prolongation.hpp — coarse-to-fine grid-state interpolation boundary
// (blueprint §4.11, §6.11).
//
// Interpolates a converged state onto a finer radial grid: odd modes in the
// scalxc-decomposed coordinate, old-axis 2*x1-x2 extrapolation, odd-m zeroed at
// the new axis, LCFS copied exactly. The legacy interpolateState (refine.cu) is
// the reference implementation.
#pragma once

#include <cuda_runtime.h>

#include "cumes/core/tensor_view.cuh"

namespace cumes {

template <class T>
class Prolongation {
 public:
  // state_old (coarse) -> state_new (fine), both physical coefficients.
  void enqueue(SpectralView<const T, PhysicalStateDomain> state_old,
               SpectralView<T, PhysicalStateDomain> state_new,
               cudaStream_t stream) const;
};

}  // namespace cumes
