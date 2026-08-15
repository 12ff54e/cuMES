// refine.cuh — grid-sequencing state interpolation (multi-radial-grid).
#pragma once
#include <cuda_runtime.h>
#include "vmec_types.h"
#include "cumes/state/spectral_storage.hpp"

// vmecpp Vmec::InterpolateToNextMultigridStep (vmec.cc:1795-2042), kLinear
// scheme: the converged PHYSICAL state on p_old is interpolated in
// s = j/(ns-1) onto p_new — odd-m in scalxc-decomposed space (axis value
// extrapolated 2*x[1]-x[2], odd-m zeroed at the new axis), LCFS copied
// exactly (boundary pinned). st_new's 12 arrays are allocated here and the
// velocities zeroed; the caller owns st_new and must freeState() it.
template <typename T>
cumes::SpectralStorage<T> interpolateState(const GridParams<T>& p_new,
                                           const cumes::SpectralStorage<T>& st_old,
                                           const GridParams<T>& p_old,
                                           cudaStream_t stream = 0);
