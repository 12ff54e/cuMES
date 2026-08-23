// prolongation.hpp — coarse-to-fine grid-state interpolation boundary
// (blueprint §4.11, §6.11).
//
// Interpolates a converged state onto a finer radial grid: odd modes in the
// scalxc-decomposed coordinate, old-axis 2*x1-x2 extrapolation, odd-m zeroed at
// the new axis, LCFS copied exactly (vmecpp
// Vmec::InterpolateToNextMultigridStep, kLinear). (Migration step 13.3: the
// legacy interpolateState free function and the refine.cuh/refine_impl.cuh
// module are gone — the body is Prolongation::enqueue, defined in
// src/prolongation_impl.cuh.)
#ifndef CUMES_INCLUDE_CUMES_NUMERICS_PROLONGATION_HPP_
#define CUMES_INCLUDE_CUMES_NUMERICS_PROLONGATION_HPP_

#include "cumes/state/spectral_storage.hpp"

#include <cuda_runtime.h>

namespace cumes {

template <class T>
class Prolongation {
   public:
    // state_old (coarse) -> state_new (fine), both physical coefficients. The
    // new state's slabs are allocated here (velocities zeroed) and returned.
    //
    // Precondition: ns_new > ns_old >= 3 with equal mnmax (validation enforces
    // this for validated problems). A violation throws cumes::CumesError — the
    // library never calls exit().
    SpectralStorage<T> enqueue(const DeviceParams<T>& p_new,
                               const SpectralStorage<T>& state_old,
                               const DeviceParams<T>& p_old,
                               cudaStream_t stream) const;
};

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_NUMERICS_PROLONGATION_HPP_
