// constraint_operator.hpp — spectral-condensation constraint boundary
// (blueprint §6.8).
//
// The constraint operator owns its ConstraintWorkspace (reference fields,
// bandpass metadata, multiplier). Transitional strangler-fig form: it still
// names the legacy FourierPlan/PreconWorkspace in the enqueue signature (for
// the geometry derivatives and preconditioner elements); those become typed
// views once the FourierPlan split lands. The de-alias backend is selected by
// passing the axisymmetric operator pointer (null -> generic cuFFT).
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "cumes/state/real_fields.cuh"
#include "constraint.cuh"
#include "fourier.cuh"
#include "precon.cuh"

namespace cumes {

template <class T> class AxisymmetricOperator;

// Versioned constraint reference (blueprint §6.8).
struct ConstraintState {
  std::uint64_t reference_state_version = 0;
  int reference_iteration = 0;
  double tcon = 1.0;
};

template <class T>
class ConstraintOperator {
 public:
  ConstraintOperator(const GridParams<T>& p, DeviceArena* arena)
      : cw_(constraintCreate(p, arena)) {}
  ~ConstraintOperator() { constraintFree(cw_); }

  ConstraintOperator(const ConstraintOperator&) = delete;
  ConstraintOperator& operator=(const ConstraintOperator&) = delete;
  ConstraintOperator(ConstraintOperator&&) noexcept = default;
  ConstraintOperator& operator=(ConstraintOperator&&) noexcept = default;

  // Reconstruct the xmpq-weighted R_con/Z_con (already produced by the fused
  // inverse / axisymmetric rzCon), compute the effective constraint force,
  // bandpass it, and add it to brmn/bzmn. `precon_updated` refreshes tcon from
  // the current preconditioner elements. `axisym` selects the direct-poloidal
  // de-alias; nullptr selects the generic cuFFT bandpass.
  void enqueue(const GridParams<T>& p, const FourierPlan<T>& fp,
               const PreconWorkspace<T>& pw, const T* sqrtS_F, bool precon_updated,
               AxisymmetricOperator<T>* axisym, cudaStream_t stream);

  // Reset rCon0/zCon0 to the LCFS-extrapolated profile (first pass / restart).
  void reset_reference(const GridParams<T>& p, const T* sqrtS_F, cudaStream_t stream);

  const ConstraintWorkspace<T>& workspace() const { return cw_; }

 private:
  ConstraintWorkspace<T> cw_;
  ConstraintState state_;
};

}  // namespace cumes
