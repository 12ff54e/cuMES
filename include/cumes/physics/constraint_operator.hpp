// constraint_operator.hpp — spectral-condensation constraint boundary
// (blueprint §6.8).
//
// The constraint operator owns its reference fields, bandpass metadata, and
// multiplier; it never borrows a raw hidden Fourier pointer. Its explicit inputs
// are the tcon0 scale, the reset cadence, and a versioned reference. The legacy
// constraintRzConCompute/constraintCompute are the reference implementation.
#pragma once

#include <cstdint>

#include "cumes/state/real_fields.cuh"
#include "cumes/state/spectral_storage.hpp"

namespace cumes {

// Versioned constraint reference (blueprint §6.8).
struct ConstraintState {
  std::uint64_t reference_state_version = 0;
  int reference_iteration = 0;
  double tcon = 1.0;
};

template <class T>
class ConstraintOperator {
 public:
  // Reconstruct the xmpq-weighted R_con/Z_con from the spectral state and add
  // the constraint force to the real-space R/Z forces. `reset_reference`
  // refreshes the LCFS-extrapolated rCon0/zCon0 (blueprint §4.8).
  void enqueue(const SpectralView<const T, PhysicalStateDomain>& state,
               const RadialProfileViews<T>& radial, ForceParityViews<T> force,
               bool reset_reference, cudaStream_t stream) const;

  ConstraintState& state() { return state_; }
  const ConstraintState& state() const { return state_; }

 private:
  ConstraintState state_;
};

}  // namespace cumes
