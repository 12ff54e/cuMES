// constraint_operator.hpp — spectral-condensation constraint boundary
// (blueprint §6.8).
//
// The constraint operator owns its ConstraintWorkspace (reference fields,
// bandpass metadata, multiplier). Transitional strangler-fig form: it still
// names the legacy PreconWorkspace in the enqueue signature (for the
// preconditioner elements feeding tcon); those become typed views once the
// mode-table extraction lands. The de-alias bandpass is dispatched through the
// unified SpectralOperator interface (no backend branch): the generic backend
// runs the compact cuFFT round trip, the axisymmetric backend its direct-
// poloidal kernel.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "cumes/state/real_fields.cuh"
#include "cumes/transforms/spectral_operator.hpp"
#include "constraint.cuh"
#include "precon.cuh"

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
  ConstraintOperator(const DeviceParams<T>& p, DeviceArena* arena)
      : cw_(constraintCreate(p, arena)) {}
  ~ConstraintOperator() { constraintFree(cw_); }

  ConstraintOperator(const ConstraintOperator&) = delete;
  ConstraintOperator& operator=(const ConstraintOperator&) = delete;
  ConstraintOperator(ConstraintOperator&&) noexcept = default;
  ConstraintOperator& operator=(ConstraintOperator&&) noexcept = default;

  // Reconstruct the xmpq-weighted R_con/Z_con (already produced by the fused
  // inverse / axisymmetric rzCon), compute the effective constraint force,
  // bandpass it, and add it to brmn/bzmn. `precon_updated` refreshes tcon from
  // the current preconditioner elements. `op` is the selected transform
  // backend whose enqueue_dealias performs the bandpass. `rs` carries the
  // parity-split geometry derivatives + force buffers.
  void enqueue(const DeviceParams<T>& p, const RealSpaceStorage<T>& rs,
               const PreconWorkspace<T>& pw, const T* sqrtS_F,
               bool precon_updated, SpectralOperator<T>* op, cudaStream_t stream);

  // Reset rCon0/zCon0 to the LCFS-extrapolated profile (first pass / restart).
  void reset_reference(const DeviceParams<T>& p, const T* sqrtS_F, cudaStream_t stream);

  const ConstraintWorkspace<T>& workspace() const { return cw_; }

  // ---- typed view accessors (blueprint §6.8) ------------------------------
  // The solver consumes these instead of naming ConstraintWorkspace's raw
  // `d_*` pointers. rCon/zCon are the xmpq-weighted reconstruction fields
  // (produced by the fused inverse / axisym rzCon); the constraint-force views
  // are the frcon/fzcon outputs folded into the forward DFT.
  RealFieldView<T> rcon_view(const DeviceParams<T>& p) const {
    return RealFieldView<T>(cw_.d_rCon, p.ns, p.ntheta, p.nzeta);
  }
  RealFieldView<T> zcon_view(const DeviceParams<T>& p) const {
    return RealFieldView<T>(cw_.d_zCon, p.ns, p.ntheta, p.nzeta);
  }
  ConstraintForceViews<const T> constraint_force_views(const DeviceParams<T>& p) const {
    auto f = [&](T* d) {
      return RealFieldView<const T>(d, p.ns, p.ntheta, p.nzeta);
    };
    ConstraintForceViews<const T> v;
    v.frcon_e = f(cw_.d_frcon_e); v.frcon_o = f(cw_.d_frcon_o);
    v.fzcon_e = f(cw_.d_fzcon_e); v.fzcon_o = f(cw_.d_fzcon_o);
    return v;
  }

 private:
  ConstraintWorkspace<T> cw_;
  ConstraintState state_;
};

}  // namespace cumes
