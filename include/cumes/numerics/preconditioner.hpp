// preconditioner.hpp — radial tridiagonal + lambda preconditioner boundary
// (blueprint §6.9).
//
// Computes the per-(m,n) lower/diagonal/upper coefficients and the lambda
// diagonal (the "ar"/"br"/"dr" naming trap is replaced with lower/diagonal/upper
// here), then solves the batched systems through a TridiagonalBackend. The
// legacy preconCompute/preconApply are the reference implementation.
//
// Transitional strangler-fig form: the operator OWNS its PreconWorkspace but
// still names the legacy FourierPlan/RadialProfiles/MetricWorkspace in the
// assemble step (it reads the mode tables and geometry). Those become typed
// views once the mode-table extraction and FourierPlan split land (blueprint
// §6.2/§6.6); the solve step already goes through the clean TridiagonalBackend.
#pragma once

#include <cuda_runtime.h>

#include "cumes/numerics/tridiagonal_backend.hpp"
#include "cumes/state/real_fields.cuh"
#include "fourier.cuh"
#include "geometry.cuh"
#include "precon.cuh"
#include "profiles.cuh"

namespace cumes {

template <class T>
class Preconditioner {
 public:
  Preconditioner(const GridParams<T>& p, DeviceArena* arena)
      : pw_(preconCreate(p, arena)) {}
  ~Preconditioner() { preconFree(pw_); }

  Preconditioner(const Preconditioner&) = delete;
  Preconditioner& operator=(const Preconditioner&) = delete;
  Preconditioner(Preconditioner&&) noexcept = default;
  Preconditioner& operator=(Preconditioner&&) noexcept = default;

  // Refresh the matrix coefficients from the current geometry/field.
  void enqueue_compute(const RealSpaceStorage<T>& rs, const FourierPlan<T>& fp,
                       const GridParams<T>& p, const RadialProfiles<T>& rp,
                       const MetricWorkspace<T>& mw, cudaStream_t stream);

  // Apply the preconditioner in place to the decomposed residual.
  void enqueue_apply(SpectralView<T, DecomposedResidualDomain> residual,
                     const GridParams<T>& p, const int* xm, const int* xn,
                     cudaStream_t stream) const;

  const PreconWorkspace<T>& workspace() const { return pw_; }

 private:
  PreconWorkspace<T> pw_;
};

}  // namespace cumes
