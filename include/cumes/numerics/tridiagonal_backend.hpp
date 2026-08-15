// tridiagonal_backend.hpp — backend-neutral batched tridiagonal interface
// (blueprint §6.9).
//
// The radial preconditioner solves one tridiagonal system per (mode, component,
// parity). The backend contract states its supported row range and numerical
// pivot policy; a backend must never silently process only a prefix of the rows.
// The legacy 128-thread grid-stride PCR is the reference backend; a scalable
// tiled PCR/Thomas backend and a library backend are Phase 8.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace cumes {

// Strided batch view over the per-mode tridiagonal systems: lower/diagonal/upper
// and rhs, each [mode][surface], surface contiguous. (The exact typed layout is
// finalized with the Phase 8 backend; this names the boundary contract.)
template <class T>
struct StridedBatchTridiagonalView {
  const T* lower = nullptr;
  const T* diagonal = nullptr;
  const T* upper = nullptr;
  T* rhs = nullptr;
  int modes = 0;
  int surfaces = 0;
};

struct BackendLimits {
  std::size_t max_rows = 0;
  std::size_t max_batch = 0;
};

template <class T>
class TridiagonalBackend {
 public:
  virtual ~TridiagonalBackend() = default;
  virtual BackendLimits limits() const noexcept = 0;
  virtual void enqueue_solve(StridedBatchTridiagonalView<const T> matrix,
                             StridedBatchTridiagonalView<T> rhs,
                             cudaStream_t stream) = 0;
};

}  // namespace cumes
