// tridiagonal_backend.hpp — backend-neutral batched tridiagonal interface
// (blueprint §6.9, §8.9).
//
// The radial preconditioner solves one tridiagonal system per (mode, component,
// parity). The backend contract states its supported row range and numerical
// pivot policy; a backend must never silently process only a prefix of the rows.
//
// The solve covers rows [first_surface[mode], last_surface) of each system with
// zero Dirichlet at both ends (x[first_surface-1] = x[last_surface] = 0), so
// the LCFS row (last_surface == ns-1) is excluded for fixed boundary. The
// lower/diagonal/upper coefficients follow the blueprint §4.9 naming: `lower`
// multiplies x[j-1], `diagonal` x[j], `upper` x[j+1]. (The historical
// "ar"/"br"/"dr" names reversed these roles; the target API keeps the
// lower/diagonal/upper names to prevent that trap.)
//
// Phase 8 provides two concrete backends:
//   - PcrBackend: the legacy 128-thread grid-stride parallel cyclic reduction,
//     the production default. Correct for arbitrary rows but uses O(ns) dynamic
//     shared memory.
//   - ThomasBackend: serial Thomas, one thread per system, O(ns) global scratch
//     but no shared-memory cap — the scalar reference and the small-batch
//     (axisymmetric) fallback.
// A tiled PCR/Thomas hybrid and a library backend are deferred to the
// benchmark-gated §8.9 follow-up.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace cumes {

// Per-solve numerical status (blueprint §4.9). A pivot below the scale-aware
// floor is *reported*, never silently converted to a positive constant. The
// status is accumulated on the device into the caller-owned int buffer passed
// to enqueue_solve (the total count of systems that hit a sub-floor pivot);
// kOk means no system broke down.
enum class TridiagonalStatus : std::uint8_t {
  kOk = 0,
  kSingular = 1,  // at least one system hit a pivot below the floor
};

// Scale-aware pivot policy (blueprint §4.9). The pivot floor is relative to the
// local coefficient scale, not the legacy absolute 1e-30:
//
//   floor = kappa * epsilon_T * scale
//
// where scale = max(|lower|, |diagonal|, |upper|) over the system's solved rows
// and epsilon_T is the scalar machine epsilon. A sub-floor pivot is guarded with
// copysign(floor, pivot) so the solve stays finite, and the breakdown is counted
// into the status. The legacy absolute clamp is reproduced by kappa = 1e30 for
// a degenerate test; production uses the relative form.
struct PivotPolicy {
  double kappa = 1.0;  // safety factor on the epsilon*scale floor
};

// Strided batch view over the per-mode tridiagonal systems. Each coefficient
// array is [mode][surface], surface contiguous (the legacy precon layout).
// `rhs` is [rhs_count][mode][surface] with consecutive rhs planes `rhs_stride`
// elements apart — production solves two spectral components per matrix with
// one shared elimination, so `rhs_count == 2` and `rhs_stride == 3*mnmax*ns`
// (comps 0/3 for R, 1/4 for Z). `first_surface` is a device int[modes] giving
// jMin per mode; `last_surface` is the shared exclusive solve end (ns-1).
template <class T>
struct StridedBatchTridiagonalView {
  const T* lower = nullptr;             // [mode][surface], x[j-1] coefficient
  const T* diagonal = nullptr;          // [mode][surface], x[j] coefficient
  const T* upper = nullptr;             // [mode][surface], x[j+1] coefficient
  T* rhs = nullptr;                     // [rhs_count][mode][surface]
  const int* first_surface = nullptr;   // device int[modes]: jMin per mode
  const T* scale = nullptr;             // device T[modes]: coefficient scale for
                                        // the pivot floor (max |lower/diagonal/
                                        // upper| over the solved rows)
  int rhs_count = 1;
  std::size_t rhs_stride = 0;           // elements between consecutive rhs planes
  int modes = 0;                        // number of systems (batch size)
  int surfaces = 0;                     // ns (full row count per system)
  int last_surface = 0;                 // exclusive solve end (ns-1, LCFS excluded)
};

struct BackendLimits {
  std::size_t max_rows = 0;    // max solved rows per system (0 = unbounded)
  std::size_t max_batch = 0;   // max systems (0 = unbounded)
};

template <class T>
class TridiagonalBackend {
 public:
  virtual ~TridiagonalBackend() = default;
  virtual BackendLimits limits() const noexcept = 0;

  // Solve the batch in place: overwrites rhs[..][mode][surface] for surfaces in
  // [first_surface[mode], last_surface); all other rhs elements are untouched.
  // Accumulates the sub-floor-pivot breakdown count into *status (device side);
  // the caller owns and resets that int. Never silently processes only a prefix
  // of the rows.
  virtual void enqueue_solve(const StridedBatchTridiagonalView<T>& matrix,
                             int* status, cudaStream_t stream) = 0;
};

// The production 128-thread grid-stride PCR (extracted bit-for-bit from the
// legacy tridiagSolveKernel). Shared memory scales as O(ns) (10*ns*sizeof(T));
// within the validated ns <= 512 range that stays under the default limit.
template <class T>
class PcrBackend : public TridiagonalBackend<T> {
 public:
  PcrBackend() = default;
  explicit PcrBackend(PivotPolicy policy) : policy_(policy) {}
  BackendLimits limits() const noexcept override;
  void enqueue_solve(const StridedBatchTridiagonalView<T>& matrix, int* status,
                     cudaStream_t stream) override;

 private:
  PivotPolicy policy_{};
};

// Serial Thomas backend (one thread per system). O(ns) dynamic shared memory
// per block (one mode per block, so bounded by the ns <= 512 shape cap, not the
// batch size). The scalar reference and the small-batch (axisymmetric) fallback;
// numerically distinct from PCR at the rounding level (different elimination
// order). Stateless — it owns no device memory.
template <class T>
class ThomasBackend : public TridiagonalBackend<T> {
 public:
  ThomasBackend() = default;
  explicit ThomasBackend(PivotPolicy policy) : policy_(policy) {}
  BackendLimits limits() const noexcept override;
  void enqueue_solve(const StridedBatchTridiagonalView<T>& matrix, int* status,
                     cudaStream_t stream) override;

 private:
  PivotPolicy policy_{};
};

}  // namespace cumes
