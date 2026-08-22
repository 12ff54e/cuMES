// preconditioner.hpp — radial tridiagonal + lambda preconditioner boundary
// (blueprint §6.9).
//
// Computes the per-(m,n) lower/diagonal/upper coefficients and the lambda
// diagonal (the "ar"/"br"/"dr" naming trap is replaced with
// lower/diagonal/upper in the TridiagonalBackend), then solves the batched
// systems. (Migration step 13.3: the operator OWNS the workspace buffers
// directly — the legacy PreconWorkspace struct +
// preconCreate/preconFree/preconCompute/preconApply are gone.)
#pragma once

#include "cumes/numerics/tridiagonal_backend.hpp"
#include "cumes/solver/control_record.hpp"
#include "cumes/state/real_fields.cuh"
#include "cumes/state/real_space_storage.hpp"

#include <cuda_runtime.h>

namespace cumes {

template <class T>
class Preconditioner {
   public:
    Preconditioner(const DeviceParams<T>& p, DeviceArena* arena);
    ~Preconditioner();

    Preconditioner(const Preconditioner&) = delete;
    Preconditioner& operator=(const Preconditioner&) = delete;
    // Moves deleted (not transfer-and-null): the destructor unconditionally
    // cudaFrees the raw workspace pointers (skipped only when arena_backed_),
    // so a defaulted move would double-free the moved-from object's pointers.
    // No call site moves a Preconditioner (stack locals / a member of the
    // non-movable EquilibriumOperator).
    Preconditioner(Preconditioner&&) = delete;
    Preconditioner& operator=(Preconditioner&&) = delete;

    // Refresh the matrix coefficients from the current geometry/field.
    // Status-guarded (completion plan step 1.4): on an invalid-Jacobian pass
    // (status->jacobian_valid == 0) every kernel no-ops and the element cache
    // is left untouched (the re-anchor makes the next pass a refresh pass).
    void enqueue_compute(const RealSpaceStorage<T>& rs,
                         const int* xm,
                         const int* xn,
                         const DeviceParams<T>& p,
                         const RadialProfileViews<T>& rpv,
                         const BaseGeometryHalfViews<T>& base,
                         const MagneticFieldViews<T>& field,
                         const ControlStatus* status,
                         cudaStream_t stream);

    // Apply the preconditioner in place to the decomposed residual.
    // Terminal-guarded (completion plan step 1.4): the tridiagonal solves and
    // the boundary/lambda finishing no-op when `gate` reports a nonfinite or
    // converged invariant residual (gate may be nullptr for direct callers).
    void enqueue_apply(SpectralView<T, DecomposedResidualDomain> residual,
                       const DeviceParams<T>& p,
                       const ControlStatus* gate,
                       cudaStream_t stream) const;

    // vmecpp applyM1Preconditioner: scale the m=1 frss/fzcs pair by the
    // odd-parity diagonal elements (ard/brd/azd/bzd) before the RZ solve. Runs
    // after the invariant residuals, before enqueue_apply. Terminal-guarded
    // like enqueue_apply.
    void enqueue_m1_scale(SpectralView<T, DecomposedResidualDomain> residual,
                          const DeviceParams<T>& p,
                          const ControlStatus* gate,
                          cudaStream_t stream) const;

    // The m=1 even-parity R/Z diagonal elements, consumed by the constraint's
    // tcon profile (the only workspace elements the constraint reads).
    T* ard() const { return d_ard_; }
    T* azd() const { return d_azd_; }

    // Dump/observability accessors (the CUMES_DUMP-gated vmecpp-comparison dump
    // machinery only; never the hot loop).
    const T* ar() const { return d_ar_; }
    const T* dr() const { return d_dr_; }
    const T* br() const { return d_br_; }
    const T* az() const { return d_az_; }
    const T* dz() const { return d_dz_; }
    const T* bz() const { return d_bz_; }
    const int* jmin() const { return d_jMin_; }
    const T* arm() const { return d_arm_; }
    const T* brm() const { return d_brm_; }
    const T* azm() const { return d_azm_; }
    const T* bzm() const { return d_bzm_; }
    const T* brd() const { return d_brd_; }
    const T* bzd() const { return d_bzd_; }
    const T* cxd() const { return d_cxd_; }
    const T* lambdaPrec() const { return d_lambdaPrec_; }

   private:
    T* d_ax_R_ = nullptr;
    T* d_ax_Z_ = nullptr;
    T* d_bx_R_ = nullptr;
    T* d_bx_Z_ = nullptr;
    T* d_cx_ = nullptr;
    T* d_arm_ = nullptr;
    T* d_brm_ = nullptr;
    T* d_azm_ = nullptr;
    T* d_bzm_ = nullptr;
    T* d_ard_ = nullptr;
    T* d_brd_ = nullptr;
    T* d_azd_ = nullptr;
    T* d_bzd_ = nullptr;
    T* d_cxd_ = nullptr;
    T* d_sm_ = nullptr;
    T* d_sp_ = nullptr;
    T* d_ar_ = nullptr;
    T* d_dr_ = nullptr;
    T* d_br_ = nullptr;
    T* d_az_ = nullptr;
    T* d_dz_ = nullptr;
    T* d_bz_ = nullptr;
    int* d_jMin_ = nullptr;
    T* d_lambdaPrec_ = nullptr;
    T* d_bLambda_ = nullptr;
    T* d_dLambda_ = nullptr;
    T* d_cLambda_ = nullptr;
    T* d_rmsPhiP_ = nullptr;
    T* d_preconScale_ = nullptr;
    int* d_preconStatus_ = nullptr;
    bool arena_backed_ = false;
};

}  // namespace cumes
