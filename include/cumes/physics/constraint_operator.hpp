// constraint_operator.hpp — spectral-condensation constraint boundary
// (blueprint §6.8).
//
// The constraint operator owns its reference fields, bandpass metadata, and
// multiplier directly (the legacy ConstraintWorkspace struct +
// constraintCreate/constraintFree/constraintCompute/constraintResetRzCon0 are
// gone). The de-alias bandpass is dispatched through the unified
// SpectralOperator interface (no backend branch): the generic backend runs the
// compact cuFFT round trip, the axisymmetric backend its direct-poloidal
// kernel.
#ifndef CUMES_INCLUDE_CUMES_PHYSICS_CONSTRAINT_OPERATOR_HPP_
#define CUMES_INCLUDE_CUMES_PHYSICS_CONSTRAINT_OPERATOR_HPP_

#include "cumes/config/device_params.hpp"
#include "cumes/solver/control_record.hpp"
#include "cumes/state/real_fields.cuh"
#include "cumes/state/real_space_storage.hpp"
#include "cumes/transforms/spectral_operator.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace cumes {

class DeviceArena;

// Versioned constraint reference (blueprint §6.8).
struct ConstraintState {
    std::uint64_t reference_state_version = 0;
    int reference_iteration = 0;
    double tcon = 1.0;
};

template <class T>
class ConstraintOperator {
   public:
    using val_type = T;

    ConstraintOperator(const DeviceParams<T>& p, DeviceArena* arena);
    ~ConstraintOperator();

    // Non-movable: the destructor frees the owned constraint buffers without
    // nulling, so a defaulted move would double-free (review finding 3.2).
    ConstraintOperator(const ConstraintOperator&) = delete;
    ConstraintOperator& operator=(const ConstraintOperator&) = delete;
    ConstraintOperator(ConstraintOperator&&) noexcept = delete;
    ConstraintOperator& operator=(ConstraintOperator&&) noexcept = delete;

    // Reconstruct the xmpq-weighted R_con/Z_con (already produced by the fused
    // inverse / axisymmetric rzCon), compute the effective constraint force,
    // bandpass it, and add it to brmn/bzmn. `precon_updated` refreshes tcon
    // from the current preconditioner elements. `op` is the selected transform
    // backend whose enqueue_dealias performs the bandpass. `rs` carries the
    // parity-split geometry derivatives + force buffers.
    // Status-guarded (completion plan step 1.4): the tcon cache, the
    // constraint-force scratch, and the brmn/bzmn += targets are not written
    // when status->jacobian_valid is clear.
    void enqueue(const DeviceParams<T>& p,
                 const RealSpaceStorage<T>& rs,
                 const T* ard,
                 const T* azd,
                 const T* sqrtS_F,
                 bool precon_updated,
                 SpectralOperator<T>* op,
                 const ControlStatus* status,
                 cudaStream_t stream);

    // Reset rCon0/zCon0 to the LCFS-extrapolated profile (first pass /
    // restart). Status-guarded: the reference cache is untouched on an invalid
    // pass.
    void reset_reference(const DeviceParams<T>& p,
                         const T* sqrtS_F,
                         const ControlStatus* status,
                         cudaStream_t stream);

    // ---- typed view accessors (blueprint §6.8) ------------------------------
    // rCon/zCon are the xmpq-weighted reconstruction fields (produced by the
    // fused inverse / axisym rzCon); the constraint-force views are the
    // frcon/fzcon outputs folded into the forward DFT.
    RealFieldView<T> rcon_view(const DeviceParams<T>& p) const {
        return RealFieldView<T>(d_rCon_, p.ns, p.ntheta, p.nzeta);
    }
    RealFieldView<T> zcon_view(const DeviceParams<T>& p) const {
        return RealFieldView<T>(d_zCon_, p.ns, p.ntheta, p.nzeta);
    }
    ConstraintForceViews<const T> constraint_force_views(
        const DeviceParams<T>& p) const {
        auto f = [&](T* d) {
            return RealFieldView<const T>(d, p.ns, p.ntheta, p.nzeta);
        };
        ConstraintForceViews<const T> v;
        v.frcon_e = f(d_frcon_e_);
        v.frcon_o = f(d_frcon_o_);
        v.fzcon_e = f(d_fzcon_e_);
        v.fzcon_o = f(d_fzcon_o_);
        return v;
    }

    // Dump/observability accessors (CUMES_DUMP-gated only).
    const T* gcon_eff() const { return d_gConEff_; }
    const T* gcon() const { return d_gCon_; }
    const T* tcon() const { return d_tcon_; }
    const T* faccon() const { return d_faccon_; }
    const T* h_faccon() const { return h_faccon_; }
    T* rcon0() const { return d_rCon0_; }
    T* zcon0() const { return d_zCon0_; }

   private:
    // Steps 0/1 (tcon refresh + gConEff) and step 3 (add to brmn/bzmn +
    // frcon/fzcon), shared by both transform backends (step 2 bandpass
    // differs).
    void enqueue_head(const DeviceParams<T>& p,
                      const RealSpaceStorage<T>& rs,
                      const T* ard,
                      const T* azd,
                      const T* sqrtS_F,
                      bool precon_updated,
                      const ControlStatus* status,
                      cudaStream_t stream);
    void enqueue_tail(const DeviceParams<T>& p,
                      const RealSpaceStorage<T>& rs,
                      const T* sqrtS_F,
                      const ControlStatus* status,
                      cudaStream_t stream);

    T* d_gConEff_ = nullptr;
    T* d_gCon_ = nullptr;
    T* d_rCon_ = nullptr;
    T* d_zCon_ = nullptr;
    T* d_rCon0_ = nullptr;
    T* d_zCon0_ = nullptr;
    T* d_tcon_ = nullptr;
    T* h_faccon_ = nullptr;  // host (pinned, never arena)
    T* d_faccon_ = nullptr;
    T* d_frcon_e_ = nullptr;
    T* d_frcon_o_ = nullptr;
    T* d_fzcon_e_ = nullptr;
    T* d_fzcon_o_ = nullptr;
    bool arena_backed_ = false;
    ConstraintState state_;
};

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_PHYSICS_CONSTRAINT_OPERATOR_HPP_
