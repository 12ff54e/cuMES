// axisymmetric_operator.hpp — the axisymmetric spectral backend (blueprint
// §8.5).
//
// For `ntor = 0, nzeta = 1` the toroidal direction is a single point and every
// folded mode has n = 0, so the product basis collapses to
//   R = Σ rmncc·cos(mθ),  Z = Σ zmnsc·sin(mθ),  λ = Σ lmnsc·sin(mθ)
// with zero toroidal derivatives and no sin(nζ) families (sin(0) = 0). The
// generic cuFFT backend produces exactly this after its length-one Z2D/D2Z, so
// the axisymmetric backend performs the same poloidal synthesis/projection
// directly and never creates or executes a length-one cuFFT plan.
//
// The operator owns only its per-mode trigonometric tables (cos/sin/mcos/msin
// plus the reduced-grid trapezoid weights) — never geometry, force, or
// diagnostics (blueprint §5.1). It is a concrete `SpectralOperator` backend;
// the generic `ToroidalFftOperator` (batched 1D ζ-cuFFT + tiled direct
// poloidal accumulation) remains the reference the axisymmetric backend is
// differentially tested against.
#pragma once

#include "cumes/core/tensor_view.cuh"
#include "cumes/runtime/device_buffer.cuh"
#include "cumes/state/real_fields.cuh"
#include "cumes/transforms/spectral_operator.hpp"
#include "vmec_types.h"

#include <cuda_runtime.h>

namespace cumes {

template <class T>
class AxisymmetricOperator : public SpectralOperator<T> {
   public:
    AxisymmetricOperator() = default;

    // Build the poloidal tables and validate that the shape is axisymmetric
    // (ntor == 0, nzeta == 1). Throws CumesError otherwise. Table construction
    // is synchronous H2D work (stage setup, not the hot loop).
    explicit AxisymmetricOperator(const DeviceParams<T>& p);

    ~AxisymmetricOperator() override = default;

    AxisymmetricOperator(const AxisymmetricOperator&) = delete;
    AxisymmetricOperator& operator=(const AxisymmetricOperator&) = delete;
    AxisymmetricOperator(AxisymmetricOperator&&) noexcept = default;
    AxisymmetricOperator& operator=(AxisymmetricOperator&&) noexcept = default;

    // Direct poloidal synthesis: coefficients -> the 18 parity-split real-space
    // geometry arrays (R/Z/λ × value/θ-deriv/ζ-deriv × even/odd). Toroidal
    // derivatives are written as zero. The xmpq-weighted rCon/zCon (when
    // non-null) are produced by the same direct-poloidal rzcon kernel (a
    // separate launch, ordered after the synthesis — see enqueue_rzcon).
    void enqueue_inverse(
        SpectralView<const T, PhysicalStateDomain> coefficients,
        GeometryParityViews<T> geometry,
        RealFieldView<T> rCon,
        RealFieldView<T> zCon,
        cudaStream_t stream) override;

    // Direct reduced-θ trapezoid projection: parity forces + constraint force
    // -> six spectral-force families. The sin(nζ) families (frss/fzcs/flcs) are
    // written as zero (no n=0 basis function); axis/LCFS rules match the
    // generic recover (axis keeps m=0 frcc only, LCFS keeps λ only).
    void enqueue_forward(ForceParityViews<const T> real_force,
                         ConstraintForceViews<const T> constraint_force,
                         SpectralView<T, DecomposedResidualDomain> residual,
                         cudaStream_t stream) override;

    // Axisymmetric constraint helpers (blueprint §8.5 "constraint/bandpass
    // backend"): the xmpq-weighted rCon/zCon reconstruction and the de-alias
    // bandpass both become direct poloidal sums when ntor=0/nzeta=1.

    // xmpq-weighted rCon/zCon synthesis: rCon = Σ m(m-1)·rmncc·cos(mθ),
    // zCon = Σ m(m-1)·zmnsc·sin(mθ) (no parity split, no scalxc — rCon/zCon are
    // full real-space fields, matching the fused inverse DFT's rCon/zCon).
    void enqueue_rzcon(SpectralView<const T, PhysicalStateDomain> coefficients,
                       RealFieldView<T> rCon,
                       RealFieldView<T> zCon,
                       cudaStream_t stream);

    // De-alias bandpass: gCon = Σ_{m=1..mpol-2} (2/nZnT)·tcon·faccon[m]·
    // (Σ_θ gConEff·sin(mθ))·sin(mθ). The axis (surface 0) is left untouched
    // (it is never consumed). `tcon`/`faccon` are the constraint's device
    // profiles (the same arrays deAliasCoeffPackKernel reads).
    void enqueue_dealias(RealFieldView<const T> gConEff,
                         const T* tcon,
                         const T* faccon,
                         RealFieldView<T> gCon,
                         cudaStream_t stream) override;

    const DeviceParams<T>& params() const { return p_; }

   private:
    DeviceParams<T> p_{};
    DeviceBuffer<T> cos_th_;   // [mpol][ntheta]  cos(mθ)
    DeviceBuffer<T> sin_th_;   // [mpol][ntheta]  sin(mθ)
    DeviceBuffer<T> mcos_th_;  // [mpol][ntheta]  m·cos(mθ)
    DeviceBuffer<T> msin_th_;  // [mpol][ntheta]  -m·sin(mθ)
    DeviceBuffer<T> fwd_w_;    // [ntheta/2+1] reduced-grid trapezoid weights
};

}  // namespace cumes
