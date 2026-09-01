// dual_spectral_operator.hpp — exact forward-mode wrapper around the linear
// double-precision spectral transforms.
#ifndef CUMES_INCLUDE_CUMES_NUMERICS_DUAL_SPECTRAL_OPERATOR_HPP_
#define CUMES_INCLUDE_CUMES_NUMERICS_DUAL_SPECTRAL_OPERATOR_HPP_

#include "cumes/numerics/forward_dual.cuh"
#include "cumes/runtime/device_buffer.cuh"
#include "cumes/state/mode_table.cuh"
#include "cumes/state/real_space_storage.hpp"
#include "cumes/state/spectral_storage.hpp"
#include "cumes/transforms/spectral_operator.hpp"
#include "cumes/transforms/toroidal_fft_operator.hpp"

#include <memory>

namespace cumes {

using ForwardDualDouble = ForwardDual<double>;

DeviceParams<ForwardDualDouble> make_forward_dual_params(
    const DeviceParams<double>& primal);

// cuFFT has no dual scalar type, but every transform used by the equilibrium
// residual is linear. Split the primal/tangent lanes, run the existing double
// transform on each lane, and merge them before the nonlinear CUDA operators.
class DualSpectralOperator final : public SpectralOperator<ForwardDualDouble> {
   public:
    using val_type = ForwardDualDouble;

    DualSpectralOperator(const DeviceParams<double>& p,
                         const DeviceModeTable& mode_table);
    ~DualSpectralOperator() override;

    DualSpectralOperator(const DualSpectralOperator&) = delete;
    DualSpectralOperator& operator=(const DualSpectralOperator&) = delete;
    DualSpectralOperator(DualSpectralOperator&&) noexcept = delete;
    DualSpectralOperator& operator=(DualSpectralOperator&&) noexcept = delete;

    void bind_stream(cudaStream_t stream);

    void enqueue_inverse(
        SpectralView<const ForwardDualDouble, PhysicalStateDomain> coefficients,
        GeometryParityViews<ForwardDualDouble> geometry,
        RealFieldView<ForwardDualDouble> rcon,
        RealFieldView<ForwardDualDouble> zcon,
        cudaStream_t stream) override;

    void enqueue_forward(
        ForceParityViews<const ForwardDualDouble> real_force,
        ConstraintForceViews<const ForwardDualDouble> constraint_force,
        SpectralView<ForwardDualDouble, DecomposedResidualDomain> residual,
        cudaStream_t stream,
        bool include_lcfs = false) override;

    void enqueue_dealias(RealFieldView<const ForwardDualDouble> gcon_eff,
                         const ForwardDualDouble* tcon,
                         const ForwardDualDouble* faccon,
                         RealFieldView<ForwardDualDouble> gcon,
                         cudaStream_t stream) override;

   private:
    DeviceParams<double> p_{};
    SpectralStorage<double> primal_state_;
    SpectralStorage<double> tangent_state_;
    RealSpaceStorage<double> primal_rs_;
    RealSpaceStorage<double> tangent_rs_;
    std::unique_ptr<ToroidalFftOperator<double>> transform_;

    DeviceBuffer<double> primal_residual_;
    DeviceBuffer<double> tangent_residual_;
    DeviceBuffer<double> primal_rcon_;
    DeviceBuffer<double> tangent_rcon_;
    DeviceBuffer<double> primal_zcon_;
    DeviceBuffer<double> tangent_zcon_;

    DeviceBuffer<double> primal_gcon_eff_;
    DeviceBuffer<double> tangent_gcon_eff_;
    DeviceBuffer<double> primal_tcon_;
    DeviceBuffer<double> tangent_tcon_;
    DeviceBuffer<double> primal_faccon_;
    DeviceBuffer<double> tangent_faccon_;
    DeviceBuffer<double> primal_gcon_;
    DeviceBuffer<double> tangent_gcon_;
    DeviceBuffer<double> tangent_gcon_term_;
    DeviceBuffer<double> tangent_faccon_term_;
};

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_NUMERICS_DUAL_SPECTRAL_OPERATOR_HPP_
