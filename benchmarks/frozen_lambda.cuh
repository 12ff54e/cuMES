// Experimental frozen-R/Z lambda map. Reuses the production transforms and
// field kernels; the lambda force below preserves the production expressions.
#ifndef CUMES_BENCHMARKS_FROZEN_LAMBDA_CUH_
#define CUMES_BENCHMARKS_FROZEN_LAMBDA_CUH_

#include "cumes/config/validated_problem.hpp"
#include "cumes/solver/stage_solver.hpp"

#include <cmath>
#include <cstddef>
#include <optional>

namespace lambda_bench {
namespace detail {

template <class T>
__global__ void lambda_force(cumes::GeometryParityViews<T> full,
                             cumes::BaseGeometryHalfViews<T> base,
                             cumes::MagneticFieldViews<T> field,
                             cumes::RadialProfileViews<T> radial,
                             cumes::ForceParityViews<T> force,
                             T lamscale,
                             int ns,
                             int nZnT) {
    const int j = blockIdx.y;
    const int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (j >= ns || k >= nZnT) return;
    const int idx_f = j * nZnT + k;
    const T sF_j = radial.sqrtS_F[j];
    const T sFull = sF_j * sF_j;
    T gsqrt_i = T(0), gsqrt_o = T(0), gvv_i = T(0), gvv_o = T(0);
    T guv_i = T(0), guv_o = T(0), bsupu_i = T(0), bsupu_o = T(0);
    T bsubu_i = T(0), bsubu_o = T(0), bsubv_i = T(0), bsubv_o = T(0);
    T sH_i = T(0), sH_o = T(0);
    if (j > 0) {
        const int h = (j - 1) * nZnT + k;
        gsqrt_i = base.gsqrt.data()[h];
        gvv_i = base.gvv.data()[h];
        guv_i = base.guv.data()[h];
        bsupu_i = field.bsupu.data()[h];
        bsubu_i = field.bsubu.data()[h];
        bsubv_i = field.bsubv.data()[h];
        sH_i = radial.sqrtS_H[j - 1];
    }
    if (j < ns - 1) {
        const int h = j * nZnT + k;
        gsqrt_o = base.gsqrt.data()[h];
        gvv_o = base.gvv.data()[h];
        guv_o = base.guv.data()[h];
        bsupu_o = field.bsupu.data()[h];
        bsubu_o = field.bsubu.data()[h];
        bsubv_o = field.bsubv.data()[h];
        sH_o = radial.sqrtS_H[j];
    }
    // forces_impl.cuh's hybrid lambda interpolation, including its boundary
    // and axis conventions. No pressure or R/Z force terms enter this map.
    const T bsubv_avg = T(0.5) * (bsubv_o + bsubv_i);
    const T gvv_gsqrt_i = j > 0 ? gvv_i / gsqrt_i : T(0);
    const T gvv_gsqrt_o = j < ns - 1 ? gvv_o / gsqrt_o : T(0);
    const T guv_bsupu_i = j > 0 ? guv_i * bsupu_i : T(0);
    const T guv_bsupu_o = j < ns - 1 ? guv_o * bsupu_o : T(0);
    const T lu_e_norm = lamscale * full.lu_e.data()[idx_f] + radial.phip_F[j];
    const T lu_o_norm = lamscale * full.lu_o.data()[idx_f];
    const T bsubv_alt =
        T(0.5) * (gvv_gsqrt_i + gvv_gsqrt_o) * lu_e_norm +
        T(0.5) * (gvv_gsqrt_i * sH_i + gvv_gsqrt_o * sH_o) * lu_o_norm +
        T(0.5) * (guv_bsupu_i + guv_bsupu_o);
    const T rb = T(2.0) * T(0.05) * (T(1.0) - sFull);
    T blmn = bsubv_avg * (T(1.0) - rb) + bsubv_alt * rb;
    T clmn = T(0.5) * (bsubu_o + bsubu_i);
    if (j > 0) {
        blmn *= -lamscale;
        clmn *= -lamscale;
    }
    force.blmn_e.data()[idx_f] = blmn;
    force.blmn_o.data()[idx_f] = blmn * sF_j;
    force.clmn_e.data()[idx_f] = clmn;
    force.clmn_o.data()[idx_f] = clmn * sF_j;
}

template <class T>
__global__ void extrapolate_base_axis(
    cumes::SpectralView<T, cumes::PhysicalStateDomain> state,
    int mnmax,
    int ntorp1) {
    const int mode = blockIdx.x * blockDim.x + threadIdx.x;
    if (mode >= mnmax) return;
    const int m = mode / ntorp1;
    if (m == 0)
        state(cumes::SpectralComponent::Lcs, mode, 0) =
            state(cumes::SpectralComponent::Lcs, mode, 1);
    if (m == 1)
        for (int c = 0; c < cumes::SPECTRAL_COMPONENT_COUNT; ++c)
            state(static_cast<cumes::SpectralComponent>(c), mode, 0) =
                state(static_cast<cumes::SpectralComponent>(c), mode, 1);
}

template <class T>
__global__ void pack_lambda(
    const T* __restrict__ d_lambda,
    cumes::SpectralView<T, cumes::PhysicalStateDomain> state,
    int ns,
    int mnmax,
    int ntorp1) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ns * mnmax) return;
    const int mode = i / ns, j = i % ns;
    const int m = mode / ntorp1, n = mode % ntorp1;
    const int source_j = j == 0 && m <= 1 ? 1 : j;
    const int source = mode * ns + source_j;
    state(cumes::SpectralComponent::Lsc, mode, j) =
        m > 0 && (j > 0 || m == 1) ? d_lambda[source] : T(0);
    state(cumes::SpectralComponent::Lcs, mode, j) =
        n > 0 && (j > 0 || m <= 1) ? d_lambda[ns * mnmax + source] : T(0);
}

template <class T>
__global__ void finish_lambda(
    cumes::SpectralView<T, cumes::DecomposedResidualDomain> raw,
    const T* __restrict__ d_sqrt_s,
    const T* __restrict__ d_lambda_prec,
    T* __restrict__ d_result,
    int ns,
    int mnmax,
    int ntorp1) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ns * mnmax) return;
    const int mode = i / ns, j = i % ns;
    const int m = mode / ntorp1, n = mode % ntorp1;
    const T odd_scale =
        m % 2 ? T(1) / fmax(d_sqrt_s[j], sqrt(T(1) / T(ns - 1))) : T(1);
    T fsc = raw(cumes::SpectralComponent::Lsc, mode, j);
    T fcs = raw(cumes::SpectralComponent::Lcs, mode, j);
    if (m % 2) {
        fsc *= odd_scale;
        fcs *= odd_scale;
    }
    if (j == 0 || m == 0) fsc = T(0);
    if (j == 0 || n == 0) fcs = T(0);
    // This diagnostic slab has only lambda residuals. There is no R/Z
    // constraint evaluation in this map and the R/Z outputs are irrelevant.
    raw(cumes::SpectralComponent::Rcc, mode, j) = T(0);
    raw(cumes::SpectralComponent::Zsc, mode, j) = T(0);
    raw(cumes::SpectralComponent::Rss, mode, j) = T(0);
    raw(cumes::SpectralComponent::Zcs, mode, j) = T(0);
    raw(cumes::SpectralComponent::Lsc, mode, j) = fsc;
    raw(cumes::SpectralComponent::Lcs, mode, j) = fcs;
    const T mscale = m == 0 ? T(1) : sqrt(T(2));
    const T nscale = n == 0 ? T(1) : sqrt(T(2));
    const T basis_scale = mscale * nscale;
    // Match the production preconditioner followed by descent's conversion
    // to physical state coefficients. Do not apply the odd scale again.
    d_result[i] = (fsc * d_lambda_prec[i]) * basis_scale;
    d_result[ns * mnmax + i] = (fcs * d_lambda_prec[i]) * basis_scale;
}

}  // namespace detail

template <class T>
class FrozenLambdaOperator {
   public:
    using val_type = T;

    // p supplies the chosen radial stage; Profiles fills its lamscale. All
    // allocations, including unused production R/Z scratch, happen here.
    FrozenLambdaOperator(DeviceParams<T> p, const cumes::ValidatedProblem& vp)
        : p_(check_params(p, vp)),
          profiles_(p_, vp, std::nullopt, false),
          state_(p_.ns, p_.mnmax),
          rs_(p_, std::nullopt),
          modes_(p_, std::nullopt),
          transform_(p_, *rs_, modes_.get()),
          geometry_(p_, std::nullopt),
          preconditioner_(p_, std::nullopt),
          d_raw_(cumes::SPECTRAL_COMPONENT_COUNT * family_size()),
          d_zero_(std::size_t(p_.ns) * p_.nZnT),
          d_jvp_chip_(p_.ns - 1),
          d_jvp_iota_(p_.ns - 1) {
        if (p_.ntor == 0 && p_.nzeta == 1) axisymmetric_.emplace(p_);
        d_zero_.zero();
        d_jvp_chip_.zero();
        d_jvp_iota_.zero();
        radial_ = profiles_.profile_views();
        homogeneous_ = radial_;
        homogeneous_.phip_F = d_zero_.data();
        homogeneous_.phip_H = d_zero_.data();
        homogeneous_.curr_H = d_zero_.data();
        homogeneous_.pres_H = d_zero_.data();
        // ncurr=1 writes chip/iota. They must not alias the immutable zero
        // current/flux arrays or the prescribed ncurr=0 chip profile.
        homogeneous_.chip_H = d_jvp_chip_.data();
        homogeneous_.iota_H = d_jvp_iota_.data();
    }

    FrozenLambdaOperator(const FrozenLambdaOperator&) = delete;
    FrozenLambdaOperator& operator=(const FrozenLambdaOperator&) = delete;

    std::size_t family_size() const { return std::size_t(p_.ns) * p_.mnmax; }
    std::size_t vector_size() const { return 2 * family_size(); }
    const DeviceParams<T>& params() const { return p_; }

    // Caller supplies a valid fixed-boundary geometry. Recompute its metric
    // and the production lambda diagonal ONCE, outside the inner iteration.
    // R/Z are copied privately and cannot be changed by lambda input vectors.
    // Use one stream through set_base and every subsequent map; synchronize
    // that stream before destroying this object.
    void set_base(cumes::SpectralView<const T, cumes::PhysicalStateDomain> base,
                  cudaStream_t stream) {
        if (!base.data() || base.ns() != p_.ns || base.mnmax() != p_.mnmax)
            throw cumes::CumesError("frozen lambda base has the wrong shape");
        transform_.bind_stream(stream);
        cumes::check_cuda(cudaMemcpyAsync(state_.state_slab(), base.data(),
                                          d_raw_.byte_size(),
                                          cudaMemcpyDeviceToDevice, stream),
                          "frozen lambda base copy");
        detail::extrapolate_base_axis<T>
            <<<(p_.mnmax + 127) / 128, 128, 0, stream>>>(state_.physical(),
                                                         p_.mnmax, p_.ntor + 1);
        inverse(stream);
        geometry_.enqueue(*rs_, p_, radial_, stream);
        const auto metric = geometry_.base_geometry_views(p_);
        const auto field = geometry_.magnetic_field_views(p_);
        cumes::MagneticFieldOperator<T>{}.enqueue(
            *rs_, p_, radial_, metric, field, nullptr, stream, false);
        preconditioner_.enqueue_compute(*rs_, transform_.xm(), transform_.xn(),
                                        p_, radial_, metric, field, nullptr,
                                        stream);
        stream_ = stream;
        ready_ = true;
        cumes::check_cuda(cudaGetLastError(), "frozen lambda base");
    }

    // Contiguous vectors [Lsc][Lcs], each [mode][surface]. Returns M F(lambda),
    // where F is the raw decomposed lambda residual INCLUDING odd scaling and
    // M is lambda_prec followed by mscale*nscale. Thus result has physical
    // state units. Inactive basis functions and every axis output are zero.
    void enqueue_residual(const T* d_lambda, T* d_result, cudaStream_t stream) {
        enqueue(d_lambda, d_result, radial_, stream);
    }

    // Returns M J direction directly, with no subtractive finite difference.
    // Fixed metrics make F affine. Zeroing flux/current offsets retains its
    // homogeneous part, including the ncurr=1 current-closure projection.
    // A=-M J and b=M F therefore give J delta=-F in the active subspace.
    void enqueue_jvp(const T* d_direction, T* d_result, cudaStream_t stream) {
        enqueue(d_direction, d_result, homogeneous_, stream);
    }

    // Last map's unpreconditioned lambda residual after odd scaling; the
    // four R/Z families are explicitly zero. Valid until the next map call.
    cumes::SpectralView<const T, cumes::DecomposedResidualDomain>
    raw_residual_view() const {
        return {d_raw_.data(), p_.ns, p_.mnmax};
    }

   private:
    static DeviceParams<T> check_params(DeviceParams<T> p,
                                        const cumes::ValidatedProblem& vp) {
        if (p.ns < 3 || p.ns > 512 || p.mnmax != p.mpol * (p.ntor + 1) ||
            vp.spec().free_boundary.lfreeb)
            throw cumes::CumesError(
                "frozen lambda requires a fixed-boundary stage");
        if (p.radius_reference != T(0))
            throw cumes::CumesError(
                "frozen lambda requires absolute-radius storage");
        return p;
    }

    cumes::SpectralOperator<T>& backend() {
        if (axisymmetric_) return *axisymmetric_;
        return transform_;
    }

    void inverse(cudaStream_t stream) {
        backend().enqueue_inverse(state_.physical_const(),
                                  rs_->geometry_views(p_), {}, {}, stream);
    }

    void enqueue(const T* d_lambda,
                 T* d_result,
                 const cumes::RadialProfileViews<T>& radial,
                 cudaStream_t stream) {
        if (!ready_ || stream != stream_ || !d_lambda || !d_result)
            throw cumes::CumesError(
                "frozen lambda map needs a base and its stream");
        const int blocks = (int(family_size()) + 255) / 256;
        detail::pack_lambda<T><<<blocks, 256, 0, stream>>>(
            d_lambda, state_.physical(), p_.ns, p_.mnmax, p_.ntor + 1);
        inverse(stream);
        const auto metric = geometry_.base_geometry_views(p_);
        const auto field = geometry_.magnetic_field_views(p_);
        cumes::MagneticFieldOperator<T>{}.enqueue(
            *rs_, p_, radial, metric, field, nullptr, stream, false);
        const dim3 force_grid((p_.nZnT + 127) / 128, p_.ns);
        detail::lambda_force<T><<<force_grid, 128, 0, stream>>>(
            rs_->geometry_views(p_), metric, field, radial,
            rs_->force_views(p_), p_.lamscale, p_.ns, p_.nZnT);
        auto view = [&](const T* d) {
            return cumes::RealFieldView<const T>(d, p_.ns, p_.ntheta, p_.nzeta);
        };
        const auto zero = view(d_zero_.data());
        cumes::ForceParityViews<const T> forces;
        forces.armn_e = forces.armn_o = forces.azmn_e = forces.azmn_o = zero;
        forces.brmn_e = forces.brmn_o = forces.bzmn_e = forces.bzmn_o = zero;
        forces.crmn_e = forces.crmn_o = forces.czmn_e = forces.czmn_o = zero;
        forces.blmn_e = view(rs_->d_blmn_e);
        forces.blmn_o = view(rs_->d_blmn_o);
        forces.clmn_e = view(rs_->d_clmn_e);
        forces.clmn_o = view(rs_->d_clmn_o);
        cumes::SpectralView<T, cumes::DecomposedResidualDomain> raw{
            d_raw_.data(), p_.ns, p_.mnmax};
        backend().enqueue_forward(forces, {zero, zero, zero, zero}, raw,
                                  stream);
        detail::finish_lambda<T><<<blocks, 256, 0, stream>>>(
            raw, radial_.sqrtS_F, preconditioner_.lambda_prec(), d_result,
            p_.ns, p_.mnmax, p_.ntor + 1);
        cumes::check_cuda(cudaGetLastError(), "frozen lambda map");
    }

    DeviceParams<T> p_;
    cumes::Profiles<T> profiles_;
    cumes::SpectralStorage<T> state_;
    cumes::stage_detail::ScopedRealSpace<T> rs_;
    cumes::stage_detail::ScopedModeTable<T> modes_;
    cumes::ToroidalFftOperator<T> transform_;
    std::optional<cumes::AxisymmetricOperator<T>> axisymmetric_;
    cumes::GeometryOperator<T> geometry_;
    cumes::Preconditioner<T> preconditioner_;
    cumes::DeviceBuffer<T> d_raw_, d_zero_, d_jvp_chip_, d_jvp_iota_;
    cumes::RadialProfileViews<T> radial_, homogeneous_;
    cudaStream_t stream_ = nullptr;
    bool ready_ = false;
};

}  // namespace lambda_bench

#endif  // CUMES_BENCHMARKS_FROZEN_LAMBDA_CUH_
