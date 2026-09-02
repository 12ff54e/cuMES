#ifndef CUMES_SRC_TANGENT_IMPL_CUH_
#define CUMES_SRC_TANGENT_IMPL_CUH_

#include "cumes/numerics/dual_spectral_operator.hpp"
#include "cumes/numerics/equilibrium_residual_jvp.hpp"
#include "cumes/runtime/cuda_status.hpp"
#include "cumes/solver/equilibrium_linearization.hpp"
#include "cumes/state/seed_state.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Dual = cumes::ForwardDualDouble;

cumes::GeometryParityViews<const double> const_geometry_views(
    const cumes::RealSpaceStorage<double>& rs,
    const DeviceParams<double>& p) {
    auto field = [&](const double* data) {
        return cumes::RealFieldView<const double>(data, p.ns, p.ntheta,
                                                  p.nzeta);
    };
    cumes::GeometryParityViews<const double> result;
#define CUMES_CONST_GEOMETRY(name) result.name = field(rs.d_##name)
    CUMES_CONST_GEOMETRY(r_e);
    CUMES_CONST_GEOMETRY(z_e);
    CUMES_CONST_GEOMETRY(l_e);
    CUMES_CONST_GEOMETRY(ru_e);
    CUMES_CONST_GEOMETRY(zu_e);
    CUMES_CONST_GEOMETRY(lu_e);
    CUMES_CONST_GEOMETRY(r_o);
    CUMES_CONST_GEOMETRY(z_o);
    CUMES_CONST_GEOMETRY(l_o);
    CUMES_CONST_GEOMETRY(ru_o);
    CUMES_CONST_GEOMETRY(zu_o);
    CUMES_CONST_GEOMETRY(lu_o);
    CUMES_CONST_GEOMETRY(rv_e);
    CUMES_CONST_GEOMETRY(zv_e);
    CUMES_CONST_GEOMETRY(lv_e);
    CUMES_CONST_GEOMETRY(rv_o);
    CUMES_CONST_GEOMETRY(zv_o);
    CUMES_CONST_GEOMETRY(lv_o);
#undef CUMES_CONST_GEOMETRY
    return result;
}

template <class Domain>
__global__ void split_spectral_kernel(
    cumes::SpectralView<const Dual, Domain> input,
    cumes::SpectralView<double, Domain> primal,
    cumes::SpectralView<double, Domain> tangent,
    int total) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const Dual value = input.data()[index];
    primal.data()[index] = value.value;
    tangent.data()[index] = value.tangent;
}

template <class Domain>
__global__ void merge_spectral_kernel(
    cumes::SpectralView<const double, Domain> primal,
    cumes::SpectralView<const double, Domain> tangent,
    cumes::SpectralView<Dual, Domain> output,
    int total) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) return;
    output.data()[index] = {primal.data()[index], tangent.data()[index]};
}

__device__ inline void merge_field(const double* primal,
                                   const double* tangent,
                                   Dual* output,
                                   int index) {
    output[index] = {primal[index], tangent[index]};
}

__global__ void merge_geometry_kernel(
    cumes::GeometryParityViews<const double> primal,
    cumes::GeometryParityViews<const double> tangent,
    cumes::GeometryParityViews<Dual> output,
    const double* primal_rcon,
    const double* tangent_rcon,
    const double* primal_zcon,
    const double* tangent_zcon,
    Dual* rcon,
    Dual* zcon,
    int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
#define CUMES_MERGE_GEOMETRY(name) \
    merge_field(primal.name.data(), tangent.name.data(), output.name.data(), i)
    CUMES_MERGE_GEOMETRY(r_e);
    CUMES_MERGE_GEOMETRY(z_e);
    CUMES_MERGE_GEOMETRY(l_e);
    CUMES_MERGE_GEOMETRY(ru_e);
    CUMES_MERGE_GEOMETRY(zu_e);
    CUMES_MERGE_GEOMETRY(lu_e);
    CUMES_MERGE_GEOMETRY(r_o);
    CUMES_MERGE_GEOMETRY(z_o);
    CUMES_MERGE_GEOMETRY(l_o);
    CUMES_MERGE_GEOMETRY(ru_o);
    CUMES_MERGE_GEOMETRY(zu_o);
    CUMES_MERGE_GEOMETRY(lu_o);
    CUMES_MERGE_GEOMETRY(rv_e);
    CUMES_MERGE_GEOMETRY(zv_e);
    CUMES_MERGE_GEOMETRY(lv_e);
    CUMES_MERGE_GEOMETRY(rv_o);
    CUMES_MERGE_GEOMETRY(zv_o);
    CUMES_MERGE_GEOMETRY(lv_o);
#undef CUMES_MERGE_GEOMETRY
    if (rcon != nullptr) rcon[i] = {primal_rcon[i], tangent_rcon[i]};
    if (zcon != nullptr) zcon[i] = {primal_zcon[i], tangent_zcon[i]};
}

__device__ inline void split_field(const Dual* input,
                                   double* primal,
                                   double* tangent,
                                   int index) {
    primal[index] = input[index].value;
    tangent[index] = input[index].tangent;
}

__global__ void split_force_kernel(
    cumes::ForceParityViews<const Dual> input,
    cumes::ConstraintForceViews<const Dual> constraint,
    cumes::ForceParityViews<double> primal,
    cumes::ForceParityViews<double> tangent,
    cumes::ConstraintForceViews<double> primal_constraint,
    cumes::ConstraintForceViews<double> tangent_constraint,
    int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
#define CUMES_SPLIT_FORCE(name) \
    split_field(input.name.data(), primal.name.data(), tangent.name.data(), i)
    CUMES_SPLIT_FORCE(armn_e);
    CUMES_SPLIT_FORCE(armn_o);
    CUMES_SPLIT_FORCE(azmn_e);
    CUMES_SPLIT_FORCE(azmn_o);
    CUMES_SPLIT_FORCE(brmn_e);
    CUMES_SPLIT_FORCE(brmn_o);
    CUMES_SPLIT_FORCE(bzmn_e);
    CUMES_SPLIT_FORCE(bzmn_o);
    CUMES_SPLIT_FORCE(blmn_e);
    CUMES_SPLIT_FORCE(blmn_o);
    CUMES_SPLIT_FORCE(clmn_e);
    CUMES_SPLIT_FORCE(clmn_o);
    CUMES_SPLIT_FORCE(crmn_e);
    CUMES_SPLIT_FORCE(crmn_o);
    CUMES_SPLIT_FORCE(czmn_e);
    CUMES_SPLIT_FORCE(czmn_o);
#undef CUMES_SPLIT_FORCE
#define CUMES_SPLIT_CONSTRAINT(name)                                   \
    split_field(constraint.name.data(), primal_constraint.name.data(), \
                tangent_constraint.name.data(), i)
    CUMES_SPLIT_CONSTRAINT(frcon_e);
    CUMES_SPLIT_CONSTRAINT(frcon_o);
    CUMES_SPLIT_CONSTRAINT(fzcon_e);
    CUMES_SPLIT_CONSTRAINT(fzcon_o);
#undef CUMES_SPLIT_CONSTRAINT
}

__global__ void split_scalar_kernel(const Dual* input,
                                    double* primal,
                                    double* tangent,
                                    int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    primal[i] = input[i].value;
    tangent[i] = input[i].tangent;
}

__global__ void add_three_and_merge_kernel(const double* primal,
                                           const double* tangent0,
                                           const double* tangent1,
                                           const double* tangent2,
                                           Dual* output,
                                           int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    output[i] = {primal[i], tangent0[i] + tangent1[i] + tangent2[i]};
}

__global__ void tangent_extrapolate_axis_kernel(
    cumes::SpectralView<Dual, cumes::PhysicalStateDomain> state,
    int mnmax,
    int ntorp1) {
    const int mode = blockIdx.x * blockDim.x + threadIdx.x;
    if (mode >= mnmax) return;
    const int m = mode / ntorp1;
    if (m == 0) {
        state(cumes::SpectralComponent::Lcs, mode, 0) =
            state(cumes::SpectralComponent::Lcs, mode, 1);
        return;
    }
    if (m != 1) return;
    for (int component = 0; component < 6; ++component) {
        state(static_cast<cumes::SpectralComponent>(component), mode, 0) =
            state(static_cast<cumes::SpectralComponent>(component), mode, 1);
    }
}

__global__ void tangent_scalxc_kernel(
    cumes::SpectralView<Dual, cumes::DecomposedResidualDomain> residual,
    const Dual* sqrt_s,
    const int* xm,
    int ns,
    int mnmax,
    Dual sqrt_s1) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ns * mnmax) return;
    const int mode = i / ns;
    if (xm[mode] % 2 == 0) return;
    const int surface = i % ns;
    const Dual scale = Dual(1.0) / fmax(sqrt_s[surface], sqrt_s1);
    for (int component = 0; component < 6; ++component) {
        residual(static_cast<cumes::SpectralComponent>(component), mode,
                 surface) *= scale;
    }
}

__global__ void tangent_m1_gauge_kernel(
    cumes::SpectralView<Dual, cumes::DecomposedResidualDomain> residual,
    int ns,
    int ntor) {
    const int surface = blockIdx.x * blockDim.x + threadIdx.x;
    if (surface >= ns) return;
    const Dual scale = Dual(1.0 / ::sqrt(2.0));
    const int first_m1 = ntor + 1;
    for (int n = 0; n < ntor + 1; ++n) {
        const int mode = first_m1 + n;
        const Dual rss = residual(cumes::SpectralComponent::Rss, mode, surface);
        const Dual zcs = residual(cumes::SpectralComponent::Zcs, mode, surface);
        residual(cumes::SpectralComponent::Rss, mode, surface) =
            (rss + zcs) * scale;
        residual(cumes::SpectralComponent::Zcs, mode, surface) = Dual(0.0);
    }
}

}  // namespace

DeviceParams<cumes::ForwardDualDouble> cumes::make_forward_dual_params(
    const DeviceParams<double>& primal) {
    DeviceParams<ForwardDualDouble> result{};
    result.ns = primal.ns;
    result.mnmax = primal.mnmax;
    result.ntheta = primal.ntheta;
    result.nzeta = primal.nzeta;
    result.nfp = primal.nfp;
    result.nZnT = primal.nZnT;
    result.mpol = primal.mpol;
    result.ntor = primal.ntor;
    result.ncurr = primal.ncurr;
    result.delt = primal.delt;
    result.ftol = primal.ftol;
    result.max_iter = primal.max_iter;
    result.tcon0 = primal.tcon0;
    result.lamscale = primal.lamscale;
    return result;
}

cumes::DualSpectralOperator::DualSpectralOperator(
    const DeviceParams<double>& p,
    const DeviceModeTable& mode_table)
    : p_(p),
      primal_state_(p.ns, p.mnmax),
      tangent_state_(p.ns, p.mnmax),
      primal_rs_(real_space_create(p)),
      tangent_rs_(real_space_create(p)),
      transform_(std::make_unique<ToroidalFftOperator<double>>(p,
                                                               primal_rs_,
                                                               mode_table)),
      primal_residual_(static_cast<std::size_t>(6) * p.ns * p.mnmax),
      tangent_residual_(static_cast<std::size_t>(6) * p.ns * p.mnmax),
      primal_rcon_(static_cast<std::size_t>(p.ns) * p.nZnT),
      tangent_rcon_(static_cast<std::size_t>(p.ns) * p.nZnT),
      primal_zcon_(static_cast<std::size_t>(p.ns) * p.nZnT),
      tangent_zcon_(static_cast<std::size_t>(p.ns) * p.nZnT),
      primal_gcon_eff_(static_cast<std::size_t>(p.ns) * p.nZnT),
      tangent_gcon_eff_(static_cast<std::size_t>(p.ns) * p.nZnT),
      primal_tcon_(p.ns),
      tangent_tcon_(p.ns),
      primal_faccon_(p.mnmax),
      tangent_faccon_(p.mnmax),
      primal_gcon_(static_cast<std::size_t>(p.ns) * p.nZnT),
      tangent_gcon_(static_cast<std::size_t>(p.ns) * p.nZnT),
      tangent_gcon_term_(static_cast<std::size_t>(p.ns) * p.nZnT),
      tangent_faccon_term_(static_cast<std::size_t>(p.ns) * p.nZnT) {}

cumes::DualSpectralOperator::~DualSpectralOperator() {
    transform_.reset();
    real_space_free(primal_rs_);
    real_space_free(tangent_rs_);
}

void cumes::DualSpectralOperator::bind_stream(cudaStream_t stream) {
    transform_->bind_stream(stream);
}

void cumes::DualSpectralOperator::enqueue_inverse(
    SpectralView<const ForwardDualDouble, PhysicalStateDomain> coefficients,
    GeometryParityViews<ForwardDualDouble> geometry,
    RealFieldView<ForwardDualDouble> rcon,
    RealFieldView<ForwardDualDouble> zcon,
    cudaStream_t stream) {
    const int spectral_count = 6 * p_.ns * p_.mnmax;
    split_spectral_kernel<<<(spectral_count + 255) / 256, 256, 0, stream>>>(
        coefficients, primal_state_.physical(), tangent_state_.physical(),
        spectral_count);
    transform_->enqueue_inverse(
        primal_state_.physical_const(), geometry_parity_views(primal_rs_, p_),
        RealFieldView<double>(primal_rcon_.data(), p_.ns, p_.ntheta, p_.nzeta),
        RealFieldView<double>(primal_zcon_.data(), p_.ns, p_.ntheta, p_.nzeta),
        stream);
    transform_->enqueue_inverse(
        tangent_state_.physical_const(), geometry_parity_views(tangent_rs_, p_),
        RealFieldView<double>(tangent_rcon_.data(), p_.ns, p_.ntheta, p_.nzeta),
        RealFieldView<double>(tangent_zcon_.data(), p_.ns, p_.ntheta, p_.nzeta),
        stream);
    const int real_count = p_.ns * p_.nZnT;
    merge_geometry_kernel<<<(real_count + 255) / 256, 256, 0, stream>>>(
        const_geometry_views(primal_rs_, p_),
        const_geometry_views(tangent_rs_, p_), geometry, primal_rcon_.data(),
        tangent_rcon_.data(), primal_zcon_.data(), tangent_zcon_.data(),
        rcon.data(), zcon.data(), real_count);
    check_cuda(cudaGetLastError(), "dual inverse transform");
}

void cumes::DualSpectralOperator::enqueue_forward(
    ForceParityViews<const ForwardDualDouble> real_force,
    ConstraintForceViews<const ForwardDualDouble> constraint_force,
    SpectralView<ForwardDualDouble, DecomposedResidualDomain> residual,
    cudaStream_t stream,
    bool include_lcfs) {
    const int real_count = p_.ns * p_.nZnT;
    split_force_kernel<<<(real_count + 255) / 256, 256, 0, stream>>>(
        real_force, constraint_force, force_parity_views(primal_rs_, p_),
        force_parity_views(tangent_rs_, p_),
        ConstraintForceViews<double>{
            RealFieldView<double>(primal_rs_.d_r_real, p_.ns, p_.ntheta,
                                  p_.nzeta),
            RealFieldView<double>(primal_rs_.d_z_real, p_.ns, p_.ntheta,
                                  p_.nzeta),
            RealFieldView<double>(primal_rs_.d_l_real, p_.ns, p_.ntheta,
                                  p_.nzeta),
            RealFieldView<double>(primal_rs_.d_ru_real, p_.ns, p_.ntheta,
                                  p_.nzeta)},
        ConstraintForceViews<double>{
            RealFieldView<double>(tangent_rs_.d_r_real, p_.ns, p_.ntheta,
                                  p_.nzeta),
            RealFieldView<double>(tangent_rs_.d_z_real, p_.ns, p_.ntheta,
                                  p_.nzeta),
            RealFieldView<double>(tangent_rs_.d_l_real, p_.ns, p_.ntheta,
                                  p_.nzeta),
            RealFieldView<double>(tangent_rs_.d_ru_real, p_.ns, p_.ntheta,
                                  p_.nzeta)},
        real_count);
    auto primal_constraint = ConstraintForceViews<const double>{
        RealFieldView<const double>(primal_rs_.d_r_real, p_.ns, p_.ntheta,
                                    p_.nzeta),
        RealFieldView<const double>(primal_rs_.d_z_real, p_.ns, p_.ntheta,
                                    p_.nzeta),
        RealFieldView<const double>(primal_rs_.d_l_real, p_.ns, p_.ntheta,
                                    p_.nzeta),
        RealFieldView<const double>(primal_rs_.d_ru_real, p_.ns, p_.ntheta,
                                    p_.nzeta)};
    auto tangent_constraint = ConstraintForceViews<const double>{
        RealFieldView<const double>(tangent_rs_.d_r_real, p_.ns, p_.ntheta,
                                    p_.nzeta),
        RealFieldView<const double>(tangent_rs_.d_z_real, p_.ns, p_.ntheta,
                                    p_.nzeta),
        RealFieldView<const double>(tangent_rs_.d_l_real, p_.ns, p_.ntheta,
                                    p_.nzeta),
        RealFieldView<const double>(tangent_rs_.d_ru_real, p_.ns, p_.ntheta,
                                    p_.nzeta)};
    transform_->enqueue_forward(
        force_views_of(static_cast<const RealSpaceStorage<double>&>(primal_rs_),
                       p_),
        primal_constraint,
        SpectralView<double, DecomposedResidualDomain>(primal_residual_.data(),
                                                       p_.ns, p_.mnmax),
        stream, include_lcfs);
    transform_->enqueue_forward(
        force_views_of(
            static_cast<const RealSpaceStorage<double>&>(tangent_rs_), p_),
        tangent_constraint,
        SpectralView<double, DecomposedResidualDomain>(tangent_residual_.data(),
                                                       p_.ns, p_.mnmax),
        stream, include_lcfs);
    const int spectral_count = 6 * p_.ns * p_.mnmax;
    merge_spectral_kernel<<<(spectral_count + 255) / 256, 256, 0, stream>>>(
        SpectralView<const double, DecomposedResidualDomain>(
            primal_residual_.data(), p_.ns, p_.mnmax),
        SpectralView<const double, DecomposedResidualDomain>(
            tangent_residual_.data(), p_.ns, p_.mnmax),
        residual, spectral_count);
    check_cuda(cudaGetLastError(), "dual forward transform");
}

void cumes::DualSpectralOperator::enqueue_dealias(
    RealFieldView<const ForwardDualDouble> gcon_eff,
    const ForwardDualDouble* tcon,
    const ForwardDualDouble* faccon,
    RealFieldView<ForwardDualDouble> gcon,
    cudaStream_t stream) {
    const int real_count = p_.ns * p_.nZnT;
    split_scalar_kernel<<<(real_count + 255) / 256, 256, 0, stream>>>(
        gcon_eff.data(), primal_gcon_eff_.data(), tangent_gcon_eff_.data(),
        real_count);
    split_scalar_kernel<<<(p_.ns + 255) / 256, 256, 0, stream>>>(
        tcon, primal_tcon_.data(), tangent_tcon_.data(), p_.ns);
    split_scalar_kernel<<<(p_.mnmax + 255) / 256, 256, 0, stream>>>(
        faccon, primal_faccon_.data(), tangent_faccon_.data(), p_.mnmax);
    transform_->dealias_bandpass(primal_gcon_eff_.data(), primal_tcon_.data(),
                                 primal_faccon_.data(), primal_gcon_.data(),
                                 stream);
    transform_->dealias_bandpass(tangent_gcon_eff_.data(), primal_tcon_.data(),
                                 primal_faccon_.data(), tangent_gcon_.data(),
                                 stream);
    transform_->dealias_bandpass(primal_gcon_eff_.data(), tangent_tcon_.data(),
                                 primal_faccon_.data(),
                                 tangent_gcon_term_.data(), stream);
    transform_->dealias_bandpass(primal_gcon_eff_.data(), primal_tcon_.data(),
                                 tangent_faccon_.data(),
                                 tangent_faccon_term_.data(), stream);
    add_three_and_merge_kernel<<<(real_count + 255) / 256, 256, 0, stream>>>(
        primal_gcon_.data(), tangent_gcon_.data(), tangent_gcon_term_.data(),
        tangent_faccon_term_.data(), gcon.data(), real_count);
    check_cuda(cudaGetLastError(), "dual de-alias transform");
}

cumes::EquilibriumResidualJvpOperator::EquilibriumResidualJvpOperator(
    const DeviceParams<double>& primal_params,
    const ValidatedProblem& problem,
    const DeviceModeTable& mode_table)
    : p_(make_forward_dual_params(primal_params)),
      profiles_(p_, problem, std::nullopt, false),
      state_(p_.ns, p_.mnmax),
      rs_(real_space_create(p_)),
      transform_(primal_params, mode_table),
      geometry_(p_, std::nullopt),
      preconditioner_(p_, std::nullopt),
      constraint_(p_, std::nullopt),
      rcon_(static_cast<std::size_t>(p_.ns) * p_.nZnT),
      zcon_(static_cast<std::size_t>(p_.ns) * p_.nZnT) {}

cumes::EquilibriumResidualJvpOperator::~EquilibriumResidualJvpOperator() {
    real_space_free(rs_);
}

void cumes::EquilibriumResidualJvpOperator::enqueue(
    SpectralView<const ForwardDualDouble, PhysicalStateDomain> state,
    SpectralView<ForwardDualDouble, DecomposedResidualDomain> residual,
    cudaStream_t stream) {
    const std::size_t state_bytes = static_cast<std::size_t>(6) * p_.ns *
                                    p_.mnmax * sizeof(ForwardDualDouble);
    check_cuda(cudaMemcpyAsync(state_.state_slab(), state.data(), state_bytes,
                               cudaMemcpyDeviceToDevice, stream),
               "tangent state copy");
    tangent_extrapolate_axis_kernel<<<(p_.mnmax + 255) / 256, 256, 0, stream>>>(
        state_.physical(), p_.mnmax, p_.ntor + 1);
    check_cuda(cudaGetLastError(), "tangent axis extrapolation");
    transform_.enqueue_inverse(
        state_.physical_const(), geometry_parity_views(rs_, p_),
        {rcon_.data(), p_.ns, p_.ntheta, p_.nzeta},
        {zcon_.data(), p_.ns, p_.ntheta, p_.nzeta}, stream);
    const auto radial = profiles_.profile_views();
    geometry_.enqueue(rs_, p_, radial, stream);
    MagneticFieldOperator<ForwardDualDouble>{}.enqueue(
        rs_, p_, radial, geometry_.base_geometry_views(p_),
        geometry_.magnetic_field_views(p_), nullptr, stream, true);
    preconditioner_.enqueue_compute(rs_, transform_.xm(), transform_.xn(), p_,
                                    radial, geometry_.base_geometry_views(p_),
                                    geometry_.magnetic_field_views(p_), nullptr,
                                    stream, false);
    ForceOperator<ForwardDualDouble>{}.enqueue(
        rs_, p_, radial, geometry_.base_geometry_views(p_),
        geometry_.magnetic_field_views(p_), nullptr, stream);
    constraint_.reset_reference(p_, radial.sqrtS_F, nullptr, stream);
    constraint_.enqueue(p_, rs_, preconditioner_.ard(), preconditioner_.azd(),
                        radial.sqrtS_F, true, &transform_, nullptr, stream);
    transform_.enqueue_forward(force_views_of(rs_, p_),
                               constraint_.constraint_force_views(p_), residual,
                               stream, false);
    tangent_scalxc_kernel<<<(p_.ns * p_.mnmax + 255) / 256, 256, 0, stream>>>(
        residual, radial.sqrtS_F, transform_.xm(), p_.ns, p_.mnmax,
        sqrt(ForwardDualDouble(1.0) / ForwardDualDouble(p_.ns - 1.0)));
    tangent_m1_gauge_kernel<<<(p_.ns + 255) / 256, 256, 0, stream>>>(
        residual, p_.ns, p_.ntor);
    check_cuda(cudaGetLastError(), "tangent residual postprocess");
}

void cumes::EquilibriumResidualJvpOperator::enqueue_precondition(
    SpectralView<ForwardDualDouble, DecomposedResidualDomain> residual,
    cudaStream_t stream) {
    preconditioner_.enqueue_m1_scale(residual, p_, nullptr, stream);
    preconditioner_.enqueue_apply(residual, p_, nullptr, stream, false);
}

class cumes::EquilibriumLinearization::Impl {
   public:
    struct ActiveDof {
        std::size_t first_state = 0;
        std::size_t second_state = std::numeric_limits<std::size_t>::max();
        std::size_t residual = 0;
    };

    Impl(const ValidatedProblem& problem,
         const EquilibriumSnapshot& equilibrium)
        : problem_(&problem), p_(init_params<double>(problem)) {
        if (problem.spec().free_boundary.lfreeb) {
            throw CumesError(
                "equilibrium forward tangents currently support fixed "
                "boundary only");
        }
        p_.ns = equilibrium.ns;
        if (p_.ns < 2 || equilibrium.mnmax != p_.mnmax ||
            equilibrium.ntheta != p_.ntheta || equilibrium.nzeta != p_.nzeta) {
            throw CumesError(
                "linearization equilibrium does not match the validated "
                "final-grid shape");
        }
        const std::size_t family_size =
            static_cast<std::size_t>(p_.ns) * p_.mnmax;
        primal_.resize(6 * family_size);
        for (std::size_t component = 0; component < 6; ++component) {
            if (equilibrium.families[component].size() != family_size) {
                throw CumesError(
                    "linearization equilibrium has an invalid spectral "
                    "family size");
            }
            std::copy(equilibrium.families[component].begin(),
                      equilibrium.families[component].end(),
                      primal_.begin() + component * family_size);
        }
        mode_table_ = mode_table_create(p_);
        evaluator_ = std::make_unique<EquilibriumResidualJvpOperator>(
            p_, problem, mode_table_);
        dual_state_.allocate(primal_.size());
        dual_residual_.allocate(primal_.size());
        build_active_dofs();
    }

    ~Impl() {
        evaluator_.reset();
        mode_table_free(mode_table_);
    }

    ResidualJvp evaluate(std::span<const double> direction) {
        if (direction.size() != primal_.size()) {
            throw CumesError(
                "state tangent has " + std::to_string(direction.size()) +
                " values; expected " + std::to_string(primal_.size()));
        }
        host_dual_.resize(primal_.size());
        for (std::size_t index = 0; index < primal_.size(); ++index) {
            host_dual_[index] = {primal_[index], direction[index]};
        }
        const std::size_t bytes = host_dual_.size() * sizeof(ForwardDualDouble);
        check_cuda(cudaMemcpy(dual_state_.data(), host_dual_.data(), bytes,
                              cudaMemcpyHostToDevice),
                   "linearization state upload");
        evaluator_->enqueue({dual_state_.data(), p_.ns, p_.mnmax},
                            {dual_residual_.data(), p_.ns, p_.mnmax}, 0);
        check_cuda(cudaMemcpy(host_dual_.data(), dual_residual_.data(), bytes,
                              cudaMemcpyDeviceToHost),
                   "linearization residual download");
        ResidualJvp result;
        result.residual.resize(primal_.size());
        result.tangent.resize(primal_.size());
        for (std::size_t index = 0; index < primal_.size(); ++index) {
            result.residual[index] = host_dual_[index].value;
            result.tangent[index] = host_dual_[index].tangent;
        }
        return result;
    }

    void build_active_dofs() {
        const std::size_t family_size =
            static_cast<std::size_t>(p_.ns) * p_.mnmax;
        auto index = [&](int component, int mode, int surface) {
            return static_cast<std::size_t>(component) * family_size +
                   static_cast<std::size_t>(mode) * p_.ns + surface;
        };
        auto add = [&](int component, int mode, int surface) {
            const std::size_t i = index(component, mode, surface);
            active_.push_back({i, std::numeric_limits<std::size_t>::max(), i});
        };
        for (int mode = 0; mode < p_.mnmax; ++mode) {
            const int m = mode / (p_.ntor + 1);
            const int n = mode % (p_.ntor + 1);
            for (int surface = m == 0 ? 0 : 1; surface < p_.ns - 1; ++surface) {
                add(0, mode, surface);  // Rcc
            }
            if (m > 0) {
                for (int surface = 1; surface < p_.ns - 1; ++surface)
                    add(1, mode, surface);  // Zsc
                for (int surface = 1; surface < p_.ns; ++surface)
                    add(2, mode, surface);  // Lsc
            }
            if (m > 0 && n > 0) {
                for (int surface = 1; surface < p_.ns - 1; ++surface) {
                    if (m == 1) {
                        active_.push_back({index(3, mode, surface),
                                           index(4, mode, surface),
                                           index(3, mode, surface)});
                    } else {
                        add(3, mode, surface);  // Rss
                    }
                }
            }
            if (n > 0 && m != 1) {
                for (int surface = m == 0 ? 0 : 1; surface < p_.ns - 1;
                     ++surface)
                    add(4, mode, surface);  // Zcs
            }
            if (n > 0) {
                for (int surface = 1; surface < p_.ns; ++surface)
                    add(5, mode, surface);  // Lcs
            }
        }
    }

    std::vector<double> expand(std::span<const double> active) const {
        std::vector<double> full(primal_.size(), 0.0);
        for (std::size_t i = 0; i < active_.size(); ++i) {
            full[active_[i].first_state] = active[i];
            if (active_[i].second_state !=
                std::numeric_limits<std::size_t>::max()) {
                full[active_[i].second_state] = active[i];
            }
        }
        return full;
    }

    std::vector<double> apply_active(std::span<const double> active) {
        const std::vector<double> full = expand(active);
        const ResidualJvp jvp = evaluate(full);
        std::vector<double> result(active_.size());
        for (std::size_t i = 0; i < active_.size(); ++i)
            result[i] = jvp.tangent[active_[i].residual];
        return result;
    }

    std::vector<double> precondition_active(
        std::span<const double> active_residual) {
        host_dual_.assign(primal_.size(), ForwardDualDouble(0.0));
        for (std::size_t i = 0; i < active_.size(); ++i)
            host_dual_[active_[i].residual].tangent = active_residual[i];
        const std::size_t bytes = host_dual_.size() * sizeof(ForwardDualDouble);
        check_cuda(cudaMemcpy(dual_residual_.data(), host_dual_.data(), bytes,
                              cudaMemcpyHostToDevice),
                   "tangent preconditioner upload");
        evaluator_->enqueue_precondition(
            {dual_residual_.data(), p_.ns, p_.mnmax}, 0);
        check_cuda(cudaMemcpy(host_dual_.data(), dual_residual_.data(), bytes,
                              cudaMemcpyDeviceToHost),
                   "tangent preconditioner download");
        std::vector<double> state(active_.size(), 0.0);
        const std::size_t family_size =
            static_cast<std::size_t>(p_.ns) * p_.mnmax;
        for (std::size_t i = 0; i < active_.size(); ++i) {
            const std::size_t within = active_[i].first_state % family_size;
            const int mode = static_cast<int>(within / p_.ns);
            const int m = mode / (p_.ntor + 1);
            const int n = mode % (p_.ntor + 1);
            const double mscale = m == 0 ? 1.0 : std::sqrt(2.0);
            const double nscale = n == 0 ? 1.0 : std::sqrt(2.0);
            state[i] =
                mscale * nscale * host_dual_[active_[i].residual].tangent;
        }
        return state;
    }

    static double dot(std::span<const double> lhs,
                      std::span<const double> rhs) {
        double result = 0.0;
        for (std::size_t i = 0; i < lhs.size(); ++i) result += lhs[i] * rhs[i];
        return result;
    }

    static double norm(std::span<const double> value) {
        return std::sqrt(dot(value, value));
    }

    std::vector<ForwardDualDouble> copy_dual(const ForwardDualDouble* device,
                                             std::size_t count,
                                             const char* label) {
        std::vector<ForwardDualDouble> result(count);
        if (count != 0) {
            check_cuda(cudaMemcpy(result.data(), device,
                                  count * sizeof(ForwardDualDouble),
                                  cudaMemcpyDeviceToHost),
                       label);
        }
        return result;
    }

    static void assign_tangent(std::vector<double>& output,
                               const std::vector<ForwardDualDouble>& input) {
        output.resize(input.size());
        for (std::size_t i = 0; i < input.size(); ++i)
            output[i] = input[i].tangent;
    }

    const ValidatedProblem* problem_ = nullptr;
    DeviceParams<double> p_{};
    DeviceModeTable mode_table_;
    std::unique_ptr<EquilibriumResidualJvpOperator> evaluator_;
    DeviceBuffer<ForwardDualDouble> dual_state_;
    DeviceBuffer<ForwardDualDouble> dual_residual_;
    std::vector<double> primal_;
    std::vector<ForwardDualDouble> host_dual_;
    std::vector<ActiveDof> active_;
};

cumes::EquilibriumLinearization::EquilibriumLinearization(
    const ValidatedProblem& problem,
    const EquilibriumSnapshot& equilibrium)
    : impl_(std::make_unique<Impl>(problem, equilibrium)) {}

cumes::EquilibriumLinearization::~EquilibriumLinearization() = default;

cumes::EquilibriumLinearization::EquilibriumLinearization(
    EquilibriumLinearization&&) noexcept = default;

cumes::EquilibriumLinearization& cumes::EquilibriumLinearization::operator=(
    EquilibriumLinearization&&) noexcept = default;

std::size_t cumes::EquilibriumLinearization::state_size() const {
    return impl_->primal_.size();
}

cumes::ResidualJvp cumes::EquilibriumLinearization::residual_jvp(
    std::span<const double> state_direction) {
    return impl_->evaluate(state_direction);
}

cumes::ResidualJvp cumes::EquilibriumLinearization::boundary_residual_jvp(
    const BoundaryTangent& direction) {
    if (!direction.matches(*impl_->problem_)) {
        throw CumesError(
            "boundary tangent does not match the validated folded basis");
    }
    std::vector<double> state_direction(impl_->primal_.size(), 0.0);
    const std::size_t family_size =
        static_cast<std::size_t>(impl_->p_.ns) * impl_->p_.mnmax;
    const int lcfs = impl_->p_.ns - 1;
    for (int mode = 0; mode < impl_->p_.mnmax; ++mode) {
        const std::size_t state_index =
            static_cast<std::size_t>(mode) * impl_->p_.ns + lcfs;
        state_direction[0 * family_size + state_index] = direction.rbcc[mode];
        state_direction[1 * family_size + state_index] = direction.zbsc[mode];
        state_direction[3 * family_size + state_index] = direction.rbss[mode];
        state_direction[4 * family_size + state_index] = direction.zbcs[mode];
    }
    return impl_->evaluate(state_direction);
}

cumes::SpectralTangentSolve
cumes::EquilibriumLinearization::solve_boundary_tangent(
    const BoundaryTangent& direction,
    const TangentLinearOptions& options) {
    if (options.max_iterations <= 0 || options.restart <= 0 ||
        options.relative_tolerance < 0.0 || options.absolute_tolerance < 0.0) {
        throw CumesError("invalid tangent linear-solver options");
    }
    const ResidualJvp boundary = boundary_residual_jvp(direction);
    const std::size_t n = impl_->active_.size();
    std::vector<double> rhs(n);
    for (std::size_t i = 0; i < n; ++i)
        rhs[i] = -boundary.tangent[impl_->active_[i].residual];
    std::vector<double> x(n, 0.0);
    std::vector<double> residual = rhs;
    auto apply_krylov = [&](std::span<const double> value) {
        if (!options.use_equilibrium_preconditioner)
            return impl_->apply_active(value);
        const std::vector<double> state = impl_->precondition_active(value);
        return impl_->apply_active(state);
    };
    const double initial = Impl::norm(residual);
    const double tolerance = std::max(options.absolute_tolerance,
                                      options.relative_tolerance * initial);
    SpectralTangentSolve result;
    result.initial_residual = initial;
    result.final_residual = initial;
    if (initial <= tolerance) { result.converged = true; }

    const int restart = std::min<int>(options.restart, static_cast<int>(n));
    while (!result.converged && result.iterations < options.max_iterations) {
        const double beta = Impl::norm(residual);
        std::vector<std::vector<double>> basis(
            static_cast<std::size_t>(restart + 1), std::vector<double>(n));
        for (std::size_t i = 0; i < n; ++i) basis[0][i] = residual[i] / beta;
        std::vector<double> h(static_cast<std::size_t>(restart + 1) * restart,
                              0.0);
        auto hij = [&](int row, int column) -> double& {
            return h[static_cast<std::size_t>(column) * (restart + 1) + row];
        };
        std::vector<double> cosine(restart, 0.0), sine(restart, 0.0);
        std::vector<double> g(restart + 1, 0.0);
        g[0] = beta;
        int used = 0;
        for (int column = 0;
             column < restart && result.iterations < options.max_iterations;
             ++column) {
            std::vector<double> w = apply_krylov(basis[column]);
            ++result.iterations;
            for (int row = 0; row <= column; ++row) {
                hij(row, column) = Impl::dot(w, basis[row]);
                for (std::size_t i = 0; i < n; ++i)
                    w[i] -= hij(row, column) * basis[row][i];
            }
            hij(column + 1, column) = Impl::norm(w);
            if (hij(column + 1, column) > 0.0) {
                for (std::size_t i = 0; i < n; ++i)
                    basis[column + 1][i] = w[i] / hij(column + 1, column);
            }
            for (int row = 0; row < column; ++row) {
                const double a = hij(row, column);
                const double b = hij(row + 1, column);
                hij(row, column) = cosine[row] * a + sine[row] * b;
                hij(row + 1, column) = -sine[row] * a + cosine[row] * b;
            }
            const double diagonal = hij(column, column);
            const double below = hij(column + 1, column);
            const double magnitude = std::hypot(diagonal, below);
            cosine[column] = magnitude == 0.0 ? 1.0 : diagonal / magnitude;
            sine[column] = magnitude == 0.0 ? 0.0 : below / magnitude;
            hij(column, column) = magnitude;
            hij(column + 1, column) = 0.0;
            g[column + 1] = -sine[column] * g[column];
            g[column] *= cosine[column];
            used = column + 1;
            result.final_residual = std::abs(g[column + 1]);
            if (result.final_residual <= tolerance || magnitude == 0.0) break;
        }
        std::vector<double> y(used, 0.0);
        for (int row = used - 1; row >= 0; --row) {
            double value = g[row];
            for (int column = row + 1; column < used; ++column)
                value -= hij(row, column) * y[column];
            const double diagonal = hij(row, row);
            y[row] = diagonal == 0.0 ? 0.0 : value / diagonal;
        }
        for (int column = 0; column < used; ++column)
            for (std::size_t i = 0; i < n; ++i)
                x[i] += basis[column][i] * y[column];
        std::vector<double> ax = apply_krylov(x);
        for (std::size_t i = 0; i < n; ++i) residual[i] = rhs[i] - ax[i];
        result.final_residual = Impl::norm(residual);
        result.converged = result.final_residual <= tolerance;
        if (used == 0) break;
    }
    const std::vector<double> active_state =
        options.use_equilibrium_preconditioner ? impl_->precondition_active(x)
                                               : x;
    result.state_tangent = impl_->expand(active_state);
    const std::size_t family_size =
        static_cast<std::size_t>(impl_->p_.ns) * impl_->p_.mnmax;
    const int lcfs = impl_->p_.ns - 1;
    for (int mode = 0; mode < impl_->p_.mnmax; ++mode) {
        const std::size_t offset =
            static_cast<std::size_t>(mode) * impl_->p_.ns + lcfs;
        result.state_tangent[0 * family_size + offset] = direction.rbcc[mode];
        result.state_tangent[1 * family_size + offset] = direction.zbsc[mode];
        result.state_tangent[3 * family_size + offset] = direction.rbss[mode];
        result.state_tangent[4 * family_size + offset] = direction.zbcs[mode];
    }
    return result;
}

cumes::EquilibriumTangent cumes::EquilibriumLinearization::materialize_tangent(
    std::span<const double> state_direction,
    const EquilibriumSnapshot& primal_equilibrium,
    const EquilibriumProfiles& primal_profiles) {
    if (primal_equilibrium.ns != impl_->p_.ns ||
        primal_equilibrium.mnmax != impl_->p_.mnmax ||
        primal_equilibrium.ntheta != impl_->p_.ntheta ||
        primal_equilibrium.nzeta != impl_->p_.nzeta) {
        throw CumesError("tangent materialization primal shape mismatch");
    }
    static_cast<void>(impl_->evaluate(state_direction));
    EquilibriumTangent result =
        EquilibriumTangent::zero_like(primal_equilibrium, primal_profiles);
    const std::size_t family_size =
        static_cast<std::size_t>(impl_->p_.ns) * impl_->p_.mnmax;
    for (std::size_t component = 0; component < 6; ++component) {
        std::copy_n(state_direction.begin() + component * family_size,
                    family_size,
                    result.equilibrium.families[component].begin());
    }
    // Match the published state convention after production axis
    // extrapolation.
    for (int mode = 0; mode < impl_->p_.mnmax; ++mode) {
        const int m = mode / (impl_->p_.ntor + 1);
        for (int component = 0; component < 6; ++component) {
            if (m != 1 && !(m == 0 && component == 5)) continue;
            auto& family = result.equilibrium.families[component];
            family[static_cast<std::size_t>(mode) * impl_->p_.ns] =
                family[static_cast<std::size_t>(mode) * impl_->p_.ns + 1];
        }
    }

    const auto& evaluator = *impl_->evaluator_;
    const auto& p = evaluator.params();
    const auto& rs = evaluator.real_space();
    const auto base = evaluator.geometry().base_geometry_views(p);
    const auto field = evaluator.geometry().magnetic_field_views(p);
    const auto radial = evaluator.profiles().profile_views();
    const std::size_t points = static_cast<std::size_t>(p.ntheta) * p.nzeta;
    const std::size_t full = static_cast<std::size_t>(p.ns) * points;
    const std::size_t half = static_cast<std::size_t>(p.ns - 1) * points;
    auto tangent_field = [&](EquilibriumSnapshot::HalfField destination,
                             const ForwardDualDouble* source,
                             const char* label) {
        Impl::assign_tangent(result.equilibrium.half_fields[destination],
                             impl_->copy_dual(source, half, label));
    };
    tangent_field(EquilibriumSnapshot::SQRTG, base.gsqrt.data(),
                  "copy tangent sqrtg");
    tangent_field(EquilibriumSnapshot::BSUPU, field.bsupu.data(),
                  "copy tangent bsupu");
    tangent_field(EquilibriumSnapshot::BSUPV, field.bsupv.data(),
                  "copy tangent bsupv");
    tangent_field(EquilibriumSnapshot::BSUBU, field.bsubu.data(),
                  "copy tangent bsubu");
    tangent_field(EquilibriumSnapshot::BSUBV, field.bsubv.data(),
                  "copy tangent bsubv");
    // B^s is identically zero for nested flux surfaces.
    std::fill(
        result.equilibrium.half_fields[EquilibriumSnapshot::BSUPS].begin(),
        result.equilibrium.half_fields[EquilibriumSnapshot::BSUPS].end(), 0.0);

    const auto r_o = impl_->copy_dual(rs.d_r_o, full, "copy tangent r_o");
    const auto z_o = impl_->copy_dual(rs.d_z_o, full, "copy tangent z_o");
    const auto rv_e = impl_->copy_dual(rs.d_rv_e, full, "copy tangent rv_e");
    const auto rv_o = impl_->copy_dual(rs.d_rv_o, full, "copy tangent rv_o");
    const auto zv_e = impl_->copy_dual(rs.d_zv_e, full, "copy tangent zv_e");
    const auto zv_o = impl_->copy_dual(rs.d_zv_o, full, "copy tangent zv_o");
    const auto radial_rs =
        impl_->copy_dual(base.rs.data(), half, "copy tangent radial rs");
    const auto radial_zs =
        impl_->copy_dual(base.zs.data(), half, "copy tangent radial zs");
    const auto ru12 =
        impl_->copy_dual(base.ru12.data(), half, "copy tangent ru12");
    const auto zu12 =
        impl_->copy_dual(base.zu12.data(), half, "copy tangent zu12");
    const auto bsupu =
        impl_->copy_dual(field.bsupu.data(), half, "copy tangent bsupu dual");
    const auto bsupv =
        impl_->copy_dual(field.bsupv.data(), half, "copy tangent bsupv dual");
    const auto bsubu =
        impl_->copy_dual(field.bsubu.data(), half, "copy tangent bsubu dual");
    const auto bsubv =
        impl_->copy_dual(field.bsubv.data(), half, "copy tangent bsubv dual");
    const auto sqrt_s_half =
        impl_->copy_dual(radial.sqrtS_H, p.ns - 1, "copy tangent sqrtS_H");
    auto& bsubs = result.equilibrium.half_fields[EquilibriumSnapshot::BSUBS];
    for (int surface = 0; surface < p.ns - 1; ++surface) {
        const ForwardDualDouble sqrt_s = sqrt_s_half[surface];
        const std::size_t inner = static_cast<std::size_t>(surface) * points;
        const std::size_t outer = inner + points;
        for (std::size_t point = 0; point < points; ++point) {
            const std::size_t hi = inner + point;
            const ForwardDualDouble ro =
                ForwardDualDouble(0.5) *
                (r_o[inner + point] + r_o[outer + point]);
            const ForwardDualDouble zo =
                ForwardDualDouble(0.5) *
                (z_o[inner + point] + z_o[outer + point]);
            const ForwardDualDouble rs_physical =
                radial_rs[hi] + ro / (ForwardDualDouble(2.0) * sqrt_s);
            const ForwardDualDouble zs_physical =
                radial_zs[hi] + zo / (ForwardDualDouble(2.0) * sqrt_s);
            const ForwardDualDouble rv =
                ForwardDualDouble(0.5) *
                ((rv_e[inner + point] + rv_e[outer + point]) +
                 sqrt_s * (rv_o[inner + point] + rv_o[outer + point]));
            const ForwardDualDouble zv =
                ForwardDualDouble(0.5) *
                ((zv_e[inner + point] + zv_e[outer + point]) +
                 sqrt_s * (zv_o[inner + point] + zv_o[outer + point]));
            const ForwardDualDouble gsu =
                rs_physical * ru12[hi] + zs_physical * zu12[hi];
            const ForwardDualDouble gsv = rs_physical * rv + zs_physical * zv;
            bsubs[hi] = (gsu * bsupu[hi] + gsv * bsupv[hi]).tangent;
        }
    }

    auto profile_tangent = [&](std::vector<double>& destination,
                               const ForwardDualDouble* source,
                               const char* label) {
        Impl::assign_tangent(destination,
                             impl_->copy_dual(source, p.ns - 1, label));
    };
    profile_tangent(result.profiles.toroidal_flux_derivative, radial.phip_H,
                    "copy tangent phip_H");
    profile_tangent(result.profiles.poloidal_flux_derivative, radial.chip_H,
                    "copy tangent chip_H");
    profile_tangent(result.profiles.rotational_transform, radial.iota_H,
                    "copy tangent iota_H");
    const double flux_scale =
        static_cast<double>(DeviceParams<ForwardDualDouble>::SIGN_JACOBIAN) *
        2.0 * M_PI;
    for (double& value : result.profiles.toroidal_flux_derivative)
        value *= flux_scale;
    for (double& value : result.profiles.poloidal_flux_derivative)
        value *= flux_scale;
    result.profiles.poloidal_covariant_field.assign(p.ns - 1, 0.0);
    result.profiles.toroidal_covariant_field.assign(p.ns - 1, 0.0);
    for (int surface = 0; surface < p.ns - 1; ++surface) {
        for (std::size_t point = 0; point < points; ++point) {
            const std::size_t index =
                static_cast<std::size_t>(surface) * points + point;
            result.profiles.poloidal_covariant_field[surface] +=
                bsubu[index].tangent;
            result.profiles.toroidal_covariant_field[surface] +=
                bsubv[index].tangent;
        }
        result.profiles.poloidal_covariant_field[surface] /= points;
        result.profiles.toroidal_covariant_field[surface] /= points;
    }
    // Current-density tangents are not consumed by the optimizer targets and
    // remain zero until curl(B) tangent output is qualified separately.
    return result;
}

#endif  // CUMES_SRC_TANGENT_IMPL_CUH_
