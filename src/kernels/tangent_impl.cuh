#ifndef CUMES_SRC_TANGENT_IMPL_CUH_
#define CUMES_SRC_TANGENT_IMPL_CUH_

#include "cumes/numerics/dual_spectral_operator.hpp"
#include "cumes/runtime/cuda_status.hpp"

#include <cstddef>
#include <memory>

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

#endif  // CUMES_SRC_TANGENT_IMPL_CUH_
