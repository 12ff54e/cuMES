// free_boundary_impl.cuh — the free-boundary vacuum coupling (step 2): the
// pimpl over vfield::VacuumFieldSolver<T> plus the bridge kernels.
//
// Included once per scalar type by free_boundary_double.cu /
// free_boundary_float.cu (explicit-instantiation split). The kernels bridge
// the two grids and representations, mirroring vmecpp's handover/assemble
// arithmetic exactly (ideal_mhd_model.cc):
//
//   surface averages : bucoH/bvcoH = sum over the REDUCED poloidal subset of
//                      bsubu/bsubv with vmecpp's trapezoid weights wInt[l] =
//                      1/(nZeta*(nThetaReduced-1)) halved at the endpoints,
//                      l-major/k-minor ascending (vmecpp's kl loop order)
//   lcfs repack      : the four spectral families at j=ns-1, divided by the
//                      state-space mscale*nscale normalization and transposed
//                      to the n-major NESTOR layout
//   axis extract     : r_axis[k] = R(j=0, l=0, k), z_axis[k] = Z(j=0, l=0, k)
//   rbsq             : outsideEdgePressure = b_sq_vac (reduced, l-major) +
//                      edgePressure; rBSq = outside * R_full / deltaS, plus
//                      the delBSq surface-mean diagnostic
//   edge force       : armn_e/o += (zu_e+zu_o)*rBSq, azmn_e/o -=
//                      (ru_e+ru_o)*rBSq at the LCFS row
//   rcon decay       : rCon0/zCon0 *= 0.9 on vacuum-active passes
//
// New-style conventions (snake_case) per the 2026-08-24 coding-style update.
#ifndef CUMES_SRC_FREE_BOUNDARY_IMPL_CUH_
#define CUMES_SRC_FREE_BOUNDARY_IMPL_CUH_

#include "cumes/physics/free_boundary_operator.hpp"
#include "cumes/runtime/cuda_status.hpp"
#include "vfield/common/sizes.hpp"
#include "vfield/free_boundary/vacuum_field_solver.hpp"
#include "vfield/makegrid/makegrid.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>

namespace cumes {
namespace {

constexpr int BLOCK_SIZE = 256;

inline int grid_size(int n) {
    return (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// One thread per half-grid surface: the buco/bvco surface averages with
// vmecpp's exact accumulation order (l-major/k-minor ascending over the
// reduced subset, per-element multiply by the trapezoid weight).
template <class T>
__global__ void surface_averages_kernel(const T* __restrict__ bsubu,
                                        const T* __restrict__ bsubv,
                                        T* __restrict__ out,
                                        int ns,
                                        int ntheta,
                                        int nzeta) {
    const int jh = blockIdx.x * blockDim.x + threadIdx.x;
    if (jh >= ns - 1) return;
    const int nred = ntheta / 2 + 1;
    const T w = T(1.0) / T(nzeta * (nred - 1));
    T buco = T(0);
    T bvco = T(0);
    for (int l = 0; l < nred; ++l) {
        T wl = w;
        if (l == 0 || l == nred - 1) wl *= T(0.5);
        for (int k = 0; k < nzeta; ++k) {
            const int idx = jh * (ntheta * nzeta) + k * ntheta + l;
            buco += bsubu[idx] * wl;
            bvco += bsubv[idx] * wl;
        }
    }
    // Two CONTIGUOUS [ns-1] halves (the caller D2Hs the buffer and passes
    // data() and data()+ns-1 as the two host arrays; the interleaved write
    // this replaced scrambled the halves and poisoned rBtor/cTor).
    out[jh] = buco;
    out[(ns - 1) + jh] = bvco;
}

// One thread per (m, n) of the n-major NESTOR layout: the LCFS row of the
// four families, transposed. The cuMES state carries vmecpp's orthonormal
// mscale*nscale factors, while NESTOR's boundary arrays use the unscaled
// Fourier coefficients, so remove those factors at the handover.
template <class T>
__global__ void lcfs_repack_kernel(const T* __restrict__ rcc,
                                   const T* __restrict__ rss,
                                   const T* __restrict__ zsc,
                                   const T* __restrict__ zcs,
                                   T* __restrict__ out,
                                   int ns,
                                   int mpol,
                                   int ntor) {
    const int mnsize = mpol * (ntor + 1);
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= mnsize) return;
    const int m = i % mpol;
    const int n = i / mpol;
    const int idx_mn = m * (ntor + 1) + n;
    const int src = idx_mn * ns + (ns - 1);
    const T mfac = (m == 0) ? T(1.0) : T(std::sqrt(2.0));
    const T nfac = (n == 0) ? T(1.0) : T(std::sqrt(2.0));
    const T inv = T(1.0) / (mfac * nfac);
    out[i] = rcc[src] * inv;
    out[mnsize + i] = rss[src] * inv;
    out[2 * mnsize + i] = zsc[src] * inv;
    out[3 * mnsize + i] = zcs[src] * inv;
}

// r_axis[k] = R(j=0, l=0, k); z_axis[k] = Z(j=0, l=0, k) — the even-parity
// part (the odd part vanishes at theta=0 by construction).
template <class T>
__global__ void axis_extract_kernel(const T* __restrict__ r_e,
                                    const T* __restrict__ z_e,
                                    T* __restrict__ out,
                                    int ntheta,
                                    int nzeta) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nzeta) return;
    out[k] = r_e[k * ntheta];
    out[nzeta + k] = z_e[k * ntheta];
}

// rBSq at the LCFS row plus the delBSq mean diagnostic. The vacuum pressure
// is l-major on the reduced grid; the full-grid poloidal index mirrors into
// the reduced half (stellarator-symmetric). Order mirrors vmecpp :806-812.
template <class T>
__global__ void rbsq_kernel(const T* __restrict__ b_sq_vac,
                            const T* __restrict__ r_e,
                            const T* __restrict__ r_o,
                            const T* __restrict__ total_pressure,
                            T* __restrict__ rbsq,
                            T* __restrict__ delbsq_sum,
                            int ns,
                            int ntheta,
                            int nzeta,
                            int nZnT,
                            T edge_pressure,
                            T delta_s) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nZnT) return;
    const int it = i % ntheta;
    const int iz = i / ntheta;
    const int nred = ntheta / 2 + 1;
    const int l_red = (it < nred) ? it : (ntheta - it);
    const T outside = b_sq_vac[l_red * nzeta + iz] + edge_pressure;
    const int base = (ns - 1) * nZnT + i;
    rbsq[i] = outside * (r_e[base] + r_o[base]) / delta_s;
    const T inside = T(1.5) * total_pressure[(ns - 2) * nZnT + i] -
                     T(0.5) * total_pressure[(ns - 3) * nZnT + i];
    atomicAdd(delbsq_sum, fabs(outside - inside) / T(nZnT));
}

// The vacuum edge force (vmecpp assembleTotalForces :2160-2167): identical
// increments on the even and odd rows at the LCFS.
template <class T>
__global__ void vacuum_edge_force_kernel(T* __restrict__ armn_e,
                                         T* __restrict__ armn_o,
                                         T* __restrict__ azmn_e,
                                         T* __restrict__ azmn_o,
                                         const T* __restrict__ zu_e,
                                         const T* __restrict__ zu_o,
                                         const T* __restrict__ ru_e,
                                         const T* __restrict__ ru_o,
                                         const T* __restrict__ rbsq,
                                         int ns,
                                         int nZnT) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nZnT) return;
    const int base = (ns - 1) * nZnT + i;
    const T zu_full = zu_e[base] + zu_o[base];
    const T ru_full = ru_e[base] + ru_o[base];
    const T r = rbsq[i];
    armn_e[base] += zu_full * r;
    armn_o[base] += zu_full * r;
    azmn_e[base] -= ru_full * r;
    azmn_o[base] -= ru_full * r;
}

// rCon0/zCon0 *= 0.9 over every surface (vmecpp :651-661; nsMaxF = ns for
// free boundary).
template <class T>
__global__ void rcon_decay_kernel(T* __restrict__ rcon0,
                                  T* __restrict__ zcon0,
                                  int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    rcon0[i] *= T(0.9);
    zcon0[i] *= T(0.9);
}

}  // namespace

// ---------------------------------------------------------------------------
// Pimpl: the persistent vacuum solver + the host state machine
// ---------------------------------------------------------------------------
template <class T>
struct FreeBoundaryOperator<T>::Impl {
    int ntheta;
    int nzeta;
    int nZnT;
    int mpol;
    int ntor;
    vfield::Sizes sizes;
    vfield::VacuumFieldSolver<T> solver;
    VacuumState state = VacuumState::OFF;
    int nvacskip = 1;
    int ivacskip = 0;
    bool run_block = false;
    bool full_update = false;
    bool decay = false;
    bool soft_restart = false;
    bool edge_gate = false;
    // True once a full update has produced a factorization. The first
    // vacuum pass is FORCED to a full update: a partial update before any
    // full one would solve with uninitialized pivots (the library now
    // throws instead of reading pivots[-1], but the run should never get
    // there — vmecpp avoids the corner because its ramp gate fires before
    // the residuals are small enough to matter).
    bool has_factors = false;
    T edge_pressure = T(0);
    double rbtor = 0.0;
    double ctor = 0.0;
    double bsubu_vac = 0.0;
    double bsubv_vac = 0.0;
    double delbsq = 0.0;

    Impl(const typename FreeBoundaryOperator<T>::HostParams& params,
         const DeviceParams<T>& p)
        : ntheta(p.ntheta),
          nzeta(p.nzeta),
          nZnT(p.nZnT),
          mpol(p.mpol),
          ntor(p.ntor),
          sizes(false, p.nfp, p.mpol, p.ntor, p.ntheta, p.nzeta),
          solver([&]() {
              typename vfield::VacuumFieldSolver<T>::Params vp(sizes);
              vp.coil_currents = params.extcur;
              if (!params.mgrid_file.empty()) {
                  vp.mgrid_file = params.mgrid_file;
              } else {
                  vfield::MakegridParameters makegrid_parameters =
                      vfield::load_makegrid_parameters(
                          params.makegrid_parameters_file);
                  if (makegrid_parameters.number_of_field_periods != p.nfp) {
                      throw cumes::CumesError(
                          "inline Makegrid number_of_field_periods must match "
                          "the equilibrium nfp");
                  }
                  if (makegrid_parameters.normalize_by_currents) {
                      throw cumes::CumesError(
                          "inline Makegrid requires "
                          "normalize_by_currents=false so extcur remains "
                          "expressed in Amperes");
                  }
                  const vfield::CoilConfiguration coils =
                      vfield::load_coil_configuration(params.coils_file);
                  vp.response_table =
                      vfield::compute_magnetic_field_response_table(
                          makegrid_parameters, coils);
              }
              return vp;
          }()) {
        // The vacuum solver raises the angular resolution to its Nyquist
        // minima; the bridge kernels map between the two grids, which must be
        // the SAME grid. The validated defaults always satisfy this; only an
        // explicit undersized/odd ntheta/nzeta trips the guard.
        if (sizes.ntheta != p.ntheta || sizes.nZeta != p.nzeta) {
            throw cumes::CumesError(
                "free-boundary run requires ntheta >= 2*mpol+6 (even) and "
                "nzeta >= 2*ntor+4 (or 1 for ntor=0): the vacuum solver would "
                "raise the angular resolution above the configured grid");
        }
        if (p.ntheta % 2 != 0) {
            throw cumes::CumesError(
                "free-boundary run requires an even ntheta (the symmetric "
                "mirror bridge assumes it)");
        }
        nvacskip = params.nvacskip;
        if (params.hot_start) state = VacuumState::INITIALIZED;
    }
};

// Launch helper: the six bridge kernels take plain scalar/pointer argument
// lists; each launch checks the launch error through the centralized
// boundary. (cudaLaunchKernel is used rather than <<< >>> so the pointer
// arithmetic stays uniform; bit-identical launch configs either way.)
template <class T>
void launch_surface_averages(const T* bsubu,
                             const T* bsubv,
                             T* out,
                             int ns,
                             int ntheta,
                             int nzeta,
                             cudaStream_t stream) {
    surface_averages_kernel<T><<<grid_size(ns - 1), BLOCK_SIZE, 0, stream>>>(
        bsubu, bsubv, out, ns, ntheta, nzeta);
    check_cuda(cudaGetLastError(), "surface averages");
}

template <class T>
void launch_lcfs_repack(const T* rcc,
                        const T* rss,
                        const T* zsc,
                        const T* zcs,
                        T* out,
                        int ns,
                        int mpol,
                        int ntor,
                        cudaStream_t stream) {
    lcfs_repack_kernel<T>
        <<<grid_size(mpol * (ntor + 1)), BLOCK_SIZE, 0, stream>>>(
            rcc, rss, zsc, zcs, out, ns, mpol, ntor);
    check_cuda(cudaGetLastError(), "lcfs repack");
}

template <class T>
void launch_axis_extract(const T* r_e,
                         const T* z_e,
                         T* out,
                         int ntheta,
                         int nzeta,
                         cudaStream_t stream) {
    axis_extract_kernel<T><<<grid_size(nzeta), BLOCK_SIZE, 0, stream>>>(
        r_e, z_e, out, ntheta, nzeta);
    check_cuda(cudaGetLastError(), "axis extract");
}

template <class T>
void launch_rbsq(const T* b_sq_vac,
                 const T* r_e,
                 const T* r_o,
                 const T* total_pressure,
                 T* rbsq,
                 T* delbsq_sum,
                 int ns,
                 int ntheta,
                 int nzeta,
                 int nZnT,
                 T edge_pressure,
                 T delta_s,
                 cudaStream_t stream) {
    rbsq_kernel<T><<<grid_size(nZnT), BLOCK_SIZE, 0, stream>>>(
        b_sq_vac, r_e, r_o, total_pressure, rbsq, delbsq_sum, ns, ntheta, nzeta,
        nZnT, edge_pressure, delta_s);
    check_cuda(cudaGetLastError(), "rbsq");
}

template <class T>
void launch_edge_force(T* armn_e,
                       T* armn_o,
                       T* azmn_e,
                       T* azmn_o,
                       const T* zu_e,
                       const T* zu_o,
                       const T* ru_e,
                       const T* ru_o,
                       const T* rbsq,
                       int ns,
                       int nZnT,
                       cudaStream_t stream) {
    vacuum_edge_force_kernel<T><<<grid_size(nZnT), BLOCK_SIZE, 0, stream>>>(
        armn_e, armn_o, azmn_e, azmn_o, zu_e, zu_o, ru_e, ru_o, rbsq, ns, nZnT);
    check_cuda(cudaGetLastError(), "vacuum edge force");
}

template <class T>
void launch_rcon_decay(T* rcon0,
                       T* zcon0,
                       int ns,
                       int ntheta,
                       int nzeta,
                       cudaStream_t stream) {
    rcon_decay_kernel<T>
        <<<grid_size(ns * ntheta * nzeta), BLOCK_SIZE, 0, stream>>>(
            rcon0, zcon0, ns * ntheta * nzeta);
    check_cuda(cudaGetLastError(), "rcon decay");
}

// ---------------------------------------------------------------------------
// FreeBoundaryOperator member definitions
// ---------------------------------------------------------------------------
template <class T>
FreeBoundaryOperator<T>::FreeBoundaryOperator(const HostParams& params,
                                              const DeviceParams<T>& p) try
    : impl_(std::make_unique<Impl>(params, p)) {
} catch (const CumesError&) { throw; } catch (const std::exception& error) {
    throw CumesError(std::string("free-boundary setup: ") + error.what());
}

template <class T>
FreeBoundaryOperator<T>::~FreeBoundaryOperator() = default;

template <class T>
VacuumState FreeBoundaryOperator<T>::state() const {
    return impl_->state;
}

template <class T>
bool FreeBoundaryOperator<T>::run_vacuum_block() const {
    return impl_->run_block;
}

template <class T>
bool FreeBoundaryOperator<T>::full_update_this_pass() const {
    return impl_->full_update;
}

template <class T>
void FreeBoundaryOperator<T>::advance(int iter2,
                                      int iter1,
                                      double fsqr,
                                      double fsqz) {
    // vmecpp ideal_mhd_model.cc :605-646, in the same order: block gate,
    // ivacskip, ramp (full update while not yet ACTIVE), nvacskip extension.
    impl_->run_block = (iter2 > 1 || impl_->state == VacuumState::INITIALIZED);
    impl_->ivacskip = 0;
    impl_->full_update = false;
    impl_->decay = false;
    impl_->soft_restart = false;
    if (!impl_->run_block) return;
    impl_->ivacskip = (iter2 - iter1) % impl_->nvacskip;
    if (impl_->state != VacuumState::ACTIVE && fsqr + fsqz < 1.0e-3) {
        impl_->ivacskip = 0;
        impl_->state =
            static_cast<VacuumState>(static_cast<int>(impl_->state) + 1);
    }
    impl_->full_update = (impl_->ivacskip == 0);
    if (!impl_->has_factors) impl_->full_update = true;
    if (impl_->full_update) {
        const int new_nvacskip =
            static_cast<int>(1.0 / std::max(0.1, 1.0e11 * (fsqr + fsqz)));
        impl_->nvacskip = std::max(impl_->nvacskip, new_nvacskip);
    }
    impl_->decay = (impl_->state != VacuumState::OFF);
}

template <class T>
void FreeBoundaryOperator<T>::run_host_update(int ns,
                                              const double* buco_h,
                                              const double* bvco_h,
                                              const T* d_lcfs_repacked,
                                              const T* d_r_axis,
                                              const T* d_z_axis,
                                              cudaStream_t /*stream*/) {
    // vmecpp :534-549: rBtor/cTor from the two outermost half-grid surface
    // averages (host scalars, D2H'd by the caller after the vacuum fence).
    const int nh = ns - 1;
    const double buco_last = buco_h[nh - 1];
    const double buco_prev = buco_h[nh - 2];
    const double bvco_last = bvco_h[nh - 1];
    const double bvco_prev = bvco_h[nh - 2];
    impl_->rbtor = 1.5 * bvco_last - 0.5 * bvco_prev;
    // cTor = extrapolation * signJ * 2*pi (signJ = kSignJacobian = -1); the
    // net toroidal current for NESTOR is cTor / MU_0, in Amperes.
    impl_->ctor = (1.5 * buco_last - 0.5 * buco_prev) *
                  DeviceParams<T>::kSignJacobian * 2.0 * M_PI;
    const double net_toroidal_current =
        impl_->ctor / static_cast<double>(DeviceParams<T>::kMu0);

    const int mnsize = impl_->mpol * (impl_->ntor + 1);
    const T* rcc = d_lcfs_repacked;
    const T* rss = d_lcfs_repacked + mnsize;
    const T* zsc = d_lcfs_repacked + 2 * mnsize;
    const T* zcs = d_lcfs_repacked + 3 * mnsize;
    T bsubu = T(0);
    T bsubv = T(0);
    impl_->solver.update(rcc, rss, nullptr, nullptr, zsc, zcs, nullptr, nullptr,
                         DeviceParams<T>::kSignJacobian, d_r_axis, d_z_axis,
                         bsubu, bsubv, static_cast<T>(net_toroidal_current),
                         impl_->full_update);
    impl_->bsubu_vac = static_cast<double>(bsubu);
    impl_->bsubv_vac = static_cast<double>(bsubv);
    if (impl_->full_update) impl_->has_factors = true;

    // Bottom promotion (vmecpp :735-737): two steps in the first vacuum pass.
    if (impl_->state == VacuumState::INITIALIZING) {
        impl_->state = VacuumState::INITIALIZED;
    }

    // Consistency checks (vmecpp :755-768) — hard errors here (cuMES has no
    // best-effort output mode; documented deviation). The values ride along
    // in the message for diagnosis.
    if (impl_->rbtor * impl_->bsubv_vac < 0.0) {
        throw cumes::CumesError(
            "rBtor and bSubVVac must have the same sign - maybe flip the "
            "sign of phiedge or the sign of the coil currents "
            "(rBtor=" +
            std::to_string(impl_->rbtor) +
            ", bSubVVac=" + std::to_string(impl_->bsubv_vac) + ")");
    }
    if (std::fabs((impl_->ctor - impl_->bsubu_vac) / impl_->rbtor) > 0.01) {
        throw cumes::CumesError(
            "VAC-VMEC I_TOR MISMATCH : BOUNDARY MAY ENCLOSE EXT. COIL "
            "(cTor=" +
            std::to_string(impl_->ctor) +
            ", bSubUVac=" + std::to_string(impl_->bsubu_vac) +
            ", rBtor=" + std::to_string(impl_->rbtor) + ")");
    }

    // Soft restart on the first vacuum-active pass (vmecpp :770-779); the
    // edge-force gate is state-dependent and INDEPENDENT of the block gate.
    impl_->soft_restart = (impl_->state == VacuumState::INITIALIZED);
    impl_->edge_gate = (impl_->state == VacuumState::INITIALIZED ||
                        impl_->state == VacuumState::ACTIVE);
}

template <class T>
bool FreeBoundaryOperator<T>::soft_restart_requested() const {
    return impl_->soft_restart;
}

template <class T>
bool FreeBoundaryOperator<T>::apply_edge_force() const {
    return impl_->edge_gate;
}

template <class T>
bool FreeBoundaryOperator<T>::decay_rcon0_zcon0() const {
    return impl_->decay;
}

template <class T>
double FreeBoundaryOperator<T>::rbtor() const {
    return impl_->rbtor;
}

template <class T>
double FreeBoundaryOperator<T>::ctor() const {
    return impl_->ctor;
}

template <class T>
double FreeBoundaryOperator<T>::bsubu_vac() const {
    return impl_->bsubu_vac;
}

template <class T>
double FreeBoundaryOperator<T>::bsubv_vac() const {
    return impl_->bsubv_vac;
}

template <class T>
double FreeBoundaryOperator<T>::delbsq_mean() const {
    return impl_->delbsq;
}

template <class T>
void FreeBoundaryOperator<T>::set_delbsq(T value) {
    impl_->delbsq = static_cast<double>(value);
}

template <class T>
void FreeBoundaryOperator<T>::on_iteration_end() {
    // vmecpp promotes INITIALIZED at the bottom of the same force iteration
    // that consumed the vacuum-activation soft restart. Leaving promotion to
    // the multigrid stage end repeats that restart on every intervening pass.
    if (impl_->state == VacuumState::INITIALIZED) {
        impl_->state = VacuumState::ACTIVE;
        impl_->soft_restart = false;
    }
}

template <class T>
void FreeBoundaryOperator<T>::on_stage_transition(int ns_old, int ns_new) {
    // vmecpp vmec.cc :536-539: the converged coarse-stage vacuum state stays
    // valid; re-mark INITIALIZED so the new stage's first pass runs the
    // vacuum block with the edge force active.
    if (ns_old != 0 && ns_old < ns_new && impl_->state == VacuumState::ACTIVE) {
        impl_->state = VacuumState::INITIALIZED;
    }
}

template <class T>
void FreeBoundaryOperator<T>::on_stage_end() {
    on_iteration_end();
}

template <class T>
void FreeBoundaryOperator<T>::set_edge_pressure(T value) {
    impl_->edge_pressure = value;
}

template <class T>
void FreeBoundaryOperator<T>::enqueue_surface_averages(
    const T* d_bsubu,
    const T* d_bsubv,
    T* d_buco_bvco,
    int ns,
    int ntheta,
    int nzeta,
    cudaStream_t stream) const {
    launch_surface_averages<T>(d_bsubu, d_bsubv, d_buco_bvco, ns, ntheta, nzeta,
                               stream);
}

template <class T>
void FreeBoundaryOperator<T>::enqueue_lcfs_repack(const T* d_rcc,
                                                  const T* d_rss,
                                                  const T* d_zsc,
                                                  const T* d_zcs,
                                                  T* d_repacked,
                                                  int ns,
                                                  int mnmax,
                                                  int mpol,
                                                  int ntor,
                                                  cudaStream_t stream) const {
    (void)mnmax;  // the repack covers mnsize = mpol*(ntor+1) <= mnmax entries
    launch_lcfs_repack<T>(d_rcc, d_rss, d_zsc, d_zcs, d_repacked, ns, mpol,
                          ntor, stream);
}

template <class T>
void FreeBoundaryOperator<T>::enqueue_axis_extract(const T* d_r_e,
                                                   const T* d_z_e,
                                                   T* d_axis,
                                                   int ntheta,
                                                   int nzeta,
                                                   cudaStream_t stream) const {
    launch_axis_extract<T>(d_r_e, d_z_e, d_axis, ntheta, nzeta, stream);
}

template <class T>
void FreeBoundaryOperator<T>::enqueue_rbsq(const T* d_r_e,
                                           const T* d_r_o,
                                           const T* d_total_pressure,
                                           T* d_rbsq,
                                           T* d_delbsq,
                                           int ns,
                                           int ntheta,
                                           int nzeta,
                                           int nZnT,
                                           T delta_s,
                                           cudaStream_t stream) const {
    launch_rbsq<T>(impl_->solver.b_sq_vac(), d_r_e, d_r_o, d_total_pressure,
                   d_rbsq, d_delbsq, ns, ntheta, nzeta, nZnT,
                   impl_->edge_pressure, delta_s, stream);
}

template <class T>
void FreeBoundaryOperator<T>::enqueue_edge_force(T* d_armn_e,
                                                 T* d_armn_o,
                                                 T* d_azmn_e,
                                                 T* d_azmn_o,
                                                 const T* d_zu_e,
                                                 const T* d_zu_o,
                                                 const T* d_ru_e,
                                                 const T* d_ru_o,
                                                 const T* d_rbsq,
                                                 int ns,
                                                 int ntheta,
                                                 int nzeta,
                                                 cudaStream_t stream) const {
    launch_edge_force<T>(d_armn_e, d_armn_o, d_azmn_e, d_azmn_o, d_zu_e, d_zu_o,
                         d_ru_e, d_ru_o, d_rbsq, ns, ntheta * nzeta, stream);
}

template <class T>
void FreeBoundaryOperator<T>::enqueue_rcon_decay(T* d_rcon0,
                                                 T* d_zcon0,
                                                 int ns,
                                                 int ntheta,
                                                 int nzeta,
                                                 cudaStream_t stream) const {
    launch_rcon_decay<T>(d_rcon0, d_zcon0, ns, ntheta, nzeta, stream);
}

}  // namespace cumes

#endif  // CUMES_SRC_FREE_BOUNDARY_IMPL_CUH_
