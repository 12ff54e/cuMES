// free_boundary_stub_impl.cuh — stub definitions for builds WITHOUT the
// vacuum-field submodule (hosted CI): the FreeBoundaryOperator symbol set
// exists so every call site compiles and links unconditionally. The
// constructor throws CumesError (the construction site in MultigridSolver
// is guarded on lfreeb, which validation already requires the submodule for,
// so the throw is a hard error only for a misconfigured lfreeb run).
//
// Included once per scalar type by free_boundary_stub_{double,float}.cu.
// New-style conventions (snake_case) per the 2026-08-24 coding-style update.
#ifndef CUMES_SRC_FREE_BOUNDARY_STUB_IMPL_CUH_
#define CUMES_SRC_FREE_BOUNDARY_STUB_IMPL_CUH_

#include "cumes/physics/free_boundary_operator.hpp"
#include "cumes/runtime/cuda_status.hpp"

namespace cumes {

template <class T>
struct FreeBoundaryOperator<T>::Impl {};

template <class T>
FreeBoundaryOperator<T>::FreeBoundaryOperator(const HostParams& /*params*/,
                                              const DeviceParams<T>& /*p*/)
    : impl_(nullptr) {
    throw cumes::CumesError(
        "lfreeb=true: the free-boundary vacuum library (deps/vacuum-field) "
        "is not built into this cuMES configuration (CUMES_USE_VACUUM_FIELD "
        "is OFF or the submodule is not checked out)");
}

template <class T>
FreeBoundaryOperator<T>::~FreeBoundaryOperator() = default;

template <class T>
VacuumState FreeBoundaryOperator<T>::state() const {
    return VacuumState::OFF;
}

template <class T>
bool FreeBoundaryOperator<T>::run_vacuum_block() const {
    return false;
}

template <class T>
bool FreeBoundaryOperator<T>::full_update_this_pass() const {
    return false;
}

template <class T>
void FreeBoundaryOperator<T>::advance(int /*iter2*/,
                                      int /*iter1*/,
                                      double /*fsqr*/,
                                      double /*fsqz*/) {}

template <class T>
void FreeBoundaryOperator<T>::run_host_update(int /*ns*/,
                                              const T* /*buco_h*/,
                                              const T* /*bvco_h*/,
                                              const T* /*d_lcfs_repacked*/,
                                              const T* /*d_r_axis*/,
                                              const T* /*d_z_axis*/,
                                              cudaStream_t /*stream*/) {}

template <class T>
bool FreeBoundaryOperator<T>::soft_restart_requested() const {
    return false;
}

template <class T>
bool FreeBoundaryOperator<T>::apply_edge_force() const {
    return false;
}

template <class T>
bool FreeBoundaryOperator<T>::decay_rcon0_zcon0() const {
    return false;
}

template <class T>
double FreeBoundaryOperator<T>::rbtor() const {
    return 0.0;
}

template <class T>
double FreeBoundaryOperator<T>::ctor() const {
    return 0.0;
}

template <class T>
double FreeBoundaryOperator<T>::bsubu_vac() const {
    return 0.0;
}

template <class T>
double FreeBoundaryOperator<T>::bsubv_vac() const {
    return 0.0;
}

template <class T>
double FreeBoundaryOperator<T>::delbsq_mean() const {
    return 0.0;
}

template <class T>
void FreeBoundaryOperator<T>::set_delbsq(T /*value*/) {}

template <class T>
void FreeBoundaryOperator<T>::on_iteration_end() {}

template <class T>
void FreeBoundaryOperator<T>::on_stage_transition(int /*ns_old*/,
                                                  int /*ns_new*/) {}

template <class T>
void FreeBoundaryOperator<T>::on_stage_end() {}

template <class T>
void FreeBoundaryOperator<T>::set_edge_pressure(T /*value*/) {}

template <class T>
void FreeBoundaryOperator<T>::enqueue_surface_averages(
    const T* /*d_bsubu*/,
    const T* /*d_bsubv*/,
    T* /*d_buco_bvco*/,
    int /*ns*/,
    int /*ntheta*/,
    int /*nzeta*/,
    cudaStream_t /*stream*/) const {}

template <class T>
void FreeBoundaryOperator<T>::enqueue_lcfs_repack(
    const T* /*d_rcc*/,
    const T* /*d_rss*/,
    const T* /*d_zsc*/,
    const T* /*d_zcs*/,
    T* /*d_repacked*/,
    int /*ns*/,
    int /*mnmax*/,
    int /*mpol*/,
    int /*ntor*/,
    cudaStream_t /*stream*/) const {}

template <class T>
void FreeBoundaryOperator<T>::enqueue_axis_extract(
    const T* /*d_r_e*/,
    const T* /*d_z_e*/,
    T* /*d_axis*/,
    int /*ntheta*/,
    int /*nzeta*/,
    cudaStream_t /*stream*/) const {}

template <class T>
void FreeBoundaryOperator<T>::enqueue_rbsq(const T* /*d_r_e*/,
                                           const T* /*d_r_o*/,
                                           const T* /*d_total_pressure*/,
                                           T* /*d_rbsq*/,
                                           T* /*d_delbsq*/,
                                           int /*ns*/,
                                           int /*ntheta*/,
                                           int /*nzeta*/,
                                           int /*nZnT*/,
                                           T /*delta_s*/,
                                           cudaStream_t /*stream*/) const {}

template <class T>
void FreeBoundaryOperator<T>::enqueue_edge_force(
    T* /*d_armn_e*/,
    T* /*d_armn_o*/,
    T* /*d_azmn_e*/,
    T* /*d_azmn_o*/,
    const T* /*d_zu_e*/,
    const T* /*d_zu_o*/,
    const T* /*d_ru_e*/,
    const T* /*d_ru_o*/,
    const T* /*d_rbsq*/,
    int /*ns*/,
    int /*ntheta*/,
    int /*nzeta*/,
    cudaStream_t /*stream*/) const {}

template <class T>
void FreeBoundaryOperator<T>::enqueue_rcon_decay(
    T* /*d_rcon0*/,
    T* /*d_zcon0*/,
    int /*ns*/,
    int /*ntheta*/,
    int /*nzeta*/,
    cudaStream_t /*stream*/) const {}

}  // namespace cumes

#endif  // CUMES_SRC_FREE_BOUNDARY_STUB_IMPL_CUH_
