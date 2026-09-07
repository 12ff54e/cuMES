// free_boundary_operator.hpp — the free-boundary vacuum coupling boundary
// (step 2 of the free-boundary work, docs/adr/0005).
//
// Pimpl boundary over the `vfield::VacuumFieldSolver<T>` from
// deps/vacuum-field: this header deliberately includes NO vacuum-field
// headers, so the solver TUs (solver_impl.cuh) can hold a nullable pointer
// to it even in builds without the submodule (the stub TU then supplies the
// definitions). The operator owns the host state machine mirroring vmecpp's
// VacuumPressureState/ivacskip/nvacskip bookkeeping; the stage-owned device
// workspaces (surface averages, LCFS repack, axis, rBSq) are allocated by
// the EquilibriumOperator and passed to the enqueue_* entry points below.
//
// New-style conventions (snake_case, #ifndef guard) per the 2026-08-24
// coding-style update; the surrounding pre-restyle code follows later at the
// rebase.
#ifndef CUMES_PHYSICS_FREE_BOUNDARY_OPERATOR_HPP_
#define CUMES_PHYSICS_FREE_BOUNDARY_OPERATOR_HPP_

#include "cumes/config/device_params.hpp"
#include "cumes/config/problem_spec.hpp"

#include <cuda_runtime.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cumes {

// vmecpp VacuumPressureState: the vacuum pressure ramps up through
// OFF -> INITIALIZING -> INITIALIZED -> ACTIVE.
enum class VacuumState {
    OFF = -1,
    INITIALIZING = 0,
    INITIALIZED = 1,
    ACTIVE = 2,
};

template <class T>
class FreeBoundaryOperator {
   public:
    using val_type = T;

    struct HostParams {
        std::string mgrid_file;  // precomputed MAKEGRID-format coil field
        std::string coils_file;  // inline Makegrid coil geometry
        std::string makegrid_parameters_file;  // inline Makegrid grid JSON
        std::optional<MakegridParametersSpec> embedded_makegrid_parameters;
        std::vector<double> extcur;  // coil currents (A)
        int nvacskip = 1;            // vacuum full-update cadence
        bool hot_start = false;      // restart from a checkpoint: state starts
                                     // INITIALIZED (vmecpp hot-restart rule)
        bool use_process_environment = true;
    };

    // Loads or generates the response table and builds the persistent vacuum
    // solver. Throws CumesError on invalid input or an angular resolution the
    // vacuum solver would raise above the configured grid.
    FreeBoundaryOperator(const HostParams& params, const DeviceParams<T>& p);
    ~FreeBoundaryOperator();

    FreeBoundaryOperator(const FreeBoundaryOperator&) = delete;
    FreeBoundaryOperator& operator=(const FreeBoundaryOperator&) = delete;

    // ---- host state machine (called from solver_run / MultigridSolver) -----
    VacuumState state() const;
    // The per-pass block gate: lfreeb && (iter2 > 1 || state == INITIALIZED).
    bool run_vacuum_block() const;
    bool full_update_this_pass() const;
    // Top-of-block advance (before the DAG prefix): ivacskip, the ramp gate
    // (full update every pass while state != ACTIVE and fsqr+fsqz < 1e-3),
    // the adaptive nvacskip extension, and the rCon0/zCon0 decay flag.
    void advance(int iter2, int iter1, double fsqr, double fsqz);
    // Mid-DAG host update (after the caller synchronized the compute stream):
    // the vfield solve on the current LCFS/axis state, the bottom promotion,
    // the rBtor/bSubVVac and cTor/bSubUVac consistency checks (CumesError on
    // failure), and the soft-restart flag. `ns` is the CURRENT stage's
    // surface count (the operator persists across stages); buco_h/bvco_h are
    // the host copies of the surface-average kernels' output ([ns-1] each).
    void run_host_update(int ns,
                         const T* buco_h,
                         const T* bvco_h,
                         const T* d_lcfs_repacked,
                         const T* d_r_axis,
                         const T* d_z_axis,
                         cudaStream_t stream);
    bool soft_restart_requested() const;
    // The vacuum edge-force gate (state in {INITIALIZED, ACTIVE} after the
    // update) — distinct from the block gate; also gates the forward-DFT
    // LCFS-row inclusion and the preconditioner boundary row.
    bool apply_edge_force() const;
    bool decay_rcon0_zcon0() const;

    // Host diagnostics for the log (rBtor/cTor in the vmecpp printout units).
    double rbtor() const;
    double ctor() const;
    double bsubu_vac() const;
    double bsubv_vac() const;
    double delbsq_mean() const;
    // Feed the delBSq surface mean back from the device scalar (read by the
    // caller at the control fence; diagnostic only).
    void set_delbsq(T value);

    // Iteration/stage hooks: INITIALIZED is a one-pass state. Promote it to
    // ACTIVE after the activation pass, matching vmecpp's loop-bottom state
    // transition. A finer stage re-marks ACTIVE as INITIALIZED so its first
    // pass repeats the handover/restart sequence.
    void on_iteration_end();
    void on_stage_transition(int ns_old, int ns_new);
    void on_stage_end();
    // Per-stage edge-pressure constant (vmecpp edgePressure; precomputed by
    // the stage driver from the mass profile and presH[ns-2]).
    void set_edge_pressure(T value);

    // ---- device-side enqueues (definitions in src/free_boundary_impl.cuh) --
    // buco/bvco surface averages: one thread per half-grid surface, serial
    // ascending reduced-subset loop with vmecpp's trapezoid weights
    // (1/(nzeta*(ntheta/2)) halved at the endpoints), per-element multiply.
    void enqueue_surface_averages(const T* d_bsubu,
                                  const T* d_bsubv,
                                  T* d_buco_bvco,
                                  int ns,
                                  int ntheta,
                                  int nzeta,
                                  cudaStream_t stream) const;
    // LCFS repack: the four spectral families at j=ns-1, divided by
    // mscale*nscale, transposed to n-major. d_repacked holds 4 contiguous
    // mnsize blocks (rCC/rSS/zSC/zCS).
    void enqueue_lcfs_repack(const T* d_rcc,
                             const T* d_rss,
                             const T* d_zsc,
                             const T* d_zcs,
                             T* d_repacked,
                             int ns,
                             int mnmax,
                             int mpol,
                             int ntor,
                             cudaStream_t stream) const;
    // Axis extraction: r_axis[k] = R(j=0, l=0, k), z_axis[k] = Z(j=0, l=0, k).
    void enqueue_axis_extract(const T* d_r_e,
                              const T* d_z_e,
                              T* d_axis,
                              int ntheta,
                              int nzeta,
                              cudaStream_t stream) const;
    // rBSq at the LCFS (reduced-grid mirror of the vacuum pressure) plus the
    // delBSq surface-mean diagnostic scalar.
    void enqueue_rbsq(const T* d_r_e,
                      const T* d_r_o,
                      const T* d_total_pressure,
                      T* d_rbsq,
                      T* d_delbsq,
                      int ns,
                      int ntheta,
                      int nzeta,
                      int nZnT,
                      T delta_s,
                      cudaStream_t stream) const;
    // The vacuum edge force (vmecpp assembleTotalForces) added to the LCFS
    // row of the parity-split force arrays.
    void enqueue_edge_force(T* d_armn_e,
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
                            cudaStream_t stream) const;
    // rCon0/zCon0 decay (x0.9, every surface) on vacuum-active passes.
    void enqueue_rcon_decay(T* d_rcon0,
                            T* d_zcon0,
                            int ns,
                            int ntheta,
                            int nzeta,
                            cudaStream_t stream) const;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cumes

#endif  // CUMES_PHYSICS_FREE_BOUNDARY_OPERATOR_HPP_
