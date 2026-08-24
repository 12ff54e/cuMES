// dump_windows.hpp — the dump/debug observability machinery for the solver
// (DUMP_CUMES_VERIFY).
//
// The per-iteration dump windows reproduce vmecpp's dump/vmecpp sequence at
// each observation point of the equilibrium DAG; the solver_run helpers write
// the init-window snapshots, the per-pass telemetry record
// (dump/cuMES/per_iter_residuals_cumes.bin), and the dump-only
// device/host-consistency diagnostics. The definitions live in
// kernels/dump_impl.cuh (included once per scalar type by the solver TUs).
//
// The machinery is compiled in only when CMake defines DUMP_CUMES_VERIFY on
// the solver TUs (CUMES_ENABLE_VERIFY_DUMP=ON — the verify/sanitizer/float
// presets), and is RUNTIME-GATED there: nothing is written and no debug
// output is produced unless the CUMES_DUMP=1 environment variable is set
// (see dump_enabled()). The windows themselves are compiled out entirely in
// production-style builds; the solver_run helpers are self-gating no-ops so
// the solver loop can call them without preprocessor guards. Dump files are
// T-native (read back by same-build tooling only); the per-pass record stays
// double.
#ifndef CUMES_INCLUDE_CUMES_SOLVER_DUMP_WINDOWS_HPP_
#define CUMES_INCLUDE_CUMES_SOLVER_DUMP_WINDOWS_HPP_

#include "cumes/config/device_params.hpp"
#include "cumes/numerics/preconditioner.hpp"
#include "cumes/physics/constraint_operator.hpp"
#include "cumes/physics/profiles.hpp"
#include "cumes/solver/pass_record.hpp"
#include "cumes/state/real_fields.cuh"
#include "cumes/state/real_space_storage.hpp"
#include "cumes/state/spectral_storage.hpp"
#include "cumes/transforms/toroidal_fft_operator.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <string_view>
#include <vector>

namespace cumes {

// Master switch for the dump/debug machinery: off unless CUMES_DUMP=1.
// Compiled out (returns false) in production-style builds.
bool dump_enabled();

// All dump output routes through dump_ensure_dir/dump_device_array, which
// no-op when disabled, so default runs write nothing to dump/ and print no
// debug noise.
void dump_ensure_dir();

// Refresh-pass force-norm dump (dump-only telemetry). The six device scalars
// hc[0..5] = {sRZ, sL, sMag, eTherm, vol, rzNorm} are printed together with
// the fNorm factors ACTUALLY used for the convergence decision — those are the
// record's device-finalized final_f_norm_* fields (completion-plan follow-up
// §2.3), consumed by the caller, not recomputed here. Self-gating no-op in
// production-style builds.
void dump_force_norms(const double* hc,
                      double delta_s,
                      int iter2,
                      double f_norm_rz,
                      double f_norm_l,
                      double f_norm1);

#ifdef DUMP_CUMES_VERIFY
// T-native dump: written as sizeof(T) elements; only read back by same-build
// tooling (e.g. tests/test_geometry_iso.cu, which is double-build-only).
template <typename T>
void dump_device_array(std::string_view filename,
                       const T* d_data,
                       std::size_t nelem);

// ---- per-iteration dump windows (7.1) ------------------------------------
// Extracted from EquilibriumOperator::enqueue so the DAG body reads as the
// arithmetic pipeline it is. Each helper reproduces its observation point's
// exact dump sequence (same files, same order, same contents) and is
// runtime-gated by dump_enabled(). The one stream side effect the windows
// carry — combine_parity at iter 0 (the *_real snapshot materialization) —
// is preserved in dump_step_a so the dump-disabled path keeps the frozen
// trajectory's launch sequence bit-for-bit.

// Iter-0 loop diagnostic: print + dump the LCFS real-space R right after the
// first inverse DFT.
template <typename T>
void dump_iter0_loop_diag(int iter,
                          const DeviceParams<T>& p,
                          const RealSpaceStorage<T>& rs);

// postinverse window (post-inverse, pre-geometry): lambda derivatives and, at
// iter 0, the full R/Z/λ real-space snapshot.
template <typename T>
void dump_step_a(int iter,
                 int iter2,
                 const DeviceParams<T>& p,
                 const RealSpaceStorage<T>& rs,
                 ToroidalFftOperator<T>& transform,
                 cudaStream_t stream);

// metric/bcontra window (post-field): metric + contravariant-B half-grid dump.
template <typename T>
void dump_step_d(int iter,
                 int iter2,
                 const DeviceParams<T>& p,
                 const BaseGeometryHalfViews<T>& base,
                 const MagneticFieldViews<T>& field);

// precon window (iter 0, on the refresh pass): tridiagonal matrices,
// jMin, intermediates, sizes.
template <typename T>
void dump_step_precon(int iter,
                      const DeviceParams<T>& p,
                      const Preconditioner<T>& precon);

// forceterm/force + halfgrid window (post-force): half-grid geometry +
// force outputs.
template <typename T>
void dump_step_ef(int iter,
                  int iter2,
                  const DeviceParams<T>& p,
                  const BaseGeometryHalfViews<T>& base,
                  const MagneticFieldViews<T>& field,
                  const RealSpaceStorage<T>& rs);

// constraint window (iter 0, post-constraint): the augmented force +
// the constraint-chain intermediates.
template <typename T>
void dump_step_g(int iter,
                 const DeviceParams<T>& p,
                 const RealSpaceStorage<T>& rs,
                 SpectralStorage<T>& storage,
                 ConstraintOperator<T>& constraint);

// scaled window (post-decomposition-scaling): decomposed force slab + the
// E2-start invariant-force window.
template <typename T>
void dump_step_h(int iter,
                 int iter2,
                 const DeviceParams<T>& p,
                 const T* f_spec);

// final window (post-m1-gauge, pre-invariant-residual): final-pass
// force-slab dump.
template <typename T>
void dump_step_final(int iter, const DeviceParams<T>& p, const T* f_spec);

// preconditioned window (post-preconditioner): preconditioned slab +
// state/velocity handoff dumps + the E2-start preconditioned-force window.
template <typename T>
void dump_step_i(int iter,
                 int iter2,
                 const DeviceParams<T>& p,
                 const T* f_spec,
                 SpectralStorage<T>& storage);
#endif  // DUMP_CUMES_VERIFY

// Inverse-DFT diagnostic (CUMES_DUMP=1 only): the combined *_real buffers are
// materialized on demand at the dump site, never read stale. Self-gating
// no-op in production-style builds.
template <typename T>
void dump_inverse_diag(ToroidalFftOperator<T>& transform,
                       SpectralStorage<T>& storage,
                       RealSpaceStorage<T>& rs,
                       const DeviceParams<T>& p,
                       cudaStream_t stream);

// Pre-loop init-window snapshots of the spectral state and the initial
// profiles. Self-gating no-op in production-style builds.
template <typename T>
void dump_step_0(const DeviceParams<T>& p,
                 SpectralStorage<T>& storage,
                 const Profiles<T>& profiles);

// Dump-only device/host rule consistency checks (the finalize kernels and the
// controller must decide identically, or the guards would suppress work the
// controller later consumes). Self-gating no-ops in production-style builds.
void dump_check_jacobian_consistency(bool device_valid,
                                     bool host_invalid,
                                     int iter2);
void dump_check_predicate_consistency(int dev_nf,
                                      int dev_cv,
                                      int prec_eval,
                                      bool verdict_nf,
                                      bool verdict_cv,
                                      int iter2);

// Restart/termination event lines, compiled out of release (fast) builds with
// the rest of the dump machinery; the iteration table still gets a row.
void dump_event_bad_jacobian(double min_oriented,
                             double max_abs,
                             double nonfinite,
                             int j_half,
                             double delt);
void dump_event_nonfinite(double delt);
void dump_event_converged(int iter2);
void dump_event_restart(bool bad_jacobian, int iter2, double delt);

// Per-pass record for convergence analysis (mirrors vmecpp's
// per_iter_residuals.bin + control scalars). Typed PassRecord; the field
// order is the frozen 15-column on-disk contract (fsqr_i fsqz_i fsql_i fsqr
// fsqz fsql delt otav dtau b1 fac iter2 iter1 reason rax). Observers read
// these scalars and cannot affect the controller's decisions — they see only
// scalars this pass already produced. Self-gating no-op in production-style
// builds.
template <typename T>
class PerIterRecorder {
   public:
    PerIterRecorder(const DeviceParams<T>& p,
                    SpectralStorage<T>& storage,
                    cudaStream_t stream,
                    int max_iter_eff);

    // Record one evaluated pass's control scalars (no-op when the dump
    // machinery is compiled out or disabled).
    void record(int reason,
                double f_ri,
                double f_zi,
                double f_li,
                double f_r,
                double f_z,
                double f_l,
                double d,
                double o,
                double dt,
                double b1v,
                double fcv,
                int it2,
                int it1);

    // Write the column-major record (dump/cuMES/per_iter_residuals_cumes.bin),
    // byte-identical to the legacy array-of-doubles layout.
    void write_file() const;

   private:
    // The frozen baseline's axis_r column was read POST-descent with a
    // synchronized strided copy; keep that exact read for the dump record.
    double axis_r_at_zeta0_sync() const;

    const DeviceParams<T>& p_;
    SpectralStorage<T>& storage_;
    cudaStream_t stream_;
    int max_iter_eff_;
    std::vector<PassRecord> per_iter_;
};

}  // namespace cumes

#endif  // CUMES_INCLUDE_CUMES_SOLVER_DUMP_WINDOWS_HPP_
