// kernels/dump_impl.cuh — dump window / observability definitions for
// solver_impl.cuh (dump_windows.hpp declarations).
//
// Included once per scalar type by solver_double.cu / solver_float.cu (through
// kernels/solver_impl.cuh); see the explicit-instantiation split
// (cumes_cuda_double / cumes_cuda_float). The DUMP_CUMES_VERIFY-gated
// definitions below are compiled in only when CMake defines that option on
// the solver TUs; everything else is self-gating (no-op in production-style
// builds) so the solver loop can call it without preprocessor guards.
#ifndef CUMES_SRC_DUMP_IMPL_CUH_
#define CUMES_SRC_DUMP_IMPL_CUH_

#include "cumes/runtime/cuda_status.hpp"
#include "cumes/solver/dump_windows.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

namespace cumes {

inline bool dump_enabled() {
#ifdef DUMP_CUMES_VERIFY
    // Read the env var once per process (the old form re-read it at every
    // one of the ~6 dump-window entry points per iteration).
    static const bool enabled = [] {
        const char* e = getenv("CUMES_DUMP");
        return e != nullptr && atoi(e) != 0;
    }();
    return enabled;
#else
    return false;
#endif
}

inline void dump_ensure_dir() {
#ifdef DUMP_CUMES_VERIFY
    if (!dump_enabled()) return;
    int rc = system("mkdir -p dump/cuMES");
    if (rc != 0)
        fprintf(stderr, "dump_ensure_dir: mkdir -p failed (rc=%d)\n", rc);
#endif
}

#ifdef DUMP_CUMES_VERIFY
template <typename T>
void dump_device_array(std::string_view filename,
                       const T* d_data,
                       size_t nelem) {
    if (!dump_enabled()) return;
    // The dump machinery reads device data on the (synchronous) default stream
    // while the hot loop produces it on the nonblocking compute stream. Sync
    // everything first so a dump never reads a stale/in-flight buffer. This is
    // compile- and runtime-gated observability, so the extra fence is free on
    // the production path.
    cudaDeviceSynchronize();
    std::vector<T> h_tmp(nelem);
    cudaError_t err = cudaMemcpy(h_tmp.data(), d_data, nelem * sizeof(T),
                                 cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "dump_device_array cudaMemcpy failed for %s: %s\n",
                filename.data(), cudaGetErrorString(err));
    }
    FILE* fp = fopen(std::string(filename).c_str(), "wb");
    if (fp) {
        uint64_t n = nelem;
        fwrite(&n, sizeof(uint64_t), 1, fp);
        fwrite(h_tmp.data(), sizeof(T), nelem, fp);
        fclose(fp);
    }
}
#endif  // DUMP_CUMES_VERIFY

inline void dump_force_norms(const double* hc,
                             double delta_s,
                             int iter2,
                             double f_norm_rz,
                             double f_norm_l,
                             double f_norm1) {
#ifndef DUMP_CUMES_VERIFY
    (void)hc;
    (void)delta_s;
    (void)iter2;
    (void)f_norm_rz;
    (void)f_norm_l;
    (void)f_norm1;
#endif
#ifdef DUMP_CUMES_VERIFY
    if (dump_enabled()) {
        double s_rz = hc[0], s_l = hc[1], s_mag = hc[2], e_therm = hc[3],
               vol = hc[4], h_rz = hc[5];
        double e_mag =
            fabs(s_mag) * delta_s;  // vmecpp: fabs(localMagneticEnergy)*deltaS
        e_therm *= delta_s;
        vol *= delta_s;
        double energy_density = std::max(e_mag, e_therm) / vol;
        // Same format as vmecpp's dump/vmecpp/force_norms_iter_<iter2>.txt
        char fn[128];
        snprintf(fn, sizeof fn, "dump/cuMES/force_norms_iter_%d.txt", iter2);
        FILE* fp2 = fopen(fn, "w");
        if (fp2) {
            fprintf(fp2,
                    "magneticEnergy %.17e\n"
                    "thermalEnergy %.17e\n"
                    "plasmaVolume %.17e\n"
                    "energyDensity %.17e\n"
                    "forceNormSumRZ %.17e\n"
                    "forceNormSumL %.17e\n"
                    "rzNorm %.17e\n"
                    "fNormRZ %.17e\n"
                    "fNormL %.17e\n"
                    "fNorm1 %.17e\n",
                    (double)e_mag, (double)e_therm, (double)vol,
                    (double)energy_density, (double)s_rz, (double)s_l,
                    (double)h_rz, (double)f_norm_rz, (double)f_norm_l,
                    (double)f_norm1);
            fclose(fp2);
        }
    }
#endif
}

#ifdef DUMP_CUMES_VERIFY
// Env-gated dump-window knobs (CUMES_DUMP_ITER / CUMES_E2_START), cached once
// per process like dump_enabled(). The CUMES_MAX_ITER override is NOT cached:
// p.max_iter is per-stage (multigrid overwrites it), so dump_max_iter folds
// the process-wide env override onto the caller's per-stage default.
struct DumpKnobs {
    int dump_iter = 150;
    int e2_start = 560;
};

static const DumpKnobs& dump_knobs() {
    static const DumpKnobs knobs = [] {
        DumpKnobs kn;
        if (const char* e = getenv("CUMES_DUMP_ITER")) kn.dump_iter = atoi(e);
        if (const char* e = getenv("CUMES_E2_START")) kn.e2_start = atoi(e);
        return kn;
    }();
    return knobs;
}

static int dump_max_iter(int default_max_iter) {
    static const bool has_override = getenv("CUMES_MAX_ITER") != nullptr;
    static const int override =
        has_override ? atoi(getenv("CUMES_MAX_ITER")) : 0;
    return has_override ? override : default_max_iter;
}

// Iter-0 loop diagnostic: print + dump the LCFS real-space R right after the
// first inverse DFT (2.1: the transform ran on the nonblocking compute
// stream, so dump_device_array's device sync is what makes the read valid).
template <typename T>
void dump_iter0_loop_diag(int iter,
                          const DeviceParams<T>& p,
                          const RealSpaceStorage<T>& rs) {
    if (iter != 0 || !dump_enabled()) return;
    dump_device_array("dump/cuMES/iter0_diag_r_e.bin", rs.d_r_e,
                      (size_t)p.nZnT * (size_t)p.ns);
    auto* h_test = new T[p.nZnT * p.ns];
    check_cuda(cudaMemcpy(h_test, rs.d_r_e, p.nZnT * p.ns * sizeof(T),
                          cudaMemcpyDeviceToHost),
               "loop test");
    int j_b = p.ns - 1;
    printf("  [loop diag] LCFS theta=0: r_e=%.4f (expect ~3.93)\n",
           (double)h_test[0 + j_b * p.nZnT]);
    delete[] h_test;
}

// postinverse window (post-inverse, pre-geometry): lambda derivatives and, at
// iter 0, the full R/Z/λ real-space snapshot.
template <typename T>
void dump_step_a(int iter,
                 int iter2,
                 const DeviceParams<T>& p,
                 const RealSpaceStorage<T>& rs,
                 ToroidalFftOperator<T>& transform,
                 cudaStream_t stream) {
    if (!dump_enabled()) {
        // combine_parity runs at iter 0 even when dumping is off — the
        // frozen launch sequence (the *_real buffers are not read by the hot
        // loop, so this is launch-order fidelity only).
        if (iter == 0) transform.combine_parity(stream);
        return;
    }
    if (iter == 0 || iter2 == 2) {
        // iter 2 = first pass with lambda != 0: dump the real-space
        // lambda derivatives for the basis-convention check.
        size_t n_real = (size_t)p.ns * (size_t)p.nZnT;
        char fn[128];
        snprintf(fn, sizeof fn, "dump/cuMES/postinverse_lu_e_iter_%d.bin",
                 iter == 0 ? 1 : 2);
        dump_device_array(fn, rs.d_lu_e, n_real);
        snprintf(fn, sizeof fn, "dump/cuMES/postinverse_lu_o_iter_%d.bin",
                 iter == 0 ? 1 : 2);
        dump_device_array(fn, rs.d_lu_o, n_real);
        snprintf(fn, sizeof fn, "dump/cuMES/postinverse_l_real_iter_%d.bin",
                 iter == 0 ? 1 : 2);
        dump_device_array(fn, rs.d_l_real, n_real);
    }
    if (iter == 0 || iter2 == 2) {
        size_t n_real = (size_t)p.ns * (size_t)p.nZnT;
        char fn[128];
        snprintf(fn, sizeof fn, "dump/cuMES/postinverse_lv_e_iter_%d.bin",
                 iter == 0 ? 1 : 2);
        dump_device_array(fn, rs.d_lv_e, n_real);
        snprintf(fn, sizeof fn, "dump/cuMES/postinverse_lv_o_iter_%d.bin",
                 iter == 0 ? 1 : 2);
        dump_device_array(fn, rs.d_lv_o, n_real);
    }
    if (iter == 0) {
        size_t n_real = (size_t)p.ns * (size_t)p.nZnT;
        // The combined *_real arrays are NOT refreshed by the hot loop
        // (inverseDFT runs with do_combine=false); materialize a fresh
        // snapshot from the current parity arrays before dumping them.
        transform.combine_parity(stream);
        // Full R, Z, lambda (even+odd)
        dump_device_array("dump/cuMES/postinverse_r_real_iter_1.bin",
                          rs.d_r_real, n_real);
        dump_device_array("dump/cuMES/postinverse_z_real_iter_1.bin",
                          rs.d_z_real, n_real);
        // Even-m parity
        dump_device_array("dump/cuMES/postinverse_r_e_iter_1.bin", rs.d_r_e,
                          n_real);
        dump_device_array("dump/cuMES/postinverse_z_e_iter_1.bin", rs.d_z_e,
                          n_real);
        dump_device_array("dump/cuMES/postinverse_l_e_iter_1.bin", rs.d_l_e,
                          n_real);
        // Odd-m parity
        dump_device_array("dump/cuMES/postinverse_r_o_iter_1.bin", rs.d_r_o,
                          n_real);
        dump_device_array("dump/cuMES/postinverse_z_o_iter_1.bin", rs.d_z_o,
                          n_real);
        dump_device_array("dump/cuMES/postinverse_l_o_iter_1.bin", rs.d_l_o,
                          n_real);
        // Poloidal derivatives
        dump_device_array("dump/cuMES/postinverse_ru_real_iter_1.bin",
                          rs.d_ru_real, n_real);
        dump_device_array("dump/cuMES/postinverse_zu_real_iter_1.bin",
                          rs.d_zu_real, n_real);
        dump_device_array("dump/cuMES/postinverse_lu_real_iter_1.bin",
                          rs.d_lu_real, n_real);
        // Toroidal derivatives
        dump_device_array("dump/cuMES/postinverse_rv_real_iter_1.bin",
                          rs.d_rv_real, n_real);
        dump_device_array("dump/cuMES/postinverse_zv_real_iter_1.bin",
                          rs.d_zv_real, n_real);
        dump_device_array("dump/cuMES/postinverse_lv_real_iter_1.bin",
                          rs.d_lv_real, n_real);
        // Even-m poloidal derivatives
        dump_device_array("dump/cuMES/postinverse_ru_e_iter_1.bin", rs.d_ru_e,
                          n_real);
        dump_device_array("dump/cuMES/postinverse_zu_e_iter_1.bin", rs.d_zu_e,
                          n_real);
        dump_device_array("dump/cuMES/postinverse_lu_e_iter_1.bin", rs.d_lu_e,
                          n_real);
        // Odd-m poloidal derivatives
        dump_device_array("dump/cuMES/postinverse_ru_o_iter_1.bin", rs.d_ru_o,
                          n_real);
        dump_device_array("dump/cuMES/postinverse_zu_o_iter_1.bin", rs.d_zu_o,
                          n_real);
        dump_device_array("dump/cuMES/postinverse_lu_o_iter_1.bin", rs.d_lu_o,
                          n_real);
    }
}

// metric/bcontra window (post-field): metric + contravariant-B half-grid dump.
template <typename T>
void dump_step_d(int iter,
                 int iter2,
                 const DeviceParams<T>& p,
                 const BaseGeometryHalfViews<T>& base,
                 const MagneticFieldViews<T>& field) {
    if (!dump_enabled()) return;
    if (iter == 0 || iter2 == 2) {
        // iter 2 = first pass with lambda != 0 (E3-D bsupv check)
        size_t n_half = (size_t)(p.ns - 1) * (size_t)p.nZnT;
        char fn[128];
        snprintf(fn, sizeof fn, "dump/cuMES/bcontra_bsupu_iter_%d.bin",
                 iter == 0 ? 1 : 2);
        dump_device_array(fn, field.bsupu.data(), n_half);
        snprintf(fn, sizeof fn, "dump/cuMES/bcontra_bsupv_iter_%d.bin",
                 iter == 0 ? 1 : 2);
        dump_device_array(fn, field.bsupv.data(), n_half);
    }
    if (iter == 0) {
        size_t n_half = (size_t)(p.ns - 1) * (size_t)p.nZnT;
        dump_device_array("dump/cuMES/metric_gsqrt_iter_1.bin",
                          base.gsqrt.data(), n_half);
        dump_device_array("dump/cuMES/metric_guu_iter_1.bin", base.guu.data(),
                          n_half);
        dump_device_array("dump/cuMES/metric_guv_iter_1.bin", base.guv.data(),
                          n_half);
        dump_device_array("dump/cuMES/metric_gvv_iter_1.bin", base.gvv.data(),
                          n_half);
    }
}

// precon window (iter 0, on the refresh pass): tridiagonal matrices,
// jMin, intermediates, sizes.
template <typename T>
void dump_step_precon(int iter,
                      const DeviceParams<T>& p,
                      const Preconditioner<T>& precon) {
    // The extracted helper gates the whole window behind dump_enabled();
    // the pre-extraction body additionally ran the jMin D2H copy and the
    // jMin/sizes fopen writes unconditionally (harmless reads/failed opens
    // on the default path — no numeric effect, Class A verified).
    if (iter != 0 || !dump_enabled()) return;
    // Dump tridiagonal preconditioner matrices for comparison
    // with vmecpp. cuMES layout: ar[mode * ns + jF] (mode-major).
    size_t n_tri = (size_t)p.mnmax * (size_t)p.ns;
    size_t n_half_2 = (size_t)2 * (size_t)(p.ns - 1);
    size_t n_full_2 = (size_t)2 * (size_t)p.ns;
    size_t n_full_1 = (size_t)p.ns;

    // Tridiagonal matrix elements (mode-major: [mode, jF])
    dump_device_array("dump/cuMES/precon_ar_iter_1.bin", precon.ar(), n_tri);
    dump_device_array("dump/cuMES/precon_dr_iter_1.bin", precon.dr(), n_tri);
    dump_device_array("dump/cuMES/precon_br_iter_1.bin", precon.br(), n_tri);
    dump_device_array("dump/cuMES/precon_az_iter_1.bin", precon.az(), n_tri);
    dump_device_array("dump/cuMES/precon_dz_iter_1.bin", precon.dz(), n_tri);
    dump_device_array("dump/cuMES/precon_bz_iter_1.bin", precon.bz(), n_tri);

    // jMin per mode (stored as int, convert to double for dump)
    {
        int* h_jmin = new int[p.mnmax];
        cudaMemcpy(h_jmin, precon.jmin(), p.mnmax * sizeof(int),
                   cudaMemcpyDeviceToHost);
        double* h_jmin_dbl = new double[p.mnmax];
        for (int i = 0; i < p.mnmax; ++i) h_jmin_dbl[i] = (double)h_jmin[i];
        FILE* fj = fopen("dump/cuMES/precon_jmin_iter_1.bin", "wb");
        if (fj) {
            uint64_t n = p.mnmax;
            fwrite(&n, sizeof(uint64_t), 1, fj);
            fwrite(h_jmin_dbl, sizeof(double), p.mnmax, fj);
            fclose(fj);
        }
        delete[] h_jmin;
        delete[] h_jmin_dbl;
    }

    // Intermediate arrays
    dump_device_array("dump/cuMES/precon_arm_iter_1.bin", precon.arm(),
                      n_half_2);
    dump_device_array("dump/cuMES/precon_ard_iter_1.bin", precon.ard(),
                      n_full_2);
    dump_device_array("dump/cuMES/precon_brm_iter_1.bin", precon.brm(),
                      n_half_2);
    dump_device_array("dump/cuMES/precon_brd_iter_1.bin", precon.brd(),
                      n_full_2);
    dump_device_array("dump/cuMES/precon_azm_iter_1.bin", precon.azm(),
                      n_half_2);
    dump_device_array("dump/cuMES/precon_azd_iter_1.bin", precon.azd(),
                      n_full_2);
    dump_device_array("dump/cuMES/precon_bzm_iter_1.bin", precon.bzm(),
                      n_half_2);
    dump_device_array("dump/cuMES/precon_bzd_iter_1.bin", precon.bzd(),
                      n_full_2);
    dump_device_array("dump/cuMES/precon_cxd_iter_1.bin", precon.cxd(),
                      n_full_1);

    // Sizes for comparison script
    double sizes_dbl[4] = {(double)p.ns, (double)(p.ns - 1), (double)p.mpol,
                           1.0};
    FILE* fs = fopen("dump/cuMES/precon_sizes_iter_1.bin", "wb");
    if (fs) {
        uint64_t n = 4;
        fwrite(&n, sizeof(uint64_t), 1, fs);
        fwrite(sizes_dbl, sizeof(double), 4, fs);
        fclose(fs);
    }
}

// forceterm/force + halfgrid window (post-force): half-grid geometry + force
// outputs.
template <typename T>
void dump_step_ef(int iter,
                  int iter2,
                  const DeviceParams<T>& p,
                  const BaseGeometryHalfViews<T>& base,
                  const MagneticFieldViews<T>& field,
                  const RealSpaceStorage<T>& rs) {
    if (!dump_enabled()) return;
    if (iter == 0 || iter2 == 2) {
        // iter 2 = first pass with lambda != 0 (E3-B blmn blending check)
        size_t n_half = (size_t)(p.ns - 1) * (size_t)p.nZnT;
        if (iter == 0) {
            dump_device_array("dump/cuMES/halfgrid_r12_iter_1.bin",
                              base.r12.data(), n_half);
            dump_device_array("dump/cuMES/halfgrid_zu12_iter_1.bin",
                              base.zu12.data(), n_half);
            dump_device_array("dump/cuMES/halfgrid_tau_iter_1.bin",
                              base.tau.data(), n_half);
            dump_device_array("dump/cuMES/halfgrid_gsqrt_iter_1.bin",
                              base.gsqrt.data(), n_half);
            dump_device_array("dump/cuMES/halfgrid_totalP_iter_1.bin",
                              field.total_pressure.data(), n_half);
            dump_device_array("dump/cuMES/halfgrid_bsupu_iter_1.bin",
                              field.bsupu.data(), n_half);
            dump_device_array("dump/cuMES/halfgrid_bsupv_iter_1.bin",
                              field.bsupv.data(), n_half);
            dump_device_array("dump/cuMES/halfgrid_bsubu_iter_1.bin",
                              field.bsubu.data(), n_half);
            dump_device_array("dump/cuMES/halfgrid_bsubv_iter_1.bin",
                              field.bsubv.data(), n_half);
            dump_device_array("dump/cuMES/halfgrid_rs_iter_1.bin",
                              base.rs.data(), n_half);
            dump_device_array("dump/cuMES/halfgrid_zs_iter_1.bin",
                              base.zs.data(), n_half);
        }

        size_t n_real = (size_t)p.ns * (size_t)p.nZnT;
        char fn[128];
        int itag = (iter == 0) ? 1 : iter2;
        snprintf(fn, sizeof fn, "dump/cuMES/force_brmn_e_iter_%d.bin", itag);
        dump_device_array(fn, rs.d_brmn_e, n_real);
        snprintf(fn, sizeof fn, "dump/cuMES/force_brmn_o_iter_%d.bin", itag);
        dump_device_array(fn, rs.d_brmn_o, n_real);
        snprintf(fn, sizeof fn, "dump/cuMES/force_bzmn_e_iter_%d.bin", itag);
        dump_device_array(fn, rs.d_bzmn_e, n_real);
        snprintf(fn, sizeof fn, "dump/cuMES/force_bzmn_o_iter_%d.bin", itag);
        dump_device_array(fn, rs.d_bzmn_o, n_real);
        snprintf(fn, sizeof fn, "dump/cuMES/force_blmn_e_iter_%d.bin", itag);
        dump_device_array(fn, rs.d_blmn_e, n_real);
        snprintf(fn, sizeof fn, "dump/cuMES/force_blmn_o_iter_%d.bin", itag);
        dump_device_array(fn, rs.d_blmn_o, n_real);
        if (iter == 0) {
            dump_device_array("dump/cuMES/forceterm_armn_e_iter_1.bin",
                              rs.d_armn_e, n_real);
            dump_device_array("dump/cuMES/forceterm_armn_o_iter_1.bin",
                              rs.d_armn_o, n_real);
            dump_device_array("dump/cuMES/forceterm_azmn_e_iter_1.bin",
                              rs.d_azmn_e, n_real);
            dump_device_array("dump/cuMES/forceterm_azmn_o_iter_1.bin",
                              rs.d_azmn_o, n_real);
            dump_device_array("dump/cuMES/forceterm_brmn_e_iter_1.bin",
                              rs.d_brmn_e, n_real);
            dump_device_array("dump/cuMES/forceterm_brmn_o_iter_1.bin",
                              rs.d_brmn_o, n_real);
            dump_device_array("dump/cuMES/forceterm_bzmn_e_iter_1.bin",
                              rs.d_bzmn_e, n_real);
            dump_device_array("dump/cuMES/forceterm_bzmn_o_iter_1.bin",
                              rs.d_bzmn_o, n_real);
            dump_device_array("dump/cuMES/forceterm_crmn_e_iter_1.bin",
                              rs.d_crmn_e, n_real);
            dump_device_array("dump/cuMES/forceterm_crmn_o_iter_1.bin",
                              rs.d_crmn_o, n_real);
            dump_device_array("dump/cuMES/forceterm_czmn_e_iter_1.bin",
                              rs.d_czmn_e, n_real);
            dump_device_array("dump/cuMES/forceterm_czmn_o_iter_1.bin",
                              rs.d_czmn_o, n_real);
            dump_device_array("dump/cuMES/forceterm_blmn_e_iter_1.bin",
                              rs.d_blmn_e, n_real);
            dump_device_array("dump/cuMES/forceterm_blmn_o_iter_1.bin",
                              rs.d_blmn_o, n_real);
            dump_device_array("dump/cuMES/forceterm_clmn_e_iter_1.bin",
                              rs.d_clmn_e, n_real);
            dump_device_array("dump/cuMES/forceterm_clmn_o_iter_1.bin",
                              rs.d_clmn_o, n_real);
            // NOTE: no combined-force dumps — the force combine buffers
            // were removed (they were allocated/dumped but never
            // produced; the parity-split arrays above are the source of
            // truth).
        }
    }
}

// constraint window (iter 0, post-constraint): the augmented force + the
// constraint-chain intermediates.
template <typename T>
void dump_step_g(int iter,
                 const DeviceParams<T>& p,
                 const RealSpaceStorage<T>& rs,
                 SpectralStorage<T>& storage,
                 ConstraintOperator<T>& constraint) {
    if (iter != 0 || !dump_enabled()) return;
    size_t n_real = (size_t)p.ns * (size_t)p.nZnT;
    dump_device_array("dump/cuMES/constraint_brmn_e_iter_1.bin", rs.d_brmn_e,
                      n_real);
    dump_device_array("dump/cuMES/constraint_brmn_o_iter_1.bin", rs.d_brmn_o,
                      n_real);
    dump_device_array("dump/cuMES/constraint_bzmn_e_iter_1.bin", rs.d_bzmn_e,
                      n_real);
    dump_device_array("dump/cuMES/constraint_bzmn_o_iter_1.bin", rs.d_bzmn_o,
                      n_real);
    // Constraint-chain intermediates (stage-by-stage vs vmecpp)
    // State as consumed by the constraint chain at iter 1 (post-descent)
    size_t n_spec2 = (size_t)p.ns * (size_t)p.mnmax;
    dump_device_array("dump/cuMES/constraint_rmncc_iter_1.bin",
                      storage.family_ptr(SpectralComponent::Rcc), n_spec2);
    dump_device_array("dump/cuMES/constraint_rmnss_iter_1.bin",
                      storage.family_ptr(SpectralComponent::Rss), n_spec2);
    dump_device_array("dump/cuMES/constraint_zmnsc_iter_1.bin",
                      storage.family_ptr(SpectralComponent::Zsc), n_spec2);
    dump_device_array("dump/cuMES/constraint_zmncs_iter_1.bin",
                      storage.family_ptr(SpectralComponent::Zcs), n_spec2);
    dump_device_array("dump/cuMES/constraint_rCon_iter_1.bin",
                      constraint.rcon_view(p).data(), n_real);
    dump_device_array("dump/cuMES/constraint_zCon_iter_1.bin",
                      constraint.zcon_view(p).data(), n_real);
    dump_device_array("dump/cuMES/constraint_gConEff_iter_1.bin",
                      constraint.gcon_eff(), n_real);
    dump_device_array("dump/cuMES/constraint_gCon_iter_1.bin",
                      constraint.gcon(), n_real);
    dump_device_array("dump/cuMES/constraint_frcon_e_iter_1.bin",
                      constraint.constraint_force_views(p).frcon_e.data(),
                      n_real);
    dump_device_array("dump/cuMES/constraint_frcon_o_iter_1.bin",
                      constraint.constraint_force_views(p).frcon_o.data(),
                      n_real);
    dump_device_array("dump/cuMES/constraint_fzcon_e_iter_1.bin",
                      constraint.constraint_force_views(p).fzcon_e.data(),
                      n_real);
    dump_device_array("dump/cuMES/constraint_fzcon_o_iter_1.bin",
                      constraint.constraint_force_views(p).fzcon_o.data(),
                      n_real);
    // tcon/faccon profiles (device arrays; h_tcon is stale -- the
    // kernel writes d_tcon directly)
    dump_device_array("dump/cuMES/constraint_tcon_iter_1.bin",
                      constraint.tcon(), p.ns);
    dump_device_array("dump/cuMES/constraint_faccon_iter_1.bin",
                      constraint.faccon(), p.mpol);
}

// scaled window (post-decomposition-scaling): decomposed force slab + the
// E2-start invariant-force window.
template <typename T>
void dump_step_h(int iter,
                 int iter2,
                 const DeviceParams<T>& p,
                 const T* f_spec) {
    if (!dump_enabled()) return;
    const DumpKnobs kn = dump_knobs();
    if (iter == 0 || iter2 == kn.dump_iter) {
        // Dump AFTER the decomposition scaling, matching vmecpp's dump
        // of m_decomposed_f (post-decomposeInto). Keyed on iter2 so a
        // handoff/plateau comparison can use the same effective counter
        // as vmecpp's dump blocks.
        size_t n_fspec = (size_t)6 * (size_t)p.mnmax * (size_t)p.ns;
        char fn[128];
        snprintf(fn, sizeof fn, "dump/cuMES/scaled_f_spec_iter_%d.bin",
                 iter == 0 ? 1 : iter2);
        dump_device_array(fn, f_spec, n_fspec);
    }
    if (iter2 >= kn.e2_start && iter2 < kn.e2_start + 40) {
        char fn[128];
        snprintf(fn, sizeof fn, "dump/cuMES/fspec_invariant_iter_%d.bin",
                 iter2);
        dump_device_array(fn, f_spec, (size_t)6 * p.mnmax * p.ns);
    }
}

// final window (post-m1-gauge, pre-invariant-residual): final-pass force-slab
// dump.
template <typename T>
void dump_step_final(int iter, const DeviceParams<T>& p, const T* f_spec) {
    if (!dump_enabled()) return;
    if (iter == dump_max_iter(p.max_iter) - 1) {
        dump_device_array("dump/cuMES/final_f_spec.bin", f_spec,
                          (size_t)6 * p.mnmax * p.ns);
    }
}

// preconditioned window (post-preconditioner): preconditioned slab +
// state/velocity handoff dumps + the E2-start preconditioned-force window.
template <typename T>
void dump_step_i(int iter,
                 int iter2,
                 const DeviceParams<T>& p,
                 const T* f_spec,
                 SpectralStorage<T>& storage) {
    if (!dump_enabled()) return;
    const DumpKnobs kn = dump_knobs();
    if (iter == 0 || iter2 == 51 ||
        (iter2 >= kn.dump_iter && iter2 <= kn.dump_iter + 2) ||
        (iter2 >= 2 && iter2 <= 4)) {
        size_t n_fspec = (size_t)6 * (size_t)p.mnmax * (size_t)p.ns;
        size_t n_spec = (size_t)p.mnmax * (size_t)p.ns;
        char fn[128];
        snprintf(fn, sizeof fn, "dump/cuMES/preconditioned_f_spec_iter_%d.bin",
                 iter == 0 ? 1 : iter2);
        dump_device_array(fn, f_spec, n_fspec);
        // State + velocities at the handoff window (pre-descent of the
        // pass, matching vmecpp's dump phase at vmec.cc). Also at the
        // iter-2..4 window (first lambda != 0 passes) for the state check.
        if (iter2 >= kn.dump_iter || iter2 == 51 ||
            (iter2 >= 2 && iter2 <= 4)) {
            snprintf(fn, sizeof fn, "dump/cuMES/state_rmncc_iter_%d.bin",
                     iter2);
            dump_device_array(fn, storage.family_ptr(SpectralComponent::Rcc),
                              n_spec);
            snprintf(fn, sizeof fn, "dump/cuMES/state_zmnsc_iter_%d.bin",
                     iter2);
            dump_device_array(fn, storage.family_ptr(SpectralComponent::Zsc),
                              n_spec);
            snprintf(fn, sizeof fn, "dump/cuMES/state_lmnsc_iter_%d.bin",
                     iter2);
            dump_device_array(fn, storage.family_ptr(SpectralComponent::Lsc),
                              n_spec);
            snprintf(fn, sizeof fn, "dump/cuMES/state_rmnss_iter_%d.bin",
                     iter2);
            dump_device_array(fn, storage.family_ptr(SpectralComponent::Rss),
                              n_spec);
            snprintf(fn, sizeof fn, "dump/cuMES/state_zmncs_iter_%d.bin",
                     iter2);
            dump_device_array(fn, storage.family_ptr(SpectralComponent::Zcs),
                              n_spec);
            snprintf(fn, sizeof fn, "dump/cuMES/state_lmncs_iter_%d.bin",
                     iter2);
            dump_device_array(fn, storage.family_ptr(SpectralComponent::Lcs),
                              n_spec);
            snprintf(fn, sizeof fn, "dump/cuMES/vel_vrmncc_iter_%d.bin", iter2);
            dump_device_array(
                fn, storage.velocity_family_ptr(SpectralComponent::Rcc),
                n_spec);
            snprintf(fn, sizeof fn, "dump/cuMES/vel_vzmnsc_iter_%d.bin", iter2);
            dump_device_array(
                fn, storage.velocity_family_ptr(SpectralComponent::Zsc),
                n_spec);
            snprintf(fn, sizeof fn, "dump/cuMES/vel_vlmnsc_iter_%d.bin", iter2);
            dump_device_array(
                fn, storage.velocity_family_ptr(SpectralComponent::Lsc),
                n_spec);
            snprintf(fn, sizeof fn, "dump/cuMES/vel_vrmnss_iter_%d.bin", iter2);
            dump_device_array(
                fn, storage.velocity_family_ptr(SpectralComponent::Rss),
                n_spec);
            snprintf(fn, sizeof fn, "dump/cuMES/vel_vzmncs_iter_%d.bin", iter2);
            dump_device_array(
                fn, storage.velocity_family_ptr(SpectralComponent::Zcs),
                n_spec);
            snprintf(fn, sizeof fn, "dump/cuMES/vel_vlmncs_iter_%d.bin", iter2);
            dump_device_array(
                fn, storage.velocity_family_ptr(SpectralComponent::Lcs),
                n_spec);
        }
    }
    if (iter2 >= kn.e2_start && iter2 < kn.e2_start + 40) {
        char fn[128];
        snprintf(fn, sizeof fn, "dump/cuMES/fspec_precon_iter_%d.bin", iter2);
        dump_device_array(fn, f_spec, (size_t)6 * p.mnmax * p.ns);
    }
}
#endif  // DUMP_CUMES_VERIFY

template <typename T>
void dump_inverse_diag(ToroidalFftOperator<T>& transform,
                       SpectralStorage<T>& storage,
                       RealSpaceStorage<T>& rs,
                       const DeviceParams<T>& p,
                       cudaStream_t stream) {
#ifdef DUMP_CUMES_VERIFY
    // Diagnostic: test inverse DFT at specified surface (CUMES_DUMP=1 only).
    // do_combine=false: the diagnostic reads only the parity arrays; the
    // combined *_real buffers are materialized on demand (fourierCombineParity)
    // at the dump site, never read stale.
    if (!dump_enabled()) return;
    transform.enqueue_inverse_dump(storage.physical_const(), stream);
    cudaDeviceSynchronize();  // dump-only read of compute-stream data
    auto* h_re = new T[p.nZnT * p.ns];
    auto* h_ro = new T[p.nZnT * p.ns];
    check_cuda(cudaMemcpy(h_re, rs.d_r_e, p.nZnT * p.ns * sizeof(T),
                          cudaMemcpyDeviceToHost),
               "diag re");
    check_cuda(cudaMemcpy(h_ro, rs.d_r_o, p.nZnT * p.ns * sizeof(T),
                          cudaMemcpyDeviceToHost),
               "diag ro");
    // Check surface j=ns-1 (LCFS): r_e should be rbc[0]*cos(0)=3.999, r_o
    // should be sum of odd m
    int j_b = p.ns - 1;
    double re_lcfs = h_re[0 + j_b * p.nZnT];  // theta=0
    double ro_lcfs = h_ro[0 + j_b * p.nZnT];
    printf(
        "  [diag] LCFS theta=0: r_e=%.4f r_o=%.4f r_total=%.4f (expect "
        "~3.93 + ~1.03 = ~4.96)\n",
        re_lcfs, ro_lcfs, re_lcfs + ro_lcfs);
    delete[] h_re;
    delete[] h_ro;
#else
    (void)transform;
    (void)storage;
    (void)rs;
    (void)p;
    (void)stream;
#endif
}

template <typename T>
void dump_step_0(const DeviceParams<T>& p,
                 SpectralStorage<T>& storage,
                 const Profiles<T>& profiles) {
#ifdef DUMP_CUMES_VERIFY
    const RadialProfileViews<T> rpv = profiles.profile_views();
    dump_ensure_dir();
    size_t n_spec = (size_t)p.ns * (size_t)p.mnmax;
    dump_device_array("dump/cuMES/init_rmncc.bin",
                      storage.family_ptr(SpectralComponent::Rcc), n_spec);
    dump_device_array("dump/cuMES/init_zmnsc.bin",
                      storage.family_ptr(SpectralComponent::Zsc), n_spec);
    dump_device_array("dump/cuMES/init_lmnsc.bin",
                      storage.family_ptr(SpectralComponent::Lsc), n_spec);
    dump_device_array("dump/cuMES/init_rmnss.bin",
                      storage.family_ptr(SpectralComponent::Rss), n_spec);
    dump_device_array("dump/cuMES/init_zmncs.bin",
                      storage.family_ptr(SpectralComponent::Zcs), n_spec);
    dump_device_array("dump/cuMES/init_lmncs.bin",
                      storage.family_ptr(SpectralComponent::Lcs), n_spec);
    dump_device_array("dump/cuMES/init_currH.bin", rpv.curr_H, p.ns - 1);
    dump_device_array("dump/cuMES/init_chipH.bin", rpv.chip_H, p.ns - 1);
    dump_device_array("dump/cuMES/init_iotaH.bin", rpv.iota_H, p.ns - 1);
    dump_device_array("dump/cuMES/init_iotaF.bin", rpv.iota_F, p.ns);
#else
    (void)p;
    (void)storage;
    (void)profiles;
#endif
}

inline void dump_check_jacobian_consistency(bool device_valid,
                                            bool host_invalid,
                                            int iter2) {
#ifdef DUMP_CUMES_VERIFY
    // Device/host rule consistency (dump-only observability): the finalize
    // kernel and the controller must decide identically, or the guards
    // would suppress work the controller later consumes.
    if (dump_enabled() && (device_valid == host_invalid)) {
        fprintf(stderr,
                "cuMES: WARNING: device jacobian status (%s) disagrees "
                "with the host gate (%s) at pass %d\n",
                device_valid ? "valid" : "invalid",
                host_invalid ? "invalid" : "valid", iter2);
    }
#else
    (void)device_valid;
    (void)host_invalid;
    (void)iter2;
#endif
}

inline void dump_check_predicate_consistency(int dev_nf,
                                             int dev_cv,
                                             int prec_eval,
                                             bool verdict_nf,
                                             bool verdict_cv,
                                             int iter2) {
#ifdef DUMP_CUMES_VERIFY
    // Device/host terminal-predicate consistency (dump-only): the device
    // bits and the host verdict must now agree on EVERY pass — on refresh
    // passes both consume the record's device-finalized factors
    // (completion-plan follow-up §2.3), on other passes the host's cached
    // factors travel by value into the predicate.
    if (dump_enabled()) {
        const bool nf_ok = (dev_nf != 0) == verdict_nf;
        const bool cv_ok = (dev_cv != 0) == verdict_cv;
        const bool ev_ok = verdict_nf || verdict_cv || (prec_eval != 0);
        if (!nf_ok || !cv_ok || !ev_ok) {
            fprintf(stderr,
                    "cuMES: WARNING: device predicates (nf=%d cv=%d "
                    "prec_eval=%d) disagree with host (%d %d) at pass %d\n",
                    (int)dev_nf, (int)dev_cv, (int)prec_eval, (int)verdict_nf,
                    (int)verdict_cv, iter2);
        }
    }
#else
    (void)dev_nf;
    (void)dev_cv;
    (void)prec_eval;
    (void)verdict_nf;
    (void)verdict_cv;
    (void)iter2;
#endif
}

inline void dump_event_bad_jacobian(double min_oriented,
                                    double max_abs,
                                    double nonfinite,
                                    int j_half,
                                    double delt) {
#ifdef DUMP_CUMES_VERIFY
    // Event line compiled out of release (fast) builds along with the rest of
    // the dump machinery; the iteration table still gets a row.
    printf(
        "  -> BAD JACOBIAN (invalid √g: min(signJ·√g)=%.3e "
        "max|√g|=%.3e nonfinite=%.0f at jH=%d) delt=%.3e\n",
        min_oriented, max_abs, nonfinite, j_half, delt);
#else
    (void)min_oriented;
    (void)max_abs;
    (void)nonfinite;
    (void)j_half;
    (void)delt;
#endif
}

inline void dump_event_nonfinite(double delt) {
#ifdef DUMP_CUMES_VERIFY
    // Event line compiled out of release (fast) builds (see above).
    printf("  -> BAD JACOBIAN (non-finite residuals) delt=%.3e\n", delt);
#else
    (void)delt;
#endif
}

inline void dump_event_converged(int iter2) {
#ifdef DUMP_CUMES_VERIFY
    // Event line compiled out of release (fast) builds (see above).
    printf("  -> CONVERGED at iter %d\n", iter2);
#else
    (void)iter2;
#endif
}

inline void dump_event_restart(bool bad_jacobian, int iter2, double delt) {
#ifdef DUMP_CUMES_VERIFY
    // The restart event lines (BAD JACOBIAN / BAD PROGRESS) are compiled out
    // of release (fast) builds with the rest of the dump machinery (see the
    // other event-line gates above).
    printf("  -> %s (iter2=%d) delt=%.3e\n",
           bad_jacobian ? "BAD JACOBIAN" : "BAD PROGRESS", iter2, delt);
#else
    (void)bad_jacobian;
    (void)iter2;
    (void)delt;
#endif
}

template <typename T>
PerIterRecorder<T>::PerIterRecorder(const DeviceParams<T>& p,
                                    SpectralStorage<T>& storage,
                                    cudaStream_t stream,
                                    int max_iter_eff)
    : p_(p), storage_(storage), stream_(stream), max_iter_eff_(max_iter_eff) {
#ifdef DUMP_CUMES_VERIFY
    per_iter_.reserve((size_t)max_iter_eff_);
#endif
}

template <typename T>
double PerIterRecorder<T>::axis_r_at_zeta0_sync() const {
#ifdef DUMP_CUMES_VERIFY
    // The dump record (per_iter_residuals_cumes.bin) must stay byte-identical
    // to the frozen baseline, whose axis_r column was read POST-descent with
    // a synchronized copy. Keep that exact read for the dump-only record
    // (dump mode already performs device-wide dumps; the extra sync costs
    // nothing there and is compiled out of production).
    cudaStreamSynchronize(stream_);
    std::vector<T> h_ax(static_cast<std::size_t>(p_.ntor) + 1);
    check_cuda(cudaMemcpy2D(h_ax.data(), sizeof(T),
                            storage_.family_ptr(SpectralComponent::Rcc),
                            (size_t)p_.ns * sizeof(T), sizeof(T), p_.ntor + 1,
                            cudaMemcpyDeviceToHost),
               "cpy Rax");
    T h = T(0.0);
    for (int n = 0; n <= p_.ntor; ++n) h += h_ax[n];
    return h;
#else
    return 0.0;
#endif
}

template <typename T>
void PerIterRecorder<T>::record(int reason,
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
                                int it1) {
#ifndef DUMP_CUMES_VERIFY
    (void)reason;
    (void)f_ri;
    (void)f_zi;
    (void)f_li;
    (void)f_r;
    (void)f_z;
    (void)f_l;
    (void)d;
    (void)o;
    (void)dt;
    (void)b1v;
    (void)fcv;
    (void)it2;
    (void)it1;
#else
    if (!dump_enabled()) return;
    if ((int)per_iter_.size() >= max_iter_eff_) return;
    PassRecord r;
    r.invariant_fsqr = f_ri;
    r.invariant_fsqz = f_zi;
    r.invariant_fsql = f_li;
    r.preconditioned_fsqr = f_r;
    r.preconditioned_fsqz = f_z;
    r.preconditioned_fsql = f_l;
    r.delta_t = d;
    r.otav = o;
    r.dtau = dt;
    r.b1 = b1v;
    r.fac = fcv;
    r.iter2 = (double)it2;
    r.iter1 = (double)it1;
    r.reason = (double)reason;
    r.axis_r = (double)axis_r_at_zeta0_sync();
    per_iter_.push_back(r);
#endif
}

template <typename T>
void PerIterRecorder<T>::write_file() const {
#ifdef DUMP_CUMES_VERIFY
    if (!dump_enabled()) return;
    // Per-pass record, column-major: 15 blocks of per_iter_.size() doubles
    // (byte-identical to the legacy array-of-doubles layout).
    FILE* fpr = fopen("dump/cuMES/per_iter_residuals_cumes.bin", "wb");
    if (fpr) {
        uint64_t n = (uint64_t)per_iter_.size();
        fwrite(&n, sizeof(uint64_t), 1, fpr);
        for (int c = 0; c < PassRecord::COLUMN_COUNT; ++c) {
            for (const PassRecord& r : per_iter_) {
                const double* base = &r.invariant_fsqr;
                fwrite(base + c, sizeof(double), 1, fpr);
            }
        }
        fclose(fpr);
    }
#endif
}

}  // namespace cumes

#endif  // CUMES_SRC_DUMP_IMPL_CUH_
