// test_force_verify.cu — converged equilibrium must sit near force balance.
//
// Runs the solver to convergence on the Solovev fixture (self-contained — no
// external vmecpp_init.bin dependency), then recomputes the forces through an
// INDEPENDENT CPU path and checks the residual is small. The gate has two
// layers:
//
//   (a) force-formula agreement: a scalar host port of the force kernel
//       (cpu_forces below, mirrors forces_kernel's weak form point-by-point) is
//       run on the solver-converged real-space state and compared against the
//       production ForceOperator output — max |diff| must be below 1e-4;
//   (b) force balance: the CPU forces are projected to spectral space by an
//       independent host implementation of the production forward path
//       (cpu_forward_project below — reduced-theta trapezoid + unnormalized D2Z
//       + mscale*nscale recovery) and the fsqr/fsqz/fsql norms must be below
//       1e-4 for the converged state.
//
// A broken force formula that still permits convergence shows up as O(1)
// disagreement in (a) and O(1) residuals in (b) — the pre-fix gate recomputed
// with the SAME production operators the solver just used and was
// tautological. A negative control corrupts the converged state (Zsc × 1e3),
// asserts the gate FIRES (O(1) residuals), then restores the state and shows
// the gate passes again — a gate that cannot fire is a tautology in another
// disguise. This is the registerable, fixture-free replacement for the old
// test that loaded an absent vmecpp_init.bin.
#include "cumes/physics/force_operator.hpp"
#include "cumes/physics/geometry_operator.hpp"
#include "cumes/physics/magnetic_field_operator.hpp"
#include "cumes/physics/profiles.hpp"
#include "cumes/runtime/device_buffer.cuh"
#include "cumes/state/spectral_storage.hpp"
#include "cumes/transforms/toroidal_fft_operator.hpp"
#include "cumes_test_cuda_helper.cuh"
#include "solver.cuh"
#include "vmec_types.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>
using namespace cumes::test;

// ---------------------------------------------------------------------------
// Independent CPU force reference (port of test_force_reference.cu's scalar
// double reference, which mirrors forces_kernel in kernels/forces_impl.cuh
// exactly).
//
// `r_e..lu_o` are the full-grid parity geometry, the half-grid metric/field
// arrays are (ns-1, nZnT), profiles are sqrtS_F/sqrtS_H/phip_F, and the 16
// force outputs are written to out (parity-split). The comparison is NOT
// bit-exact: the GPU compiles with -use_fast_math, so the double leg compares
// at 1e-4 (far above the ~1e-11 FMA-fusion difference of the ~1e0 force
// scale).
template <typename T>
static void cpu_forces(const std::vector<T>& r_e,
                       const std::vector<T>& r_o,
                       const std::vector<T>& z_o,
                       const std::vector<T>& ru_e,
                       const std::vector<T>& ru_o,
                       const std::vector<T>& zu_e,
                       const std::vector<T>& zu_o,
                       const std::vector<T>& rv_e,
                       const std::vector<T>& rv_o,
                       const std::vector<T>& zv_e,
                       const std::vector<T>& zv_o,
                       const std::vector<T>& lu_e,
                       const std::vector<T>& lu_o,
                       const std::vector<T>& r12,
                       const std::vector<T>& ru12,
                       const std::vector<T>& zu12,
                       const std::vector<T>& rs,
                       const std::vector<T>& zs,
                       const std::vector<T>& tau,
                       const std::vector<T>& gsqrt,
                       const std::vector<T>& guv,
                       const std::vector<T>& gvv,
                       const std::vector<T>& bsupu,
                       const std::vector<T>& bsupv,
                       const std::vector<T>& bsubu,
                       const std::vector<T>& bsubv,
                       const std::vector<T>& totalP,
                       const std::vector<T>& sqrtS_F,
                       const std::vector<T>& sqrtS_H,
                       const std::vector<T>& phip_F,
                       int ns,
                       int nZnT,
                       T lamscale,
                       T delta_s,
                       std::vector<T>& armn_e,
                       std::vector<T>& armn_o,
                       std::vector<T>& azmn_e,
                       std::vector<T>& azmn_o,
                       std::vector<T>& brmn_e,
                       std::vector<T>& brmn_o,
                       std::vector<T>& bzmn_e,
                       std::vector<T>& bzmn_o,
                       std::vector<T>& crmn_e,
                       std::vector<T>& crmn_o,
                       std::vector<T>& czmn_e,
                       std::vector<T>& czmn_o,
                       std::vector<T>& blmn_e,
                       std::vector<T>& blmn_o,
                       std::vector<T>& clmn_e,
                       std::vector<T>& clmn_o) {
    for (int j = 0; j < ns; ++j) {
        T sF_j = sqrtS_F[j];
        T sFull = sF_j * sF_j;
        for (int k = 0; k < nZnT; ++k) {
            const int idx_f = k + j * nZnT;
            T re_j = r_e[idx_f], ro_j = r_o[idx_f], zo_j = z_o[idx_f];
            T rue_j = ru_e[idx_f], ruo_j = ru_o[idx_f];
            T zue_j = zu_e[idx_f], zuo_j = zu_o[idx_f];
            T rve_j = rv_e[idx_f], rvo_j = rv_o[idx_f];
            T zve_j = zv_e[idx_f], zvo_j = zv_o[idx_f];

            T r12_i = T(0), ru12_i = T(0), zu12_i = T(0), rs_i = T(0),
              zs_i = T(0), tau_i = T(0);
            T gsqrt_i = T(0), bsupu_i = T(0), bsupv_i = T(0);
            T bsubu_i = T(0), bsubv_i = T(0), totalP_i = T(0);
            T sH_i = T(0), gvv_i = T(0), guv_i = T(0);
            T r12_o = T(0), ru12_o = T(0), zu12_o = T(0), rs_o = T(0),
              zs_o = T(0), tau_o = T(0);
            T gsqrt_o = T(0), bsupu_o = T(0), bsupv_o = T(0);
            T bsubu_o = T(0), bsubv_o = T(0), totalP_o = T(0);
            T sH_o = T(0), gvv_o = T(0), guv_o = T(0);

            if (j > 0) {
                int h_i = k + (j - 1) * nZnT;
                r12_i = r12[h_i];
                ru12_i = ru12[h_i];
                zu12_i = zu12[h_i];
                rs_i = rs[h_i];
                zs_i = zs[h_i];
                tau_i = tau[h_i];
                gsqrt_i = gsqrt[h_i];
                gvv_i = gvv[h_i];
                guv_i = guv[h_i];
                bsupu_i = bsupu[h_i];
                bsupv_i = bsupv[h_i];
                bsubu_i = bsubu[h_i];
                bsubv_i = bsubv[h_i];
                totalP_i = totalP[h_i];
                sH_i = sqrtS_H[j - 1];
            }
            if (j < ns - 1) {
                int h_o = k + j * nZnT;
                r12_o = r12[h_o];
                ru12_o = ru12[h_o];
                zu12_o = zu12[h_o];
                rs_o = rs[h_o];
                zs_o = zs[h_o];
                tau_o = tau[h_o];
                gsqrt_o = gsqrt[h_o];
                gvv_o = gvv[h_o];
                guv_o = guv[h_o];
                bsupu_o = bsupu[h_o];
                bsupv_o = bsupv[h_o];
                bsubu_o = bsubu[h_o];
                bsubv_o = bsubv[h_o];
                totalP_o = totalP[h_o];
                sH_o = sqrtS_H[j];
            }

            T P_i = r12_i * totalP_i, P_o = r12_o * totalP_o;
            T zup_i = zu12_i * P_i, zup_o = zu12_o * P_o;
            T rup_i = ru12_i * P_i, rup_o = ru12_o * P_o;
            T rsp_i = rs_i * P_i, rsp_o = rs_o * P_o;
            T zsp_i = zs_i * P_i, zsp_o = zs_o * P_o;
            T taup_i = tau_i * totalP_i, taup_o = tau_o * totalP_o;

            T gbubu_i = gsqrt_i * bsupu_i * bsupu_i;
            T gbubu_o = gsqrt_o * bsupu_o * bsupu_o;
            T gbvbv_i = gsqrt_i * bsupv_i * bsupv_i;
            T gbvbv_o = gsqrt_o * bsupv_o * bsupv_o;
            T gbubv_i = gsqrt_i * bsupu_i * bsupv_i;
            T gbubv_o = gsqrt_o * bsupu_o * bsupv_o;

            T inv_ds = T(1.0) / delta_s;
            T inv_sH_i = (j > 0) ? (T(1.0) / sH_i) : T(0.0);
            T inv_sH_o = (j < ns - 1) ? (T(1.0) / sH_o) : T(0.0);

            T P_avg = T(0.5) * (P_o + P_i);
            T P_wavg = T(0.5) * (P_o * inv_sH_o + P_i * inv_sH_i);
            T gbubu_avg = T(0.5) * (gbubu_o + gbubu_i);
            T gbubu_wavg = T(0.5) * (gbubu_o * sH_o + gbubu_i * sH_i);
            T gbvbv_avg = T(0.5) * (gbvbv_o + gbvbv_i);
            T gbvbv_wavg = T(0.5) * (gbvbv_o * sH_o + gbvbv_i * sH_i);
            T gbubv_avg = T(0.5) * (gbubv_o + gbubv_i);
            T gbubv_wavg = T(0.5) * (gbubv_o * sH_o + gbubv_i * sH_i);

            T armn_e_v = (zup_o - zup_i) * inv_ds + T(0.5) * (taup_o + taup_i) -
                         gbvbv_avg * re_j - gbvbv_wavg * ro_j;
            T armn_o_v = (zup_o * sH_o - zup_i * sH_i) * inv_ds -
                         T(0.5) * P_wavg * zue_j - T(0.5) * P_avg * zuo_j +
                         T(0.5) * (taup_o * sH_o + taup_i * sH_i) -
                         gbvbv_wavg * re_j - gbvbv_avg * ro_j * sFull;
            T azmn_e_v = -(rup_o - rup_i) * inv_ds;
            T azmn_o_v = -(rup_o * sH_o - rup_i * sH_i) * inv_ds +
                         T(0.5) * P_wavg * rue_j + T(0.5) * P_avg * ruo_j;
            T brmn_e_v = T(0.5) * (zsp_o + zsp_i) + T(0.5) * P_wavg * zo_j -
                         gbubu_avg * rue_j - gbubu_wavg * ruo_j -
                         gbubv_avg * rve_j - gbubv_wavg * rvo_j;
            T brmn_o_v = T(0.5) * (zsp_o * sH_o + zsp_i * sH_i) +
                         T(0.5) * P_avg * zo_j - gbubu_wavg * rue_j -
                         gbubu_avg * ruo_j * sFull - gbubv_wavg * rve_j -
                         gbubv_avg * rvo_j * sFull;
            T bzmn_e_v = -T(0.5) * (rsp_o + rsp_i) - T(0.5) * P_wavg * ro_j -
                         gbubu_avg * zue_j - gbubu_wavg * zuo_j -
                         gbubv_avg * zve_j - gbubv_wavg * zvo_j;
            T bzmn_o_v = -T(0.5) * (rsp_o * sH_o + rsp_i * sH_i) -
                         T(0.5) * P_avg * ro_j - gbubu_wavg * zue_j -
                         gbubu_avg * zuo_j * sFull - gbubv_wavg * zve_j -
                         gbubv_avg * zvo_j * sFull;
            T crmn_e_v = gbubv_avg * rue_j + gbubv_wavg * ruo_j +
                         gbvbv_avg * rve_j + gbvbv_wavg * rvo_j;
            T crmn_o_v = gbubv_wavg * rue_j + gbubv_avg * ruo_j * sFull +
                         gbvbv_wavg * rve_j + gbvbv_avg * rvo_j * sFull;
            T czmn_e_v = gbubv_avg * zue_j + gbubv_wavg * zuo_j +
                         gbvbv_avg * zve_j + gbvbv_wavg * zvo_j;
            T czmn_o_v = gbubv_wavg * zue_j + gbubv_avg * zuo_j * sFull +
                         gbvbv_wavg * zve_j + gbvbv_avg * zvo_j * sFull;

            T bsubv_avg = T(0.5) * (bsubv_o + bsubv_i);
            T gvv_gsqrt_i = (j > 0) ? (gvv_i / gsqrt_i) : T(0.0);
            T gvv_gsqrt_o = (j < ns - 1) ? (gvv_o / gsqrt_o) : T(0.0);
            T guv_bsupu_i = (j > 0) ? (guv_i * bsupu_i) : T(0.0);
            T guv_bsupu_o = (j < ns - 1) ? (guv_o * bsupu_o) : T(0.0);
            T lu_e_norm = lamscale * lu_e[idx_f] + phip_F[j];
            T lu_o_norm = lamscale * lu_o[idx_f];
            T bsubv_alt =
                T(0.5) * (gvv_gsqrt_i + gvv_gsqrt_o) * lu_e_norm +
                T(0.5) * (gvv_gsqrt_i * sH_i + gvv_gsqrt_o * sH_o) * lu_o_norm +
                T(0.5) * (guv_bsupu_i + guv_bsupu_o);
            T rb = T(2.0) * T(0.05) * (T(1.0) - sFull);
            T _blmn = bsubv_avg * (T(1.0) - rb) + bsubv_alt * rb;
            if (j > 0) _blmn *= -lamscale;
            T blmn_e_v = _blmn;
            T blmn_o_v = _blmn * sF_j;

            T _clmn = T(0.5) * (bsubu_o + bsubu_i);
            if (j > 0) _clmn *= -lamscale;
            T clmn_e_v = _clmn;
            T clmn_o_v = _clmn * sF_j;

            armn_e[idx_f] = armn_e_v;
            armn_o[idx_f] = armn_o_v;
            azmn_e[idx_f] = azmn_e_v;
            azmn_o[idx_f] = azmn_o_v;
            brmn_e[idx_f] = brmn_e_v;
            brmn_o[idx_f] = brmn_o_v;
            bzmn_e[idx_f] = bzmn_e_v;
            bzmn_o[idx_f] = bzmn_o_v;
            crmn_e[idx_f] = crmn_e_v;
            crmn_o[idx_f] = crmn_o_v;
            czmn_e[idx_f] = czmn_e_v;
            czmn_o[idx_f] = czmn_o_v;
            blmn_e[idx_f] = blmn_e_v;
            blmn_o[idx_f] = blmn_o_v;
            clmn_e[idx_f] = clmn_e_v;
            clmn_o[idx_f] = clmn_o_v;
        }
    }
}

// ---------------------------------------------------------------------------
// Independent CPU forward projection: mirrors the production forward path
// (forward_reduce_kernel + the unnormalized real D2Z + forward_recover_kernel
// in kernels/fourier_impl.cuh) on the host. Consumes the 16 parity-split
// real-space force families (full grid, column-major idx = j*nZnT + k*ntheta +
// l) and writes the six spectral families in DecomposedResidualDomain order
// (Rcc, Zsc, Lsc, Rss, Zcs, Lcs; [c*ns*mnmax + mode*ns + j]) with the
// production normalization (reduced-theta trapezoid weight with endpoint
// halving, mscale*nscale recovery, axis/LCFS row rules). The constraint-force
// inputs (frcon/fzcon) are zero in this test — matching the zeroed buffers
// the production forward below is run with.
static void cpu_forward_project(const std::vector<double>& armn_e,
                                const std::vector<double>& armn_o,
                                const std::vector<double>& azmn_e,
                                const std::vector<double>& azmn_o,
                                const std::vector<double>& brmn_e,
                                const std::vector<double>& brmn_o,
                                const std::vector<double>& bzmn_e,
                                const std::vector<double>& bzmn_o,
                                const std::vector<double>& crmn_e,
                                const std::vector<double>& crmn_o,
                                const std::vector<double>& czmn_e,
                                const std::vector<double>& czmn_o,
                                const std::vector<double>& blmn_e,
                                const std::vector<double>& blmn_o,
                                const std::vector<double>& clmn_e,
                                const std::vector<double>& clmn_o,
                                int ns,
                                int mpol,
                                int ntor,
                                int nfp,
                                int ntheta,
                                int nzeta,
                                int nZnT,
                                std::vector<double>& spec) {
    const int mnmax = mpol * (ntor + 1);
    const int nThetaRed = ntheta / 2 + 1;
    const double intNorm = 1.0 / ((double)nzeta * (nThetaRed - 1));
    spec.assign(6 * (size_t)ns * mnmax, 0.0);
    for (int m = 0; m < mpol; ++m) {
        const bool mEven = (m % 2 == 0);
        const std::vector<double>& armn = mEven ? armn_e : armn_o;
        const std::vector<double>& azmn = mEven ? azmn_e : azmn_o;
        const std::vector<double>& brmn = mEven ? brmn_e : brmn_o;
        const std::vector<double>& bzmn = mEven ? bzmn_e : bzmn_o;
        const std::vector<double>& crmn = mEven ? crmn_e : crmn_o;
        const std::vector<double>& czmn = mEven ? czmn_e : czmn_o;
        const std::vector<double>& blmn = mEven ? blmn_e : blmn_o;
        const std::vector<double>& clmn = mEven ? clmn_e : clmn_o;
        for (int j = 0; j < ns; ++j) {
            // The 12 slot signals (forward_reduce_kernel's shuffle output).
            std::vector<double> vs[12];
            for (int s = 0; s < 12; ++s) vs[s].assign(nzeta, 0.0);
            for (int k = 0; k < nzeta; ++k) {
                double v0 = 0, v1 = 0, v2 = 0, v3 = 0, v4 = 0, v5 = 0;
                double v6 = 0, v7 = 0, v8 = 0, v9 = 0, v10 = 0, v11 = 0;
                for (int l = 0; l < nThetaRed; ++l) {
                    double w = intNorm;
                    if (l == 0 || l == nThetaRed - 1) w *= 0.5;
                    const double th = 2.0 * M_PI * l / ntheta;
                    const double cosm = w * cos(m * th), sinm = w * sin(m * th);
                    const double mcos = w * m * cos(m * th),
                                 msin = w * (-m) * sin(m * th);
                    const int idx = j * nZnT + k * ntheta + l;
                    const double tempR = armn[idx];  // frcon == 0 here
                    const double tempZ = azmn[idx];
                    const double br = brmn[idx], bz = bzmn[idx];
                    const double cr = crmn[idx], cz = czmn[idx];
                    const double bl = blmn[idx], cl = clmn[idx];
                    v0 += tempR * cosm + br * msin;
                    v1 += tempR * sinm + br * mcos;
                    v2 += -cr * cosm;
                    v3 += -cr * sinm;
                    v4 += tempZ * sinm + bz * mcos;
                    v5 += tempZ * cosm + bz * msin;
                    v6 += -cz * sinm;
                    v7 += -cz * cosm;
                    v8 += bl * mcos;
                    v9 += bl * msin;
                    v10 += -cl * sinm;
                    v11 += -cl * cosm;
                }
                vs[0][k] = v0;
                vs[1][k] = v1;
                vs[2][k] = v2;
                vs[3][k] = v3;
                vs[4][k] = v4;
                vs[5][k] = v5;
                vs[6][k] = v6;
                vs[7][k] = v7;
                vs[8][k] = v8;
                vs[9][k] = v9;
                vs[10][k] = v10;
                vs[11][k] = v11;
            }
            const double ms = (m == 0) ? 1.0 : std::sqrt(2.0);
            for (int n = 0; n <= ntor; ++n) {
                // Unnormalized real D2Z: F_s(n) = Σ_k v_s[k]·(cos - i·sin).
                double Fx[12] = {}, Fy[12] = {};
                for (int k = 0; k < nzeta; ++k) {
                    const double ang = 2.0 * M_PI * n * k / nzeta;
                    for (int s = 0; s < 12; ++s) {
                        Fx[s] += vs[s][k] * cos(ang);
                        Fy[s] += -vs[s][k] * sin(ang);
                    }
                }
                const double nsq = (n == 0) ? 1.0 : std::sqrt(2.0);
                const double mn = ms * nsq;
                const double nf = (double)(n * nfp);
                const int mode = m * (ntor + 1) + n;
                const size_t base = (size_t)mode * ns + j;
                const size_t stride = (size_t)ns * mnmax;
                if (j == 0) {
                    // axis: m=0 keeps frcc/fzcs; everything else stays zero
                    if (m == 0) {
                        spec[0 * stride + base] = mn * (Fx[0] + nf * Fy[2]);
                        spec[4 * stride + base] = mn * (-Fy[5] + nf * Fx[7]);
                    }
                } else if (j == ns - 1) {
                    // LCFS: lambda only (R/Z are fixed-boundary)
                    spec[2 * stride + base] = mn * (Fx[8] + nf * Fy[10]);
                    spec[5 * stride + base] = mn * (-Fy[9] + nf * Fx[11]);
                } else {
                    spec[0 * stride + base] = mn * (Fx[0] + nf * Fy[2]);
                    spec[1 * stride + base] = mn * (Fx[4] + nf * Fy[6]);
                    spec[2 * stride + base] = mn * (Fx[8] + nf * Fy[10]);
                    spec[3 * stride + base] = mn * (-Fy[1] + nf * Fx[3]);
                    spec[4 * stride + base] = mn * (-Fy[5] + nf * Fx[7]);
                    spec[5 * stride + base] = mn * (-Fy[9] + nf * Fx[11]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// One full recompute pass + the independent gate on the CURRENT device state.
// ---------------------------------------------------------------------------
struct ForceGate {
    double fsqr_prod, fsqz_prod, fsql_prod;  // production-forward residuals
    double fsqr_cpu, fsqz_cpu, fsql_cpu;     // independent CPU residuals
    double maxdiff;  // max |CPU - production| over the 16 force families
};

static ForceGate run_force_gate(cumes::SpectralStorage<double>& storage,
                                const DeviceParams<double>& p,
                                cumes::Profiles<double>& profiles,
                                const cumes::RadialProfileViews<double>& rp,
                                cumes::ToroidalFftOperator<double>& transform,
                                cumes::RealSpaceStorage<double>& rs,
                                cumes::GeometryOperator<double>& geometry) {
    ForceGate g{};
    const size_t nF = (size_t)p.ns * p.nZnT;
    const size_t nH = (size_t)(p.ns - 1) * p.nZnT;

    // ---- Production path (the operators the solver itself uses) ----
    transform.inverse(storage.physical_const(), /*do_combine=*/true);
    geometry.enqueue(rs, p, rp, 0);
    cumes::MagneticFieldOperator<double>{}.enqueue(
        rs, p, rp, geometry.base_geometry_views(p),
        geometry.magnetic_field_views(p), nullptr, 0, true);
    cumes::ForceOperator<double>{}.enqueue(
        rs, p, rp, geometry.base_geometry_views(p),
        geometry.magnetic_field_views(p), nullptr, 0);

    // Forward DFT to get the production spectral forces (6 families), with
    // zeroed constraint-force inputs (no constraint pass in this recompute).
    const size_t n6 = (size_t)6 * p.ns * p.mnmax;
    cumes::DeviceBuffer<double> d_f_spec(n6);
    cumes::DeviceBuffer<double> frcon_e(nF), frcon_o(nF);
    cumes::DeviceBuffer<double> fzcon_e(nF), fzcon_o(nF);
    frcon_e.zero();
    frcon_o.zero();
    fzcon_e.zero();
    fzcon_o.zero();
    transform.forward(
        cumes::SpectralView<double, cumes::DecomposedResidualDomain>(
            d_f_spec.data(), p.ns, p.mnmax),
        frcon_e.data(), frcon_o.data(), fzcon_e.data(), fzcon_o.data());

    std::vector<double> h_f(n6);
    cc(cudaMemcpy(h_f.data(), d_f_spec.data(), n6 * sizeof(double),
                  cudaMemcpyDeviceToHost),
       "cpy f");
    for (int c = 0; c < 6; ++c) {
        double sum = 0;
        for (size_t i = 0; i < (size_t)p.ns * p.mnmax; ++i)
            sum += h_f[c * (size_t)p.ns * p.mnmax + i] *
                   h_f[c * (size_t)p.ns * p.mnmax + i];
        sum /= (double)(p.ns * p.mnmax);
        if (c == 0 || c == 3)
            g.fsqr_prod += sum;
        else if (c == 1 || c == 4)
            g.fsqz_prod += sum;
        else if (c == 2 || c == 5)
            g.fsql_prod += sum;
    }

    // ---- Independent CPU recompute on the production real-space state ----
    auto get = [&](const double* d, size_t n) {
        std::vector<double> v(n);
        cc(cudaMemcpy(v.data(), d, n * sizeof(double), cudaMemcpyDeviceToHost),
           "get");
        return v;
    };
    // production force outputs (the comparison target)
    std::vector<double> g_armn_e = get(rs.d_armn_e, nF),
                        g_armn_o = get(rs.d_armn_o, nF);
    std::vector<double> g_azmn_e = get(rs.d_azmn_e, nF),
                        g_azmn_o = get(rs.d_azmn_o, nF);
    std::vector<double> g_brmn_e = get(rs.d_brmn_e, nF),
                        g_brmn_o = get(rs.d_brmn_o, nF);
    std::vector<double> g_bzmn_e = get(rs.d_bzmn_e, nF),
                        g_bzmn_o = get(rs.d_bzmn_o, nF);
    std::vector<double> g_crmn_e = get(rs.d_crmn_e, nF),
                        g_crmn_o = get(rs.d_crmn_o, nF);
    std::vector<double> g_czmn_e = get(rs.d_czmn_e, nF),
                        g_czmn_o = get(rs.d_czmn_o, nF);
    std::vector<double> g_blmn_e = get(rs.d_blmn_e, nF),
                        g_blmn_o = get(rs.d_blmn_o, nF);
    std::vector<double> g_clmn_e = get(rs.d_clmn_e, nF),
                        g_clmn_o = get(rs.d_clmn_o, nF);
    // full-grid parity geometry
    std::vector<double> r_e = get(rs.d_r_e, nF), r_o = get(rs.d_r_o, nF),
                        z_o = get(rs.d_z_o, nF);
    std::vector<double> ru_e = get(rs.d_ru_e, nF), ru_o = get(rs.d_ru_o, nF);
    std::vector<double> zu_e = get(rs.d_zu_e, nF), zu_o = get(rs.d_zu_o, nF);
    std::vector<double> rv_e = get(rs.d_rv_e, nF), rv_o = get(rs.d_rv_o, nF);
    std::vector<double> zv_e = get(rs.d_zv_e, nF), zv_o = get(rs.d_zv_o, nF);
    std::vector<double> lu_e = get(rs.d_lu_e, nF), lu_o = get(rs.d_lu_o, nF);
    // half-grid
    auto hg = geometry.base_geometry_views(p);
    std::vector<double> r12 = get(hg.r12.data(), nH),
                        ru12 = get(hg.ru12.data(), nH);
    std::vector<double> zu12 = get(hg.zu12.data(), nH),
                        rs_h = get(hg.rs.data(), nH);
    std::vector<double> zs_h = get(hg.zs.data(), nH),
                        tau = get(hg.tau.data(), nH);
    std::vector<double> gsqrt = get(hg.gsqrt.data(), nH),
                        guv = get(hg.guv.data(), nH);
    std::vector<double> gvv = get(hg.gvv.data(), nH);
    auto mf = geometry.magnetic_field_views(p);
    std::vector<double> bsupu = get(mf.bsupu.data(), nH),
                        bsupv = get(mf.bsupv.data(), nH);
    std::vector<double> bsubu = get(mf.bsubu.data(), nH),
                        bsubv = get(mf.bsubv.data(), nH);
    std::vector<double> totalP = get(mf.total_pressure.data(), nH);
    // profiles
    std::vector<double> sqrtS_F = get(rp.sqrtS_F, p.ns),
                        sqrtS_H = get(rp.sqrtS_H, p.ns - 1);
    std::vector<double> phip_F = get(rp.phip_F, p.ns);

    std::vector<double> c_armn_e(nF), c_armn_o(nF), c_azmn_e(nF), c_azmn_o(nF);
    std::vector<double> c_brmn_e(nF), c_brmn_o(nF), c_bzmn_e(nF), c_bzmn_o(nF);
    std::vector<double> c_crmn_e(nF), c_crmn_o(nF), c_czmn_e(nF), c_czmn_o(nF);
    std::vector<double> c_blmn_e(nF), c_blmn_o(nF), c_clmn_e(nF), c_clmn_o(nF);
    cpu_forces(r_e, r_o, z_o, ru_e, ru_o, zu_e, zu_o, rv_e, rv_o, zv_e, zv_o,
               lu_e, lu_o, r12, ru12, zu12, rs_h, zs_h, tau, gsqrt, guv, gvv,
               bsupu, bsupv, bsubu, bsubv, totalP, sqrtS_F, sqrtS_H, phip_F,
               p.ns, p.nZnT, p.lamscale, profiles.delta_s(), c_armn_e, c_armn_o,
               c_azmn_e, c_azmn_o, c_brmn_e, c_brmn_o, c_bzmn_e, c_bzmn_o,
               c_crmn_e, c_crmn_o, c_czmn_e, c_czmn_o, c_blmn_e, c_blmn_o,
               c_clmn_e, c_clmn_o);

    // (a) pointwise CPU-vs-production force agreement
    g.maxdiff = 0.0;
    g.maxdiff = std::max(g.maxdiff, max_diff(g_armn_e, c_armn_e));
    g.maxdiff = std::max(g.maxdiff, max_diff(g_armn_o, c_armn_o));
    g.maxdiff = std::max(g.maxdiff, max_diff(g_azmn_e, c_azmn_e));
    g.maxdiff = std::max(g.maxdiff, max_diff(g_azmn_o, c_azmn_o));
    g.maxdiff = std::max(g.maxdiff, max_diff(g_brmn_e, c_brmn_e));
    g.maxdiff = std::max(g.maxdiff, max_diff(g_brmn_o, c_brmn_o));
    g.maxdiff = std::max(g.maxdiff, max_diff(g_bzmn_e, c_bzmn_e));
    g.maxdiff = std::max(g.maxdiff, max_diff(g_bzmn_o, c_bzmn_o));
    g.maxdiff = std::max(g.maxdiff, max_diff(g_crmn_e, c_crmn_e));
    g.maxdiff = std::max(g.maxdiff, max_diff(g_crmn_o, c_crmn_o));
    g.maxdiff = std::max(g.maxdiff, max_diff(g_czmn_e, c_czmn_e));
    g.maxdiff = std::max(g.maxdiff, max_diff(g_czmn_o, c_czmn_o));
    g.maxdiff = std::max(g.maxdiff, max_diff(g_blmn_e, c_blmn_e));
    g.maxdiff = std::max(g.maxdiff, max_diff(g_blmn_o, c_blmn_o));
    g.maxdiff = std::max(g.maxdiff, max_diff(g_clmn_e, c_clmn_e));
    g.maxdiff = std::max(g.maxdiff, max_diff(g_clmn_o, c_clmn_o));

    // (b) independent CPU projection -> spectral residual norms
    std::vector<double> spec;
    cpu_forward_project(c_armn_e, c_armn_o, c_azmn_e, c_azmn_o, c_brmn_e,
                        c_brmn_o, c_bzmn_e, c_bzmn_o, c_crmn_e, c_crmn_o,
                        c_czmn_e, c_czmn_o, c_blmn_e, c_blmn_o, c_clmn_e,
                        c_clmn_o, p.ns, p.mpol, p.ntor, p.nfp, p.ntheta,
                        p.nzeta, p.nZnT, spec);
    for (int c = 0; c < 6; ++c) {
        double sum = 0;
        for (size_t i = 0; i < (size_t)p.ns * p.mnmax; ++i)
            sum += spec[c * (size_t)p.ns * p.mnmax + i] *
                   spec[c * (size_t)p.ns * p.mnmax + i];
        sum /= (double)(p.ns * p.mnmax);
        if (c == 0 || c == 3)
            g.fsqr_cpu += sum;
        else if (c == 1 || c == 4)
            g.fsqz_cpu += sum;
        else if (c == 2 || c == 5)
            g.fsql_cpu += sum;
    }
    return g;
}

int main() {
    // ---- Initial Solovev state (ns=55 = the Solovev final grid, ntor=0) ----
    const int ns = 55, mpol = 6, ntor = 0, ntheta = 18, nzeta = 1;
    DeviceParams<double> p;
    p.ns = ns;
    p.mnmax = mpol * (ntor + 1);
    p.ntheta = ntheta;
    p.nzeta = nzeta;
    p.nfp = 1;
    p.nZnT = ntheta * nzeta;
    p.mpol = mpol;
    p.ntor = ntor;
    p.ncurr = 0;
    p.delt = 0.9;
    p.ftol = 1e-14;
    p.max_iter = 2000;
    p.tcon0 = 1.0;
    p.lamscale = 0.0;

    // ---- Initial state from vmecpp interpFromBoundaryAndAxis (same logic as
    // main.cu initState): m=0 linear in s between axis and boundary, m>0 with
    // a s^(m/2) radial envelope. Uses the folded boundary from the JSON.
    cumes::ValidatedProblem vp = load_validated();
    const cumes::FoldedBoundary& bnd = vp.boundary();
    const cumes::ProblemSpec& spec = vp.spec();
    const int ntorp1 = p.ntor + 1;
    cumes::SpectralStorage<double> storage(ns, p.mnmax);
    size_t nb = (size_t)ns * p.mnmax * sizeof(double);
    std::vector<double> h_rmncc(ns * p.mnmax);
    std::vector<double> h_zmnsc(ns * p.mnmax);
    std::vector<double> h_lmnsc(ns * p.mnmax);
    std::vector<double> h_rmnss(ns * p.mnmax);
    std::vector<double> h_zmncs(ns * p.mnmax);
    std::vector<double> h_lmncs(ns * p.mnmax);
    for (int j = 0; j < ns; ++j) {
        double sFlux = (double)j / (ns - 1.0);
        double sqrtS = std::sqrt(sFlux);
        for (int m = 0; m < p.mpol; ++m) {
            for (int n = 0; n < p.ntor + 1; ++n) {
                int mn = m * (p.ntor + 1) + n;
                if (m == 0) {
                    h_rmncc[j + mn * ns] = sFlux * bnd.rbcc[0 * ntorp1 + n] +
                                           (1.0 - sFlux) * spec.raxis_c[n];
                    h_zmncs[j + mn * ns] = sFlux * bnd.zbcs[0 * ntorp1 + n] -
                                           (1.0 - sFlux) * spec.zaxis_s[n];
                } else if (m == 1) {
                    double w = sqrtS;
                    h_rmncc[j + mn * ns] = w * bnd.rbcc[m * ntorp1 + n];
                    h_rmnss[j + mn * ns] = w * bnd.rbss[m * ntorp1 + n];
                    h_zmnsc[j + mn * ns] = w * bnd.zbsc[m * ntorp1 + n];
                    h_zmncs[j + mn * ns] = w * bnd.zbcs[m * ntorp1 + n];
                } else {
                    double w = std::pow(sqrtS, m);
                    h_rmncc[j + mn * ns] = w * bnd.rbcc[m * ntorp1 + n];
                    h_rmnss[j + mn * ns] = w * bnd.rbss[m * ntorp1 + n];
                    h_zmnsc[j + mn * ns] = w * bnd.zbsc[m * ntorp1 + n];
                    h_zmncs[j + mn * ns] = w * bnd.zbcs[m * ntorp1 + n];
                }
            }
        }
    }

    cc(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Rcc),
                  h_rmncc.data(), nb, cudaMemcpyHostToDevice),
       "cpy rmncc");
    cc(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Zsc),
                  h_zmnsc.data(), nb, cudaMemcpyHostToDevice),
       "cpy zmnsc");
    cc(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Lsc),
                  h_lmnsc.data(), nb, cudaMemcpyHostToDevice),
       "cpy lmnsc");
    cc(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Rss),
                  h_rmnss.data(), nb, cudaMemcpyHostToDevice),
       "cpy rmnss");
    cc(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Zcs),
                  h_zmncs.data(), nb, cudaMemcpyHostToDevice),
       "cpy zmncs");
    cc(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Lcs),
                  h_lmncs.data(), nb, cudaMemcpyHostToDevice),
       "cpy lmncs");

    // ---- Profiles / plan / workspace ----
    cumes::Profiles<double> profiles(p, vp, std::nullopt);
    const cumes::RadialProfileViews<double> rp = profiles.profile_views();
    cumes::RealSpaceStorage<double> rs = real_space_create(p);
    cumes::DeviceModeTable mt = cumes::mode_table_create<double>(p);
    cumes::ToroidalFftOperator<double> transform(p, rs, mt, std::nullopt);
    cumes::GeometryOperator<double> geometry(p, std::nullopt);

    // ---- Converge: the solver drives the MHD residual to ftol ----
    SolverResult<double> res =
        solver_run(storage, p, profiles, transform, rs, geometry);
    std::cout << format(
        "solver: converged={} iterations={} fsqr={:.3e} fsqz={:.3e} "
        "fsql={:.3e}\n",
        int(res.converged), res.iterations, res.fsqr, res.fsqz, res.fsql);
    check(res.converged, "converged equilibrium reached");

    // ---- Capture the converged state (host) for the sensitivity control ----
    auto get_fam = [&](cumes::SpectralComponent c) {
        std::vector<double> v((size_t)ns * p.mnmax);
        cc(cudaMemcpy(v.data(), storage.family_ptr(c), nb,
                      cudaMemcpyDeviceToHost),
           "get family");
        return v;
    };
    std::vector<double> h_rmncc_c = get_fam(cumes::SpectralComponent::Rcc);
    std::vector<double> h_zmnsc_c = get_fam(cumes::SpectralComponent::Zsc);
    std::vector<double> h_lmnsc_c = get_fam(cumes::SpectralComponent::Lsc);
    std::vector<double> h_rmnss_c = get_fam(cumes::SpectralComponent::Rss);
    std::vector<double> h_zmncs_c = get_fam(cumes::SpectralComponent::Zcs);
    std::vector<double> h_lmncs_c = get_fam(cumes::SpectralComponent::Lcs);

    // The whole point: a converged equilibrium must sit near a force balance.
    // The balance CHECKs below are computed by the INDEPENDENT CPU path, so a
    // broken production force (or forward-transform) formula that still
    // permits convergence shows up as O(1) residuals / O(1) pointwise
    // disagreement instead of passing on the shared-kernel margin.
    const double FAIL_THRESH = 1e-4;
    const double AGREE_THRESH = 1e-4;

    // ---- Independent gate on the converged state ----
    ForceGate g0 =
        run_force_gate(storage, p, profiles, rp, transform, rs, geometry);
    std::cout << format(
        "Converged state — production forward residuals:   FSQR = {:.3e}  FSQZ "
        "= {:.3e}  FSQL = {:.3e}\n",
        g0.fsqr_prod, g0.fsqz_prod, g0.fsql_prod);
    std::cout << format(
        "Converged state — independent CPU residuals:      FSQR = {:.3e}  FSQZ "
        "= {:.3e}  FSQL = {:.3e}\n",
        g0.fsqr_cpu, g0.fsqz_cpu, g0.fsql_cpu);
    std::cout << format(
        "CPU-vs-production force agreement: max |diff| = {:.3e}\n", g0.maxdiff);
    check(g0.maxdiff <= AGREE_THRESH,
          "CPU force formula agrees with the production force path (1e-4)");
    check(g0.fsqr_cpu <= FAIL_THRESH,
          "FSQR small for converged equilibrium (independent CPU path)");
    check(g0.fsqz_cpu <= FAIL_THRESH,
          "FSQZ small for converged equilibrium (independent CPU path)");
    check(g0.fsql_cpu <= FAIL_THRESH,
          "FSQL small for converged equilibrium (independent CPU path)");

    // ---- Sensitivity control: the gate must actually fire. Corrupt one
    // spectral family of the converged state (Zsc x 1e3) — the corrupted
    // state is far from balance, so the independent gate must report O(1)
    // residuals — then restore the state and show the gate passes again. A
    // gate that cannot fire on a corrupted state would be a tautology in
    // another disguise.
    std::vector<double> corrupt = h_zmnsc_c;
    for (auto& v : corrupt) v *= 1e3;
    cc(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Zsc),
                  corrupt.data(), nb, cudaMemcpyHostToDevice),
       "corrupt zsc");
    ForceGate g1 =
        run_force_gate(storage, p, profiles, rp, transform, rs, geometry);
    std::cout << format(
        "CORRUPTED state (Zsc x 1e3) — CPU residuals:      FSQR = {:.3e}  FSQZ "
        "= {:.3e}  FSQL = {:.3e}  maxdiff = {:.3e}\n",
        g1.fsqr_cpu, g1.fsqz_cpu, g1.fsql_cpu, g1.maxdiff);
    check(std::max(g1.fsqr_cpu, g1.fsqz_cpu) > FAIL_THRESH,
          "corrupted state: the independent gate FIRES (O(1) residuals)");

    // Restore the full converged state (only Zsc was changed, but re-uploading
    // all six families is a cheap and unambiguous full restore).
    cc(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Rcc),
                  h_rmncc_c.data(), nb, cudaMemcpyHostToDevice),
       "restore rmncc");
    cc(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Zsc),
                  h_zmnsc_c.data(), nb, cudaMemcpyHostToDevice),
       "restore zmnsc");
    cc(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Lsc),
                  h_lmnsc_c.data(), nb, cudaMemcpyHostToDevice),
       "restore lmnsc");
    cc(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Rss),
                  h_rmnss_c.data(), nb, cudaMemcpyHostToDevice),
       "restore rmnss");
    cc(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Zcs),
                  h_zmncs_c.data(), nb, cudaMemcpyHostToDevice),
       "restore zmncs");
    cc(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Lcs),
                  h_lmncs_c.data(), nb, cudaMemcpyHostToDevice),
       "restore lmncs");
    ForceGate g2 =
        run_force_gate(storage, p, profiles, rp, transform, rs, geometry);
    std::cout << format(
        "RESTORED state — CPU residuals:                   FSQR = {:.3e}  FSQZ "
        "= {:.3e}  FSQL = {:.3e}\n",
        g2.fsqr_cpu, g2.fsqz_cpu, g2.fsql_cpu);
    check(g2.fsqr_cpu <= FAIL_THRESH && g2.fsqz_cpu <= FAIL_THRESH &&
              g2.fsql_cpu <= FAIL_THRESH,
          "restored state: the independent gate passes again");

    // Cleanup
    real_space_free(rs);
    // profiles/fp/mw owned by Profiles/ToroidalFftOperator/GeometryOperator
    // (RAII)
    cumes::mode_table_free(mt);

    return summary();
}
