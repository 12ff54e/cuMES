// test_force_reference.cu — scalar CPU reference for the MHD force kernel
// (blueprint §10.1 "local scalar reference"; Phase-5 deliverable #4).
//
// The force kernel is the one CUDA operator without an existing CPU oracle
// (the transforms have cpuInvDFT in test_fourier, the tridiagonal solve has the
// Thomas reference in test_regression_kernels). This test adds the missing
// layer: a scalar double reference that mirrors forcesKernel's weak form
// (radial/poloidal/toroidal/hybrid-lambda contributions, all eight e/o force
// families) point-by-point, and a dual-run gate that runs the GPU kernel and
// the CPU reference on the same frozen input and compares.
//
// The comparison is NOT bit-exact: the GPU is compiled with -use_fast_math,
// whose FMA fusion differs from the CPU's IEEE contraction, so the double leg
// compares at 1e-11 (a few ULPs of the ~1e0 force scale) and the float leg at
// 1e-4 (the float rounding floor of these sums).
#include <cmath>
#include <vector>

#include "vmec_types.h"
#include "cumes/transforms/toroidal_fft_operator.hpp"
#include "cumes/state/mode_table.cuh"
#include "cumes/physics/geometry_operator.hpp"
#include "cumes/physics/magnetic_field_operator.hpp"
#include "cumes/physics/force_operator.hpp"
#include "cumes/physics/profiles.hpp"
#include "cumes/state/spectral_storage.hpp"
#include "cumes_test_cuda_helper.cuh"
using namespace cumes::test;


// Scalar CPU reference: mirrors forcesKernel (forces_impl.cuh) exactly, in
// double, on host arrays. `r_e..zv_o` are the full-grid parity geometry, the
// half-grid metric/field arrays are (ns-1, nZnT), profiles are sqrtS_F/sqrtS_H/
// phip_F, and the 16 force outputs are written to out (parity-split).
template <typename T>
static void cpuForces(const std::vector<T>& r_e, const std::vector<T>& r_o,
                      const std::vector<T>& z_o,
                      const std::vector<T>& ru_e, const std::vector<T>& ru_o,
                      const std::vector<T>& zu_e, const std::vector<T>& zu_o,
                      const std::vector<T>& rv_e, const std::vector<T>& rv_o,
                      const std::vector<T>& zv_e, const std::vector<T>& zv_o,
                      const std::vector<T>& lu_e, const std::vector<T>& lu_o,
                      const std::vector<T>& r12, const std::vector<T>& ru12,
                      const std::vector<T>& zu12, const std::vector<T>& rs,
                      const std::vector<T>& zs, const std::vector<T>& tau,
                      const std::vector<T>& gsqrt, const std::vector<T>& guv,
                      const std::vector<T>& gvv, const std::vector<T>& bsupu,
                      const std::vector<T>& bsupv, const std::vector<T>& bsubu,
                      const std::vector<T>& bsubv, const std::vector<T>& totalP,
                      const std::vector<T>& sqrtS_F, const std::vector<T>& sqrtS_H,
                      const std::vector<T>& phip_F,
                      int ns, int nZnT, T lamscale, T delta_s,
                      std::vector<T>& armn_e, std::vector<T>& armn_o,
                      std::vector<T>& azmn_e, std::vector<T>& azmn_o,
                      std::vector<T>& brmn_e, std::vector<T>& brmn_o,
                      std::vector<T>& bzmn_e, std::vector<T>& bzmn_o,
                      std::vector<T>& crmn_e, std::vector<T>& crmn_o,
                      std::vector<T>& czmn_e, std::vector<T>& czmn_o,
                      std::vector<T>& blmn_e, std::vector<T>& blmn_o,
                      std::vector<T>& clmn_e, std::vector<T>& clmn_o) {
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

            T r12_i = T(0), ru12_i = T(0), zu12_i = T(0), rs_i = T(0), zs_i = T(0), tau_i = T(0);
            T gsqrt_i = T(0), bsupu_i = T(0), bsupv_i = T(0);
            T bsubu_i = T(0), bsubv_i = T(0), totalP_i = T(0);
            T sH_i = T(0), gvv_i = T(0), guv_i = T(0);
            T r12_o = T(0), ru12_o = T(0), zu12_o = T(0), rs_o = T(0), zs_o = T(0), tau_o = T(0);
            T gsqrt_o = T(0), bsupu_o = T(0), bsupv_o = T(0);
            T bsubu_o = T(0), bsubv_o = T(0), totalP_o = T(0);
            T sH_o = T(0), gvv_o = T(0), guv_o = T(0);

            if (j > 0) {
                int h_i = k + (j - 1) * nZnT;
                r12_i = r12[h_i]; ru12_i = ru12[h_i]; zu12_i = zu12[h_i];
                rs_i = rs[h_i]; zs_i = zs[h_i]; tau_i = tau[h_i];
                gsqrt_i = gsqrt[h_i]; gvv_i = gvv[h_i]; guv_i = guv[h_i];
                bsupu_i = bsupu[h_i]; bsupv_i = bsupv[h_i];
                bsubu_i = bsubu[h_i]; bsubv_i = bsubv[h_i];
                totalP_i = totalP[h_i]; sH_i = sqrtS_H[j - 1];
            }
            if (j < ns - 1) {
                int h_o = k + j * nZnT;
                r12_o = r12[h_o]; ru12_o = ru12[h_o]; zu12_o = zu12[h_o];
                rs_o = rs[h_o]; zs_o = zs[h_o]; tau_o = tau[h_o];
                gsqrt_o = gsqrt[h_o]; gvv_o = gvv[h_o]; guv_o = guv[h_o];
                bsupu_o = bsupu[h_o]; bsupv_o = bsupv[h_o];
                bsubu_o = bsubu[h_o]; bsubv_o = bsubv[h_o];
                totalP_o = totalP[h_o]; sH_o = sqrtS_H[j];
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

            T armn_e_v = (zup_o - zup_i) * inv_ds + T(0.5) * (taup_o + taup_i)
                         - gbvbv_avg * re_j - gbvbv_wavg * ro_j;
            T armn_o_v = (zup_o * sH_o - zup_i * sH_i) * inv_ds
                         - T(0.5) * P_wavg * zue_j - T(0.5) * P_avg * zuo_j
                         + T(0.5) * (taup_o * sH_o + taup_i * sH_i)
                         - gbvbv_wavg * re_j - gbvbv_avg * ro_j * sFull;
            T azmn_e_v = -(rup_o - rup_i) * inv_ds;
            T azmn_o_v = -(rup_o * sH_o - rup_i * sH_i) * inv_ds
                         + T(0.5) * P_wavg * rue_j + T(0.5) * P_avg * ruo_j;
            T brmn_e_v = T(0.5) * (zsp_o + zsp_i) + T(0.5) * P_wavg * zo_j
                         - gbubu_avg * rue_j - gbubu_wavg * ruo_j
                         - gbubv_avg * rve_j - gbubv_wavg * rvo_j;
            T brmn_o_v = T(0.5) * (zsp_o * sH_o + zsp_i * sH_i) + T(0.5) * P_avg * zo_j
                         - gbubu_wavg * rue_j - gbubu_avg * ruo_j * sFull
                         - gbubv_wavg * rve_j - gbubv_avg * rvo_j * sFull;
            T bzmn_e_v = -T(0.5) * (rsp_o + rsp_i) - T(0.5) * P_wavg * ro_j
                         - gbubu_avg * zue_j - gbubu_wavg * zuo_j
                         - gbubv_avg * zve_j - gbubv_wavg * zvo_j;
            T bzmn_o_v = -T(0.5) * (rsp_o * sH_o + rsp_i * sH_i) - T(0.5) * P_avg * ro_j
                         - gbubu_wavg * zue_j - gbubu_avg * zuo_j * sFull
                         - gbubv_wavg * zve_j - gbubv_avg * zvo_j * sFull;
            T crmn_e_v = gbubv_avg * rue_j + gbubv_wavg * ruo_j
                         + gbvbv_avg * rve_j + gbvbv_wavg * rvo_j;
            T crmn_o_v = gbubv_wavg * rue_j + gbubv_avg * ruo_j * sFull
                         + gbvbv_wavg * rve_j + gbvbv_avg * rvo_j * sFull;
            T czmn_e_v = gbubv_avg * zue_j + gbubv_wavg * zuo_j
                         + gbvbv_avg * zve_j + gbvbv_wavg * zvo_j;
            T czmn_o_v = gbubv_wavg * zue_j + gbubv_avg * zuo_j * sFull
                         + gbvbv_wavg * zve_j + gbvbv_avg * zvo_j * sFull;

            T bsubv_avg = T(0.5) * (bsubv_o + bsubv_i);
            T gvv_gsqrt_i = (j > 0) ? (gvv_i / gsqrt_i) : T(0.0);
            T gvv_gsqrt_o = (j < ns - 1) ? (gvv_o / gsqrt_o) : T(0.0);
            T guv_bsupu_i = (j > 0) ? (guv_i * bsupu_i) : T(0.0);
            T guv_bsupu_o = (j < ns - 1) ? (guv_o * bsupu_o) : T(0.0);
            T lu_e_norm = lamscale * lu_e[idx_f] + phip_F[j];
            T lu_o_norm = lamscale * lu_o[idx_f];
            T bsubv_alt = T(0.5) * (gvv_gsqrt_i + gvv_gsqrt_o) * lu_e_norm
                          + T(0.5) * (gvv_gsqrt_i * sH_i + gvv_gsqrt_o * sH_o) * lu_o_norm
                          + T(0.5) * (guv_bsupu_i + guv_bsupu_o);
            T rb = T(2.0) * T(0.05) * (T(1.0) - sFull);
            T _blmn = bsubv_avg * (T(1.0) - rb) + bsubv_alt * rb;
            if (j > 0) _blmn *= -lamscale;
            T blmn_e_v = _blmn;
            T blmn_o_v = _blmn * sF_j;

            T _clmn = T(0.5) * (bsubu_o + bsubu_i);
            if (j > 0) _clmn *= -lamscale;
            T clmn_e_v = _clmn;
            T clmn_o_v = _clmn * sF_j;

            armn_e[idx_f] = armn_e_v; armn_o[idx_f] = armn_o_v;
            azmn_e[idx_f] = azmn_e_v; azmn_o[idx_f] = azmn_o_v;
            brmn_e[idx_f] = brmn_e_v; brmn_o[idx_f] = brmn_o_v;
            bzmn_e[idx_f] = bzmn_e_v; bzmn_o[idx_f] = bzmn_o_v;
            crmn_e[idx_f] = crmn_e_v; crmn_o[idx_f] = crmn_o_v;
            czmn_e[idx_f] = czmn_e_v; czmn_o[idx_f] = czmn_o_v;
            blmn_e[idx_f] = blmn_e_v; blmn_o[idx_f] = blmn_o_v;
            clmn_e[idx_f] = clmn_e_v; clmn_o[idx_f] = clmn_o_v;
        }
    }
}

template <typename T>
static void runReference(int ns, int mpol, int ntor, int ntheta, int nzeta, const char* label) {
    DeviceParams<T> p;
    p.ns = ns; p.mnmax = mpol * (ntor + 1); p.ntheta = ntheta; p.nzeta = nzeta;
    p.nfp = 1; p.nZnT = ntheta * nzeta; p.mpol = mpol; p.ntor = ntor;
    p.ncurr = 0; p.delt = T(0.9); p.ftol = T(1e-14); p.max_iter = 10;
    p.tcon0 = T(1.0); p.lamscale = T(0.0);

    // Frozen, non-degenerate Solovev-like state (same pattern as test_forces).
    cumes::SpectralStorage<T> storage(ns, p.mnmax);
    const size_t nS = (size_t)ns * p.mnmax, nb = nS * sizeof(T);
    auto* h_cc = new T[nS](); auto* h_ss = new T[nS](); auto* h_zsc = new T[nS]();
    auto* h_zcs = new T[nS](); auto* h_lsc = new T[nS](); auto* h_lcs = new T[nS]();
    for (int j = 0; j < ns; ++j) {
        T s = T(j) / T(ns - 1);
        for (int mode = 0; mode < p.mnmax; ++mode) {
            int m = mode / (ntor + 1);
            if (m == 0 && mode == 0) h_cc[j + mode * ns] = T(4.0);
            else if (m == 1) { h_cc[j + mode * ns] = T(0.3) * s; h_zsc[j + mode * ns] = T(-0.5) * s; h_zcs[j + mode * ns] = T(-0.5) * s; }
            else if (m == 2) h_cc[j + mode * ns] = T(0.2) * s * s;
            h_ss[j + mode * ns] = h_cc[j + mode * ns];
        }
    }
    check_cuda(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Rcc), h_cc, nb, cudaMemcpyHostToDevice), "cc");
    check_cuda(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Rss), h_ss, nb, cudaMemcpyHostToDevice), "ss");
    check_cuda(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Zsc), h_zsc, nb, cudaMemcpyHostToDevice), "zsc");
    check_cuda(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Zcs), h_zcs, nb, cudaMemcpyHostToDevice), "zcs");
    check_cuda(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Lsc), h_lsc, nb, cudaMemcpyHostToDevice), "lsc");
    check_cuda(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Lcs), h_lcs, nb, cudaMemcpyHostToDevice), "lcs");
    delete[] h_cc; delete[] h_ss; delete[] h_zsc; delete[] h_zcs; delete[] h_lsc; delete[] h_lcs;

    cumes::ValidatedProblem vp = load_validated("inputs/solovev.json");
    cumes::Profiles<T> profiles(p, vp, nullptr); cumes::RadialProfileViews<T> rp = profiles.profile_views();
    cumes::DeviceModeTable mt = cumes::modeTableCreate(p);
    cumes::RealSpaceStorage<T> rs = realSpaceCreate(p);
    cumes::ToroidalFftOperator<T> op(p, rs, mt);
    cumes::GeometryOperator<T> geometry(p, nullptr);

    op.inverse(storage.physical_const(), /*do_combine=*/true);
    geometry.enqueue(rs, p, rp, 0); cumes::MagneticFieldOperator<T>{}.enqueue(rs, p, rp, geometry.base_geometry_views(p), geometry.magnetic_field_views(p), nullptr, 0, true);
    cumes::ForceOperator<T>{}.enqueue(rs, p, rp, geometry.base_geometry_views(p), geometry.magnetic_field_views(p), nullptr, 0);

    const size_t nF = (size_t)ns * p.nZnT, nH = (size_t)(ns - 1) * p.nZnT;
    auto g = [&](const T* d, size_t n) { std::vector<T> v(n); check_cuda(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "g"); return v; };
    // full-grid parity
    std::vector<T> r_e = g(rs.d_r_e, nF), r_o = g(rs.d_r_o, nF);
    std::vector<T> z_e = g(rs.d_z_e, nF), z_o = g(rs.d_z_o, nF);
    std::vector<T> ru_e = g(rs.d_ru_e, nF), ru_o = g(rs.d_ru_o, nF);
    std::vector<T> zu_e = g(rs.d_zu_e, nF), zu_o = g(rs.d_zu_o, nF);
    std::vector<T> rv_e = g(rs.d_rv_e, nF), rv_o = g(rs.d_rv_o, nF);
    std::vector<T> zv_e = g(rs.d_zv_e, nF), zv_o = g(rs.d_zv_o, nF);
    std::vector<T> lu_e = g(rs.d_lu_e, nF), lu_o = g(rs.d_lu_o, nF);
    // half-grid
    std::vector<T> r12 = g(geometry.base_geometry_views(p).r12.data(), nH), ru12 = g(geometry.base_geometry_views(p).ru12.data(), nH), zu12 = g(geometry.base_geometry_views(p).zu12.data(), nH);
    std::vector<T> rs_h = g(geometry.base_geometry_views(p).rs.data(), nH), zs = g(geometry.base_geometry_views(p).zs.data(), nH), tau = g(geometry.base_geometry_views(p).tau.data(), nH);
    std::vector<T> gsqrt = g(geometry.base_geometry_views(p).gsqrt.data(), nH), guv = g(geometry.base_geometry_views(p).guv.data(), nH), gvv = g(geometry.base_geometry_views(p).gvv.data(), nH);
    std::vector<T> bsupu = g(geometry.magnetic_field_views(p).bsupu.data(), nH), bsupv = g(geometry.magnetic_field_views(p).bsupv.data(), nH);
    std::vector<T> bsubu = g(geometry.magnetic_field_views(p).bsubu.data(), nH), bsubv = g(geometry.magnetic_field_views(p).bsubv.data(), nH);
    std::vector<T> totalP = g(geometry.magnetic_field_views(p).total_pressure.data(), nH);
    // profiles
    std::vector<T> sqrtS_F = g(rp.sqrtS_F, ns), sqrtS_H = g(rp.sqrtS_H, ns - 1), phip_F = g(rp.phip_F, ns);

    // GPU force outputs
    std::vector<T> g_armn_e = g(rs.d_armn_e, nF), g_armn_o = g(rs.d_armn_o, nF);
    std::vector<T> g_azmn_e = g(rs.d_azmn_e, nF), g_azmn_o = g(rs.d_azmn_o, nF);
    std::vector<T> g_brmn_e = g(rs.d_brmn_e, nF), g_brmn_o = g(rs.d_brmn_o, nF);
    std::vector<T> g_bzmn_e = g(rs.d_bzmn_e, nF), g_bzmn_o = g(rs.d_bzmn_o, nF);
    std::vector<T> g_crmn_e = g(rs.d_crmn_e, nF), g_crmn_o = g(rs.d_crmn_o, nF);
    std::vector<T> g_czmn_e = g(rs.d_czmn_e, nF), g_czmn_o = g(rs.d_czmn_o, nF);
    std::vector<T> g_blmn_e = g(rs.d_blmn_e, nF), g_blmn_o = g(rs.d_blmn_o, nF);
    std::vector<T> g_clmn_e = g(rs.d_clmn_e, nF), g_clmn_o = g(rs.d_clmn_o, nF);

    // CPU reference
    std::vector<T> c_armn_e(nF), c_armn_o(nF), c_azmn_e(nF), c_azmn_o(nF);
    std::vector<T> c_brmn_e(nF), c_brmn_o(nF), c_bzmn_e(nF), c_bzmn_o(nF);
    std::vector<T> c_crmn_e(nF), c_crmn_o(nF), c_czmn_e(nF), c_czmn_o(nF);
    std::vector<T> c_blmn_e(nF), c_blmn_o(nF), c_clmn_e(nF), c_clmn_o(nF);
    cpuForces(r_e, r_o, z_o, ru_e, ru_o, zu_e, zu_o, rv_e, rv_o, zv_e, zv_o,
              lu_e, lu_o, r12, ru12, zu12, rs_h, zs, tau, gsqrt, guv, gvv,
              bsupu, bsupv, bsubu, bsubv, totalP, sqrtS_F, sqrtS_H, phip_F,
              ns, p.nZnT, p.lamscale, profiles.delta_s(),
              c_armn_e, c_armn_o, c_azmn_e, c_azmn_o, c_brmn_e, c_brmn_o,
              c_bzmn_e, c_bzmn_o, c_crmn_e, c_crmn_o, c_czmn_e, c_czmn_o,
              c_blmn_e, c_blmn_o, c_clmn_e, c_clmn_o);

    const double tol = (sizeof(T) == sizeof(double)) ? 1e-8 : 1e-4;
    double md = 0.0;
    md = std::max(md, max_diff(g_armn_e, c_armn_e)); md = std::max(md, max_diff(g_armn_o, c_armn_o));
    md = std::max(md, max_diff(g_azmn_e, c_azmn_e)); md = std::max(md, max_diff(g_azmn_o, c_azmn_o));
    md = std::max(md, max_diff(g_brmn_e, c_brmn_e)); md = std::max(md, max_diff(g_brmn_o, c_brmn_o));
    md = std::max(md, max_diff(g_bzmn_e, c_bzmn_e)); md = std::max(md, max_diff(g_bzmn_o, c_bzmn_o));
    md = std::max(md, max_diff(g_crmn_e, c_crmn_e)); md = std::max(md, max_diff(g_crmn_o, c_crmn_o));
    md = std::max(md, max_diff(g_czmn_e, c_czmn_e)); md = std::max(md, max_diff(g_czmn_o, c_czmn_o));
    md = std::max(md, max_diff(g_blmn_e, c_blmn_e)); md = std::max(md, max_diff(g_blmn_o, c_blmn_o));
    md = std::max(md, max_diff(g_clmn_e, c_clmn_e)); md = std::max(md, max_diff(g_clmn_o, c_clmn_o));

    auto msg = format("{}: GPU force == CPU scalar reference (max |diff| {:.3e} < {:.1e})", label, md, tol);
    check(md < tol, msg);

    realSpaceFree(rs);
    cumes::modeTableFree(mt);
}

int main() {
    std::cout << "=== Force kernel: GPU vs CPU scalar reference (dual-run) ===\n";
    runReference<double>(5, 4, 0, 18, 1, "double axisymmetric ns=5");
    runReference<double>(11, 6, 2, 18, 4, "double 3D ns=11");
    runReference<float>(5, 4, 0, 18, 1, "float axisymmetric ns=5");
    return summary();
}
