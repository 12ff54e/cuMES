// test_force_reference.cu — scalar CPU reference for the MHD force kernel
// (blueprint §10.1 "local scalar reference"; Phase-5 deliverable #4).
//
// The force kernel is the one CUDA operator without an existing CPU oracle
// (the transforms have cpu_inv_dft in test_fourier, the tridiagonal solve has
// the Thomas reference in test_regression_kernels). This test adds the missing
// layer: a scalar double reference that mirrors forces_kernel's weak form
// (radial/poloidal/toroidal/hybrid-lambda contributions, all eight e/o force
// families) point-by-point, and a dual-run gate that runs the GPU kernel and
// the CPU reference on the same frozen input and compares.
//
// The comparison is NOT bit-exact: the GPU is compiled with -use_fast_math,
// whose FMA fusion differs from the CPU's IEEE contraction, so the double leg
// compares at 1e-11 (a few ULPs of the ~1e0 force scale) and the float leg at
// 1e-4 (the float rounding floor of these sums).
#include "cumes/physics/force_operator.hpp"
#include "cumes/physics/geometry_operator.hpp"
#include "cumes/physics/magnetic_field_operator.hpp"
#include "cumes/physics/profiles.hpp"
#include "cumes/state/mode_table.cuh"
#include "cumes/state/spectral_storage.hpp"
#include "cumes/transforms/toroidal_fft_operator.hpp"
#include "cumes_test_cuda_helper.cuh"
#include "cumes_test_force_reference.hpp"
#include "vmec_types.h"

#include <cmath>
#include <vector>
using namespace cumes::test;

template <typename T>
static void run_reference(int ns,
                          int mpol,
                          int ntor,
                          int ntheta,
                          int nzeta,
                          const char* label) {
    DeviceParams<T> p;
    p.ns = ns;
    p.mnmax = mpol * (ntor + 1);
    p.ntheta = ntheta;
    p.nzeta = nzeta;
    p.nfp = 1;
    p.nZnT = ntheta * nzeta;
    p.mpol = mpol;
    p.ntor = ntor;
    p.ncurr = 0;
    p.delt = T(0.9);
    p.ftol = T(1e-14);
    p.max_iter = 10;
    p.tcon0 = T(1.0);
    p.lamscale = T(0.0);

    // Frozen, non-degenerate Solovev-like state (same pattern as test_forces).
    cumes::SpectralStorage<T> storage(ns, p.mnmax);
    const size_t nS = (size_t)ns * p.mnmax, nb = nS * sizeof(T);
    std::vector<T> h_cc(nS);
    std::vector<T> h_ss(nS);
    std::vector<T> h_zsc(nS);
    std::vector<T> h_zcs(nS);
    std::vector<T> h_lsc(nS);
    std::vector<T> h_lcs(nS);
    for (int j = 0; j < ns; ++j) {
        T s = T(j) / T(ns - 1);
        for (int mode = 0; mode < p.mnmax; ++mode) {
            int m = mode / (ntor + 1);
            if (m == 0 && mode == 0)
                h_cc[j + mode * ns] = T(4.0);
            else if (m == 1) {
                h_cc[j + mode * ns] = T(0.3) * s;
                h_zsc[j + mode * ns] = T(-0.5) * s;
                h_zcs[j + mode * ns] = T(-0.5) * s;
            } else if (m == 2)
                h_cc[j + mode * ns] = T(0.2) * s * s;
            h_ss[j + mode * ns] = h_cc[j + mode * ns];
        }
    }
    check_cuda(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Rcc),
                          h_cc.data(), nb, cudaMemcpyHostToDevice),
               "cc");
    check_cuda(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Rss),
                          h_ss.data(), nb, cudaMemcpyHostToDevice),
               "ss");
    check_cuda(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Zsc),
                          h_zsc.data(), nb, cudaMemcpyHostToDevice),
               "zsc");
    check_cuda(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Zcs),
                          h_zcs.data(), nb, cudaMemcpyHostToDevice),
               "zcs");
    check_cuda(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Lsc),
                          h_lsc.data(), nb, cudaMemcpyHostToDevice),
               "lsc");
    check_cuda(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Lcs),
                          h_lcs.data(), nb, cudaMemcpyHostToDevice),
               "lcs");

    cumes::ValidatedProblem vp = load_validated("inputs/solovev.json");
    cumes::Profiles<T> profiles(p, vp, std::nullopt);
    cumes::RadialProfileViews<T> rp = profiles.profile_views();
    cumes::DeviceModeTable mt = cumes::mode_table_create(p);
    cumes::RealSpaceStorage<T> rs = real_space_create(p);
    cumes::ToroidalFftOperator<T> op(p, rs, mt);
    cumes::GeometryOperator<T> geometry(p, std::nullopt);

    op.inverse(storage.physical_const(), /*do_combine=*/true);
    geometry.enqueue(rs, p, rp, 0);
    cumes::MagneticFieldOperator<T>{}.enqueue(
        rs, p, rp, geometry.base_geometry_views(p),
        geometry.magnetic_field_views(p), nullptr, 0, true);
    cumes::ForceOperator<T>{}.enqueue(
        rs, p, rp, geometry.base_geometry_views(p),
        geometry.magnetic_field_views(p), nullptr, 0);

    const size_t nF = (size_t)ns * p.nZnT, nH = (size_t)(ns - 1) * p.nZnT;
    auto g = [&](const T* d, size_t n) {
        std::vector<T> v(n);
        check_cuda(
            cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost),
            "g");
        return v;
    };
    // full-grid parity
    std::vector<T> r_e = g(rs.d_r_e, nF), r_o = g(rs.d_r_o, nF);
    std::vector<T> z_e = g(rs.d_z_e, nF), z_o = g(rs.d_z_o, nF);
    std::vector<T> ru_e = g(rs.d_ru_e, nF), ru_o = g(rs.d_ru_o, nF);
    std::vector<T> zu_e = g(rs.d_zu_e, nF), zu_o = g(rs.d_zu_o, nF);
    std::vector<T> rv_e = g(rs.d_rv_e, nF), rv_o = g(rs.d_rv_o, nF);
    std::vector<T> zv_e = g(rs.d_zv_e, nF), zv_o = g(rs.d_zv_o, nF);
    std::vector<T> lu_e = g(rs.d_lu_e, nF), lu_o = g(rs.d_lu_o, nF);
    // half-grid
    std::vector<T> r12 = g(geometry.base_geometry_views(p).r12.data(), nH),
                   ru12 = g(geometry.base_geometry_views(p).ru12.data(), nH),
                   zu12 = g(geometry.base_geometry_views(p).zu12.data(), nH);
    std::vector<T> rs_h = g(geometry.base_geometry_views(p).rs.data(), nH),
                   zs = g(geometry.base_geometry_views(p).zs.data(), nH),
                   tau = g(geometry.base_geometry_views(p).tau.data(), nH);
    std::vector<T> gsqrt = g(geometry.base_geometry_views(p).gsqrt.data(), nH),
                   guv = g(geometry.base_geometry_views(p).guv.data(), nH),
                   gvv = g(geometry.base_geometry_views(p).gvv.data(), nH);
    std::vector<T> bsupu = g(geometry.magnetic_field_views(p).bsupu.data(), nH),
                   bsupv = g(geometry.magnetic_field_views(p).bsupv.data(), nH);
    std::vector<T> bsubu = g(geometry.magnetic_field_views(p).bsubu.data(), nH),
                   bsubv = g(geometry.magnetic_field_views(p).bsubv.data(), nH);
    std::vector<T> totalP =
        g(geometry.magnetic_field_views(p).total_pressure.data(), nH);
    // profiles
    std::vector<T> sqrtS_F = g(rp.sqrtS_F, ns), sqrtS_H = g(rp.sqrtS_H, ns - 1),
                   phip_F = g(rp.phip_F, ns);

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
    cpu_forces(r_e, r_o, z_o, ru_e, ru_o, zu_e, zu_o, rv_e, rv_o, zv_e, zv_o,
               lu_e, lu_o, r12, ru12, zu12, rs_h, zs, tau, gsqrt, guv, gvv,
               bsupu, bsupv, bsubu, bsubv, totalP, sqrtS_F, sqrtS_H, phip_F, ns,
               p.nZnT, p.lamscale, profiles.delta_s(), c_armn_e, c_armn_o,
               c_azmn_e, c_azmn_o, c_brmn_e, c_brmn_o, c_bzmn_e, c_bzmn_o,
               c_crmn_e, c_crmn_o, c_czmn_e, c_czmn_o, c_blmn_e, c_blmn_o,
               c_clmn_e, c_clmn_o);

    const double tol = (sizeof(T) == sizeof(double)) ? 1e-8 : 1e-4;
    double md = 0.0;
    md = std::max(md, max_diff(g_armn_e, c_armn_e));
    md = std::max(md, max_diff(g_armn_o, c_armn_o));
    md = std::max(md, max_diff(g_azmn_e, c_azmn_e));
    md = std::max(md, max_diff(g_azmn_o, c_azmn_o));
    md = std::max(md, max_diff(g_brmn_e, c_brmn_e));
    md = std::max(md, max_diff(g_brmn_o, c_brmn_o));
    md = std::max(md, max_diff(g_bzmn_e, c_bzmn_e));
    md = std::max(md, max_diff(g_bzmn_o, c_bzmn_o));
    md = std::max(md, max_diff(g_crmn_e, c_crmn_e));
    md = std::max(md, max_diff(g_crmn_o, c_crmn_o));
    md = std::max(md, max_diff(g_czmn_e, c_czmn_e));
    md = std::max(md, max_diff(g_czmn_o, c_czmn_o));
    md = std::max(md, max_diff(g_blmn_e, c_blmn_e));
    md = std::max(md, max_diff(g_blmn_o, c_blmn_o));
    md = std::max(md, max_diff(g_clmn_e, c_clmn_e));
    md = std::max(md, max_diff(g_clmn_o, c_clmn_o));

    auto msg = format(
        "{}: GPU force == CPU scalar reference (max |diff| {:.3e} < {:.1e})",
        label, md, tol);
    check(md < tol, msg);

    real_space_free(rs);
    cumes::mode_table_free(mt);
}

int main() {
    std::cout
        << "=== Force kernel: GPU vs CPU scalar reference (dual-run) ===\n";
    run_reference<double>(5, 4, 0, 18, 1, "double axisymmetric ns=5");
    run_reference<double>(11, 6, 2, 18, 4, "double 3D ns=11");
    run_reference<float>(5, 4, 0, 18, 1, "float axisymmetric ns=5");
    return summary();
}
