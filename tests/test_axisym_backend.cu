// test_axisym_backend.cu — differential test: the axisymmetric direct-poloidal
// backend (AxisymmetricOperator) vs the generic cuFFT backend on identical
// frozen axisymmetric inputs (blueprint §8.5 gate).
//
// For ntor=0, nzeta=1 the generic backend executes length-one cuFFT plans that
// are the identity transform; the axisymmetric backend replaces them with
// direct poloidal sums over cos(mθ)/sin(mθ). This test runs BOTH backends on
// the same state / forces / constraint input and compares every transform
// product:
//   - inverse: 18 parity-split real-space geometry arrays (R/Z/λ ×
//     value/θ-deriv/ζ-deriv × even/odd)
//   - forward: 6 spectral-force families
//   - constraint rzCon: rCon/zCon
//   - constraint bandpass: gCon
// The comparison is Class B (the two paths differ in summation order / the
// length-one FFT elision), so the gate is a ULP-level tolerance, not bytes.
#include "cumes/state/mode_table.cuh"
#include "cumes/state/spectral_storage.hpp"
#include "cumes/transforms/axisymmetric_operator.hpp"
#include "cumes/transforms/toroidal_fft_operator.hpp"
#include "cumes_test_cuda_helper.cuh"
#include "vmec_types.h"

#include <cmath>
#include <vector>
using namespace cumes::test;

constexpr int kNs = 5, kMpol = 6, kNtor = 0, kNtheta = 18, kNzeta = 1, kNfp = 1;
constexpr int kMnmax = kMpol * (kNtor + 1), kNZnT = kNtheta * kNzeta;

// Local scratch bundle replacing the deleted ConstraintWorkspace (the A/B test
// allocates its own constraint buffers rather than borrowing the operator's).
template <typename T>
struct Cw {
    using val_type = T;

    T* d_frcon_e;
    T* d_frcon_o;
    T* d_fzcon_e;
    T* d_fzcon_o;
    T* d_tcon;
    T* d_gConEff;
    T* d_faccon;
    T* d_gCon;
    T* d_rCon;
    T* d_zCon;
};

template <typename T>
static constexpr double tolNear() {
    return sizeof(T) == sizeof(float) ? 1e-3 : 1e-11;
}

// Compare two device arrays elementwise at an absolute tolerance.
template <typename T>
static void compareArrays(const char* label,
                          const T* a,
                          const T* b,
                          int n,
                          double tol) {
    std::vector<T> ha((size_t)n), hb((size_t)n);
    cc(cudaMemcpy(ha.data(), a, (size_t)n * sizeof(T), cudaMemcpyDeviceToHost),
       "cp a");
    cc(cudaMemcpy(hb.data(), b, (size_t)n * sizeof(T), cudaMemcpyDeviceToHost),
       "cp b");
    double maxd = 0.0;
    for (int i = 0; i < n; ++i)
        maxd = fmax(maxd, fabs((double)ha[i] - (double)hb[i]));
    if (maxd > tol) {
        std::cerr << format("FAIL [{}] max abs diff = {:.3e} (tol {:.3e})\n",
                            label, maxd, tol);
        ++failures();
    } else {
        std::cout << format("  PASS {} (max abs diff {:.3e})\n", label, maxd);
    }
}

template <typename T>
static void fillState(cumes::SpectralStorage<T>& storage, int ns, int mnmax) {
    // Deterministic frozen pattern; fills all six families (the sin(nζ)
    // families Rss/Zcs/Lcs are nonzero so the test proves both backends ignore
    // them identically for n=0).
    std::vector<T> v((size_t)6 * ns * mnmax);
    for (int c = 0; c < 6; ++c)
        for (int m = 0; m < mnmax; ++m)
            for (int j = 0; j < ns; ++j)
                v[(size_t)c * mnmax * ns + m * ns + j] =
                    T(0.001 * (c + 1) * (m + 1) * (j + 1));
    cc(cudaMemcpy(storage.state_slab(), v.data(), v.size() * sizeof(T),
                  cudaMemcpyHostToDevice),
       "fill state");
}

template <typename T>
static void fillForces(cumes::RealSpaceStorage<T>& rs,
                       Cw<T>& cw,
                       int ns,
                       int nZnT) {
    const int n = ns * nZnT;
    std::vector<T> v((size_t)n);
    auto fill = [&](T* dst, double base) {
        for (int i = 0; i < n; ++i) v[i] = T(base * (1.0 + 0.01 * (i % 7)));
        cc(cudaMemcpy(dst, v.data(), (size_t)n * sizeof(T),
                      cudaMemcpyHostToDevice),
           "fill force");
    };
    fill(rs.d_armn_e, 0.5);
    fill(rs.d_armn_o, 0.4);
    fill(rs.d_azmn_e, 0.3);
    fill(rs.d_azmn_o, 0.2);
    fill(rs.d_brmn_e, 0.1);
    fill(rs.d_brmn_o, 0.05);
    fill(rs.d_bzmn_e, 0.03);
    fill(rs.d_bzmn_o, 0.02);
    fill(rs.d_blmn_e, 0.01);
    fill(rs.d_blmn_o, 0.005);
    // The constraint MHD-force families too: the forward reduction reads all
    // twelve parity views, and initcheck flags reads of never-written bytes.
    fill(rs.d_clmn_e, 0.004);
    fill(rs.d_clmn_o, 0.003);
    fill(rs.d_crmn_e, 0.002);
    fill(rs.d_crmn_o, 0.001);
    fill(rs.d_czmn_e, 0.0005);
    fill(rs.d_czmn_o, 0.0004);
    fill(cw.d_frcon_e, 0.7);
    fill(cw.d_frcon_o, 0.6);
    fill(cw.d_fzcon_e, 0.5);
    fill(cw.d_fzcon_o, 0.4);
}

template <typename T>
static void fillTcon(Cw<T>& cw, int ns) {
    std::vector<T> v((size_t)ns);
    for (int j = 0; j < ns; ++j) v[j] = T(0.5 + 0.1 * j);
    cc(cudaMemcpy(cw.d_tcon, v.data(), (size_t)ns * sizeof(T),
                  cudaMemcpyHostToDevice),
       "fill tcon");
}

template <typename T>
static void fillFaccon(Cw<T>& cw, int mpol) {
    // The production faccon[m] = 0.25/(xmpq[m+1]^2) profile (ConstraintOperator
    // ctor); the old test left this buffer as cudaMalloc garbage and passed
    // only because the stale pages happened to read zero.
    std::vector<T> v((size_t)mpol);
    for (int m = 0; m < mpol; ++m) {
        T xmpq = T((m + 1) * m);
        v[m] = (m > 0) ? (T(0.25) / (xmpq * xmpq)) : T(0.0);
    }
    cc(cudaMemcpy(cw.d_faccon, v.data(), (size_t)mpol * sizeof(T),
                  cudaMemcpyHostToDevice),
       "fill faccon");
}

template <typename T>
static void fillGconEff(Cw<T>& cw, int ns, int nZnT) {
    std::vector<T> v((size_t)ns * nZnT);
    for (int i = 0; i < ns * nZnT; ++i) v[i] = T(sin(0.3 * i) + 0.1 * cos(i));
    cc(cudaMemcpy(cw.d_gConEff, v.data(), v.size() * sizeof(T),
                  cudaMemcpyHostToDevice),
       "fill gConEff");
}

// Inverse: compare the 18 parity-split geometry arrays.
template <typename T>
static void testInverse(DeviceParams<T>& p,
                        cumes::RealSpaceStorage<T>& rs,
                        cumes::ToroidalFftOperator<T>& generic,
                        cumes::SpectralStorage<T>& storage,
                        cumes::AxisymmetricOperator<T>& op) {
    std::cout << "  inverse (18 geometry arrays) ...\n";
    const int n = p.ns * p.nZnT;
    // Generic backend.
    generic.inverse(storage.physical_const(), /*do_combine=*/false);
    // Axisymmetric backend -> one contiguous scratch carved into 18 views.
    std::vector<T> scratch((size_t)18 * n, T(0));
    T* d_ax = nullptr;
    cc(cudaMalloc(&d_ax, (size_t)18 * n * sizeof(T)), "ax geom");
    cc(cudaMemcpy(d_ax, scratch.data(), (size_t)18 * n * sizeof(T),
                  cudaMemcpyHostToDevice),
       "ax geom seed");
    auto view = [&](int k) {
        return cumes::RealFieldView<T>(d_ax + (size_t)k * n, p.ns, p.ntheta,
                                       p.nzeta);
    };
    cumes::GeometryParityViews<T> g;
    g.r_e = view(0);
    g.z_e = view(1);
    g.l_e = view(2);
    g.ru_e = view(3);
    g.zu_e = view(4);
    g.lu_e = view(5);
    g.r_o = view(6);
    g.z_o = view(7);
    g.l_o = view(8);
    g.ru_o = view(9);
    g.zu_o = view(10);
    g.lu_o = view(11);
    g.rv_e = view(12);
    g.zv_e = view(13);
    g.lv_e = view(14);
    g.rv_o = view(15);
    g.zv_o = view(16);
    g.lv_o = view(17);
    op.enqueue_inverse(storage.physical_const(), g, cumes::RealFieldView<T>(),
                       cumes::RealFieldView<T>(), 0);

    const T* gen[18] = {rs.d_r_e,  rs.d_z_e,  rs.d_l_e,  rs.d_ru_e, rs.d_zu_e,
                        rs.d_lu_e, rs.d_r_o,  rs.d_z_o,  rs.d_l_o,  rs.d_ru_o,
                        rs.d_zu_o, rs.d_lu_o, rs.d_rv_e, rs.d_zv_e, rs.d_lv_e,
                        rs.d_rv_o, rs.d_zv_o, rs.d_lv_o};
    const char* names[18] = {"r_e",  "z_e",  "l_e",  "ru_e", "zu_e", "lu_e",
                             "r_o",  "z_o",  "l_o",  "ru_o", "zu_o", "lu_o",
                             "rv_e", "zv_e", "lv_e", "rv_o", "zv_o", "lv_o"};
    const T* ax[18] = {
        d_ax + (size_t)0 * n,  d_ax + (size_t)1 * n,  d_ax + (size_t)2 * n,
        d_ax + (size_t)3 * n,  d_ax + (size_t)4 * n,  d_ax + (size_t)5 * n,
        d_ax + (size_t)6 * n,  d_ax + (size_t)7 * n,  d_ax + (size_t)8 * n,
        d_ax + (size_t)9 * n,  d_ax + (size_t)10 * n, d_ax + (size_t)11 * n,
        d_ax + (size_t)12 * n, d_ax + (size_t)13 * n, d_ax + (size_t)14 * n,
        d_ax + (size_t)15 * n, d_ax + (size_t)16 * n, d_ax + (size_t)17 * n};
    for (int k = 0; k < 18; ++k)
        compareArrays(names[k], gen[k], ax[k], n, tolNear<T>());
    cudaFree(d_ax);
}

// Forward: compare the 6 spectral-force families.
template <typename T>
static void testForward(DeviceParams<T>& p,
                        cumes::RealSpaceStorage<T>& rs,
                        cumes::ToroidalFftOperator<T>& gen,
                        Cw<T>& cw,
                        cumes::AxisymmetricOperator<T>& op) {
    std::cout << "  forward (6 spectral families) ...\n";
    const int n = p.ns * p.mnmax;
    T *d_gen = nullptr, *d_ax = nullptr;
    cc(cudaMalloc(&d_gen, (size_t)6 * n * sizeof(T)), "gen fwd");
    cc(cudaMalloc(&d_ax, (size_t)6 * n * sizeof(T)), "ax fwd");
    // The forward reductions leave the boundary/axis zero rows to the kernels;
    // zero both outputs so the full-slab differential comparison stays
    // initcheck-defined on every entry either backend may skip.
    cc(cudaMemset(d_gen, 0, (size_t)6 * n * sizeof(T)), "gen fwd zero");
    cc(cudaMemset(d_ax, 0, (size_t)6 * n * sizeof(T)), "ax fwd zero");
    cumes::SpectralView<T, cumes::DecomposedResidualDomain> gen_v(d_gen, p.ns,
                                                                  p.mnmax);
    cumes::SpectralView<T, cumes::DecomposedResidualDomain> ax_v(d_ax, p.ns,
                                                                 p.mnmax);
    gen.forward(gen_v, cw.d_frcon_e, cw.d_frcon_o, cw.d_fzcon_e, cw.d_fzcon_o);

    cumes::ForceParityViews<const T> f;
    f.armn_e =
        cumes::RealFieldView<const T>(rs.d_armn_e, p.ns, p.ntheta, p.nzeta);
    f.armn_o =
        cumes::RealFieldView<const T>(rs.d_armn_o, p.ns, p.ntheta, p.nzeta);
    f.azmn_e =
        cumes::RealFieldView<const T>(rs.d_azmn_e, p.ns, p.ntheta, p.nzeta);
    f.azmn_o =
        cumes::RealFieldView<const T>(rs.d_azmn_o, p.ns, p.ntheta, p.nzeta);
    f.brmn_e =
        cumes::RealFieldView<const T>(rs.d_brmn_e, p.ns, p.ntheta, p.nzeta);
    f.brmn_o =
        cumes::RealFieldView<const T>(rs.d_brmn_o, p.ns, p.ntheta, p.nzeta);
    f.bzmn_e =
        cumes::RealFieldView<const T>(rs.d_bzmn_e, p.ns, p.ntheta, p.nzeta);
    f.bzmn_o =
        cumes::RealFieldView<const T>(rs.d_bzmn_o, p.ns, p.ntheta, p.nzeta);
    f.blmn_e =
        cumes::RealFieldView<const T>(rs.d_blmn_e, p.ns, p.ntheta, p.nzeta);
    f.blmn_o =
        cumes::RealFieldView<const T>(rs.d_blmn_o, p.ns, p.ntheta, p.nzeta);
    f.crmn_e =
        cumes::RealFieldView<const T>(rs.d_crmn_e, p.ns, p.ntheta, p.nzeta);
    f.crmn_o =
        cumes::RealFieldView<const T>(rs.d_crmn_o, p.ns, p.ntheta, p.nzeta);
    f.czmn_e =
        cumes::RealFieldView<const T>(rs.d_czmn_e, p.ns, p.ntheta, p.nzeta);
    f.czmn_o =
        cumes::RealFieldView<const T>(rs.d_czmn_o, p.ns, p.ntheta, p.nzeta);
    f.clmn_e =
        cumes::RealFieldView<const T>(rs.d_clmn_e, p.ns, p.ntheta, p.nzeta);
    f.clmn_o =
        cumes::RealFieldView<const T>(rs.d_clmn_o, p.ns, p.ntheta, p.nzeta);
    cumes::ConstraintForceViews<const T> cf;
    cf.frcon_e =
        cumes::RealFieldView<const T>(cw.d_frcon_e, p.ns, p.ntheta, p.nzeta);
    cf.frcon_o =
        cumes::RealFieldView<const T>(cw.d_frcon_o, p.ns, p.ntheta, p.nzeta);
    cf.fzcon_e =
        cumes::RealFieldView<const T>(cw.d_fzcon_e, p.ns, p.ntheta, p.nzeta);
    cf.fzcon_o =
        cumes::RealFieldView<const T>(cw.d_fzcon_o, p.ns, p.ntheta, p.nzeta);
    op.enqueue_forward(f, cf, ax_v, 0);

    const char* names[6] = {"frcc", "fzsc", "flsc", "frss", "fzcs", "flcs"};
    for (int c = 0; c < 6; ++c)
        compareArrays(names[c], d_gen + (size_t)c * n, d_ax + (size_t)c * n, n,
                      tolNear<T>());
    cudaFree(d_gen);
    cudaFree(d_ax);
}

// Constraint rzCon: rCon/zCon.
template <typename T>
static void testRzCon(DeviceParams<T>& p,
                      cumes::ToroidalFftOperator<T>& gen,
                      Cw<T>& cw,
                      cumes::SpectralStorage<T>& storage,
                      cumes::AxisymmetricOperator<T>& op) {
    std::cout << "  constraint rzCon (rCon/zCon) ...\n";
    const int n = p.ns * p.nZnT;
    // rCon/zCon reference is the fused inverse DFT (the production path); the
    // axisymmetric enqueue_rzcon is Class B ULP-equivalent to it.
    gen.inverse_fused(storage.physical_const(), /*do_combine=*/false, cw.d_rCon,
                      cw.d_zCon);
    T *d_ax_r = nullptr, *d_ax_z = nullptr;
    cc(cudaMalloc(&d_ax_r, (size_t)n * sizeof(T)), "ax rcon");
    cc(cudaMalloc(&d_ax_z, (size_t)n * sizeof(T)), "ax zcon");
    // The axisymmetric rzCon synthesis leaves the axis row zero (no poloidal
    // angle there); zero the outputs so the full-grid comparisons stay
    // initcheck-defined.
    cc(cudaMemset(d_ax_r, 0, (size_t)n * sizeof(T)), "ax rcon zero");
    cc(cudaMemset(d_ax_z, 0, (size_t)n * sizeof(T)), "ax zcon zero");
    op.enqueue_rzcon(storage.physical_const(),
                     cumes::RealFieldView<T>(d_ax_r, p.ns, p.ntheta, p.nzeta),
                     cumes::RealFieldView<T>(d_ax_z, p.ns, p.ntheta, p.nzeta),
                     0);
    compareArrays("rCon", cw.d_rCon, d_ax_r, n, tolNear<T>());
    compareArrays("zCon", cw.d_zCon, d_ax_z, n, tolNear<T>());
    cudaFree(d_ax_r);
    cudaFree(d_ax_z);
}

// One-null rzCon (review 3.3): the SpectralOperator interface documents
// rCon/zCon as "may be null views to skip that output". The axisymmetric
// backend must write only the requested output — an rCon-only (or zCon-only)
// caller must not fault and the produced output must match the generic fused
// inverse. Runs through enqueue_inverse (the documented entry point) with a
// valid geometry scratch so only the rzCon side is one-null.
template <typename T>
static void testRzConOneNull(DeviceParams<T>& p,
                             cumes::ToroidalFftOperator<T>& gen,
                             Cw<T>& cw,
                             cumes::SpectralStorage<T>& storage,
                             cumes::AxisymmetricOperator<T>& op) {
    std::cout << "  constraint rzCon one-null (skip-one-output) ...\n";
    const int n = p.ns * p.nZnT;
    gen.inverse_fused(storage.physical_const(), /*do_combine=*/false, cw.d_rCon,
                      cw.d_zCon);
    T* d_geom = nullptr;
    cc(cudaMalloc(&d_geom, (size_t)18 * n * sizeof(T)), "ax geom (one-null)");
    auto view = [&](int k) {
        return cumes::RealFieldView<T>(d_geom + (size_t)k * n, p.ns, p.ntheta,
                                       p.nzeta);
    };
    cumes::GeometryParityViews<T> g;
    g.r_e = view(0);
    g.z_e = view(1);
    g.l_e = view(2);
    g.ru_e = view(3);
    g.zu_e = view(4);
    g.lu_e = view(5);
    g.r_o = view(6);
    g.z_o = view(7);
    g.l_o = view(8);
    g.ru_o = view(9);
    g.zu_o = view(10);
    g.lu_o = view(11);
    g.rv_e = view(12);
    g.zv_e = view(13);
    g.lv_e = view(14);
    g.rv_o = view(15);
    g.zv_o = view(16);
    g.lv_o = view(17);

    auto run_one_null = [&](const char* label, bool rcon_null, const T* ref) {
        T* d_one = nullptr;
        cc(cudaMalloc(&d_one, (size_t)n * sizeof(T)), "ax one output");
        std::vector<T> poison((size_t)n, T(123.5));
        cc(cudaMemcpy(d_one, poison.data(), (size_t)n * sizeof(T),
                      cudaMemcpyHostToDevice),
           "poison one");
        op.enqueue_inverse(
            storage.physical_const(), g,
            rcon_null ? cumes::RealFieldView<T>()
                      : cumes::RealFieldView<T>(d_one, p.ns, p.ntheta, p.nzeta),
            rcon_null ? cumes::RealFieldView<T>(d_one, p.ns, p.ntheta, p.nzeta)
                      : cumes::RealFieldView<T>(),
            0);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            std::cerr << format("FAIL [{}] kernel error: {}\n", label,
                                cudaGetErrorString(e));
            ++failures();
        } else {
            compareArrays(label, ref, d_one, n, tolNear<T>());
        }
        cudaFree(d_one);
    };
    run_one_null("zCon (rCon null)", /*rcon_null=*/true, cw.d_zCon);
    run_one_null("rCon (zCon null)", /*rcon_null=*/false, cw.d_rCon);
    cudaFree(d_geom);
}

// Constraint bandpass: gCon (skip the axis row, which neither backend writes).
template <typename T>
static void testDealias(DeviceParams<T>& p,
                        cumes::ToroidalFftOperator<T>& gen,
                        Cw<T>& cw,
                        cumes::AxisymmetricOperator<T>& op) {
    std::cout << "  constraint bandpass (gCon) ...\n";
    const int n = p.ns * p.nZnT;
    gen.dealias_bandpass(cw.d_gConEff, cw.d_tcon, cw.d_faccon, cw.d_gCon);
    T* d_ax = nullptr;
    cc(cudaMalloc(&d_ax, (size_t)n * sizeof(T)), "ax gcon");
    cc(cudaMemset(d_ax, 0, (size_t)n * sizeof(T)), "ax gcon zero");
    op.enqueue_dealias(
        cumes::RealFieldView<const T>(cw.d_gConEff, p.ns, p.ntheta, p.nzeta),
        cw.d_tcon, cw.d_faccon,
        cumes::RealFieldView<T>(d_ax, p.ns, p.ntheta, p.nzeta), 0);
    // Both kernels leave the axis (surface 0) untouched; compare surfaces 1..
    std::vector<T> hg((size_t)n), ha((size_t)n);
    cc(cudaMemcpy(hg.data(), cw.d_gCon, (size_t)n * sizeof(T),
                  cudaMemcpyDeviceToHost),
       "cp g");
    cc(cudaMemcpy(ha.data(), d_ax, (size_t)n * sizeof(T),
                  cudaMemcpyDeviceToHost),
       "cp a");
    double maxd = 0.0;
    for (int j = 1; j < p.ns; ++j)
        for (int l = 0; l < p.nZnT; ++l)
            maxd = fmax(maxd, fabs((double)hg[(size_t)j * p.nZnT + l] -
                                   (double)ha[(size_t)j * p.nZnT + l]));
    if (maxd > tolNear<T>()) {
        std::cerr << format("FAIL [gCon] max abs diff = {:.3e}\n", maxd);
        ++failures();
    } else {
        std::cout << format("  PASS gCon (interior, max abs diff {:.3e})\n",
                            maxd);
    }
    cudaFree(d_ax);
}

template <typename T>
static int runTests() {
    std::cout << format("--- {} precision ---\n",
                        sizeof(T) == sizeof(double) ? "double" : "float");
    DeviceParams<T> p;
    p.ns = kNs;
    p.mnmax = kMnmax;
    p.ntheta = kNtheta;
    p.nzeta = kNzeta;
    p.nfp = kNfp;
    p.nZnT = kNZnT;
    p.mpol = kMpol;
    p.ntor = kNtor;
    p.ncurr = 0;
    p.delt = T(1.0);
    p.ftol = T(1e-14);
    p.max_iter = 10;
    p.lamscale = T(1.0);

    cumes::DeviceModeTable mt = cumes::modeTableCreate<T>(p);
    cumes::RealSpaceStorage<T> rs = realSpaceCreate(p);
    cumes::ToroidalFftOperator<T> gen(p, rs, mt);
    const size_t nF = (size_t)p.ns * p.nZnT;
    Cw<T> cw{};
    auto balloc = [&](T*& d, const char* tag) {
        cc(cudaMalloc(&d, nF * sizeof(T)), tag);
    };
    balloc(cw.d_frcon_e, "frcon_e");
    balloc(cw.d_frcon_o, "frcon_o");
    balloc(cw.d_fzcon_e, "fzcon_e");
    balloc(cw.d_fzcon_o, "fzcon_o");
    balloc(cw.d_tcon, "tcon");
    balloc(cw.d_gConEff, "gConEff");
    balloc(cw.d_faccon, "faccon");
    balloc(cw.d_gCon, "gCon");
    // The bandpass skips the axis row (no constraint on axis); zero the
    // output so the full-grid readback below stays initcheck-defined.
    cc(cudaMemset(cw.d_gCon, 0, nF * sizeof(T)), "gCon zero");
    balloc(cw.d_rCon, "rCon");
    balloc(cw.d_zCon, "zCon");
    // The fused-inverse rCon/zCon synthesis leaves the axis row to the
    // reconstruction's natural zero; zero the buffers so the full-grid
    // comparisons stay initcheck-defined.
    cc(cudaMemset(cw.d_rCon, 0, nF * sizeof(T)), "rCon zero");
    cc(cudaMemset(cw.d_zCon, 0, nF * sizeof(T)), "zCon zero");
    cumes::SpectralStorage<T> storage(p.ns, p.mnmax);
    cumes::AxisymmetricOperator<T> op(p);

    fillState(storage, p.ns, p.mnmax);
    fillForces(rs, cw, p.ns, p.nZnT);
    fillTcon(cw, p.ns);
    fillFaccon(cw, p.mpol);
    fillGconEff(cw, p.ns, p.nZnT);

    testInverse(p, rs, gen, storage, op);
    testForward(p, rs, gen, cw, op);
    testRzCon(p, gen, cw, storage, op);
    testRzConOneNull(p, gen, cw, storage, op);
    testDealias(p, gen, cw, op);

    realSpaceFree(rs);
    cudaFree(cw.d_frcon_e);
    cudaFree(cw.d_frcon_o);
    cudaFree(cw.d_fzcon_e);
    cudaFree(cw.d_fzcon_o);
    cudaFree(cw.d_tcon);
    cudaFree(cw.d_gConEff);
    cudaFree(cw.d_faccon);
    cudaFree(cw.d_gCon);
    cudaFree(cw.d_rCon);
    cudaFree(cw.d_zCon);
    cumes::modeTableFree(mt);
    return 0;
}

int main() {
    std::cout << "=== Axisymmetric backend differential test ===\n";
    runTests<double>();
    runTests<float>();
    std::cout << format(failures() == 0 ? "ALL PASS\n" : "{} FAILURES\n",
                        failures());
    return summary();
}
