// test_fourier.cu — algorithm correctness tests for parity-split DFT
// transforms. Build: cd cuMES/build && cmake .. && make test_fourier &&
// ./test_fourier Conventions (matching the solver): folded mode table mode =
// m*(ntor+1)+n, n = 0..ntor; inverse DFT with the plus-zmncs convention, lmncs
// coefficient, lv = -∂λ/∂ζ, per-mode mscale*nscale factor on lambda; forward
// DFT with w = mscale*nscale/nZnT and 6 spectral force components.
//
// The whole suite runs in BOTH double and float (the modules are templated on
// the scalar type T). The host CPU reference always computes in double from
// the T inputs; the float leg compares at ~1e-4, the double leg at the
// original 1e-12/1e-13/1e-10 tolerances.
#include "cumes/state/mode_table.cuh"
#include "cumes/state/spectral_storage.hpp"
#include "cumes/transforms/toroidal_fft_operator.hpp"
#include "cumes_test_cuda_helper.cuh"
#include "vmec_types.h"

#include <cmath>
#include <cstdlib>
#include <vector>
using namespace cumes::test;

constexpr int NS = 3, MPOL = 3, NTOR = 2, NTHETA = 16, NZETA = 8, NFP = 1;
constexpr int MNMAX = MPOL * (NTOR + 1), NZNT = NTHETA * NZETA;

static void check_near(double g,
                       double e,
                       double t,
                       const char* s,
                       int j,
                       int k) {
    if (fabs(g - e) > t) {
        std::cerr << format("FAIL [{}] j={} k={} got={:.15e} exp={:.15e}\n", s,
                            j, k, g, e);
        ++failures();
    }
}
static void check_mode(double g,
                       double e,
                       double t,
                       const char* s,
                       int j,
                       int m) {
    if (fabs(g - e) > t) {
        std::cerr << format("FAIL [{}] j={} m={} got={:.15e} exp={:.15e}\n", s,
                            j, m, g, e);
        ++failures();
    }
}

// Per-type comparison tolerances (float arithmetic cannot reach the double
// reference at 1e-12; 1e-4 is ~100x the float rounding floor of these sums).
template <typename T>
static constexpr double tol_inv() {
    return sizeof(T) == sizeof(float) ? 1e-4 : 1e-12;
}
template <typename T>
static constexpr double tol_fwd() {
    return sizeof(T) == sizeof(float) ? 1e-4 : 1e-13;
}
template <typename T>
static constexpr double tol_axis() {
    return sizeof(T) == sizeof(float) ? 1e-4 : 1e-10;
}

// Host copy of the basis tables (the test cannot read device pointers).
// Computed in double: the CPU reference always evaluates in double from the
// (possibly float) T inputs.
template <typename T>
static void host_basis(const DeviceParams<T>& p,
                       std::vector<double>& hcc,
                       std::vector<double>& hss,
                       std::vector<double>& hsc,
                       std::vector<double>& hcs,
                       std::vector<int>& hxm,
                       std::vector<int>& hxn) {
    hcc.assign(p.nZnT * p.mnmax, 0);
    hss.assign(p.nZnT * p.mnmax, 0);
    hsc.assign(p.nZnT * p.mnmax, 0);
    hcs.assign(p.nZnT * p.mnmax, 0);
    hxm.resize(p.mnmax);
    hxn.resize(p.mnmax);
    for (int m = 0; m < p.mpol; ++m)
        for (int n = 0; n < p.ntor + 1; ++n) {
            int mode = m * (p.ntor + 1) + n;
            hxm[mode] = m;
            hxn[mode] = n * p.nfp;
        }
    for (int it = 0; it < p.ntheta; ++it) {
        double th = 2 * M_PI * it / p.ntheta;
        for (int iz = 0; iz < p.nzeta; ++iz) {
            double ze = 2 * M_PI * iz / p.nzeta;
            int idx = iz * p.ntheta + it;
            for (int mode = 0; mode < p.mnmax; ++mode) {
                int m = hxm[mode], n = hxn[mode];
                double cm = cos(m * th), sm = sin(m * th), cn = cos(n * ze),
                       sn = sin(n * ze);
                hcc[idx + mode * p.nZnT] = cm * cn;
                hss[idx + mode * p.nZnT] = sm * sn;
                hsc[idx + mode * p.nZnT] = sm * cn;
                hcs[idx + mode * p.nZnT] = cm * sn;
            }
        }
    }
}

// CPU: parity inverse DFT matching inverseDFTKernel (always in double).
static void cpu_inv_dft(const double* cc_,
                        const double* ss_,
                        const double* zsc_,
                        const double* zcs_,
                        const double* lsc_,
                        const double* lcs_,
                        const double* pcc,
                        const double* pss,
                        const double* psc,
                        const double* pcs,
                        const int* xm,
                        const int* xn,
                        int ns,
                        int mnmax,
                        int nZnT,
                        double* r,
                        double* z,
                        double* l,
                        double* ru,
                        double* zu,
                        double* lu,
                        double* rv,
                        double* zv,
                        double* lv) {
    for (int j = 0; j < ns; ++j)
        for (int k = 0; k < nZnT; ++k) {
            double re = 0, ro = 0, ze = 0, zo = 0, le = 0, lo = 0, rue = 0,
                   ruo = 0, zue = 0, zuo = 0, lue = 0, luo = 0;
            double rve = 0, rvo = 0, zve = 0, zvo = 0, lve = 0, lvo = 0;
            for (int m = 0; m < mnmax; ++m) {
                double cc_v = pcc[k + m * nZnT], ss_v = pss[k + m * nZnT],
                       sc_v = psc[k + m * nZnT], cs_v = pcs[k + m * nZnT];
                int mi = xm[m], ni = xn[m];
                double rc = cc_[j + m * ns], rs = ss_[j + m * ns];
                double zs = zsc_[j + m * ns], zc = zcs_[j + m * ns];
                double ls = lsc_[j + m * ns], lcs = lcs_[j + m * ns];
                // No mscale*nscale on the lambda reconstruction: the lambda
                // STATE already carries ms*ns (state = vmecpp-decomposed *
                // ms*ns), so the real-space uses the raw basis (FIXED
                // 2026-08-02 to match the vmecpp-verified convention; the old
                // lfac double-counted).
                if (mi % 2 == 0) {
                    re += rc * cc_v + rs * ss_v;
                    rue += -mi * rc * sc_v + mi * rs * cs_v;
                    rve += -ni * rc * cs_v + ni * rs * sc_v;
                    ze += zs * sc_v + zc * cs_v;
                    zue += mi * zs * cc_v - mi * zc * ss_v;
                    zve += -ni * zs * ss_v + ni * zc * cc_v;
                    le += ls * sc_v + lcs * cs_v;
                    lue += mi * ls * cc_v - mi * lcs * ss_v;
                    lve += ni * ls * ss_v - ni * lcs * cc_v;
                } else {
                    ro += rc * cc_v + rs * ss_v;
                    ruo += -mi * rc * sc_v + mi * rs * cs_v;
                    rvo += -ni * rc * cs_v + ni * rs * sc_v;
                    zo += zs * sc_v + zc * cs_v;
                    zuo += mi * zs * cc_v - mi * zc * ss_v;
                    zvo += -ni * zs * ss_v + ni * zc * cc_v;
                    lo += ls * sc_v + lcs * cs_v;
                    luo += mi * ls * cc_v - mi * lcs * ss_v;
                    lvo += ni * ls * ss_v - ni * lcs * cc_v;
                }
            }
            double maxsc =
                fmax(sqrt((double)j / (ns - 1.0)), sqrt(1.0 / (ns - 1.0)));
            ro /= maxsc;
            ruo /= maxsc;
            rvo /= maxsc;
            zo /= maxsc;
            zuo /= maxsc;
            zvo /= maxsc;
            lo /= maxsc;
            luo /= maxsc;
            lvo /= maxsc;
            int idx = k + j * nZnT;
            r[idx] = re + ro;
            z[idx] = ze + zo;
            l[idx] = le + lo;
            ru[idx] = rue + ruo;
            zu[idx] = zue + zuo;
            lu[idx] = lue + luo;
            rv[idx] = rve + rvo;
            zv[idx] = zve + zvo;
            lv[idx] = lve + lvo;
        }
}

template <typename T>
static void gpu_inv(cumes::SpectralStorage<T>& storage,
                    cumes::ToroidalFftOperator<T>& op,
                    const DeviceParams<T>& p,
                    const T* cc_,
                    const T* ss_,
                    const T* zsc_,
                    const T* zcs_,
                    const T* lsc_,
                    const T* lcs_) {
    size_t nb = p.ns * p.mnmax * sizeof(T);
    cc(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Rcc), cc_, nb,
                  cudaMemcpyHostToDevice),
       "up cc");
    cc(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Rss), ss_, nb,
                  cudaMemcpyHostToDevice),
       "up ss");
    cc(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Zsc), zsc_, nb,
                  cudaMemcpyHostToDevice),
       "up zsc");
    cc(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Zcs), zcs_, nb,
                  cudaMemcpyHostToDevice),
       "up zcs");
    cc(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Lsc), lsc_, nb,
                  cudaMemcpyHostToDevice),
       "up lsc");
    cc(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Lcs), lcs_, nb,
                  cudaMemcpyHostToDevice),
       "up lcs");
    // family_ptr(Rcc) is the contiguous state-slab base (Rcc is component 0),
    // so a component-major view over it matches
    // SpectralStorage::physical_const().
    op.inverse(
        cumes::SpectralView<const T, cumes::PhysicalStateDomain>(
            storage.family_ptr(cumes::SpectralComponent::Rcc), p.ns, p.mnmax),
        /*do_combine=*/true);
}

template <typename T>
static int t_inv_constR(DeviceParams<T>& p,
                        cumes::ToroidalFftOperator<T>& op,
                        cumes::RealSpaceStorage<T>& rs,
                        cumes::SpectralStorage<T>& storage) {
    int lf = failures();
    std::cout << "  test_inverseDFT_constantR ... ";
    std::vector<T> cc_(p.ns * p.mnmax, T(0)), ss_(p.ns * p.mnmax, T(0)),
        zs(p.ns * p.mnmax, T(0)), zc(p.ns * p.mnmax, T(0)),
        ls_(p.ns * p.mnmax, T(0)), lcs(p.ns * p.mnmax, T(0));
    for (int j = 0; j < p.ns; ++j) cc_[j + 0 * p.ns] = T(4.0);  // R_00
    T *h_r = new T[p.ns * p.nZnT], *h_rv = new T[p.ns * p.nZnT];
    std::vector<double> r(p.ns * p.nZnT), z(p.ns * p.nZnT), l(p.ns * p.nZnT);
    std::vector<double> ru(p.ns * p.nZnT), zu(p.ns * p.nZnT), lu(p.ns * p.nZnT);
    std::vector<double> rv(p.ns * p.nZnT), zv(p.ns * p.nZnT), lv(p.ns * p.nZnT);
    gpu_inv(storage, op, p, cc_.data(), ss_.data(), zs.data(), zc.data(),
            ls_.data(), lcs.data());
    cc(cudaMemcpy(h_r, rs.d_r_real, p.ns * p.nZnT * sizeof(T),
                  cudaMemcpyDeviceToHost),
       "get r");
    cc(cudaMemcpy(h_rv, rs.d_rv_real, p.ns * p.nZnT * sizeof(T),
                  cudaMemcpyDeviceToHost),
       "get rv");
    std::vector<double> hcc, hss, hsc, hcs;
    std::vector<int> hxm, hxn;
    host_basis(p, hcc, hss, hsc, hcs, hxm, hxn);
    std::vector<double> cc_d(cc_.begin(), cc_.end()),
        ss_d(ss_.begin(), ss_.end());
    std::vector<double> zs_d(zs.begin(), zs.end()), zc_d(zc.begin(), zc.end());
    std::vector<double> ls_d(ls_.begin(), ls_.end()),
        lcs_d(lcs.begin(), lcs.end());
    cpu_inv_dft(cc_d.data(), ss_d.data(), zs_d.data(), zc_d.data(), ls_d.data(),
                lcs_d.data(), hcc.data(), hss.data(), hsc.data(), hcs.data(),
                hxm.data(), hxn.data(), p.ns, p.mnmax, p.nZnT, r.data(),
                z.data(), l.data(), ru.data(), zu.data(), lu.data(), rv.data(),
                zv.data(), lv.data());
    for (int i = 0; i < p.ns * p.nZnT; ++i) {
        check_near(h_r[i], r[i], tol_inv<T>(), "R", i / p.nZnT, i % p.nZnT);
        check_near(h_rv[i], rv[i], tol_inv<T>(), "Rv", i / p.nZnT, i % p.nZnT);
    }
    delete[] h_r;
    delete[] h_rv;
    std::cout << (failures() == lf ? "PASS\n" : "FAIL\n");
    return failures() - lf;
}

template <typename T>
static int t_inv_theta(DeviceParams<T>& p,
                       cumes::ToroidalFftOperator<T>& op,
                       cumes::RealSpaceStorage<T>& rs,
                       cumes::SpectralStorage<T>& storage) {
    int lf = failures();
    std::cout << "  test_inverseDFT_thetaDerivative ... ";
    std::vector<T> cc_(p.ns * p.mnmax, T(0)), ss_(p.ns * p.mnmax, T(0)),
        zs(p.ns * p.mnmax, T(0)), zc(p.ns * p.mnmax, T(0)),
        ls_(p.ns * p.mnmax, T(0)), lcs(p.ns * p.mnmax, T(0));
    int m1 = 1 * (p.ntor + 1) + 0;
    for (int j = 0; j < p.ns; ++j) cc_[j + m1 * p.ns] = T(0.3);  // R_10
    T *h_r = new T[p.ns * p.nZnT], *h_ru = new T[p.ns * p.nZnT];
    std::vector<double> r(p.ns * p.nZnT), z(p.ns * p.nZnT), l(p.ns * p.nZnT);
    std::vector<double> ru(p.ns * p.nZnT), zu(p.ns * p.nZnT), lu(p.ns * p.nZnT);
    std::vector<double> rv(p.ns * p.nZnT), zv(p.ns * p.nZnT), lv(p.ns * p.nZnT);
    gpu_inv(storage, op, p, cc_.data(), ss_.data(), zs.data(), zc.data(),
            ls_.data(), lcs.data());
    cc(cudaMemcpy(h_r, rs.d_r_real, p.ns * p.nZnT * sizeof(T),
                  cudaMemcpyDeviceToHost),
       "get r");
    cc(cudaMemcpy(h_ru, rs.d_ru_real, p.ns * p.nZnT * sizeof(T),
                  cudaMemcpyDeviceToHost),
       "get ru");
    std::vector<double> hcc, hss, hsc, hcs;
    std::vector<int> hxm, hxn;
    host_basis(p, hcc, hss, hsc, hcs, hxm, hxn);
    std::vector<double> cc_d(cc_.begin(), cc_.end()),
        ss_d(ss_.begin(), ss_.end());
    std::vector<double> zs_d(zs.begin(), zs.end()), zc_d(zc.begin(), zc.end());
    std::vector<double> ls_d(ls_.begin(), ls_.end()),
        lcs_d(lcs.begin(), lcs.end());
    cpu_inv_dft(cc_d.data(), ss_d.data(), zs_d.data(), zc_d.data(), ls_d.data(),
                lcs_d.data(), hcc.data(), hss.data(), hsc.data(), hcs.data(),
                hxm.data(), hxn.data(), p.ns, p.mnmax, p.nZnT, r.data(),
                z.data(), l.data(), ru.data(), zu.data(), lu.data(), rv.data(),
                zv.data(), lv.data());
    for (int i = 0; i < p.ns * p.nZnT; ++i) {
        check_near(h_r[i], r[i], tol_inv<T>(), "R", i / p.nZnT, i % p.nZnT);
        check_near(h_ru[i], ru[i], tol_inv<T>(), "Ru", i / p.nZnT, i % p.nZnT);
    }
    delete[] h_r;
    delete[] h_ru;
    std::cout << (failures() == lf ? "PASS\n" : "FAIL\n");
    return failures() - lf;
}

template <typename T>
static int t_inv_zeta(DeviceParams<T>& p,
                      cumes::ToroidalFftOperator<T>& op,
                      cumes::RealSpaceStorage<T>& rs,
                      cumes::SpectralStorage<T>& storage) {
    int lf = failures();
    std::cout << "  test_inverseDFT_zetaDerivative ... ";
    std::vector<T> cc_(p.ns * p.mnmax, T(0)), ss_(p.ns * p.mnmax, T(0)),
        zs(p.ns * p.mnmax, T(0)), zc(p.ns * p.mnmax, T(0)),
        ls_(p.ns * p.mnmax, T(0)), lcs(p.ns * p.mnmax, T(0));
    int m1 = 1 * (p.ntor + 1) + 1;  // R_11 (cos(θ-ζ)): folded rmncc=rmnss=0.2
    for (int j = 0; j < p.ns; ++j) {
        cc_[j + m1 * p.ns] = T(0.2);
        ss_[j + m1 * p.ns] = T(0.2);
    }
    T* h_rv = new T[p.ns * p.nZnT];
    std::vector<double> r(p.ns * p.nZnT), z(p.ns * p.nZnT), l(p.ns * p.nZnT);
    std::vector<double> ru(p.ns * p.nZnT), zu(p.ns * p.nZnT), lu(p.ns * p.nZnT);
    std::vector<double> rv(p.ns * p.nZnT), zv(p.ns * p.nZnT), lv(p.ns * p.nZnT);
    gpu_inv(storage, op, p, cc_.data(), ss_.data(), zs.data(), zc.data(),
            ls_.data(), lcs.data());
    cc(cudaMemcpy(h_rv, rs.d_rv_real, p.ns * p.nZnT * sizeof(T),
                  cudaMemcpyDeviceToHost),
       "get rv");
    std::vector<double> hcc, hss, hsc, hcs;
    std::vector<int> hxm, hxn;
    host_basis(p, hcc, hss, hsc, hcs, hxm, hxn);
    std::vector<double> cc_d(cc_.begin(), cc_.end()),
        ss_d(ss_.begin(), ss_.end());
    std::vector<double> zs_d(zs.begin(), zs.end()), zc_d(zc.begin(), zc.end());
    std::vector<double> ls_d(ls_.begin(), ls_.end()),
        lcs_d(lcs.begin(), lcs.end());
    cpu_inv_dft(cc_d.data(), ss_d.data(), zs_d.data(), zc_d.data(), ls_d.data(),
                lcs_d.data(), hcc.data(), hss.data(), hsc.data(), hcs.data(),
                hxm.data(), hxn.data(), p.ns, p.mnmax, p.nZnT, r.data(),
                z.data(), l.data(), ru.data(), zu.data(), lu.data(), rv.data(),
                zv.data(), lv.data());
    for (int i = 0; i < p.ns * p.nZnT; ++i)
        check_near(h_rv[i], rv[i], tol_inv<T>(), "Rv", i / p.nZnT, i % p.nZnT);
    delete[] h_rv;
    std::cout << (failures() == lf ? "PASS\n" : "FAIL\n");
    return failures() - lf;
}

template <typename T>
static int t_fwd_const(DeviceParams<T>& p,
                       cumes::ToroidalFftOperator<T>& op,
                       cumes::RealSpaceStorage<T>& rs) {
    int lf = failures();
    std::cout << "  test_forwardDFT_constant ... ";
    size_t nbr = p.ns * p.nZnT * sizeof(T);
    std::vector<T> fr(p.ns * p.nZnT, T(3.0));
    std::vector<T> fs(6 * p.ns * p.mnmax, T(0));
    T* d_fs;
    cc(cudaMalloc(&d_fs, 6 * p.ns * p.mnmax * sizeof(T)), "fs");
    T *frcon_e, *frcon_o, *fzcon_e, *fzcon_o;
    size_t nfc = (size_t)p.ns * p.nZnT * sizeof(T);
    cudaMalloc(&frcon_e, nfc);
    cudaMemset(frcon_e, 0, nfc);
    cudaMalloc(&frcon_o, nfc);
    cudaMemset(frcon_o, 0, nfc);
    cudaMalloc(&fzcon_e, nfc);
    cudaMemset(fzcon_e, 0, nfc);
    cudaMalloc(&fzcon_o, nfc);
    cudaMemset(fzcon_o, 0, nfc);
    // armn_e = 3.0 on all surfaces; everything else zero.
    cc(cudaMemset(rs.d_armn_e, 0, nbr), "ms");
    cc(cudaMemcpy(rs.d_armn_e, fr.data(), nbr, cudaMemcpyHostToDevice),
       "armn_e");
    cc(cudaMemset(rs.d_armn_o, 0, nbr), "ms");
    cc(cudaMemset(rs.d_azmn_e, 0, nbr), "ms");
    cc(cudaMemset(rs.d_azmn_o, 0, nbr), "ms");
    cc(cudaMemset(rs.d_brmn_e, 0, nbr), "ms");
    cc(cudaMemset(rs.d_brmn_o, 0, nbr), "ms");
    cc(cudaMemset(rs.d_bzmn_e, 0, nbr), "ms");
    cc(cudaMemset(rs.d_bzmn_o, 0, nbr), "ms");
    cc(cudaMemset(rs.d_blmn_e, 0, nbr), "ms");
    cc(cudaMemset(rs.d_blmn_o, 0, nbr), "ms");
    cc(cudaMemset(rs.d_crmn_e, 0, nbr), "ms");
    cc(cudaMemset(rs.d_crmn_o, 0, nbr), "ms");
    cc(cudaMemset(rs.d_czmn_e, 0, nbr), "ms");
    cc(cudaMemset(rs.d_czmn_o, 0, nbr), "ms");
    cc(cudaMemset(rs.d_clmn_e, 0, nbr), "ms");
    cc(cudaMemset(rs.d_clmn_o, 0, nbr), "ms");
    op.forward(cumes::SpectralView<T, cumes::DecomposedResidualDomain>(
                   d_fs, p.ns, p.mnmax),
               frcon_e, frcon_o, fzcon_e, fzcon_o);
    cc(cudaMemcpy(fs.data(), d_fs, 6 * p.ns * p.mnmax * sizeof(T),
                  cudaMemcpyDeviceToHost),
       "get fs");
    for (int j = 0; j < p.ns - 1; ++j) {
        check_near((double)fs[j + 0 * p.mnmax * p.ns], 3.0, tol_fwd<T>(),
                   "fR_cc", j, 0);
        for (int m = 1; m < p.mnmax; ++m) {
            check_mode((double)fs[j + m * p.ns + 0 * p.mnmax * p.ns], 0.0,
                       tol_fwd<T>(), "fR0", j, m);
            check_mode((double)fs[j + m * p.ns + 3 * p.mnmax * p.ns], 0.0,
                       tol_fwd<T>(), "fR0_ss", j, m);
        }
    }
    cudaFree(d_fs);
    cudaFree(frcon_e);
    cudaFree(frcon_o);
    cudaFree(fzcon_e);
    cudaFree(fzcon_o);
    std::cout << (failures() == lf ? "PASS\n" : "FAIL\n");
    return failures() - lf;
}

template <typename T>
static int t_fwd_sine(DeviceParams<T>& p,
                      cumes::ToroidalFftOperator<T>& op,
                      cumes::RealSpaceStorage<T>& rs) {
    int lf = failures();
    std::cout << "  test_forwardDFT_sine ... ";
    size_t nbr = p.ns * p.nZnT * sizeof(T);
    std::vector<T> fz(p.ns * p.nZnT, T(0));
    std::vector<T> fs(6 * p.ns * p.mnmax, T(0));
    T* d_fs;
    cc(cudaMalloc(&d_fs, 6 * p.ns * p.mnmax * sizeof(T)), "fs");
    T *frcon_e, *frcon_o, *fzcon_e, *fzcon_o;
    size_t nfc = (size_t)p.ns * p.nZnT * sizeof(T);
    cudaMalloc(&frcon_e, nfc);
    cudaMemset(frcon_e, 0, nfc);
    cudaMalloc(&frcon_o, nfc);
    cudaMemset(frcon_o, 0, nfc);
    cudaMalloc(&fzcon_e, nfc);
    cudaMemset(fzcon_e, 0, nfc);
    cudaMalloc(&fzcon_o, nfc);
    cudaMemset(fzcon_o, 0, nfc);
    // F_Z = sin(θ)cos(ζ) on interior surfaces: picked up by fzsc of (1,1).
    // m=1 is ODD parity: the input lives in azmn_o (parity-split arrays).
    for (int j = 0; j < p.ns - 1; ++j)
        for (int k = 0; k < p.nZnT; ++k) {
            int it = k % p.ntheta, iz = k / p.ntheta;
            double th = 2 * M_PI * it / p.ntheta, ze = 2 * M_PI * iz / p.nzeta;
            fz[k + j * p.nZnT] = T(sin(th) * cos(ze));
        }
    cc(cudaMemset(rs.d_armn_e, 0, nbr), "ms");
    cc(cudaMemset(rs.d_armn_o, 0, nbr), "ms");
    cc(cudaMemset(rs.d_azmn_e, 0, nbr), "ms");
    cc(cudaMemcpy(rs.d_azmn_o, fz.data(), nbr, cudaMemcpyHostToDevice),
       "azmn_o");
    cc(cudaMemset(rs.d_brmn_e, 0, nbr), "ms");
    cc(cudaMemset(rs.d_brmn_o, 0, nbr), "ms");
    cc(cudaMemset(rs.d_bzmn_e, 0, nbr), "ms");
    cc(cudaMemset(rs.d_bzmn_o, 0, nbr), "ms");
    cc(cudaMemset(rs.d_blmn_e, 0, nbr), "ms");
    cc(cudaMemset(rs.d_blmn_o, 0, nbr), "ms");
    cc(cudaMemset(rs.d_crmn_e, 0, nbr), "ms");
    cc(cudaMemset(rs.d_crmn_o, 0, nbr), "ms");
    cc(cudaMemset(rs.d_czmn_e, 0, nbr), "ms");
    cc(cudaMemset(rs.d_czmn_o, 0, nbr), "ms");
    cc(cudaMemset(rs.d_clmn_e, 0, nbr), "ms");
    cc(cudaMemset(rs.d_clmn_o, 0, nbr), "ms");
    op.forward(cumes::SpectralView<T, cumes::DecomposedResidualDomain>(
                   d_fs, p.ns, p.mnmax),
               frcon_e, frcon_o, fzcon_e, fzcon_o);
    cc(cudaMemcpy(fs.data(), d_fs, 6 * p.ns * p.mnmax * sizeof(T),
                  cudaMemcpyDeviceToHost),
       "get fs");
    int m11 = 1 * (p.ntor + 1) + 1;
    // vmecpp convention: the reduced-grid trapezoid with mscale*nscale
    // weights gives the normalized coefficient 0.5 for a unit raw mode
    // (mscale*nscale = 2, sum sin^2*cos^2 over the reduced grid = 1/2).
    // The axis (j=0) is m=0 only (vmecpp dft_ForcesToFourier mmax=1).
    for (int j = 0; j < p.ns - 1; ++j) {
        double exp = (j == 0) ? 0.0 : 0.5;
        check_near((double)fs[j + m11 * p.ns + 1 * p.mnmax * p.ns], exp,
                   tol_fwd<T>(), "fZ_sc", j, m11);
    }
    cudaFree(d_fs);
    cudaFree(frcon_e);
    cudaFree(frcon_o);
    cudaFree(fzcon_e);
    cudaFree(fzcon_o);
    std::cout << (failures() == lf ? "PASS\n" : "FAIL\n");
    return failures() - lf;
}

template <typename T>
static int t_gpuVcpu_inv(DeviceParams<T>& p,
                         cumes::ToroidalFftOperator<T>& op,
                         cumes::RealSpaceStorage<T>& rs,
                         cumes::SpectralStorage<T>& storage) {
    int lf = failures();
    std::cout << "  test_gpuVcpu_inverseDFT ... ";
    std::vector<T> cc_(p.ns * p.mnmax, T(0)), ss_(p.ns * p.mnmax, T(0)),
        zs(p.ns * p.mnmax, T(0)), zc(p.ns * p.mnmax, T(0)),
        ls_(p.ns * p.mnmax, T(0)), lcs(p.ns * p.mnmax, T(0));
    for (int j = 0; j < p.ns; ++j)
        for (int m = 0; m < p.mnmax; ++m) {
            cc_[j + m * p.ns] = T(0.001 * (m + 1) * (j + 1));
            ss_[j + m * p.ns] = T(0.002 * (m + 1) * (j + 1));
            zs[j + m * p.ns] = T(0.003 * (m + 1) * (j + 1));
            zc[j + m * p.ns] = T(0.004 * (m + 1) * (j + 1));
            ls_[j + m * p.ns] = T(0.005 * (m + 1) * (j + 1));
            lcs[j + m * p.ns] = T(0.006 * (m + 1) * (j + 1));
        }
    T *h_r = new T[p.ns * p.nZnT], *h_z = new T[p.ns * p.nZnT];
    T *h_l = new T[p.ns * p.nZnT], *h_ru = new T[p.ns * p.nZnT];
    T *h_zu = new T[p.ns * p.nZnT], *h_lu = new T[p.ns * p.nZnT];
    T *h_rv = new T[p.ns * p.nZnT], *h_zv = new T[p.ns * p.nZnT];
    T* h_lv = new T[p.ns * p.nZnT];
    std::vector<double> r(p.ns * p.nZnT), z(p.ns * p.nZnT), l(p.ns * p.nZnT);
    std::vector<double> ru(p.ns * p.nZnT), zu(p.ns * p.nZnT), lu(p.ns * p.nZnT);
    std::vector<double> rv(p.ns * p.nZnT), zv(p.ns * p.nZnT), lv(p.ns * p.nZnT);
    gpu_inv(storage, op, p, cc_.data(), ss_.data(), zs.data(), zc.data(),
            ls_.data(), lcs.data());
    cc(cudaMemcpy(h_r, rs.d_r_real, p.ns * p.nZnT * sizeof(T),
                  cudaMemcpyDeviceToHost),
       "get r");
    cc(cudaMemcpy(h_z, rs.d_z_real, p.ns * p.nZnT * sizeof(T),
                  cudaMemcpyDeviceToHost),
       "get z");
    cc(cudaMemcpy(h_l, rs.d_l_real, p.ns * p.nZnT * sizeof(T),
                  cudaMemcpyDeviceToHost),
       "get l");
    cc(cudaMemcpy(h_ru, rs.d_ru_real, p.ns * p.nZnT * sizeof(T),
                  cudaMemcpyDeviceToHost),
       "get ru");
    cc(cudaMemcpy(h_zu, rs.d_zu_real, p.ns * p.nZnT * sizeof(T),
                  cudaMemcpyDeviceToHost),
       "get zu");
    cc(cudaMemcpy(h_lu, rs.d_lu_real, p.ns * p.nZnT * sizeof(T),
                  cudaMemcpyDeviceToHost),
       "get lu");
    cc(cudaMemcpy(h_rv, rs.d_rv_real, p.ns * p.nZnT * sizeof(T),
                  cudaMemcpyDeviceToHost),
       "get rv");
    cc(cudaMemcpy(h_zv, rs.d_zv_real, p.ns * p.nZnT * sizeof(T),
                  cudaMemcpyDeviceToHost),
       "get zv");
    cc(cudaMemcpy(h_lv, rs.d_lv_real, p.ns * p.nZnT * sizeof(T),
                  cudaMemcpyDeviceToHost),
       "get lv");
    std::vector<double> hcc, hss, hsc, hcs;
    std::vector<int> hxm, hxn;
    host_basis(p, hcc, hss, hsc, hcs, hxm, hxn);
    std::vector<double> cc_d(cc_.begin(), cc_.end()),
        ss_d(ss_.begin(), ss_.end());
    std::vector<double> zs_d(zs.begin(), zs.end()), zc_d(zc.begin(), zc.end());
    std::vector<double> ls_d(ls_.begin(), ls_.end()),
        lcs_d(lcs.begin(), lcs.end());
    cpu_inv_dft(cc_d.data(), ss_d.data(), zs_d.data(), zc_d.data(), ls_d.data(),
                lcs_d.data(), hcc.data(), hss.data(), hsc.data(), hcs.data(),
                hxm.data(), hxn.data(), p.ns, p.mnmax, p.nZnT, r.data(),
                z.data(), l.data(), ru.data(), zu.data(), lu.data(), rv.data(),
                zv.data(), lv.data());
    for (int i = 0; i < p.ns * p.nZnT; ++i) {
        check_near(h_r[i], r[i], tol_inv<T>(), "R", i / p.nZnT, i % p.nZnT);
        check_near(h_z[i], z[i], tol_inv<T>(), "Z", i / p.nZnT, i % p.nZnT);
        check_near(h_l[i], l[i], tol_inv<T>(), "L", i / p.nZnT, i % p.nZnT);
        check_near(h_ru[i], ru[i], tol_inv<T>(), "Ru", i / p.nZnT, i % p.nZnT);
        check_near(h_zu[i], zu[i], tol_inv<T>(), "Zu", i / p.nZnT, i % p.nZnT);
        check_near(h_lu[i], lu[i], tol_inv<T>(), "Lu", i / p.nZnT, i % p.nZnT);
        check_near(h_rv[i], rv[i], tol_inv<T>(), "Rv", i / p.nZnT, i % p.nZnT);
        check_near(h_zv[i], zv[i], tol_inv<T>(), "Zv", i / p.nZnT, i % p.nZnT);
        check_near(h_lv[i], lv[i], tol_inv<T>(), "Lv", i / p.nZnT, i % p.nZnT);
    }
    delete[] h_r;
    delete[] h_z;
    delete[] h_l;
    delete[] h_ru;
    delete[] h_zu;
    delete[] h_lu;
    delete[] h_rv;
    delete[] h_zv;
    delete[] h_lv;
    std::cout << (failures() == lf ? "PASS\n" : "FAIL\n");
    return failures() - lf;
}

// A/B cross-check: the direct-sum and cuFFT backends must agree on identical
// input at ~1e-10 relative (both are the same linear map, differing only in
// floating-point summation order).
// host with the same reduced-grid trapezoid as the kernels.
template <typename T>
static int t_fwd_axis(DeviceParams<T>& p,
                      cumes::ToroidalFftOperator<T>& op,
                      cumes::RealSpaceStorage<T>& rs) {
    int lf = failures();
    std::cout << "  test_forwardDFT_axis ... ";
    size_t nbr = p.ns * p.nZnT * sizeof(T);
    std::vector<T> fr(p.ns * p.nZnT, T(0));
    for (int k = 0; k < p.nZnT; ++k) {
        double ze = 2 * M_PI * (k / p.ntheta) / p.nzeta;
        fr[k + 0 * p.nZnT] = T(2.0 + cos(ze));  // armn_e at axis
    }
    cc(cudaMemset(rs.d_armn_e, 0, nbr), "ms");
    cc(cudaMemcpy(rs.d_armn_e, fr.data(), nbr, cudaMemcpyHostToDevice),
       "armn_e");
    for (int k = 0; k < p.nZnT; ++k) {
        double ze = 2 * M_PI * (k / p.ntheta) / p.nzeta;
        fr[k + 0 * p.nZnT] = T(sin(ze));  // crmn_e at axis
    }
    cc(cudaMemset(rs.d_crmn_e, 0, nbr), "ms");
    cc(cudaMemcpy(rs.d_crmn_e, fr.data(), nbr, cudaMemcpyHostToDevice),
       "crmn_e");
    for (int k = 0; k < p.nZnT; ++k) {
        double ze = 2 * M_PI * (k / p.ntheta) / p.nzeta;
        fr[k + 0 * p.nZnT] = T(sin(ze));  // azmn_e at axis
    }
    cc(cudaMemset(rs.d_azmn_e, 0, nbr), "ms");
    cc(cudaMemcpy(rs.d_azmn_e, fr.data(), nbr, cudaMemcpyHostToDevice),
       "azmn_e");
    for (int k = 0; k < p.nZnT; ++k) {
        double ze = 2 * M_PI * (k / p.ntheta) / p.nzeta;
        fr[k + 0 * p.nZnT] = T(cos(ze));  // czmn_e at axis
    }
    cc(cudaMemset(rs.d_czmn_e, 0, nbr), "ms");
    cc(cudaMemcpy(rs.d_czmn_e, fr.data(), nbr, cudaMemcpyHostToDevice),
       "czmn_e");
    T* d_fs;
    cc(cudaMalloc(&d_fs, 6 * p.ns * p.mnmax * sizeof(T)), "fs");
    T *frcon_e, *frcon_o, *fzcon_e, *fzcon_o;
    size_t nfc = (size_t)p.ns * p.nZnT * sizeof(T);
    cudaMalloc(&frcon_e, nfc);
    cudaMemset(frcon_e, 0, nfc);
    cudaMalloc(&frcon_o, nfc);
    cudaMemset(frcon_o, 0, nfc);
    cudaMalloc(&fzcon_e, nfc);
    cudaMemset(fzcon_e, 0, nfc);
    cudaMalloc(&fzcon_o, nfc);
    cudaMemset(fzcon_o, 0, nfc);
    op.forward(cumes::SpectralView<T, cumes::DecomposedResidualDomain>(
                   d_fs, p.ns, p.mnmax),
               frcon_e, frcon_o, fzcon_e, fzcon_o);
    std::vector<T> fs(6 * p.ns * p.mnmax);
    cc(cudaMemcpy(fs.data(), d_fs, 6 * p.ns * p.mnmax * sizeof(T),
                  cudaMemcpyDeviceToHost),
       "get fs");
    std::vector<double> hcc, hss, hsc, hcs;
    std::vector<int> hxm, hxn;
    host_basis(p, hcc, hss, hsc, hcs, hxm, hxn);
    // Host expectation for mode (0,1) at the axis (nfp=1 -> n_ibp=1).
    int mode = 0 * (p.ntor + 1) + 1;
    int nThetaRed = p.ntheta / 2 + 1;
    double intNorm = 1.0 / ((double)p.nzeta * (nThetaRed - 1));
    double frcc = 0, fzcs = 0;
    for (int k = 0; k < p.nZnT; ++k) {
        int it = k % p.ntheta;
        if (it >= nThetaRed) continue;
        double w = intNorm;
        if (it == 0 || it == nThetaRed - 1) w *= 0.5;
        int iz = k / p.ntheta;
        double ze = 2 * M_PI * iz / p.nzeta;
        frcc += ((2.0 + cos(ze)) * hcc[k + mode * p.nZnT] +
                 sin(ze) * hcs[k + mode * p.nZnT]) *
                w;
        fzcs += (sin(ze) * hcs[k + mode * p.nZnT] -
                 cos(ze) * hcc[k + mode * p.nZnT]) *
                w;
    }
    const double sq2 = sqrt(2.0);  // nscale for n=1; mscale=1 for m=0
    frcc *= sq2;
    fzcs *= sq2;
    check_mode((double)fs[0 * p.ns + mode * p.ns + 0 * p.mnmax * p.ns], frcc,
               tol_axis<T>(), "axis_frcc", 0, mode);
    check_mode((double)fs[0 * p.ns + mode * p.ns + 4 * p.mnmax * p.ns], fzcs,
               tol_axis<T>(), "axis_fzcs", 0, mode);
    // Poloidal m>0 at the axis must stay zero (mode index >= ntor+1; the
    // (0,n) modes ARE computed at the axis).
    for (int m = p.ntor + 1; m < p.mnmax; ++m) {
        check_mode((double)fs[0 + m * p.ns + 0 * p.mnmax * p.ns], 0.0,
                   tol_fwd<T>(), "axis_m>0", 0, m);
        check_mode((double)fs[0 + m * p.ns + 4 * p.mnmax * p.ns], 0.0,
                   tol_fwd<T>(), "axis_m>0_z", 0, m);
    }
    cudaFree(d_fs);
    cudaFree(frcon_e);
    cudaFree(frcon_o);
    cudaFree(fzcon_e);
    cudaFree(fzcon_o);
    std::cout << (failures() == lf ? "PASS\n" : "FAIL\n");
    return failures() - lf;
}

// LCFS branch (j=ns-1 keeps only the λ components flsc/flcs).
template <typename T>
static int t_fwd_lcfs(DeviceParams<T>& p,
                      cumes::ToroidalFftOperator<T>& op,
                      cumes::RealSpaceStorage<T>& rs) {
    int lf = failures();
    std::cout << "  test_forwardDFT_lcfs ... ";
    size_t nbr = p.ns * p.nZnT * sizeof(T);
    int jB = p.ns - 1;
    std::vector<T> fr(p.ns * p.nZnT, T(0));
    // blmn_o (m=1 is odd) at the LCFS: 1 + sin(θ)cos(ζ); clmn_o: cos(θ)sin(ζ)
    for (int k = 0; k < p.nZnT; ++k) {
        int it = k % p.ntheta, iz = k / p.ntheta;
        double th = 2 * M_PI * it / p.ntheta, ze = 2 * M_PI * iz / p.nzeta;
        fr[k + jB * p.nZnT] = T(1.0 + sin(th) * cos(ze));
    }
    cc(cudaMemset(rs.d_blmn_o, 0, nbr), "ms");
    cc(cudaMemcpy(rs.d_blmn_o, fr.data(), nbr, cudaMemcpyHostToDevice),
       "blmn_o");
    for (int k = 0; k < p.nZnT; ++k) {
        int it = k % p.ntheta, iz = k / p.ntheta;
        double th = 2 * M_PI * it / p.ntheta, ze = 2 * M_PI * iz / p.nzeta;
        fr[k + jB * p.nZnT] = T(cos(th) * sin(ze));
    }
    cc(cudaMemset(rs.d_clmn_o, 0, nbr), "ms");
    cc(cudaMemcpy(rs.d_clmn_o, fr.data(), nbr, cudaMemcpyHostToDevice),
       "clmn_o");
    T* d_fs;
    cc(cudaMalloc(&d_fs, 6 * p.ns * p.mnmax * sizeof(T)), "fs");
    T *frcon_e, *frcon_o, *fzcon_e, *fzcon_o;
    size_t nfc = (size_t)p.ns * p.nZnT * sizeof(T);
    cudaMalloc(&frcon_e, nfc);
    cudaMemset(frcon_e, 0, nfc);
    cudaMalloc(&frcon_o, nfc);
    cudaMemset(frcon_o, 0, nfc);
    cudaMalloc(&fzcon_e, nfc);
    cudaMemset(fzcon_e, 0, nfc);
    cudaMalloc(&fzcon_o, nfc);
    cudaMemset(fzcon_o, 0, nfc);
    op.forward(cumes::SpectralView<T, cumes::DecomposedResidualDomain>(
                   d_fs, p.ns, p.mnmax),
               frcon_e, frcon_o, fzcon_e, fzcon_o);
    std::vector<T> fs(6 * p.ns * p.mnmax);
    cc(cudaMemcpy(fs.data(), d_fs, 6 * p.ns * p.mnmax * sizeof(T),
                  cudaMemcpyDeviceToHost),
       "get fs");
    std::vector<double> hcc, hss, hsc, hcs;
    std::vector<int> hxm, hxn;
    host_basis(p, hcc, hss, hsc, hcs, hxm, hxn);
    int mode = 1 * (p.ntor + 1) + 1;  // (m,n)=(1,1)
    int nThetaRed = p.ntheta / 2 + 1;
    double intNorm = 1.0 / ((double)p.nzeta * (nThetaRed - 1));
    double flsc = 0, flcs = 0;
    for (int k = 0; k < p.nZnT; ++k) {
        int it = k % p.ntheta;
        if (it >= nThetaRed) continue;
        double w = intNorm;
        if (it == 0 || it == nThetaRed - 1) w *= 0.5;
        int iz = k / p.ntheta;
        double th = 2 * M_PI * it / p.ntheta, ze = 2 * M_PI * iz / p.nzeta;
        double blmn = 1.0 + sin(th) * cos(ze), clmn = cos(th) * sin(ze);
        flsc += (blmn * (1.0 * hcc[k + mode * p.nZnT]) +
                 clmn * (1.0 * hss[k + mode * p.nZnT])) *
                w;
        flcs += (blmn * (-1.0 * hss[k + mode * p.nZnT]) +
                 clmn * (-1.0 * hcc[k + mode * p.nZnT])) *
                w;
    }
    const double sq2 = sqrt(2.0);  // mscale=nscale=√2 for (1,1)
    flsc *= 2.0;
    flcs *= 2.0;
    check_mode((double)fs[jB + mode * p.ns + 2 * p.mnmax * p.ns], flsc,
               tol_axis<T>(), "lcfs_flsc", jB, mode);
    check_mode((double)fs[jB + mode * p.ns + 5 * p.mnmax * p.ns], flcs,
               tol_axis<T>(), "lcfs_flcs", jB, mode);
    // R/Z components at the LCFS stay zero.
    for (int c = 0; c < 6; ++c) {
        if (c == 2 || c == 5) continue;
        check_mode((double)fs[jB + mode * p.ns + c * p.mnmax * p.ns], 0.0,
                   tol_fwd<T>(), "lcfs_rz", jB, mode);
    }
    cudaFree(d_fs);
    cudaFree(frcon_e);
    cudaFree(frcon_o);
    cudaFree(fzcon_e);
    cudaFree(fzcon_o);
    std::cout << (failures() == lf ? "PASS\n" : "FAIL\n");
    return failures() - lf;
}

// ---------------------------------------------------------------------------
// Regression tests added 2026-08-17 for the code-review findings:
//   1.1 forward_reduce_kernel dropped reduced-theta points beyond lane 15
//       (fixed 16-lane block) -> t_fwd_theta32 below (nThetaRed = 17).
//   1.2 mpol=2 made the de-alias plan batch 2*(mpol-2)*(ns-1) zero ->
//       cufftPlanMany(batch=0) threw at construction -> t_mpol2_construct.
//   3.6 enqueue_inverse/enqueue_forward ignored the passed view bundles
//       (used the captured rs_) -> t_enqueue_*_views below.
// ---------------------------------------------------------------------------

// 1.1: forward quadrature over nThetaRed = ntheta/2+1 = 17 > 16 lanes.
// F_R = cos(theta)cos(zeta) on the interior surface (m=1 -> odd parity
// armn_o). The reduced-grid endpoint theta=pi contributes cos^2 = 1, so a
// truncated quadrature (dropping l >= 16) measurably changes frcc(1,1).
template <typename T>
static int t_fwd_theta32() {
    int lf = failures();
    std::cout << "  test_forwardDFT_theta32 (nThetaRed>16 coverage) ... ";
    DeviceParams<T> p;
    p.ns = 3;
    p.mnmax = 3 * (2 + 1);
    p.ntheta = 32;
    p.nzeta = 8;  // nThetaRed = 17
    p.nfp = 1;
    p.nZnT = 32 * 8;
    p.mpol = 3;
    p.ntor = 2;
    p.ncurr = 0;
    p.delt = T(1.0);
    p.ftol = T(1e-14);
    p.max_iter = 10;
    p.lamscale = T(1.0);
    cumes::DeviceModeTable mt = cumes::mode_table_create<T>(p);
    cumes::RealSpaceStorage<T> rs = real_space_create(p);
    cumes::ToroidalFftOperator<T> op(p, rs, mt);
    size_t nbr = (size_t)p.ns * p.nZnT * sizeof(T);
    std::vector<T> fr(p.ns * p.nZnT, T(0));
    for (int k = 0; k < p.nZnT; ++k) {
        int it = k % p.ntheta, iz = k / p.ntheta;
        double th = 2 * M_PI * it / p.ntheta, ze = 2 * M_PI * iz / p.nzeta;
        fr[k + 1 * p.nZnT] = T(cos(th) * cos(ze));
    }
    cc(cudaMemset(rs.d_armn_e, 0, nbr), "ms");
    cc(cudaMemset(rs.d_armn_o, 0, nbr), "ms");
    cc(cudaMemset(rs.d_azmn_e, 0, nbr), "ms");
    cc(cudaMemset(rs.d_azmn_o, 0, nbr), "ms");
    cc(cudaMemset(rs.d_brmn_e, 0, nbr), "ms");
    cc(cudaMemset(rs.d_brmn_o, 0, nbr), "ms");
    cc(cudaMemset(rs.d_bzmn_e, 0, nbr), "ms");
    cc(cudaMemset(rs.d_bzmn_o, 0, nbr), "ms");
    cc(cudaMemset(rs.d_blmn_e, 0, nbr), "ms");
    cc(cudaMemset(rs.d_blmn_o, 0, nbr), "ms");
    cc(cudaMemset(rs.d_crmn_e, 0, nbr), "ms");
    cc(cudaMemset(rs.d_crmn_o, 0, nbr), "ms");
    cc(cudaMemset(rs.d_czmn_e, 0, nbr), "ms");
    cc(cudaMemset(rs.d_czmn_o, 0, nbr), "ms");
    cc(cudaMemset(rs.d_clmn_e, 0, nbr), "ms");
    cc(cudaMemset(rs.d_clmn_o, 0, nbr), "ms");
    cc(cudaMemcpy(rs.d_armn_o, fr.data(), nbr, cudaMemcpyHostToDevice),
       "armn_o");
    T* d_fs;
    cc(cudaMalloc(&d_fs, 6 * p.ns * p.mnmax * sizeof(T)), "fs");
    T *frcon_e, *frcon_o, *fzcon_e, *fzcon_o;
    size_t nfc = (size_t)p.ns * p.nZnT * sizeof(T);
    cudaMalloc(&frcon_e, nfc);
    cudaMemset(frcon_e, 0, nfc);
    cudaMalloc(&frcon_o, nfc);
    cudaMemset(frcon_o, 0, nfc);
    cudaMalloc(&fzcon_e, nfc);
    cudaMemset(fzcon_e, 0, nfc);
    cudaMalloc(&fzcon_o, nfc);
    cudaMemset(fzcon_o, 0, nfc);
    op.forward(cumes::SpectralView<T, cumes::DecomposedResidualDomain>(
                   d_fs, p.ns, p.mnmax),
               frcon_e, frcon_o, fzcon_e, fzcon_o);
    std::vector<T> fs(6 * p.ns * p.mnmax);
    cc(cudaMemcpy(fs.data(), d_fs, 6 * p.ns * p.mnmax * sizeof(T),
                  cudaMemcpyDeviceToHost),
       "get fs");
    // CPU reference (the reduced-grid trapezoid of CLAUDE.md "Forward DFT
    // normalization"): frcc(1,1) = mscale*nscale * [sum_l w_l cos^2(theta_l)]
    // * [sum_iz cos^2(zeta_iz)] with w_l = intNorm, endpoints halved, over
    // ALL nThetaRed = 17 points (including the half-weight theta=pi point).
    int m11 = 1 * (p.ntor + 1) + 1;
    int nThetaRed = p.ntheta / 2 + 1;
    double intNorm = 1.0 / ((double)p.nzeta * (nThetaRed - 1));
    double sumth = 0, sumze = 0;
    for (int l = 0; l < nThetaRed; ++l) {
        double w = intNorm;
        if (l == 0 || l == nThetaRed - 1) w *= 0.5;
        double th = 2 * M_PI * l / p.ntheta;
        sumth += w * cos(th) * cos(th);
    }
    for (int iz = 0; iz < p.nzeta; ++iz) {
        double ze = 2 * M_PI * iz / p.nzeta;
        sumze += cos(ze) * cos(ze);
    }
    double expect =
        2.0 * sumth * sumze;  // mscale*nscale = sqrt2*sqrt2 for (1,1)
    check_mode((double)fs[1 + m11 * p.ns + 0 * p.mnmax * p.ns], expect,
               tol_fwd<T>(), "theta32_frcc", 1, m11);
    cudaFree(d_fs);
    cudaFree(frcon_e);
    cudaFree(frcon_o);
    cudaFree(fzcon_e);
    cudaFree(fzcon_o);
    real_space_free(rs);
    cumes::mode_table_free(mt);
    std::cout << (failures() == lf ? "PASS\n" : "FAIL\n");
    return failures() - lf;
}

// 1.2: the validated minimum mpol=2 must construct (no batch-0 de-alias
// plans) and the de-alias bandpass (empty pass band) must zero gCon on j>=1.
template <typename T>
static int t_mpol2_construct() {
    int lf = failures();
    std::cout << "  test_mpol2_dealias_plans ... ";
    DeviceParams<T> p;
    p.ns = 3;
    p.mnmax = 2 * (2 + 1);
    p.ntheta = 16;
    p.nzeta = 8;
    p.nfp = 1;
    p.nZnT = 16 * 8;
    p.mpol = 2;
    p.ntor = 2;  // validated minimum mpol
    p.ncurr = 0;
    p.delt = T(1.0);
    p.ftol = T(1e-14);
    p.max_iter = 10;
    p.lamscale = T(1.0);
    cumes::DeviceModeTable mt = cumes::mode_table_create<T>(p);
    cumes::RealSpaceStorage<T> rs = real_space_create(p);
    bool threw = false, ok = true;
    try {
        cumes::ToroidalFftOperator<T> op(p, rs, mt);
        // The skipped de-alias plans must not break stream binding.
        op.bind_stream(0);
        // Empty pass band (no m in [1, mpol-2]): gCon rows j>=1 -> zero,
        // the axis row untouched (the bandpass kernels skip jF==0).
        std::vector<T> zero(p.ns * p.nZnT, T(0));
        T *d_eff, *d_g, *d_tcon, *d_faccon;
        cc(cudaMalloc(&d_eff, zero.size() * sizeof(T)), "eff");
        cc(cudaMalloc(&d_g, zero.size() * sizeof(T)), "g");
        cc(cudaMalloc(&d_tcon, p.ns * sizeof(T)), "tcon");
        cc(cudaMalloc(&d_faccon, p.mpol * sizeof(T)), "faccon");
        cc(cudaMemcpy(d_eff, zero.data(), zero.size() * sizeof(T),
                      cudaMemcpyHostToDevice),
           "up eff");
        std::vector<T> tone(p.ns, T(1)), fone(p.mpol, T(1));
        cc(cudaMemcpy(d_tcon, tone.data(), p.ns * sizeof(T),
                      cudaMemcpyHostToDevice),
           "up tcon");
        cc(cudaMemcpy(d_faccon, fone.data(), p.mpol * sizeof(T),
                      cudaMemcpyHostToDevice),
           "up faccon");
        std::vector<T> seed(p.ns * p.nZnT, T(7));
        cc(cudaMemcpy(d_g, seed.data(), seed.size() * sizeof(T),
                      cudaMemcpyHostToDevice),
           "up g");
        op.dealias_bandpass(d_eff, d_tcon, d_faccon, d_g, 0);
        std::vector<T> back(p.ns * p.nZnT);
        cc(cudaMemcpy(back.data(), d_g, back.size() * sizeof(T),
                      cudaMemcpyDeviceToHost),
           "get g");
        for (int j = 1; j < p.ns && ok; ++j)
            for (int k = 0; k < p.nZnT && ok; ++k)
                if (back[j * p.nZnT + k] != T(0)) ok = false;
        if (!ok) std::cerr << "FAIL [mpol2] gCon not zeroed on j>=1\n";
        cudaFree(d_eff);
        cudaFree(d_g);
        cudaFree(d_tcon);
        cudaFree(d_faccon);
    } catch (const std::exception& e) {
        threw = true;
        std::cerr << format("FAIL [mpol2] construction threw: {}\n", e.what());
    }
    if (threw)
        ++failures();
    else if (!ok)
        ++failures();
    real_space_free(rs);
    cumes::mode_table_free(mt);
    std::cout << (failures() == lf ? "PASS\n" : "FAIL\n");
    return failures() - lf;
}

// 3.6: enqueue_inverse must write the CALLER-PASSED geometry views (not the
// captured rs_). Reference: the rs_-backed inverse(); target: a non-aliasing
// scratch bundle seeded with -1 (unwritten before the fix).
template <typename T>
static int t_enqueue_inverse_views(DeviceParams<T>& p,
                                   cumes::ToroidalFftOperator<T>& op,
                                   cumes::RealSpaceStorage<T>& rs,
                                   cumes::SpectralStorage<T>& storage) {
    int lf = failures();
    std::cout << "  test_enqueue_inverse_honors_views ... ";
    size_t nb = p.ns * p.mnmax, nn = p.ns * p.nZnT;
    std::vector<T> cc_(nb, T(0)), ss_(nb, T(0)), zs(nb, T(0)), zc(nb, T(0)),
        ls_(nb, T(0)), lcs(nb, T(0));
    for (int j = 0; j < p.ns; ++j)
        for (int m = 0; m < p.mnmax; ++m) {
            cc_[j + m * p.ns] = T(0.001 * (m + 1) * (j + 1));
            ss_[j + m * p.ns] = T(0.002 * (m + 1) * (j + 1));
            zs[j + m * p.ns] = T(0.003 * (m + 1) * (j + 1));
            zc[j + m * p.ns] = T(0.004 * (m + 1) * (j + 1));
            ls_[j + m * p.ns] = T(0.005 * (m + 1) * (j + 1));
            lcs[j + m * p.ns] = T(0.006 * (m + 1) * (j + 1));
        }
    size_t nbT = nb * sizeof(T);
    cc(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Rcc), cc_.data(),
                  nbT, cudaMemcpyHostToDevice),
       "up cc");
    cc(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Rss), ss_.data(),
                  nbT, cudaMemcpyHostToDevice),
       "up ss");
    cc(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Zsc), zs.data(),
                  nbT, cudaMemcpyHostToDevice),
       "up zsc");
    cc(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Zcs), zc.data(),
                  nbT, cudaMemcpyHostToDevice),
       "up zcs");
    cc(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Lsc), ls_.data(),
                  nbT, cudaMemcpyHostToDevice),
       "up lsc");
    cc(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Lcs), lcs.data(),
                  nbT, cudaMemcpyHostToDevice),
       "up lcs");
    cumes::SpectralView<const T, cumes::PhysicalStateDomain> coeff(
        storage.family_ptr(cumes::SpectralComponent::Rcc), p.ns, p.mnmax);
    // Reference: the rs_-backed primitive.
    op.inverse(coeff, /*do_combine=*/false);
    std::vector<T> ref(18 * nn);
    const T* rp[18] = {rs.d_r_e,  rs.d_z_e,  rs.d_l_e,  rs.d_ru_e, rs.d_zu_e,
                       rs.d_lu_e, rs.d_r_o,  rs.d_z_o,  rs.d_l_o,  rs.d_ru_o,
                       rs.d_zu_o, rs.d_lu_o, rs.d_rv_e, rs.d_zv_e, rs.d_lv_e,
                       rs.d_rv_o, rs.d_zv_o, rs.d_lv_o};
    const char* nm[18] = {"r_e",  "z_e",  "l_e",  "ru_e", "zu_e", "lu_e",
                          "r_o",  "z_o",  "l_o",  "ru_o", "zu_o", "lu_o",
                          "rv_e", "zv_e", "lv_e", "rv_o", "zv_o", "lv_o"};
    for (int c = 0; c < 18; ++c)
        cc(cudaMemcpy(ref.data() + c * nn, rp[c], nn * sizeof(T),
                      cudaMemcpyDeviceToHost),
           "get ref");
    // Non-aliasing scratch bundle seeded with -1 (proves the views are
    // written).
    T* d_ax = nullptr;
    cc(cudaMalloc(&d_ax, (size_t)18 * nn * sizeof(T)), "ax geom");
    std::vector<T> seed(18 * nn, T(-1));
    cc(cudaMemcpy(d_ax, seed.data(), (size_t)18 * nn * sizeof(T),
                  cudaMemcpyHostToDevice),
       "seed ax");
    auto view = [&](int k) {
        return cumes::RealFieldView<T>(d_ax + (size_t)k * nn, p.ns, p.ntheta,
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
    op.enqueue_inverse(coeff, g, cumes::RealFieldView<T>(),
                       cumes::RealFieldView<T>(), 0);
    std::vector<T> ax(18 * nn);
    cc(cudaMemcpy(ax.data(), d_ax, (size_t)18 * nn * sizeof(T),
                  cudaMemcpyDeviceToHost),
       "get ax");
    for (int c = 0; c < 18; ++c)
        for (size_t i = 0; i < nn; ++i)
            check_near((double)ax[c * nn + i], (double)ref[c * nn + i],
                       tol_inv<T>(), nm[c], (int)(i / p.nZnT),
                       (int)(i % p.nZnT));
    cudaFree(d_ax);
    std::cout << (failures() == lf ? "PASS\n" : "FAIL\n");
    return failures() - lf;
}

// 3.6: enqueue_forward must read the CALLER-PASSED force views. rs_ holds a
// nonzero armn_e pattern; the passed views alias an all-zero buffer, so the
// honoring implementation returns an all-zero residual (the pre-fix one reads
// rs_ and returns the nonzero reference instead).
template <typename T>
static int t_enqueue_forward_views(DeviceParams<T>& p,
                                   cumes::ToroidalFftOperator<T>& op,
                                   cumes::RealSpaceStorage<T>& rs) {
    int lf = failures();
    std::cout << "  test_enqueue_forward_honors_views ... ";
    size_t nn = p.ns * p.nZnT, nbr = nn * sizeof(T);
    std::vector<T> fr(nn, T(3.0));
    cc(cudaMemset(rs.d_armn_e, 0, nbr), "ms");
    cc(cudaMemcpy(rs.d_armn_e, fr.data(), nbr, cudaMemcpyHostToDevice),
       "armn_e");
    T* d_fs;
    cc(cudaMalloc(&d_fs, 6 * p.ns * p.mnmax * sizeof(T)), "fs");
    // Zero views over a separate buffer + zero constraint force.
    T* d_zero = nullptr;
    cc(cudaMalloc(&d_zero, nn * sizeof(T)), "zero");
    cc(cudaMemset(d_zero, 0, nn * sizeof(T)), "ms zero");
    auto view = [&](int) {
        return cumes::RealFieldView<const T>(d_zero, p.ns, p.ntheta, p.nzeta);
    };
    cumes::ForceParityViews<const T> f;
    f.armn_e = view(0);
    f.armn_o = view(1);
    f.azmn_e = view(2);
    f.azmn_o = view(3);
    f.brmn_e = view(4);
    f.brmn_o = view(5);
    f.bzmn_e = view(6);
    f.bzmn_o = view(7);
    f.blmn_e = view(8);
    f.blmn_o = view(9);
    f.crmn_e = view(10);
    f.crmn_o = view(11);
    f.czmn_e = view(12);
    f.czmn_o = view(13);
    f.clmn_e = view(14);
    f.clmn_o = view(15);
    cumes::ConstraintForceViews<const T> cf;
    cf.frcon_e = view(16);
    cf.frcon_o = view(16);
    cf.fzcon_e = view(16);
    cf.fzcon_o = view(16);
    op.enqueue_forward(f, cf,
                       cumes::SpectralView<T, cumes::DecomposedResidualDomain>(
                           d_fs, p.ns, p.mnmax),
                       0);
    std::vector<T> fs(6 * p.ns * p.mnmax);
    cc(cudaMemcpy(fs.data(), d_fs, 6 * p.ns * p.mnmax * sizeof(T),
                  cudaMemcpyDeviceToHost),
       "get fs");
    // Zero input -> zero residual (the recover kernel writes all six families
    // explicitly, so the exact-zero check is meaningful).
    int bad = 0;
    for (size_t i = 0; i < fs.size(); ++i)
        if (fs[i] != T(0)) ++bad;
    if (bad > 0)
        std::cerr << format("FAIL [fwdviews] {} nonzero entries (expected 0)\n",
                            bad);
    failures() += (bad > 0) ? 1 : 0;
    cudaFree(d_fs);
    cudaFree(d_zero);
    std::cout << (failures() == lf ? "PASS\n" : "FAIL\n");
    return failures() - lf;
}

// Run the whole suite for one scalar type T.
template <typename T>
static int run_tests() {
    std::cout << format("--- {} precision ---\n",
                        sizeof(T) == sizeof(double) ? "double" : "float");
    DeviceParams<T> p;
    p.ns = NS;
    p.mnmax = MNMAX;
    p.ntheta = NTHETA;
    p.nzeta = NZETA;
    p.nfp = NFP;
    p.nZnT = NZNT;
    p.mpol = MPOL;
    p.ntor = NTOR;
    p.ncurr = 0;
    p.delt = T(1.0);
    p.ftol = T(1e-14);
    p.max_iter = 10;
    p.lamscale = T(1.0);

    cumes::DeviceModeTable mt = cumes::mode_table_create<T>(p);
    cumes::RealSpaceStorage<T> rs = real_space_create(p);
    cumes::SpectralStorage<T> storage(p.ns, p.mnmax);
    cumes::ToroidalFftOperator<T> op(p, rs, mt);

    // All tests compare the cuFFT backend against the backend-independent
    // host CPU reference (the direct-sum kernels were removed after the
    // cuFFT A/B validation; the git history has them).
    int nf = 0;
    nf += t_inv_constR(p, op, rs, storage);
    nf += t_inv_theta(p, op, rs, storage);
    nf += t_inv_zeta(p, op, rs, storage);
    nf += t_fwd_const(p, op, rs);
    nf += t_fwd_sine(p, op, rs);
    nf += t_gpuVcpu_inv(p, op, rs, storage);
    nf += t_fwd_axis(p, op, rs);
    nf += t_fwd_lcfs(p, op, rs);
    nf += t_fwd_theta32<T>();
    nf += t_mpol2_construct<T>();
    nf += t_enqueue_inverse_views(p, op, rs, storage);
    nf += t_enqueue_forward_views(p, op, rs);

    real_space_free(rs);
    cumes::mode_table_free(mt);

    return nf;
}

int main() {
    std::cout << "=== Fourier Transform Tests (folded n>=0 basis) ===\n";
    int nf = 0;
    nf += run_tests<double>();
    nf += run_tests<float>();
    failures() = nf;
    std::cout << (failures() == 0 ? "ALL PASS\n"
                                  : format("{} FAILURES\n", failures()));
    return summary();
}
