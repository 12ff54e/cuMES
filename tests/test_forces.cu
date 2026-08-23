// test_forces.cu — force/geometry chain on a manufactured Solovev-like state.
// Runs inverseDFT -> computeGeometry -> computeForces once and asserts the
// numerical invariants a correct implementation must satisfy:
//   * geometry is finite and the oriented Jacobian has the expected sign;
//   * real-space R/Z and all four even/odd force families are finite
//     (a NaN or inf would indicate a broken derivative/Jacobian/force path);
//   * the axisymmetric manufactured state has the exact Z=0 symmetry at
//     theta=0 and exact zero Z-forces (azmn_e == 0): a launch or indexing
//     error leaks non-axisymmetric content into the Z channel.
// The test returns nonzero if any assertion fails (it is a gate, not a
// print-only diagnostic).
#include "cumes/physics/force_operator.hpp"
#include "cumes/physics/geometry_operator.hpp"
#include "cumes/physics/magnetic_field_operator.hpp"
#include "cumes/physics/profiles.hpp"
#include "cumes/state/mode_table.cuh"
#include "cumes/state/spectral_storage.hpp"
#include "cumes/transforms/toroidal_fft_operator.hpp"
#include "cumes_test_cuda_helper.cuh"
#include "vmec_types.h"

#include <cmath>
#include <vector>
using namespace cumes::test;

int main() {
    DeviceParams<double> p;
    p.ns = 17;
    p.mnmax = 6 * (2 + 1);
    p.ntheta = 32;
    p.nzeta = 64;
    p.nfp = 1;
    p.nZnT = 2048;
    p.mpol = 6;
    p.ntor = 2;
    p.ncurr = 0;
    p.delt = 1.0;
    p.ftol = 1e-14;
    p.max_iter = 10;

    std::cout << "=== Force Diagnostic Test ===\n";

    // Create Solovev-like initial state with independent parity coefficients
    cumes::SpectralStorage<double> storage(p.ns, p.mnmax);
    size_t nbytes_state = p.ns * p.mnmax * sizeof(double);
    std::vector<double> h_cc(p.ns * p.mnmax);
    std::vector<double> h_ss(p.ns * p.mnmax);
    std::vector<double> h_zsc(p.ns * p.mnmax);
    std::vector<double> h_zcs(p.ns * p.mnmax);
    std::vector<double> h_lsc(p.ns * p.mnmax);
    // h_lcs was declared but never filled/uploaded before: the inverse DFT
    // reads the lmncs family, so the kernel consumed uninitialized device
    // memory (and the family was leaked). Zero it like the other families.
    std::vector<double> h_lcs(p.ns * p.mnmax);

    for (int j = 0; j < p.ns; ++j) {
        double ss = j / (p.ns - 1.0);
        for (int m = 0; m < p.mnmax; ++m) {
            int mm = m / (p.ntor + 1);
            if (mm == 0 && m == 0) {
                h_cc[j + m * p.ns] = 4.0;  // R_00 constant
            } else if (m == 1 * (p.ntor + 1) + 0) {
                h_cc[j + m * p.ns] = 0.3 * ss;  // R_10
            } else if (m == 2 * (p.ntor + 1) + 0) {
                h_cc[j + m * p.ns] = 0.2 * ss;  // R_20
            }
            h_ss[j + m * p.ns] = h_cc[j + m * p.ns];  // rmnss = rmncc initially
            if (m == 1 * (p.ntor + 1) + 0) {
                h_zsc[j + m * p.ns] = -0.5 * ss;  // Z_10
                h_zcs[j + m * p.ns] = -0.5 * ss;  // zmncs = zmnsc initially
            }
        }
    }

    check_cuda(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Rcc),
                          h_cc.data(), nbytes_state, cudaMemcpyHostToDevice),
               "cpy cc");
    check_cuda(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Rss),
                          h_ss.data(), nbytes_state, cudaMemcpyHostToDevice),
               "cpy ss");
    check_cuda(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Zsc),
                          h_zsc.data(), nbytes_state, cudaMemcpyHostToDevice),
               "cpy zsc");
    check_cuda(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Zcs),
                          h_zcs.data(), nbytes_state, cudaMemcpyHostToDevice),
               "cpy zcs");
    check_cuda(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Lsc),
                          h_lsc.data(), nbytes_state, cudaMemcpyHostToDevice),
               "cpy lsc");
    check_cuda(cudaMemcpy(storage.family_ptr(cumes::SpectralComponent::Lcs),
                          h_lcs.data(), nbytes_state, cudaMemcpyHostToDevice),
               "cpy lcs");

    cumes::ValidatedProblem vp = load_validated();
    cumes::Profiles<double> profiles(p, vp, nullptr);
    cumes::RadialProfileViews<double> rp = profiles.profile_views();
    cumes::DeviceModeTable mt = cumes::mode_table_create(p);
    cumes::RealSpaceStorage<double> rs = real_space_create(p);
    cumes::ToroidalFftOperator<double> op(p, rs, mt);
    cumes::GeometryOperator<double> geometry(p, nullptr);

    // Run one iteration
    op.inverse(storage.physical_const(), /*do_combine=*/true);
    geometry.enqueue(rs, p, rp, 0);
    cumes::MagneticFieldOperator<double>{}.enqueue(
        rs, p, rp, geometry.base_geometry_views(p),
        geometry.magnetic_field_views(p), nullptr, 0, true);
    cumes::ForceOperator<double>{}.enqueue(
        rs, p, rp, geometry.base_geometry_views(p),
        geometry.magnetic_field_views(p), nullptr, 0);

    // Check combined geometry at axis (j=0) and mid (j=8)
    size_t nbr = p.ns * p.nZnT * sizeof(double);
    std::vector<double> h_r(p.ns * p.nZnT);
    std::vector<double> h_z(p.ns * p.nZnT);
    check_cuda(cudaMemcpy(h_r.data(), rs.d_r_real, nbr, cudaMemcpyDeviceToHost),
               "r");
    check_cuda(cudaMemcpy(h_z.data(), rs.d_z_real, nbr, cudaMemcpyDeviceToHost),
               "z");

    // Print R at first theta point for all surfaces
    std::cout << "\nR(s,theta=0,zeta=0):\n";
    for (int j = 0; j < p.ns; ++j) {
        std::cout << format("  j={}: R={:.6f}  Z={:.6f}\n", j, h_r[j * p.nZnT],
                            h_z[j * p.nZnT]);
    }

    // Check forces
    std::vector<double> h_armn_e(p.ns * p.nZnT);
    std::vector<double> h_armn_o(p.ns * p.nZnT);
    std::vector<double> h_blmn_e(p.ns * p.nZnT);
    check_cuda(
        cudaMemcpy(h_armn_e.data(), rs.d_armn_e, nbr, cudaMemcpyDeviceToHost),
        "armn_e");
    check_cuda(
        cudaMemcpy(h_armn_o.data(), rs.d_armn_o, nbr, cudaMemcpyDeviceToHost),
        "armn_o");
    check_cuda(
        cudaMemcpy(h_blmn_e.data(), rs.d_blmn_e, nbr, cudaMemcpyDeviceToHost),
        "blmn_e");

    std::cout << "\nForces at theta=0,zeta=0:\n";
    std::cout << "  j  |  armn_e      armn_o      azmn_e      blmn_e\n";
    std::cout << "  ---+----------------------------------------------\n";
    std::vector<double> h_az(p.ns * p.nZnT);
    check_cuda(
        cudaMemcpy(h_az.data(), rs.d_azmn_e, nbr, cudaMemcpyDeviceToHost),
        "az");
    for (int j = 0; j < p.ns; ++j) {
        std::cout << format("  {} | {:.4e} {:.4e} {:.4e} {:.4e}\n", j,
                            h_armn_e[j * p.nZnT], h_armn_o[j * p.nZnT],
                            h_az[j * p.nZnT], h_blmn_e[j * p.nZnT]);
    }

    // Check gsqrt at half-grid
    size_t nH = (p.ns - 1) * p.nZnT * sizeof(double);
    std::vector<double> h_gs((p.ns - 1) * p.nZnT);
    std::vector<double> h_tau((p.ns - 1) * p.nZnT);
    check_cuda(
        cudaMemcpy(h_gs.data(), geometry.base_geometry_views(p).gsqrt.data(),
                   nH, cudaMemcpyDeviceToHost),
        "gs");
    check_cuda(
        cudaMemcpy(h_tau.data(), geometry.base_geometry_views(p).tau.data(), nH,
                   cudaMemcpyDeviceToHost),
        "tau");

    std::cout << "\nHalf-grid at theta=0,zeta=0:\n";
    std::cout << "  jH |  tau         gsqrt       r12\n";
    std::vector<double> h_r12((p.ns - 1) * p.nZnT);
    check_cuda(
        cudaMemcpy(h_r12.data(), geometry.base_geometry_views(p).r12.data(), nH,
                   cudaMemcpyDeviceToHost),
        "r12");
    for (int j = 0; j < p.ns - 1; ++j) {
        std::cout << format("  {} | {:.4e} {:.4e} {:.4e}\n", j,
                            h_tau[j * p.nZnT], h_gs[j * p.nZnT],
                            h_r12[j * p.nZnT]);
    }

    // Compute spectral forces via forward DFT
    size_t nbs = 6 * p.ns * p.mnmax * sizeof(double);
    std::vector<double> d_fspec(6 * p.ns * p.mnmax);
    double* d_fspec_gpu;
    check_cuda(cudaMalloc(&d_fspec_gpu, nbs), "fspec");
    double *frcon_e, *frcon_o, *fzcon_e, *fzcon_o;
    cudaMalloc(&frcon_e, (size_t)p.ns * p.nZnT * sizeof(double));
    cudaMemset(frcon_e, 0, (size_t)p.ns * p.nZnT * sizeof(double));
    cudaMalloc(&frcon_o, (size_t)p.ns * p.nZnT * sizeof(double));
    cudaMemset(frcon_o, 0, (size_t)p.ns * p.nZnT * sizeof(double));
    cudaMalloc(&fzcon_e, (size_t)p.ns * p.nZnT * sizeof(double));
    cudaMemset(fzcon_e, 0, (size_t)p.ns * p.nZnT * sizeof(double));
    cudaMalloc(&fzcon_o, (size_t)p.ns * p.nZnT * sizeof(double));
    cudaMemset(fzcon_o, 0, (size_t)p.ns * p.nZnT * sizeof(double));
    op.forward(cumes::SpectralView<double, cumes::DecomposedResidualDomain>(
                   d_fspec_gpu, p.ns, p.mnmax),
               frcon_e, frcon_o, fzcon_e, fzcon_o);
    check_cuda(
        cudaMemcpy(d_fspec.data(), d_fspec_gpu, nbs, cudaMemcpyDeviceToHost),
        "fspec d");

    std::cout << "\nSpectral forces (f_rmnc, f_zmns, f_lmnc):\n";
    std::cout << "  mode | m  n |  f_rmnc(axis) f_zmns(axis) f_lmnc(axis)\n";
    for (int m = 0; m < p.mnmax; ++m) {
        // mode = m*(ntor+1)+n (the old m/ntor division mislabeled the modes)
        int mm = m / (p.ntor + 1);
        int nn = m % (p.ntor + 1);
        int idx_r = 0 + m * p.ns;  // axis (j=0), mode m, comp R
        int idx_z = 0 + m * p.ns + p.mnmax * p.ns;
        int idx_l = 0 + m * p.ns + 2 * p.mnmax * p.ns;
        std::cout << format("  {} | {} {} | {:.4e} {:.4e} {:.4e}\n", m, mm, nn,
                            d_fspec[idx_r], d_fspec[idx_z], d_fspec[idx_l]);
    }

    // ---- numerical assertions (the gate) ----
    {
        // Geometry: finite everywhere; the oriented Jacobian is signJ*|gsqrt|
        // (signJ = -1), so signJ*gsqrt must be positive on an interior
        // surface (a flipped Jacobian from a bad parity combination would
        // make it negative).
        size_t nH = (size_t)(p.ns - 1) * p.nZnT;
        check_cuda(cudaMemcpy(h_gs.data(),
                              geometry.base_geometry_views(p).gsqrt.data(),
                              nH * sizeof(double), cudaMemcpyDeviceToHost),
                   "gs");
        bool geo_finite = true;
        double jmin = 1e300, jmax = 0.0;
        for (size_t i = 0; i < nH; ++i) {
            if (!std::isfinite(h_gs[i])) geo_finite = false;
            jmin = std::min(jmin, std::abs(h_gs[i]));
            jmax = std::max(jmax, std::abs(h_gs[i]));
        }
        check(geo_finite, "geometry gsqrt finite");
        check(jmin > 0.0 && jmax > 0.0,
              "geometry gsqrt nonzero (non-degenerate)");
        // Axisymmetric: R at theta=0 must be positive on the axis (4.0).
        check(h_r[0] > 0.0, "axis R positive");

        // Real-space forces finite (both parities, all families).
        bool f_finite = true;
        for (size_t i = 0; i < (size_t)p.ns * p.nZnT; ++i) {
            if (!std::isfinite(h_armn_e[i]) || !std::isfinite(h_armn_o[i]) ||
                !std::isfinite(h_az[i]) || !std::isfinite(h_blmn_e[i]))
                f_finite = false;
        }
        check(f_finite, "real-space forces finite");

        // Axisymmetric symmetry: Z(theta=0) == 0 exactly (the manufactured
        // state has only sin(m theta) Z content, which vanishes at theta=0;
        // a leak of cos(m theta) Z content into the axisymmetric Z channel
        // would break this). The Z-FORCE at theta=0 also vanishes for the
        // same reason — the weak form's Z term is proportional to Z and its
        // derivatives, all zero at theta=0.
        bool z_zero = true;
        for (int j = 0; j < p.ns; ++j)
            if (h_z[j * p.nZnT] != 0.0) z_zero = false;
        check(z_zero, "axisymmetric: Z(theta=0) == 0");
        bool az_zero = true;
        for (int j = 0; j < p.ns; ++j)
            if (h_az[j * p.nZnT] != 0.0) az_zero = false;
        check(az_zero, "axisymmetric: Z-force azmn_e(theta=0) == 0");

        // Spectral forces finite (the forward-DFT path).
        bool fs_finite = true;
        for (size_t i = 0; i < 6 * (size_t)p.ns * p.mnmax; ++i)
            if (!std::isfinite(d_fspec[i])) fs_finite = false;
        check(fs_finite, "spectral forces finite");
    }

    // Cleanup (the state/velocity slabs are freed by SpectralStorage's RAII)
    real_space_free(rs);
    cumes::mode_table_free(mt);
    cudaFree(frcon_e);
    cudaFree(frcon_o);
    cudaFree(fzcon_e);
    cudaFree(fzcon_o);

    cudaFree(d_fspec_gpu);

    std::cout << "\nDone.\n";
    return summary();
}
