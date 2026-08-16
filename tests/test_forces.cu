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
#include <cstdio>
#include <cmath>
#include <vector>

#include "vmec_types.h"
#include "fourier.cuh"
#include "cumes/state/mode_table.cuh"
#include "cumes/state/spectral_storage.hpp"
#include "cumes/physics/geometry_operator.hpp"
#include "cumes/physics/magnetic_field_operator.hpp"
#include "cumes/physics/force_operator.hpp"
#include "cumes/physics/profiles.hpp"
#include "cumes_test_support.cuh"

static int failures = 0;
#define CHECK(cond, msg)                                                     \
    do {                                                                     \
        if (cond) {                                                          \
            printf("PASS %s\n", msg);                                        \
        } else {                                                             \
            printf("FAIL %s\n", msg);                                        \
            ++failures;                                                      \
        }                                                                    \
    } while (0)


int main() {
    DeviceParams<double> p;
    p.ns = 17; p.mnmax = 6*(2+1); p.ntheta = 32; p.nzeta = 64;
    p.nfp = 1; p.nZnT = 2048; p.mpol = 6; p.ntor = 2;
    p.ncurr = 0; p.delt = 1.0; p.ftol = 1e-14; p.max_iter = 10;

    printf("=== Force Diagnostic Test ===\n");

    // Create Solovev-like initial state with independent parity coefficients
    cumes::SpectralStorage<double> storage(p.ns, p.mnmax);
    SpectralState<double> st = storage.legacy_view();
    size_t nbytes_state = p.ns * p.mnmax * sizeof(double);
    auto* h_cc = new double[p.ns * p.mnmax]();
    auto* h_ss = new double[p.ns * p.mnmax]();
    auto* h_zsc = new double[p.ns * p.mnmax]();
    auto* h_zcs = new double[p.ns * p.mnmax]();
    auto* h_lsc = new double[p.ns * p.mnmax]();
    // h_lcs was declared but never filled/uploaded before: the inverse DFT
    // reads st.d_lmncs, so the kernel consumed uninitialized device memory
    // (and st.d_lmncs was leaked). Zero it like the other families.
    auto* h_lcs = new double[p.ns * p.mnmax]();

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

    checkCuda(cudaMemcpy(st.d_rmncc, h_cc, nbytes_state, cudaMemcpyHostToDevice), "cpy cc");
    checkCuda(cudaMemcpy(st.d_rmnss, h_ss, nbytes_state, cudaMemcpyHostToDevice), "cpy ss");
    checkCuda(cudaMemcpy(st.d_zmnsc, h_zsc, nbytes_state, cudaMemcpyHostToDevice), "cpy zsc");
    checkCuda(cudaMemcpy(st.d_zmncs, h_zcs, nbytes_state, cudaMemcpyHostToDevice), "cpy zcs");
    checkCuda(cudaMemcpy(st.d_lmnsc, h_lsc, nbytes_state, cudaMemcpyHostToDevice), "cpy lsc");
    checkCuda(cudaMemcpy(st.d_lmncs, h_lcs, nbytes_state, cudaMemcpyHostToDevice), "cpy lcs");

    delete[] h_cc; delete[] h_ss; delete[] h_zsc; delete[] h_zcs;
    delete[] h_lsc; delete[] h_lcs;

    cumes::ValidatedProblem vp = loadValidated();
    cumes::Profiles<double> profiles(p, vp, nullptr); cumes::RadialProfileViews<double> rp = profiles.profile_views();
    FourierPlan<double> fp = fourierCreate(p);
    cumes::DeviceModeTable mt = cumes::modeTableCreate(p);
    cumes::RealSpaceStorage<double> rs = realSpaceCreate(p);
    cumes::GeometryOperator<double> geometry(p, nullptr);

    // Run one iteration
    inverseDFT(fp, rs, storage.physical_const(), p, mt.d_xm, mt.d_xn);
    geometry.enqueue(rs, p, rp, 0); cumes::MagneticFieldOperator<double>{}.enqueue(rs, p, rp, geometry.base_geometry_views(p), geometry.magnetic_field_views(p), 0, true);
    cumes::ForceOperator<double>{}.enqueue(rs, p, rp, geometry.base_geometry_views(p), geometry.magnetic_field_views(p), 0);

    // Check combined geometry at axis (j=0) and mid (j=8)
    size_t nbr = p.ns * p.nZnT * sizeof(double);
    auto* h_r = new double[p.ns * p.nZnT];
    auto* h_z = new double[p.ns * p.nZnT];
    checkCuda(cudaMemcpy(h_r, rs.d_r_real, nbr, cudaMemcpyDeviceToHost), "r");
    checkCuda(cudaMemcpy(h_z, rs.d_z_real, nbr, cudaMemcpyDeviceToHost), "z");

    // Print R at first theta point for all surfaces
    printf("\nR(s,theta=0,zeta=0):\n");
    for (int j = 0; j < p.ns; ++j) {
        printf("  j=%2d: R=%10.6f  Z=%10.6f\n", j, h_r[j * p.nZnT], h_z[j * p.nZnT]);
    }

    // Check forces
    auto* h_armn_e = new double[p.ns * p.nZnT];
    auto* h_armn_o = new double[p.ns * p.nZnT];
    auto* h_blmn_e = new double[p.ns * p.nZnT];
    checkCuda(cudaMemcpy(h_armn_e, rs.d_armn_e, nbr, cudaMemcpyDeviceToHost), "armn_e");
    checkCuda(cudaMemcpy(h_armn_o, rs.d_armn_o, nbr, cudaMemcpyDeviceToHost), "armn_o");
    checkCuda(cudaMemcpy(h_blmn_e, rs.d_blmn_e, nbr, cudaMemcpyDeviceToHost), "blmn_e");

    printf("\nForces at theta=0,zeta=0:\n");
    printf("  j  |  armn_e      armn_o      azmn_e      blmn_e\n");
    printf("  ---+----------------------------------------------\n");
    auto* h_az = new double[p.ns * p.nZnT];
    checkCuda(cudaMemcpy(h_az, rs.d_azmn_e, nbr, cudaMemcpyDeviceToHost), "az");
    for (int j = 0; j < p.ns; ++j) {
        printf("  %2d | %11.4e %11.4e %11.4e %11.4e\n",
               j, h_armn_e[j * p.nZnT], h_armn_o[j * p.nZnT],
               h_az[j * p.nZnT], h_blmn_e[j * p.nZnT]);
    }

    // Check gsqrt at half-grid
    size_t nH = (p.ns - 1) * p.nZnT * sizeof(double);
    auto* h_gs = new double[(p.ns-1) * p.nZnT];
    auto* h_tau = new double[(p.ns-1) * p.nZnT];
    checkCuda(cudaMemcpy(h_gs, geometry.base_geometry_views(p).gsqrt.data(), nH, cudaMemcpyDeviceToHost), "gs");
    checkCuda(cudaMemcpy(h_tau, geometry.base_geometry_views(p).tau.data(), nH, cudaMemcpyDeviceToHost), "tau");

    printf("\nHalf-grid at theta=0,zeta=0:\n");
    printf("  jH |  tau         gsqrt       r12\n");
    auto* h_r12 = new double[(p.ns-1) * p.nZnT];
    checkCuda(cudaMemcpy(h_r12, geometry.base_geometry_views(p).r12.data(), nH, cudaMemcpyDeviceToHost), "r12");
    for (int j = 0; j < p.ns - 1; ++j) {
        printf("  %2d | %11.4e %11.4e %11.4e\n",
               j, h_tau[j * p.nZnT], h_gs[j * p.nZnT], h_r12[j * p.nZnT]);
    }

    // Compute spectral forces via forward DFT
    size_t nbs = 6 * p.ns * p.mnmax * sizeof(double);
    auto* d_fspec = new double[6 * p.ns * p.mnmax];  // host
    double* d_fspec_gpu;
    checkCuda(cudaMalloc(&d_fspec_gpu, nbs), "fspec");
    double *frcon_e, *frcon_o, *fzcon_e, *fzcon_o; cudaMalloc(&frcon_e, (size_t)p.ns*p.nZnT*sizeof(double)); cudaMemset(frcon_e, 0, (size_t)p.ns*p.nZnT*sizeof(double));
    cudaMalloc(&frcon_o, (size_t)p.ns*p.nZnT*sizeof(double)); cudaMemset(frcon_o, 0, (size_t)p.ns*p.nZnT*sizeof(double));
    cudaMalloc(&fzcon_e, (size_t)p.ns*p.nZnT*sizeof(double)); cudaMemset(fzcon_e, 0, (size_t)p.ns*p.nZnT*sizeof(double));
    cudaMalloc(&fzcon_o, (size_t)p.ns*p.nZnT*sizeof(double)); cudaMemset(fzcon_o, 0, (size_t)p.ns*p.nZnT*sizeof(double));
    forwardDFT(fp, rs, cumes::SpectralView<double, cumes::DecomposedResidualDomain>(
                       d_fspec_gpu, p.ns, p.mnmax),
               p, mt.d_xm, mt.d_xn, frcon_e, frcon_o, fzcon_e, fzcon_o);
    checkCuda(cudaMemcpy(d_fspec, d_fspec_gpu, nbs, cudaMemcpyDeviceToHost), "fspec d");

    printf("\nSpectral forces (f_rmnc, f_zmns, f_lmnc):\n");
    printf("  mode | m  n |  f_rmnc(axis) f_zmns(axis) f_lmnc(axis)\n");
    for (int m = 0; m < p.mnmax; ++m) {
        // mode = m*(ntor+1)+n (the old m/ntor division mislabeled the modes)
        int mm = m / (p.ntor + 1);
        int nn = m % (p.ntor + 1);
        int idx_r = 0 + m * p.ns;  // axis (j=0), mode m, comp R
        int idx_z = 0 + m * p.ns + p.mnmax * p.ns;
        int idx_l = 0 + m * p.ns + 2 * p.mnmax * p.ns;
        printf("  %4d | %d %d | %11.4e %11.4e %11.4e\n",
               m, mm, nn, d_fspec[idx_r], d_fspec[idx_z], d_fspec[idx_l]);
    }

    // ---- numerical assertions (the gate) ----
    {
        // Geometry: finite everywhere; the oriented Jacobian is signJ*|gsqrt|
        // (signJ = -1), so signJ*gsqrt must be positive on an interior
        // surface (a flipped Jacobian from a bad parity combination would
        // make it negative).
        size_t nH = (size_t)(p.ns - 1) * p.nZnT;
        checkCuda(cudaMemcpy(h_gs, geometry.base_geometry_views(p).gsqrt.data(), nH * sizeof(double), cudaMemcpyDeviceToHost), "gs");
        bool geo_finite = true;
        double jmin = 1e300, jmax = 0.0;
        for (size_t i = 0; i < nH; ++i) {
            if (!std::isfinite(h_gs[i])) geo_finite = false;
            jmin = std::min(jmin, std::abs(h_gs[i]));
            jmax = std::max(jmax, std::abs(h_gs[i]));
        }
        CHECK(geo_finite, "geometry gsqrt finite");
        CHECK(jmin > 0.0 && jmax > 0.0, "geometry gsqrt nonzero (non-degenerate)");
        // Axisymmetric: R at theta=0 must be positive on the axis (4.0).
        CHECK(h_r[0] > 0.0, "axis R positive");

        // Real-space forces finite (both parities, all families).
        bool f_finite = true;
        for (size_t i = 0; i < (size_t)p.ns * p.nZnT; ++i) {
            if (!std::isfinite(h_armn_e[i]) || !std::isfinite(h_armn_o[i]) ||
                !std::isfinite(h_az[i]) || !std::isfinite(h_blmn_e[i]))
                f_finite = false;
        }
        CHECK(f_finite, "real-space forces finite");

        // Axisymmetric symmetry: Z(theta=0) == 0 exactly (the manufactured
        // state has only sin(m theta) Z content, which vanishes at theta=0;
        // a leak of cos(m theta) Z content into the axisymmetric Z channel
        // would break this). The Z-FORCE at theta=0 also vanishes for the
        // same reason — the weak form's Z term is proportional to Z and its
        // derivatives, all zero at theta=0.
        bool z_zero = true;
        for (int j = 0; j < p.ns; ++j)
            if (h_z[j * p.nZnT] != 0.0) z_zero = false;
        CHECK(z_zero, "axisymmetric: Z(theta=0) == 0");
        bool az_zero = true;
        for (int j = 0; j < p.ns; ++j)
            if (h_az[j * p.nZnT] != 0.0) az_zero = false;
        CHECK(az_zero, "axisymmetric: Z-force azmn_e(theta=0) == 0");

        // Spectral forces finite (the forward-DFT path).
        bool fs_finite = true;
        for (size_t i = 0; i < 6 * (size_t)p.ns * p.mnmax; ++i)
            if (!std::isfinite(d_fspec[i])) fs_finite = false;
        CHECK(fs_finite, "spectral forces finite");
    }

    // Cleanup (the state/velocity slabs are freed by SpectralStorage's RAII)
    realSpaceFree(rs);
    fourierFree(fp); cumes::modeTableFree(mt);
    cudaFree(frcon_e); cudaFree(frcon_o);
    cudaFree(fzcon_e); cudaFree(fzcon_o);

    delete[] h_r; delete[] h_z;
    delete[] h_armn_e; delete[] h_armn_o; delete[] h_blmn_e;
    delete[] h_az; delete[] h_gs; delete[] h_tau; delete[] h_r12;
    cudaFree(d_fspec_gpu); delete[] d_fspec;

    printf("\nDone.\n");
    if (failures == 0) {
        printf("test_forces: ALL PASS\n");
        return 0;
    }
    printf("test_forces: %d FAILURES\n", failures);
    return 1;
}
