// test_forces.cu — diagnostic: check force sign and magnitude for cylinder
#include <cstdio>
#include <cmath>
#include <cublas_v2.h>
#include <vector>

#include "vmec_types.h"
#include "input_json.h"
#include "constraint.cuh"
#include "fourier.cuh"
#include "geometry.cuh"
#include "forces.cuh"
#include "profiles.cuh"

static void checkCuda(cudaError_t err, const char* tag) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error [%s]: %s\n", tag, cudaGetErrorString(err));
        exit(1);
    }
}

int main() {
    GridParams<double> p;
    p.ns = 17; p.mnmax = 6*(2+1); p.ntheta = 32; p.nzeta = 64;
    p.nfp = 1; p.nZnT = 2048; p.mpol = 6; p.ntor = 2;
    p.ncurr = 0; p.delt = 1.0; p.ftol = 1e-14; p.max_iter = 10;

    printf("=== Force Diagnostic Test ===\n");

    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);

    // Create Solovev-like initial state with independent parity coefficients
    SpectralState<double> st{};
    size_t nbytes_state = p.ns * p.mnmax * sizeof(double);
    auto* h_cc = new double[p.ns * p.mnmax]();
    auto* h_ss = new double[p.ns * p.mnmax]();
    auto* h_zsc = new double[p.ns * p.mnmax]();
    auto* h_zcs = new double[p.ns * p.mnmax]();
    auto* h_lsc = new double[p.ns * p.mnmax]();

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

    checkCuda(cudaMalloc(&st.d_rmncc, nbytes_state), "rmncc");
    checkCuda(cudaMalloc(&st.d_rmnss, nbytes_state), "rmnss");
    checkCuda(cudaMalloc(&st.d_zmnsc, nbytes_state), "zmnsc");
    checkCuda(cudaMalloc(&st.d_zmncs, nbytes_state), "zmncs");
    checkCuda(cudaMalloc(&st.d_lmnsc, nbytes_state), "lmnsc");
    checkCuda(cudaMalloc(&st.d_lmncs, nbytes_state), "lmncs");
    checkCuda(cudaMalloc(&st.d_v_rmncc, nbytes_state), "vcc");
    checkCuda(cudaMalloc(&st.d_v_rmnss, nbytes_state), "vss");
    checkCuda(cudaMalloc(&st.d_v_zmnsc, nbytes_state), "vzsc");
    checkCuda(cudaMalloc(&st.d_v_zmncs, nbytes_state), "vzcs");
    checkCuda(cudaMalloc(&st.d_v_lmnsc, nbytes_state), "vlsc");
    checkCuda(cudaMalloc(&st.d_v_lmncs, nbytes_state), "vlcs");
    checkCuda(cudaMemcpy(st.d_rmncc, h_cc, nbytes_state, cudaMemcpyHostToDevice), "cpy cc");
    checkCuda(cudaMemcpy(st.d_rmnss, h_ss, nbytes_state, cudaMemcpyHostToDevice), "cpy ss");
    checkCuda(cudaMemcpy(st.d_zmnsc, h_zsc, nbytes_state, cudaMemcpyHostToDevice), "cpy zsc");
    checkCuda(cudaMemcpy(st.d_zmncs, h_zcs, nbytes_state, cudaMemcpyHostToDevice), "cpy zcs");
    checkCuda(cudaMemcpy(st.d_lmnsc, h_lsc, nbytes_state, cudaMemcpyHostToDevice), "cpy lsc");
    checkCuda(cudaMemset(st.d_v_rmncc, 0, nbytes_state), "vcc");
    checkCuda(cudaMemset(st.d_v_rmnss, 0, nbytes_state), "vss");
    checkCuda(cudaMemset(st.d_v_zmnsc, 0, nbytes_state), "vzsc");
    checkCuda(cudaMemset(st.d_v_zmncs, 0, nbytes_state), "vzcs");
    checkCuda(cudaMemset(st.d_v_lmnsc, 0, nbytes_state), "vlsc");
    checkCuda(cudaMemset(st.d_v_lmncs, 0, nbytes_state), "vlcs");

    delete[] h_cc; delete[] h_ss; delete[] h_zsc; delete[] h_zcs; delete[] h_lsc;

    InputParams ip = initInputParams();
    RadialProfiles<double> rp = profilesCreate(p, ip);
    FourierPlan<double> fp = fourierCreate(p, cublasHandle);
    MetricWorkspace<double> mw = metricCreate(p);

    // Run one iteration
    inverseDFT(fp, st, p);
    computeGeometry(fp, p, rp, mw);
    computeForces(fp, p, rp, mw);

    // Check combined geometry at axis (j=0) and mid (j=8)
    size_t nbr = p.ns * p.nZnT * sizeof(double);
    auto* h_r = new double[p.ns * p.nZnT];
    auto* h_z = new double[p.ns * p.nZnT];
    checkCuda(cudaMemcpy(h_r, fp.d_r_real, nbr, cudaMemcpyDeviceToHost), "r");
    checkCuda(cudaMemcpy(h_z, fp.d_z_real, nbr, cudaMemcpyDeviceToHost), "z");

    // Print R at first theta point for all surfaces
    printf("\nR(s,theta=0,zeta=0):\n");
    for (int j = 0; j < p.ns; ++j) {
        printf("  j=%2d: R=%10.6f  Z=%10.6f\n", j, h_r[j * p.nZnT], h_z[j * p.nZnT]);
    }

    // Check forces
    auto* h_armn_e = new double[p.ns * p.nZnT];
    auto* h_armn_o = new double[p.ns * p.nZnT];
    auto* h_blmn_e = new double[p.ns * p.nZnT];
    checkCuda(cudaMemcpy(h_armn_e, fp.d_armn_e, nbr, cudaMemcpyDeviceToHost), "armn_e");
    checkCuda(cudaMemcpy(h_armn_o, fp.d_armn_o, nbr, cudaMemcpyDeviceToHost), "armn_o");
    checkCuda(cudaMemcpy(h_blmn_e, fp.d_blmn_e, nbr, cudaMemcpyDeviceToHost), "blmn_e");

    printf("\nForces at theta=0,zeta=0:\n");
    printf("  j  |  armn_e      armn_o      azmn_e      blmn_e\n");
    printf("  ---+----------------------------------------------\n");
    auto* h_az = new double[p.ns * p.nZnT];
    checkCuda(cudaMemcpy(h_az, fp.d_azmn_e, nbr, cudaMemcpyDeviceToHost), "az");
    for (int j = 0; j < p.ns; ++j) {
        printf("  %2d | %11.4e %11.4e %11.4e %11.4e\n",
               j, h_armn_e[j * p.nZnT], h_armn_o[j * p.nZnT],
               h_az[j * p.nZnT], h_blmn_e[j * p.nZnT]);
    }

    // Check gsqrt at half-grid
    size_t nH = (p.ns - 1) * p.nZnT * sizeof(double);
    auto* h_gs = new double[(p.ns-1) * p.nZnT];
    auto* h_tau = new double[(p.ns-1) * p.nZnT];
    checkCuda(cudaMemcpy(h_gs, mw.d_gsqrt, nH, cudaMemcpyDeviceToHost), "gs");
    checkCuda(cudaMemcpy(h_tau, mw.d_tau, nH, cudaMemcpyDeviceToHost), "tau");

    printf("\nHalf-grid at theta=0,zeta=0:\n");
    printf("  jH |  tau         gsqrt       r12\n");
    auto* h_r12 = new double[(p.ns-1) * p.nZnT];
    checkCuda(cudaMemcpy(h_r12, mw.d_r12, nH, cudaMemcpyDeviceToHost), "r12");
    for (int j = 0; j < p.ns - 1; ++j) {
        printf("  %2d | %11.4e %11.4e %11.4e\n",
               j, h_tau[j * p.nZnT], h_gs[j * p.nZnT], h_r12[j * p.nZnT]);
    }

    // Compute spectral forces via forward DFT
    size_t nbs = 6 * p.ns * p.mnmax * sizeof(double);
    auto* d_fspec = (double*)malloc(nbs);  // host
    double* d_fspec_gpu;
    checkCuda(cudaMalloc(&d_fspec_gpu, nbs), "fspec");
    ConstraintWorkspace<double> cw_zero{}; cudaMalloc(&cw_zero.d_frcon_e, (size_t)p.ns*p.nZnT*sizeof(double)); cudaMemset(cw_zero.d_frcon_e, 0, (size_t)p.ns*p.nZnT*sizeof(double));
    cudaMalloc(&cw_zero.d_frcon_o, (size_t)p.ns*p.nZnT*sizeof(double)); cudaMemset(cw_zero.d_frcon_o, 0, (size_t)p.ns*p.nZnT*sizeof(double));
    cudaMalloc(&cw_zero.d_fzcon_e, (size_t)p.ns*p.nZnT*sizeof(double)); cudaMemset(cw_zero.d_fzcon_e, 0, (size_t)p.ns*p.nZnT*sizeof(double));
    cudaMalloc(&cw_zero.d_fzcon_o, (size_t)p.ns*p.nZnT*sizeof(double)); cudaMemset(cw_zero.d_fzcon_o, 0, (size_t)p.ns*p.nZnT*sizeof(double));
    forwardDFT(fp, d_fspec_gpu, p, cw_zero);
    checkCuda(cudaMemcpy(d_fspec, d_fspec_gpu, nbs, cudaMemcpyDeviceToHost), "fspec d");

    printf("\nSpectral forces (f_rmnc, f_zmns, f_lmnc):\n");
    printf("  mode | m  n |  f_rmnc(axis) f_zmns(axis) f_lmnc(axis)\n");
    for (int m = 0; m < p.mnmax; ++m) {
        int mm = m / p.ntor;
        int nn = m % p.ntor;
        int idx_r = 0 + m * p.ns;  // axis (j=0), mode m, comp R
        int idx_z = 0 + m * p.ns + p.mnmax * p.ns;
        int idx_l = 0 + m * p.ns + 2 * p.mnmax * p.ns;
        printf("  %4d | %d %d | %11.4e %11.4e %11.4e\n",
               m, mm, nn, d_fspec[idx_r], d_fspec[idx_z], d_fspec[idx_l]);
    }

    // Cleanup
    cudaFree(st.d_rmncc); cudaFree(st.d_rmnss); cudaFree(st.d_zmnsc);
    cudaFree(st.d_zmncs); cudaFree(st.d_lmnsc);
    cudaFree(st.d_v_rmncc); cudaFree(st.d_v_rmnss); cudaFree(st.d_v_zmnsc);
    cudaFree(st.d_v_zmncs); cudaFree(st.d_v_lmnsc);
    fourierFree(fp); metricFree(mw); profilesFree(rp);
    cublasDestroy(cublasHandle);

    delete[] h_r; delete[] h_z;
    delete[] h_armn_e; delete[] h_armn_o; delete[] h_blmn_e;
    delete[] h_az; delete[] h_gs; delete[] h_tau; delete[] h_r12;
    cudaFree(d_fspec_gpu); free(d_fspec);

    printf("\nDone.\n");
    return 0;
}
