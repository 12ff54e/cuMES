// test_force_verify.cu — load vmecpp equilibrium and check forces.
// If the force formulas are correct, the forces should be near zero
// for the converged vmecpp solution.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cublas_v2.h>

#include "input.h"
#include "constraint.cuh"
#include "vmec_types.h"
#include "fourier.cuh"
#include "geometry.cuh"
#include "forces.cuh"
#include "profiles.cuh"

static void cc(cudaError_t e, const char* t) {
    if(e!=cudaSuccess){fprintf(stderr,"CUDA[%s]:%s\n",t,cudaGetErrorString(e));exit(1);}
}

int main() {
    // Read vmecpp-initialized state
    FILE* fp = fopen("vmecpp_init.bin", "rb");
    if (!fp) { printf("Cannot open vmecpp_init.bin\n"); return 1; }
    int ns, mnmax;
    if (fread(&ns, sizeof(int), 1, fp) != 1 || fread(&mnmax, sizeof(int), 1, fp) != 1) {
        printf("Truncated vmecpp_init.bin (header)\n");
        fclose(fp);
        return 1;
    }
    printf("Read vmecpp state: ns=%d mnmax=%d\n", ns, mnmax);

    GridParams<double> p;
    p.ns = ns; p.mnmax = mnmax; p.ntheta = 18; p.nzeta = 1;
    p.nfp = 1; p.nZnT = p.ntheta * p.nzeta;
    p.mpol = mnmax;  // ntor=0 (axisymmetric): mnmax = mpol*(ntor+1)
    p.ntor = 0;
    p.ncurr = 0; p.delt = 1.0; p.ftol = 1e-14; p.max_iter = 10;

    size_t nb = ns * mnmax * sizeof(double);
    auto* h_rmncc = new double[ns*mnmax];
    auto* h_zmnsc = new double[ns*mnmax];
    auto* h_lmnsc = new double[ns*mnmax];
    auto* h_rmnss = new double[ns*mnmax];
    auto* h_zmncs = new double[ns*mnmax];
    auto* h_lmncs = new double[ns*mnmax];

    size_t ncoef = (size_t)ns * mnmax;
    if (fread(h_rmncc, sizeof(double), ncoef, fp) != ncoef ||
        fread(h_zmnsc, sizeof(double), ncoef, fp) != ncoef ||
        fread(h_lmnsc, sizeof(double), ncoef, fp) != ncoef ||
        fread(h_rmnss, sizeof(double), ncoef, fp) != ncoef ||
        fread(h_zmncs, sizeof(double), ncoef, fp) != ncoef ||
        fread(h_lmncs, sizeof(double), ncoef, fp) != ncoef) {
        printf("Truncated vmecpp_init.bin (state)\n");
        fclose(fp);
        return 1;
    }
    fclose(fp);

    printf("R_00 axis=%.6f edge=%.6f\n", h_rmncc[0], h_rmncc[ns-1]);
    printf("Z_10 axis=%.6f edge=%.6f\n", h_zmnsc[ns], h_zmnsc[ns-1+ns]);
    printf("R_10 axis=%.6f edge=%.6f\n", h_rmncc[ns], h_rmncc[ns-1+ns]);

    // Allocate GPU state
    SpectralState<double> st{};
    cc(cudaMalloc(&st.d_rmncc, nb), "rmncc");
    cc(cudaMalloc(&st.d_zmnsc, nb), "zmnsc");
    cc(cudaMalloc(&st.d_lmnsc, nb), "lmnsc");
    cc(cudaMalloc(&st.d_lmncs, nb), "lmncs");
    cc(cudaMalloc(&st.d_rmnss, nb), "rmnss");
    cc(cudaMalloc(&st.d_zmncs, nb), "zmncs");
    cc(cudaMemcpy(st.d_rmncc, h_rmncc, nb, cudaMemcpyHostToDevice), "cpy rmncc");
    cc(cudaMemcpy(st.d_zmnsc, h_zmnsc, nb, cudaMemcpyHostToDevice), "cpy zmnsc");
    cc(cudaMemcpy(st.d_lmnsc, h_lmnsc, nb, cudaMemcpyHostToDevice), "cpy lmnsc");
    cc(cudaMemcpy(st.d_rmnss, h_rmnss, nb, cudaMemcpyHostToDevice), "cpy rmnss");
    cc(cudaMemcpy(st.d_zmncs, h_zmncs, nb, cudaMemcpyHostToDevice), "cpy zmncs");

    // Create profiles and Fourier plan
    InputParams ip = initInputParams();
    RadialProfiles<double> rp = profilesCreate(p, ip);
    cublasHandle_t handle;
    cublasCreate(&handle);
    FourierPlan<double> fpl = fourierCreate(p, handle);
    MetricWorkspace<double> mw = metricCreate(p);

    // Run one iteration: inverse DFT + geometry + forces + forward DFT
    inverseDFT(fpl, st, p);
    printf("dbg: invDFT err=%s\n", cudaGetErrorString(cudaGetLastError()));
    cudaDeviceSynchronize();
    computeGeometry(fpl, p, rp, mw);
    printf("dbg: geom err=%s\n", cudaGetErrorString(cudaGetLastError()));
    cudaDeviceSynchronize();
    computeForces(fpl, p, rp, mw);
    printf("dbg: forces err=%s\n", cudaGetErrorString(cudaGetLastError()));
    cudaDeviceSynchronize();

    // Forward DFT to get spectral forces
    double* d_f_spec;
    cc(cudaMalloc(&d_f_spec, 5*ns*mnmax*sizeof(double)), "f_spec");
    ConstraintWorkspace<double> cw_zero{}; cudaMalloc(&cw_zero.d_frcon_e, (size_t)p.ns*p.nZnT*sizeof(double)); cudaMemset(cw_zero.d_frcon_e, 0, (size_t)p.ns*p.nZnT*sizeof(double));
    cudaMalloc(&cw_zero.d_frcon_o, (size_t)p.ns*p.nZnT*sizeof(double)); cudaMemset(cw_zero.d_frcon_o, 0, (size_t)p.ns*p.nZnT*sizeof(double));
    cudaMalloc(&cw_zero.d_fzcon_e, (size_t)p.ns*p.nZnT*sizeof(double)); cudaMemset(cw_zero.d_fzcon_e, 0, (size_t)p.ns*p.nZnT*sizeof(double));
    cudaMalloc(&cw_zero.d_fzcon_o, (size_t)p.ns*p.nZnT*sizeof(double)); cudaMemset(cw_zero.d_fzcon_o, 0, (size_t)p.ns*p.nZnT*sizeof(double));
    forwardDFT(fpl, d_f_spec, p, cw_zero);

    // Copy forces to host
    auto* h_f = new double[5*ns*mnmax];
    cc(cudaMemcpy(h_f, d_f_spec, 5*ns*mnmax*sizeof(double), cudaMemcpyDeviceToHost), "cpy f");

    // Compute residuals
    double fsqr=0, fsqz=0, fsql=0;
    for (int c = 0; c < 5; ++c) {
        double sum = 0;
        for (int i = 0; i < ns*mnmax; ++i) sum += h_f[c*ns*mnmax + i] * h_f[c*ns*mnmax + i];
        sum /= (ns*mnmax);
        if (c == 0) fsqr = sum;
        else if (c == 1) fsqz = sum;
        else if (c == 2) fsql = sum;
    }
    printf("\nForce residuals for vmecpp equilibrium:\n");
    printf("  FSQR = %.3e\n", fsqr);
    printf("  FSQZ = %.3e\n", fsqz);
    printf("  FSQL = %.3e\n", fsql);

    // Print forces for mode 0 (R_00) at axis
    printf("\nMode 0 (R_00) forces at each surface:\n");
    for (int j = 0; j < ns; ++j) {
        printf("  j=%2d: f_rmncc=%.6e  f_rmnss=%.6e  f_zmnsc=%.6e  f_zmncs=%.6e  f_lmnsc=%.6e\n",
               j, h_f[j], h_f[j+ns], h_f[j+2*ns], h_f[j+3*ns], h_f[j+4*ns]);
    }

    // Print forces for mode 2 (R_10, Z_10) at a few surfaces
    int m2 = 1;  // mode index for m=1,n=0 (since ntor=1)
    printf("\nMode %d (m=1,n=0) forces:\n", m2);
    for (int j = 0; j < ns; j += 2) {
        int idx = j + m2*ns;
        printf("  j=%2d: f_rmncc=%.6e  f_rmnss=%.6e  f_zmnsc=%.6e  f_zmncs=%.6e\n",
               j, h_f[idx], h_f[idx+ns*mnmax], h_f[idx+2*ns*mnmax], h_f[idx+3*ns*mnmax]);
    }

    // Print geometry diagnostics for first half-grid
    auto* h_r12 = new double[p.nZnT];
    auto* h_tau = new double[p.nZnT];
    auto* h_gsqrt = new double[p.nZnT];
    auto* h_bsupu = new double[p.nZnT];
    auto* h_bsupv = new double[p.nZnT];
    auto* h_totalP = new double[p.nZnT];
    auto* h_guu = new double[p.nZnT];
    auto* h_gvv = new double[p.nZnT];
    auto* h_ru12 = new double[p.nZnT];
    auto* h_zu12 = new double[p.nZnT];
    auto* h_rs = new double[p.nZnT];
    auto* h_zs = new double[p.nZnT];
    cc(cudaMemcpy(h_r12, mw.d_r12, p.nZnT*sizeof(double), cudaMemcpyDeviceToHost), "cpy r12");
    cc(cudaMemcpy(h_tau, mw.d_tau, p.nZnT*sizeof(double), cudaMemcpyDeviceToHost), "cpy tau");
    cc(cudaMemcpy(h_gsqrt, mw.d_gsqrt, p.nZnT*sizeof(double), cudaMemcpyDeviceToHost), "cpy gsqrt");
    cc(cudaMemcpy(h_bsupu, mw.d_bsupu, p.nZnT*sizeof(double), cudaMemcpyDeviceToHost), "cpy bsupu");
    cc(cudaMemcpy(h_bsupv, mw.d_bsupv, p.nZnT*sizeof(double), cudaMemcpyDeviceToHost), "cpy bsupv");
    cc(cudaMemcpy(h_totalP, mw.d_totalPressure, p.nZnT*sizeof(double), cudaMemcpyDeviceToHost), "cpy tP");
    cc(cudaMemcpy(h_guu, mw.d_guu, p.nZnT*sizeof(double), cudaMemcpyDeviceToHost), "cpy guu");
    cc(cudaMemcpy(h_gvv, mw.d_gvv, p.nZnT*sizeof(double), cudaMemcpyDeviceToHost), "cpy gvv");
    cc(cudaMemcpy(h_ru12, mw.d_ru12, p.nZnT*sizeof(double), cudaMemcpyDeviceToHost), "cpy ru12");
    cc(cudaMemcpy(h_zu12, mw.d_zu12, p.nZnT*sizeof(double), cudaMemcpyDeviceToHost), "cpy zu12");
    cc(cudaMemcpy(h_rs, mw.d_rs, p.nZnT*sizeof(double), cudaMemcpyDeviceToHost), "cpy rs");
    cc(cudaMemcpy(h_zs, mw.d_zs, p.nZnT*sizeof(double), cudaMemcpyDeviceToHost), "cpy zs");

    printf("\nHalf-grid jH=0 diagnostics (theta=0, zeta=0):\n");
    printf("  r12=%.6f  tau=%.6e  gsqrt=%.6e\n", h_r12[0], h_tau[0], h_gsqrt[0]);
    printf("  ru12=%.6f  zu12=%.6f  rs=%.6f  zs=%.6f\n", h_ru12[0], h_zu12[0], h_rs[0], h_zs[0]);
    printf("  bsupu=%.6e  bsupv=%.6e\n", h_bsupu[0], h_bsupv[0]);
    printf("  guu=%.6e  gvv=%.6e\n", h_guu[0], h_gvv[0]);
    printf("  totalP=%.6e\n", h_totalP[0]);
    // Cross-check: tau1 = ru12*zs - rs*zu12
    double tau1_check = h_ru12[0]*h_zs[0] - h_rs[0]*h_zu12[0];
    printf("  tau1=%.6e (check: %.6e)\n", h_tau[0], tau1_check);
    // gsqrt check: sqrt(tau^2 + 0.01^2) * r12
    double gs_check = sqrt(h_tau[0]*h_tau[0] + 0.0001) * h_r12[0];
    printf("  gsqrt check: %.6e\n", gs_check);

    // Print iota, pres, phip at first half-grid
    auto* h_iotaH = new double[p.ns-1];
    auto* h_presH = new double[p.ns-1];
    auto* h_phipH = new double[p.ns-1];
    cc(cudaMemcpy(h_iotaH, rp.d_iota_H, (p.ns-1)*sizeof(double), cudaMemcpyDeviceToHost), "cpy iota");
    cc(cudaMemcpy(h_presH, rp.d_pres_H, (p.ns-1)*sizeof(double), cudaMemcpyDeviceToHost), "cpy pres");
    cc(cudaMemcpy(h_phipH, rp.d_phip_H, (p.ns-1)*sizeof(double), cudaMemcpyDeviceToHost), "cpy phip");
    printf("  iota_H[0]=%.6f  pres_H[0]=%.6e  phip_H[0]=%.6f\n", h_iotaH[0], h_presH[0], h_phipH[0]);

    delete[] h_r12; delete[] h_tau; delete[] h_gsqrt; delete[] h_bsupu; delete[] h_bsupv;
    delete[] h_totalP; delete[] h_guu; delete[] h_gvv;
    delete[] h_ru12; delete[] h_zu12; delete[] h_rs; delete[] h_zs;
    delete[] h_iotaH; delete[] h_presH; delete[] h_phipH;

    // Cleanup
    cudaFree(d_f_spec);
    fourierFree(fpl); metricFree(mw); profilesFree(rp);
    cudaFree(st.d_rmncc); cudaFree(st.d_zmnsc); cudaFree(st.d_lmnsc);
    cudaFree(st.d_rmnss); cudaFree(st.d_zmncs);
    cublasDestroy(handle);
    delete[] h_rmncc; delete[] h_zmnsc; delete[] h_lmnsc; delete[] h_rmnss; delete[] h_zmncs;
    delete[] h_f;
    return 0;
}
