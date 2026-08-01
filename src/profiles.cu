// profiles.cu — allocate and initialise radial profiles on GPU.
#include "profiles.cuh"
#include "input.h"
#include <cstdio>

static void checkCuda(cudaError_t err, const char* tag) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error [%s]: %s\n", tag, cudaGetErrorString(err));
        exit(1);
    }
}

RadialProfiles profilesCreate(const GridParams& p) {
    RadialProfiles rp{};
    rp.delta_s = 1.0 / (p.ns - 1);

    size_t nF = p.ns * sizeof(double);
    size_t nH = (p.ns - 1) * sizeof(double);

    checkCuda(cudaMalloc(&rp.d_iota_F, nF), "iota_F");
    checkCuda(cudaMalloc(&rp.d_pres_F, nF), "pres_F");
    checkCuda(cudaMalloc(&rp.d_phip_F, nF), "phip_F");
    checkCuda(cudaMalloc(&rp.d_chi_F,  nF), "chi_F");
    checkCuda(cudaMalloc(&rp.d_sqrtS_F, nF), "sqrtS_F");
    checkCuda(cudaMalloc(&rp.d_iota_H, nH), "iota_H");
    checkCuda(cudaMalloc(&rp.d_pres_H, nH), "pres_H");
    checkCuda(cudaMalloc(&rp.d_phip_H, nH), "phip_H");
    checkCuda(cudaMalloc(&rp.d_mass_H, nH), "mass_H");
    checkCuda(cudaMalloc(&rp.d_dVds_H, nH), "dVds_H");
    checkCuda(cudaMalloc(&rp.d_sqrtS_H, nH), "sqrtS_H");

    // Evaluate on host and upload
    auto* h = new double[p.ns];
    // Full grid
    for (int j = 0; j < p.ns; ++j) {
        double s = rp.delta_s * j;
        h[j] = iotaProfile(s);
    }
    checkCuda(cudaMemcpy(rp.d_iota_F, h, nF, cudaMemcpyHostToDevice), "iota_F cpy");
    for (int j = 0; j < p.ns; ++j) {
        double s = rp.delta_s * j;
        h[j] = presProfile(s);
    }
    checkCuda(cudaMemcpy(rp.d_pres_F, h, nF, cudaMemcpyHostToDevice), "pres_F cpy");
    // Match vmecpp: maxToroidalFlux = signJ * phiedge / (2*pi)
    // signJ = -1 (kSignJacobian), phiedge = 1.0
    // gsqrt = tau*r12 is negative (tau < 0), so chipH must also be negative
    // to give positive bsupu = chipH/gsqrt = (-)/(-) = (+).
    double maxToroidalFlux = -1.0 / (2.0 * M_PI);  // signJ=-1, phiedge=1.0
    for (int j = 0; j < p.ns; ++j) h[j] = maxToroidalFlux;
    checkCuda(cudaMemcpy(rp.d_phip_F, h, nF, cudaMemcpyHostToDevice), "phip_F cpy");
    for (int j = 0; j < p.ns; ++j) h[j] = 0.0;  // chi placeholder
    checkCuda(cudaMemcpy(rp.d_chi_F, h, nF, cudaMemcpyHostToDevice), "chi_F cpy");
    for (int j = 0; j < p.ns; ++j) h[j] = sqrt(rp.delta_s * j + 1e-12);
    checkCuda(cudaMemcpy(rp.d_sqrtS_F, h, nF, cudaMemcpyHostToDevice), "sqrtS_F cpy");

    // Half grid — average of neighbouring full-grid values
    for (int j = 0; j < p.ns - 1; ++j) {
        double sh = rp.delta_s * (j + 0.5);
        h[j] = iotaProfile(sh);
    }
    checkCuda(cudaMemcpy(rp.d_iota_H, h, nH, cudaMemcpyHostToDevice), "iota_H cpy");
    for (int j = 0; j < p.ns - 1; ++j) {
        double sh = rp.delta_s * (j + 0.5);
        h[j] = presProfile(sh);
    }
    checkCuda(cudaMemcpy(rp.d_pres_H, h, nH, cudaMemcpyHostToDevice), "pres_H cpy");
    for (int j = 0; j < p.ns - 1; ++j) h[j] = maxToroidalFlux;
    checkCuda(cudaMemcpy(rp.d_phip_H, h, nH, cudaMemcpyHostToDevice), "phip_H cpy");
    for (int j = 0; j < p.ns - 1; ++j) h[j] = massProfile(rp.delta_s * (j + 0.5));
    checkCuda(cudaMemcpy(rp.d_mass_H, h, nH, cudaMemcpyHostToDevice), "mass_H cpy");
    for (int j = 0; j < p.ns - 1; ++j) h[j] = 1.0;
    checkCuda(cudaMemcpy(rp.d_dVds_H, h, nH, cudaMemcpyHostToDevice), "dVds_H cpy");
    // sqrt(s) on half-grid for parity mixing
    for (int j = 0; j < p.ns - 1; ++j) {
        h[j] = sqrt(rp.delta_s * (j + 0.5));
    }
    checkCuda(cudaMemcpy(rp.d_sqrtS_H, h, nH, cudaMemcpyHostToDevice), "sqrtS_H cpy");

    delete[] h;
    return rp;
}

void profilesFree(RadialProfiles& rp) {
    cudaFree(rp.d_iota_F); cudaFree(rp.d_pres_F);
    cudaFree(rp.d_phip_F); cudaFree(rp.d_chi_F);
    cudaFree(rp.d_sqrtS_F);
    cudaFree(rp.d_iota_H); cudaFree(rp.d_pres_H);
    cudaFree(rp.d_phip_H); cudaFree(rp.d_mass_H);
    cudaFree(rp.d_dVds_H); cudaFree(rp.d_sqrtS_H);
}
