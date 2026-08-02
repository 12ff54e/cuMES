// profiles.cu — evaluate radial profiles on host from the input parameters
// and upload to GPU. Matches vmecpp's evalRadialProfiles (radial_profiles.cc
// lines 1149-1220) and computeMagneticFluxes (440-455):
//   maxToroidalFlux = signJ*phiedge/(2π) / torflux(1)
//   phip = maxToroidalFlux * torfluxDeriv(s)
//   torflux(x) = x*Σ aphi_i*x^i, torfluxDeriv(x) = Σ (i+1)*aphi_i*x^i
//   mass = μ0*pres_scale * Σ am_i*tf^i   (tf = toroidal flux coordinate)
//   pres = mass / dVds^gamma  (gamma = adiabatic_index; dVds placeholder
//          for gamma != 0 — the geometry-dependent dVds is not implemented)
//   iota = Σ ai_i*tf^i  (ncurr=0; for ncurr=1 iotaH/chipH are recomputed
//          every iteration from the prescribed current, see geometry.cu)
//   curr = Itor * Σ ac_i*tf^(i+1)/(i+1), Itor = signJ*μ0*curtor/(2π*I(1))
// lamscale = sqrt(deltaS * Σ_j phipH[j]^2), the vmecpp constants_.lamscale.
#include "profiles.cuh"
#include "input.h"
#include <cstdio>
#include <cmath>

static void checkCuda(cudaError_t err, const char* tag) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error [%s]: %s\n", tag, cudaGetErrorString(err));
        exit(1);
    }
}

// Horner evaluation of a power series; with integrate=true gives the
// integral ∫₀ˣ Σ c_i t^i dt = x*Σ c_i*x^i/(i+1) (vmecpp evalPowerSeries).
static double evalPowerSeries(const double* c, int n, double x, bool integrate) {
    double ret = 0.0;
    for (int i = n - 1; i >= 0; --i) {
        if (integrate) {
            ret = x * ret + c[i] / static_cast<double>(i + 1);
        } else {
            ret = x * ret + c[i];
        }
    }
    if (integrate) ret *= x;
    return ret;
}

static double torflux(const InputParams& ip, double x) {
    return x * evalPowerSeries(ip.aphi, ip.aphi_n, x, false);
}

static double torfluxDeriv(const InputParams& ip, double x) {
    double ret = 0.0;
    for (int i = 0; i < ip.aphi_n; ++i) {
        ret += static_cast<double>(i + 1) * ip.aphi[i] * pow(x, i);
    }
    return ret;
}

static double evalIotaProfile(const InputParams& ip, double x) {
    return evalPowerSeries(ip.ai, ip.ai_n, x, false);
}

static double evalMassProfile(const InputParams& ip, double x) {
    double normX = fmin(fabs(x * ip.bloat), 1.0);
    return evalPowerSeries(ip.am, ip.am_n, normX, false) *
           (GridParams::kMu0 * ip.pres_scale);
}

static double evalCurrProfile(const InputParams& ip, double x) {
    double normX = fmin(fabs(x * ip.bloat), 1.0);
    return evalPowerSeries(ip.ac, ip.ac_n, normX, true);
}

RadialProfiles profilesCreate(GridParams& p, const InputParams& ip) {
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
    checkCuda(cudaMalloc(&rp.d_curr_H, nH), "curr_H");
    checkCuda(cudaMalloc(&rp.d_chip_H, nH), "chip_H");

    // maxToroidalFlux = signJ * phiedge / (2π) / torflux(1)
    // (signJ = -1, so phiedge < 0 gives a positive flux, e.g. w7x).
    double maxToroidalFlux = GridParams::kSignJacobian * ip.phiedge / (2.0 * M_PI);
    double tf1 = torflux(ip, 1.0);
    if (tf1 != 0.0) maxToroidalFlux /= tf1;

    // ncurr=1: normalize the enclosed toroidal current profile
    // Itor = signJ * μ0*curtor / (2π * I(1)), I(s) = ∫₀ˢ ac
    double Itor = 0.0;
    if (ip.ncurr == 1) {
        double edgeCurrent = evalCurrProfile(ip, 1.0);
        Itor = GridParams::kSignJacobian * GridParams::kMu0 * ip.curtor /
               (2.0 * M_PI * edgeCurrent);
    }

    auto* h = new double[p.ns];
    // ---- Full grid ----
    for (int j = 0; j < p.ns; ++j) {
        double s = rp.delta_s * j;
        double tf = fmin(torflux(ip, s), 1.0);
        h[j] = evalIotaProfile(ip, tf);
    }
    checkCuda(cudaMemcpy(rp.d_iota_F, h, nF, cudaMemcpyHostToDevice), "iota_F cpy");
    for (int j = 0; j < p.ns; ++j) {
        double s = rp.delta_s * j;
        double tf = fmin(torflux(ip, fmin(s, ip.spres_ped)), 1.0);
        h[j] = evalMassProfile(ip, tf);  // pres = mass (gamma = 0)
    }
    checkCuda(cudaMemcpy(rp.d_pres_F, h, nF, cudaMemcpyHostToDevice), "pres_F cpy");
    for (int j = 0; j < p.ns; ++j) {
        double s = rp.delta_s * j;
        h[j] = maxToroidalFlux * torfluxDeriv(ip, s);
    }
    checkCuda(cudaMemcpy(rp.d_phip_F, h, nF, cudaMemcpyHostToDevice), "phip_F cpy");
    for (int j = 0; j < p.ns; ++j) {
        double s = rp.delta_s * j;
        double tf = fmin(torflux(ip, s), 1.0);
        h[j] = maxToroidalFlux * evalIotaProfile(ip, tf) * torfluxDeriv(ip, s);
    }
    checkCuda(cudaMemcpy(rp.d_chi_F, h, nF, cudaMemcpyHostToDevice), "chi_F cpy");
    for (int j = 0; j < p.ns; ++j) h[j] = sqrt(rp.delta_s * j + 1e-12);
    checkCuda(cudaMemcpy(rp.d_sqrtS_F, h, nF, cudaMemcpyHostToDevice), "sqrtS_F cpy");

    // ---- Half grid ----
    for (int j = 0; j < p.ns - 1; ++j) {
        double sh = rp.delta_s * (j + 0.5);
        double tf = fmin(torflux(ip, sh), 1.0);
        h[j] = evalIotaProfile(ip, tf);
    }
    checkCuda(cudaMemcpy(rp.d_iota_H, h, nH, cudaMemcpyHostToDevice), "iota_H cpy");
    for (int j = 0; j < p.ns - 1; ++j) {
        double sh = rp.delta_s * (j + 0.5);
        double tf = fmin(torflux(ip, fmin(sh, ip.spres_ped)), 1.0);
        h[j] = evalMassProfile(ip, tf);  // pres = mass (gamma = 0)
    }
    checkCuda(cudaMemcpy(rp.d_pres_H, h, nH, cudaMemcpyHostToDevice), "pres_H cpy");
    for (int j = 0; j < p.ns - 1; ++j) {
        double sh = rp.delta_s * (j + 0.5);
        h[j] = maxToroidalFlux * torfluxDeriv(ip, sh);
    }
    checkCuda(cudaMemcpy(rp.d_phip_H, h, nH, cudaMemcpyHostToDevice), "phip_H cpy");
    for (int j = 0; j < p.ns - 1; ++j) {
        double sh = rp.delta_s * (j + 0.5);
        double tf = fmin(torflux(ip, sh), 1.0);
        h[j] = maxToroidalFlux * evalIotaProfile(ip, tf) * torfluxDeriv(ip, sh);
    }
    checkCuda(cudaMemcpy(rp.d_chip_H, h, nH, cudaMemcpyHostToDevice), "chip_H cpy");
    for (int j = 0; j < p.ns - 1; ++j) h[j] = 1.0;  // dVds placeholder (gamma=0)
    checkCuda(cudaMemcpy(rp.d_dVds_H, h, nH, cudaMemcpyHostToDevice), "dVds_H cpy");
    for (int j = 0; j < p.ns - 1; ++j) {
        double sh = rp.delta_s * (j + 0.5);
        h[j] = Itor * evalCurrProfile(ip, fmin(torflux(ip, sh), 1.0));
    }
    checkCuda(cudaMemcpy(rp.d_curr_H, h, nH, cudaMemcpyHostToDevice), "curr_H cpy");
    // sqrt(s) on half-grid for parity mixing
    for (int j = 0; j < p.ns - 1; ++j) {
        h[j] = sqrt(rp.delta_s * (j + 0.5));
    }
    checkCuda(cudaMemcpy(rp.d_sqrtS_H, h, nH, cudaMemcpyHostToDevice), "sqrtS_H cpy");

    delete[] h;

    // lamscale = sqrt(deltaS * Σ phipH²) (vmecpp constants_.lamscale)
    double rmsPhiP = 0.0;
    double* h_phip = new double[p.ns - 1];
    checkCuda(cudaMemcpy(h_phip, rp.d_phip_H, nH, cudaMemcpyDeviceToHost), "phip_H get");
    for (int j = 0; j < p.ns - 1; ++j) rmsPhiP += h_phip[j] * h_phip[j];
    delete[] h_phip;
    p.lamscale = sqrt(rmsPhiP * rp.delta_s);
    printf("  profiles: ns=%d phip=%.6e lamscale=%.6e maxToroidalFlux=%.6e\n",
           p.ns, maxToroidalFlux * torfluxDeriv(ip, 0.5), p.lamscale,
           maxToroidalFlux);

    return rp;
}

void profilesFree(RadialProfiles& rp) {
    cudaFree(rp.d_iota_F); cudaFree(rp.d_pres_F);
    cudaFree(rp.d_phip_F); cudaFree(rp.d_chi_F);
    cudaFree(rp.d_sqrtS_F);
    cudaFree(rp.d_iota_H); cudaFree(rp.d_pres_H);
    cudaFree(rp.d_phip_H); cudaFree(rp.d_mass_H);
    cudaFree(rp.d_dVds_H); cudaFree(rp.d_sqrtS_H);
    cudaFree(rp.d_curr_H); cudaFree(rp.d_chip_H);
}
