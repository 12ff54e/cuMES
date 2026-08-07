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
//
// All computation is templated on the scalar type T (double or float); the
// InputParams coefficients stay double (host config) and are converted at
// the point of use.
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
template <typename T>
static T evalPowerSeries(const double* c, int n, T x, bool integrate) {
    T ret = T(0.0);
    for (int i = n - 1; i >= 0; --i) {
        if (integrate) {
            ret = x * ret + T(c[i]) / T(i + 1);
        } else {
            ret = x * ret + T(c[i]);
        }
    }
    if (integrate) ret *= x;
    return ret;
}

template <typename T>
static T torflux(const InputParams& ip, T x) {
    return x * evalPowerSeries<T>(ip.aphi, ip.aphi_n, x, false);
}

template <typename T>
static T torfluxDeriv(const InputParams& ip, T x) {
    T ret = T(0.0);
    for (int i = 0; i < ip.aphi_n; ++i) {
        ret += T(i + 1) * T(ip.aphi[i]) * pow(x, i);
    }
    return ret;
}

template <typename T>
static T evalIotaProfile(const InputParams& ip, T x) {
    return evalPowerSeries<T>(ip.ai, ip.ai_n, x, false);
}

template <typename T>
static T evalMassProfile(const InputParams& ip, T x) {
    T normX = fmin(fabs(x * T(ip.bloat)), T(1.0));
    return evalPowerSeries<T>(ip.am, ip.am_n, normX, false) *
           (GridParams<T>::kMu0 * T(ip.pres_scale));
}

template <typename T>
static T evalCurrProfile(const InputParams& ip, T x) {
    T normX = fmin(fabs(x * T(ip.bloat)), T(1.0));
    return evalPowerSeries<T>(ip.ac, ip.ac_n, normX, true);
}

template <typename T>
RadialProfiles<T> profilesCreate(GridParams<T>& p, const InputParams& ip) {
    RadialProfiles<T> rp{};
    rp.delta_s = T(1.0) / T(p.ns - 1);

    size_t nF = p.ns * sizeof(T);
    size_t nH = (p.ns - 1) * sizeof(T);

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
    T maxToroidalFlux = T(GridParams<T>::kSignJacobian * ip.phiedge) / T(2.0 * M_PI);
    T tf1 = torflux<T>(ip, T(1.0));
    if (tf1 != T(0.0)) maxToroidalFlux /= tf1;

    // ncurr=1: normalize the enclosed toroidal current profile
    // Itor = signJ * μ0*curtor / (2π * I(1)), I(s) = ∫₀ˢ ac
    T Itor = T(0.0);
    if (ip.ncurr == 1) {
        T edgeCurrent = evalCurrProfile<T>(ip, T(1.0));
        Itor = T(GridParams<T>::kSignJacobian) * GridParams<T>::kMu0 * T(ip.curtor) /
               (T(2.0 * M_PI) * edgeCurrent);
    }

    auto* h = new T[p.ns];
    // ---- Full grid ----
    for (int j = 0; j < p.ns; ++j) {
        T s = rp.delta_s * T(j);
        T tf = fmin(torflux<T>(ip, s), T(1.0));
        h[j] = evalIotaProfile<T>(ip, tf);
    }
    checkCuda(cudaMemcpy(rp.d_iota_F, h, nF, cudaMemcpyHostToDevice), "iota_F cpy");
    for (int j = 0; j < p.ns; ++j) {
        T s = rp.delta_s * T(j);
        T tf = fmin(torflux<T>(ip, fmin(s, T(ip.spres_ped))), T(1.0));
        h[j] = evalMassProfile<T>(ip, tf);  // pres = mass (gamma = 0)
    }
    checkCuda(cudaMemcpy(rp.d_pres_F, h, nF, cudaMemcpyHostToDevice), "pres_F cpy");
    for (int j = 0; j < p.ns; ++j) {
        T s = rp.delta_s * T(j);
        h[j] = maxToroidalFlux * torfluxDeriv<T>(ip, s);
    }
    checkCuda(cudaMemcpy(rp.d_phip_F, h, nF, cudaMemcpyHostToDevice), "phip_F cpy");
    for (int j = 0; j < p.ns; ++j) {
        T s = rp.delta_s * T(j);
        T tf = fmin(torflux<T>(ip, s), T(1.0));
        h[j] = maxToroidalFlux * evalIotaProfile<T>(ip, tf) * torfluxDeriv<T>(ip, s);
    }
    checkCuda(cudaMemcpy(rp.d_chi_F, h, nF, cudaMemcpyHostToDevice), "chi_F cpy");
    for (int j = 0; j < p.ns; ++j) h[j] = sqrt(rp.delta_s * T(j) + T(1e-12));
    checkCuda(cudaMemcpy(rp.d_sqrtS_F, h, nF, cudaMemcpyHostToDevice), "sqrtS_F cpy");

    // ---- Half grid ----
    for (int j = 0; j < p.ns - 1; ++j) {
        T sh = rp.delta_s * (T(j) + T(0.5));
        T tf = fmin(torflux<T>(ip, sh), T(1.0));
        h[j] = evalIotaProfile<T>(ip, tf);
    }
    checkCuda(cudaMemcpy(rp.d_iota_H, h, nH, cudaMemcpyHostToDevice), "iota_H cpy");
    for (int j = 0; j < p.ns - 1; ++j) {
        T sh = rp.delta_s * (T(j) + T(0.5));
        T tf = fmin(torflux<T>(ip, fmin(sh, T(ip.spres_ped))), T(1.0));
        h[j] = evalMassProfile<T>(ip, tf);  // pres = mass (gamma = 0)
    }
    checkCuda(cudaMemcpy(rp.d_pres_H, h, nH, cudaMemcpyHostToDevice), "pres_H cpy");
    for (int j = 0; j < p.ns - 1; ++j) {
        T sh = rp.delta_s * (T(j) + T(0.5));
        h[j] = maxToroidalFlux * torfluxDeriv<T>(ip, sh);
    }
    checkCuda(cudaMemcpy(rp.d_phip_H, h, nH, cudaMemcpyHostToDevice), "phip_H cpy");
    for (int j = 0; j < p.ns - 1; ++j) {
        T sh = rp.delta_s * (T(j) + T(0.5));
        T tf = fmin(torflux<T>(ip, sh), T(1.0));
        h[j] = maxToroidalFlux * evalIotaProfile<T>(ip, tf) * torfluxDeriv<T>(ip, sh);
    }
    checkCuda(cudaMemcpy(rp.d_chip_H, h, nH, cudaMemcpyHostToDevice), "chip_H cpy");
    for (int j = 0; j < p.ns - 1; ++j) h[j] = T(1.0);  // dVds placeholder (gamma=0)
    checkCuda(cudaMemcpy(rp.d_dVds_H, h, nH, cudaMemcpyHostToDevice), "dVds_H cpy");
    for (int j = 0; j < p.ns - 1; ++j) {
        T sh = rp.delta_s * (T(j) + T(0.5));
        h[j] = Itor * evalCurrProfile<T>(ip, fmin(torflux<T>(ip, sh), T(1.0)));
    }
    checkCuda(cudaMemcpy(rp.d_curr_H, h, nH, cudaMemcpyHostToDevice), "curr_H cpy");
    // sqrt(s) on half-grid for parity mixing
    for (int j = 0; j < p.ns - 1; ++j) {
        h[j] = sqrt(rp.delta_s * (T(j) + T(0.5)));
    }
    checkCuda(cudaMemcpy(rp.d_sqrtS_H, h, nH, cudaMemcpyHostToDevice), "sqrtS_H cpy");

    delete[] h;

    // lamscale = sqrt(deltaS * Σ phipH²) (vmecpp constants_.lamscale)
    T rmsPhiP = T(0.0);
    auto* h_phip = new T[p.ns - 1];
    checkCuda(cudaMemcpy(h_phip, rp.d_phip_H, nH, cudaMemcpyDeviceToHost), "phip_H get");
    for (int j = 0; j < p.ns - 1; ++j) rmsPhiP += h_phip[j] * h_phip[j];
    delete[] h_phip;
    p.lamscale = sqrt(rmsPhiP * rp.delta_s);
    printf("  profiles: ns=%d phip=%.6e lamscale=%.6e maxToroidalFlux=%.6e\n",
           p.ns, (double)(maxToroidalFlux * torfluxDeriv<T>(ip, T(0.5))),
           (double)p.lamscale, (double)maxToroidalFlux);

    return rp;
}

template <typename T>
void profilesFree(RadialProfiles<T>& rp) {
    cudaFree(rp.d_iota_F); cudaFree(rp.d_pres_F);
    cudaFree(rp.d_phip_F); cudaFree(rp.d_chi_F);
    cudaFree(rp.d_sqrtS_F);
    cudaFree(rp.d_iota_H); cudaFree(rp.d_pres_H);
    cudaFree(rp.d_phip_H); cudaFree(rp.d_mass_H);
    cudaFree(rp.d_dVds_H); cudaFree(rp.d_sqrtS_H);
    cudaFree(rp.d_curr_H); cudaFree(rp.d_chip_H);
}

// ---- Explicit instantiation (double + float) ----------------------------
template RadialProfiles<double> profilesCreate<double>(GridParams<double>&, const InputParams&);
template RadialProfiles<float>  profilesCreate<float>(GridParams<float>&, const InputParams&);
template void profilesFree<double>(RadialProfiles<double>&);
template void profilesFree<float>(RadialProfiles<float>&);
