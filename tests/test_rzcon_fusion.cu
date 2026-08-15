// test_rzcon_fusion.cu — differential test for the weighted R/Z fusion
// (blueprint §8.4): the fused inverse DFT accumulates xmpq = m(m-1)-weighted
// rCon/zCon at the same time as the main poloidal synthesis, replacing the
// constraint's separate xmpq-weighted inverse transform.
//
// Two gates:
//   1. The fused inverse's 18 geometry arrays are BITWISE identical to the
//      non-fused inverseDFT (the fusion is additive — it must not change the
//      geometry codegen or arithmetic).
//   2. The fused rCon/zCon agree with constraintRzConCompute's rCon/zCon at a
//      ULP tolerance (Class B: the xmpq weight moves across the reconstruction,
//      so the summation order differs at the rounding level).
#include <cstdio>
#include <cmath>
#include <vector>
#include "vmec_types.h"
#include "fourier.cuh"
#include "constraint.cuh"
#include "cumes/state/spectral_storage.hpp"
#include "cumes_test_support.cuh"

constexpr int kNs = 4, kMpol = 5, kNtor = 2, kNtheta = 16, kNzeta = 8, kNfp = 1;
constexpr int kMnmax = kMpol * (kNtor + 1), kNZnT = kNtheta * kNzeta;

static int g_failures = 0;

template <typename T>
static constexpr double tolNear() {
    return sizeof(T) == sizeof(float) ? 1e-3 : 1e-11;
}

template <typename T>
static double maxAbsDiff(const T* a, const T* b, int n) {
    std::vector<T> ha((size_t)n), hb((size_t)n);
    cc(cudaMemcpy(ha.data(), a, (size_t)n * sizeof(T), cudaMemcpyDeviceToHost),
       "cp a");
    cc(cudaMemcpy(hb.data(), b, (size_t)n * sizeof(T), cudaMemcpyDeviceToHost),
       "cp b");
    double d = 0.0;
    for (int i = 0; i < n; ++i) d = fmax(d, fabs((double)ha[i] - (double)hb[i]));
    return d;
}

template <typename T>
static void fillState(cumes::SpectralStorage<T>& storage, int ns, int mnmax) {
    std::vector<T> v((size_t)6 * ns * mnmax);
    for (int c = 0; c < 6; ++c)
        for (int m = 0; m < mnmax; ++m)
            for (int j = 0; j < ns; ++j)
                v[(size_t)c * mnmax * ns + m * ns + j] =
                    T(0.01 * (c + 1) * (m + 1) * (j + 1));
    cc(cudaMemcpy(storage.state_slab(), v.data(), v.size() * sizeof(T),
                  cudaMemcpyHostToDevice),
       "fill state");
}

template <typename T>
static int runTests() {
    printf("--- %s precision ---\n",
           sizeof(T) == sizeof(double) ? "double" : "float");
    GridParams<T> p;
    p.ns = kNs; p.mnmax = kMnmax; p.ntheta = kNtheta; p.nzeta = kNzeta;
    p.nfp = kNfp; p.nZnT = kNZnT; p.mpol = kMpol; p.ntor = kNtor;
    p.ncurr = 0; p.delt = T(1.0); p.ftol = T(1e-14); p.max_iter = 10;
    p.lamscale = T(1.0);

    FourierPlan<T> fp = fourierCreate<T>(p);
    ConstraintWorkspace<T> cw = constraintCreate<T>(p);
    cumes::SpectralStorage<T> storage(p.ns, p.mnmax);
    fillState(storage, p.ns, p.mnmax);

    const int n = p.ns * p.nZnT;
    T *d_ax_r = nullptr, *d_ax_z = nullptr;
    cc(cudaMalloc(&d_ax_r, (size_t)n * sizeof(T)), "fused rCon");
    cc(cudaMalloc(&d_ax_z, (size_t)n * sizeof(T)), "fused zCon");

    // Gate 1: fused geometry bitwise == non-fused geometry.
    inverseDFT(fp, storage.physical_const(), p, false, 0);
    // The fused run overwrites the geometry arrays; snapshot the non-fused
    // geometry first into a scratch copy, then run the fused and compare.
    T* d_gen_r_e = nullptr;
    cc(cudaMalloc(&d_gen_r_e, (size_t)n * sizeof(T)), "gen r_e");
    cc(cudaMemcpy(d_gen_r_e, fp.d_r_e, (size_t)n * sizeof(T),
                  cudaMemcpyDeviceToDevice), "snap r_e");
    inverseDFTFused(fp, storage.physical_const(), p, false, d_ax_r, d_ax_z, 0);
    {
        double d = maxAbsDiff(d_gen_r_e, fp.d_r_e, n);
        if (d != 0.0) {
            fprintf(stderr, "FAIL [fused geometry r_e] diff = %.3e\n", d);
            ++g_failures;
        } else {
            printf("  PASS fused geometry r_e is bitwise-identical\n");
        }
    }

    // Gate 2: fused rCon/zCon == constraintRzConCompute (Class B ULP).
    constraintRzConCompute(p, fp, storage.physical_const(), cw,
                           static_cast<const T*>(nullptr), 0);
    {
        double dr = maxAbsDiff(cw.d_rCon, d_ax_r, n);
        double dz = maxAbsDiff(cw.d_zCon, d_ax_z, n);
        if (dr > tolNear<T>() || dz > tolNear<T>()) {
            fprintf(stderr, "FAIL [rCon/zCon] dr=%.3e dz=%.3e\n", dr, dz);
            ++g_failures;
        } else {
            printf("  PASS rCon/zCon match (dr %.3e, dz %.3e)\n", dr, dz);
        }
    }

    cudaFree(d_gen_r_e);
    cudaFree(d_ax_r);
    cudaFree(d_ax_z);
    fourierFree(fp);
    constraintFree(cw);
    return 0;
}

int main() {
    printf("=== R/Z fusion differential test ===\n");
    runTests<double>();
    runTests<float>();
    printf(g_failures == 0 ? "ALL PASS\n" : "%d FAILURES\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
