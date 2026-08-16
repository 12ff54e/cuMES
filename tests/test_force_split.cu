// test_force_split.cu — differential test for the §8.10 R/Z vs lambda force split.
//
// Runs the monolithic forcesKernel and the split rzForcesKernel+lambdaForcesKernel
// on the same frozen geometry and compares all sixteen force families, then
// times both. The split is a verbatim extraction of the monolith's per-family
// arithmetic, so the two should be bit-identical (max diff 0); if -use_fast_math
// contracts an expression differently across the two kernels the comparison
// degrades to a few ULPs (still far inside the tolerance). The timing median
// answers the §8.10 question: does the reduced register set (108 -> 82/54) win
// wall time, or is the kernel bandwidth/latency-bound?
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <vector>

#include "vmec_types.h"
#include "input_json.h"
#include "fourier.cuh"
#include "geometry.cuh"
#include "forces.cuh"
#include "profiles.cuh"
#include "cumes/state/spectral_storage.hpp"
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

static double median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

template <typename T>
static void runSplit(int ns, int mpol, int ntor, int ntheta, int nzeta, const char* label) {
    GridParams<T> p;
    p.ns = ns; p.mnmax = mpol * (ntor + 1); p.ntheta = ntheta; p.nzeta = nzeta;
    p.nfp = 1; p.nZnT = ntheta * nzeta; p.mpol = mpol; p.ntor = ntor;
    p.ncurr = 0; p.delt = T(0.9); p.ftol = T(1e-14); p.max_iter = 10;
    p.tcon0 = T(1.0); p.lamscale = T(0.0);

    // Frozen non-degenerate Solovev-like state (same pattern as test_force_reference).
    cumes::SpectralStorage<T> storage(ns, p.mnmax);
    SpectralState<T> st = storage.legacy_view();
    const size_t nS = (size_t)ns * p.mnmax, nb = nS * sizeof(T);
    auto* h_cc = new T[nS](); auto* h_ss = new T[nS](); auto* h_zsc = new T[nS]();
    auto* h_zcs = new T[nS](); auto* h_lsc = new T[nS](); auto* h_lcs = new T[nS]();
    for (int j = 0; j < ns; ++j) {
        T s = T(j) / T(ns - 1);
        for (int mode = 0; mode < p.mnmax; ++mode) {
            int m = mode / (ntor + 1);
            if (m == 0 && mode == 0) h_cc[j + mode * ns] = T(4.0);
            else if (m == 1) { h_cc[j + mode * ns] = T(0.3) * s; h_zsc[j + mode * ns] = T(-0.5) * s; h_zcs[j + mode * ns] = T(-0.5) * s; }
            else if (m == 2) h_cc[j + mode * ns] = T(0.2) * s * s;
            h_ss[j + mode * ns] = h_cc[j + mode * ns];
        }
    }
    checkCuda(cudaMemcpy(st.d_rmncc, h_cc, nb, cudaMemcpyHostToDevice), "cc");
    checkCuda(cudaMemcpy(st.d_rmnss, h_ss, nb, cudaMemcpyHostToDevice), "ss");
    checkCuda(cudaMemcpy(st.d_zmnsc, h_zsc, nb, cudaMemcpyHostToDevice), "zsc");
    checkCuda(cudaMemcpy(st.d_zmncs, h_zcs, nb, cudaMemcpyHostToDevice), "zcs");
    checkCuda(cudaMemcpy(st.d_lmnsc, h_lsc, nb, cudaMemcpyHostToDevice), "lsc");
    checkCuda(cudaMemcpy(st.d_lmncs, h_lcs, nb, cudaMemcpyHostToDevice), "lcs");
    delete[] h_cc; delete[] h_ss; delete[] h_zsc; delete[] h_zcs; delete[] h_lsc; delete[] h_lcs;

    InputParams ip = initInputParams("inputs/solovev.json");
    RadialProfiles<T> rp = profilesCreate(p, ip);
    FourierPlan<T> fp = fourierCreate(p);
    MetricWorkspace<T> mw = metricCreate(p);

    inverseDFT(fp, storage.physical_const(), p);
    computeGeometry(fp, p, rp, mw);

    // ---- time both force paths (monolith vs split) ----------------------
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    const int reps = 200, warmup = 20;
    std::vector<float> mono_us, split_us;
    mono_us.reserve(reps); split_us.reserve(reps);
    for (int r = 0; r < reps; ++r) {
        cudaEventRecord(e0);
        computeForces(fp, p, rp, mw);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0.0f; cudaEventElapsedTime(&ms, e0, e1);
        if (r >= warmup) mono_us.push_back(ms * 1000.0f);
    }
    for (int r = 0; r < reps; ++r) {
        cudaEventRecord(e0);
        computeForcesSplit(fp, p, rp, mw);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0.0f; cudaEventElapsedTime(&ms, e0, e1);
        if (r >= warmup) split_us.push_back(ms * 1000.0f);
    }

    // ---- capture the two outputs (they share the fp force buffers) ------
    const size_t nF = (size_t)ns * p.nZnT;
    auto g = [&](const T* d, size_t n) { std::vector<T> v(n); checkCuda(cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "g"); return v; };
    computeForces(fp, p, rp, mw);
    cudaDeviceSynchronize();
    std::vector<T> m_armn_e = g(fp.d_armn_e, nF), m_armn_o = g(fp.d_armn_o, nF);
    std::vector<T> m_azmn_e = g(fp.d_azmn_e, nF), m_azmn_o = g(fp.d_azmn_o, nF);
    std::vector<T> m_brmn_e = g(fp.d_brmn_e, nF), m_brmn_o = g(fp.d_brmn_o, nF);
    std::vector<T> m_bzmn_e = g(fp.d_bzmn_e, nF), m_bzmn_o = g(fp.d_bzmn_o, nF);
    std::vector<T> m_crmn_e = g(fp.d_crmn_e, nF), m_crmn_o = g(fp.d_crmn_o, nF);
    std::vector<T> m_czmn_e = g(fp.d_czmn_e, nF), m_czmn_o = g(fp.d_czmn_o, nF);
    std::vector<T> m_blmn_e = g(fp.d_blmn_e, nF), m_blmn_o = g(fp.d_blmn_o, nF);
    std::vector<T> m_clmn_e = g(fp.d_clmn_e, nF), m_clmn_o = g(fp.d_clmn_o, nF);

    computeForcesSplit(fp, p, rp, mw);
    cudaDeviceSynchronize();
    std::vector<T> s_armn_e = g(fp.d_armn_e, nF), s_armn_o = g(fp.d_armn_o, nF);
    std::vector<T> s_azmn_e = g(fp.d_azmn_e, nF), s_azmn_o = g(fp.d_azmn_o, nF);
    std::vector<T> s_brmn_e = g(fp.d_brmn_e, nF), s_brmn_o = g(fp.d_brmn_o, nF);
    std::vector<T> s_bzmn_e = g(fp.d_bzmn_e, nF), s_bzmn_o = g(fp.d_bzmn_o, nF);
    std::vector<T> s_crmn_e = g(fp.d_crmn_e, nF), s_crmn_o = g(fp.d_crmn_o, nF);
    std::vector<T> s_czmn_e = g(fp.d_czmn_e, nF), s_czmn_o = g(fp.d_czmn_o, nF);
    std::vector<T> s_blmn_e = g(fp.d_blmn_e, nF), s_blmn_o = g(fp.d_blmn_o, nF);
    std::vector<T> s_clmn_e = g(fp.d_clmn_e, nF), s_clmn_o = g(fp.d_clmn_o, nF);

    auto maxDiff = [&](const std::vector<T>& a, const std::vector<T>& b) {
        double m = 0.0;
        for (size_t i = 0; i < nF; ++i) m = std::max(m, std::fabs((double)a[i] - (double)b[i]));
        return m;
    };
    double md = 0.0;
    md = std::max(md, maxDiff(m_armn_e, s_armn_e)); md = std::max(md, maxDiff(m_armn_o, s_armn_o));
    md = std::max(md, maxDiff(m_azmn_e, s_azmn_e)); md = std::max(md, maxDiff(m_azmn_o, s_azmn_o));
    md = std::max(md, maxDiff(m_brmn_e, s_brmn_e)); md = std::max(md, maxDiff(m_brmn_o, s_brmn_o));
    md = std::max(md, maxDiff(m_bzmn_e, s_bzmn_e)); md = std::max(md, maxDiff(m_bzmn_o, s_bzmn_o));
    md = std::max(md, maxDiff(m_crmn_e, s_crmn_e)); md = std::max(md, maxDiff(m_crmn_o, s_crmn_o));
    md = std::max(md, maxDiff(m_czmn_e, s_czmn_e)); md = std::max(md, maxDiff(m_czmn_o, s_czmn_o));
    md = std::max(md, maxDiff(m_blmn_e, s_blmn_e)); md = std::max(md, maxDiff(m_blmn_o, s_blmn_o));
    md = std::max(md, maxDiff(m_clmn_e, s_clmn_e)); md = std::max(md, maxDiff(m_clmn_o, s_clmn_o));

    const double tol = (sizeof(T) == sizeof(double)) ? 1e-12 : 1e-5;
    const double mmed = median(mono_us), smed = median(split_us);
    printf("\n[%s]\n  monolith vs split: max |diff| = %.3e (tol %.1e, bitwise=%s)\n",
           label, md, tol, md == 0.0 ? "yes" : "no");
    printf("  force kernel median: monolith %.2f us, split %.2f us (split/mono %.3fx)\n",
           mmed, smed, smed / (mmed > 0.0 ? mmed : 1.0));
    char msg[160];
    snprintf(msg, sizeof msg, "%s: split == monolith (max |diff| %.3e < %.1e)", label, md, tol);
    CHECK(md < tol, msg);

    fourierFree(fp); metricFree(mw); profilesFree(rp);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
}

int main() {
    printf("=== Force split prototype: monolith vs R/Z+lambda (differential + timing) ===\n");
    runSplit<double>(5, 4, 0, 18, 1, "double axisymmetric ns=5");
    runSplit<double>(11, 6, 2, 18, 4, "double 3D ns=11");
    runSplit<float>(5, 4, 0, 18, 1, "float axisymmetric ns=5");
    if (failures == 0) {
        printf("\ntest_force_split: ALL PASS\n");
        return 0;
    }
    printf("\ntest_force_split: %d FAILURES\n", failures);
    return 1;
}
