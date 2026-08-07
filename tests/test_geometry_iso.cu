// test_geometry_iso.cu — isolated check of the ncurr=1 geometry chain:
// loads the w7x iter-1 state (from dump/cuMES/step_0_*.bin), runs
// computeGeometry only, and verifies the bsupu write coverage.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cublas_v2.h>

#include "input.h"
#include "vmec_types.h"
#include "fourier.cuh"
#include "geometry.cuh"
#include "profiles.cuh"

static void cc(cudaError_t e, const char* t) {
    if (e != cudaSuccess) { fprintf(stderr, "CUDA[%s]: %s\n", t, cudaGetErrorString(e)); exit(1); }
}

static void loadState(SpectralState<double>& st, const GridParams<double>& p, const char* base) {
    size_t nb = (size_t)p.ns * p.mnmax * sizeof(double);
    auto* h = new double[p.ns * p.mnmax];
    auto rd = [&](double* d, const char* fn) {
        FILE* fp = fopen(fn, "rb");
        if (!fp) { fprintf(stderr, "cannot open %s\n", fn); exit(1); }
        uint64_t n;
        if (fread(&n, sizeof(uint64_t), 1, fp) != 1 ||
            fread(h, sizeof(double), p.ns * p.mnmax, fp) != (size_t)(p.ns * p.mnmax)) {
            fprintf(stderr, "truncated %s\n", fn);
            fclose(fp);
            exit(1);
        }
        fclose(fp);
        cc(cudaMemcpy(d, h, nb, cudaMemcpyHostToDevice), "cpy");
    };
    char fn[256];
    snprintf(fn, sizeof fn, "%s_rmncc.bin", base); rd(st.d_rmncc, fn);
    snprintf(fn, sizeof fn, "%s_zmnsc.bin", base); rd(st.d_zmnsc, fn);
    snprintf(fn, sizeof fn, "%s_lmnsc.bin", base); rd(st.d_lmnsc, fn);
    snprintf(fn, sizeof fn, "%s_rmnss.bin", base); rd(st.d_rmnss, fn);
    snprintf(fn, sizeof fn, "%s_zmncs.bin", base); rd(st.d_zmncs, fn);
    snprintf(fn, sizeof fn, "%s_lmncs.bin", base); rd(st.d_lmncs, fn);
    delete[] h;
}

int main() {
    InputParams ip = initInputParams();
    GridParams<double> p{};
    p.ns = ip.ns; p.mpol = ip.mpol; p.ntor = ip.ntor;
    p.ntheta = ip.ntheta; p.nzeta = ip.nzeta; p.nfp = ip.nfp;
    p.nZnT = p.ntheta * p.nzeta;
    p.mnmax = p.mpol * (p.ntor + 1);
    p.ncurr = ip.ncurr;
    p.delt = ip.delt; p.ftol = ip.ftol; p.max_iter = ip.max_iter;
    p.lamscale = 0.0;

    cublasHandle_t cb; cublasCreate(&cb);
    SpectralState<double> st{};
    size_t nb = (size_t)p.ns * p.mnmax * sizeof(double);
    cc(cudaMalloc(&st.d_rmncc, nb), "cc"); cc(cudaMalloc(&st.d_rmnss, nb), "ss");
    cc(cudaMalloc(&st.d_zmnsc, nb), "zsc"); cc(cudaMalloc(&st.d_zmncs, nb), "zcs");
    cc(cudaMalloc(&st.d_lmnsc, nb), "lsc"); cc(cudaMalloc(&st.d_lmncs, nb), "lcs");
    cc(cudaMalloc(&st.d_v_rmncc, nb), "vcc"); cc(cudaMalloc(&st.d_v_rmnss, nb), "vss");
    cc(cudaMalloc(&st.d_v_zmnsc, nb), "vzsc"); cc(cudaMalloc(&st.d_v_zmncs, nb), "vzcs");
    cc(cudaMalloc(&st.d_v_lmnsc, nb), "vlsc"); cc(cudaMalloc(&st.d_v_lmncs, nb), "vlcs");

    loadState(st, p, "dump/cuMES/step_0");
    RadialProfiles<double> rp = profilesCreate(p, ip);
    FourierPlan<double> fp = fourierCreate(p, cb);
    MetricWorkspace<double> mw = metricCreate(p);

    // extrapolate m=1 to the axis (as the solver does each iteration)
    {
        // reuse the solver's kernel via a small inline copy is not possible;
        // do it on the host instead
        double* hcc = new double[p.ns * p.mnmax];
        cudaMemcpy(hcc, st.d_rmncc, nb, cudaMemcpyDeviceToHost);
        for (int n = 0; n < p.ntor + 1; ++n) {
            int mn = 1 * (p.ntor + 1) + n;
            hcc[0 + mn * p.ns] = hcc[1 + mn * p.ns];
        }
        cudaMemcpy(st.d_rmncc, hcc, nb, cudaMemcpyHostToDevice);
        delete[] hcc;
    }

    inverseDFT(fp, st, p);
    computeGeometry(fp, p, rp, mw);

    // check bsupu coverage
    int nZnT = p.nZnT;
    double hb[8];
    int ks[8] = {0, 31, 32, 255, 256, 287, 951, 1079};
    for (int i = 0; i < 8; ++i)
        cudaMemcpy(&hb[i], mw.d_bsupu + 50 * nZnT + ks[i], sizeof(double), cudaMemcpyDeviceToHost);
    int nz = 0;
    double* h_all = new double[(p.ns - 1) * nZnT];
    cudaMemcpy(h_all, mw.d_bsupu, (p.ns - 1) * nZnT * sizeof(double), cudaMemcpyDeviceToHost);
    for (int k = 0; k < nZnT; ++k) if (h_all[50 * nZnT + k] == 0.0) ++nz;
    printf("bsupu[50] zeros: %d/1080\n", nz);
    printf("k=0:% .6f k=31:% .6f k=32:% .6f k=255:% .6f k=256:% .6f k=287:% .6f k=951:% .6f k=1079:% .6f\n",
           hb[0], hb[1], hb[2], hb[3], hb[4], hb[5], hb[6], hb[7]);
    // also check the bsubu coverage
    cudaMemcpy(h_all, mw.d_bsubu, (p.ns - 1) * nZnT * sizeof(double), cudaMemcpyDeviceToHost);
    int nz2 = 0;
    for (int k = 0; k < nZnT; ++k) if (h_all[50 * nZnT + k] == 0.0) ++nz2;
    printf("bsubu[50] zeros: %d/1080\n", nz2);

    fourierFree(fp); metricFree(mw); profilesFree(rp);
    cublasDestroy(cb);
    return 0;
}
