// test_geometry_iso.cu — isolated check of the ncurr=1 geometry chain:
// loads the w7x iter-1 state (from dump/cuMES/step_0_*.bin), runs
// computeGeometry only, and verifies the bsupu/bsubu write coverage.
// The coverage checks ASSERT (nonzero exit on failure) — a passing exit code
// must mean every angular point of the surface was written.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "input_json.h"
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
        if (fread(&n, sizeof(uint64_t), 1, fp) != 1 || n != (uint64_t)(p.ns * p.mnmax)) {
            fprintf(stderr, "size mismatch %s: file has %llu elements, expected %d\n",
                    fn, (unsigned long long)n, p.ns * p.mnmax);
            fclose(fp);
            exit(1);
        }
        if (fread(h, sizeof(double), p.ns * p.mnmax, fp) != (size_t)(p.ns * p.mnmax)) {
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
    // W7-X config (CWD = repo root, same as the dump/ paths below).
    InputParams ip = initInputParams("inputs/w7x.json");
    GridParams<double> p{};
    // The dump/cuMES/step_0_* files are left by the LAST grid stage of a
    // multigrid run (per-stage dumps overwrite), so use the final grid ns.
    p.ns = ip.ns_array[ip.n_grids - 1]; p.mpol = ip.mpol; p.ntor = ip.ntor;
    p.ntheta = ip.ntheta; p.nzeta = ip.nzeta; p.nfp = ip.nfp;
    p.nZnT = p.ntheta * p.nzeta;
    p.mnmax = p.mpol * (p.ntor + 1);
    p.ncurr = ip.ncurr;
    p.delt = ip.delt; p.ftol = ip.ftol; p.max_iter = ip.max_iter;
    p.lamscale = 0.0;

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
    FourierPlan<double> fp = fourierCreate(p);
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

    // check bsupu coverage on a mid-volume surface. All indices are computed
    // from the actual grid (the old hardcoded ks/1080 were W7-X-specific).
    int nZnT = p.nZnT;
    int jMid = (p.ns - 1) / 2;  // a surface in the middle of the volume
    int nks = 8;
    int ks[8];
    for (int i = 0; i < nks; ++i) ks[i] = (i * nZnT) / nks;  // spread over the plane
    double hb[8];
    for (int i = 0; i < nks; ++i)
        cudaMemcpy(&hb[i], mw.d_bsupu + jMid * nZnT + ks[i], sizeof(double), cudaMemcpyDeviceToHost);
    int nz = 0;
    double* h_all = new double[(p.ns - 1) * nZnT];
    cudaMemcpy(h_all, mw.d_bsupu, (p.ns - 1) * nZnT * sizeof(double), cudaMemcpyDeviceToHost);
    for (int k = 0; k < nZnT; ++k) if (h_all[jMid * nZnT + k] == 0.0) ++nz;
    printf("bsupu[jMid=%d] zeros: %d/%d\n", jMid, nz, nZnT);
    for (int i = 0; i < nks; ++i)
        printf("  k=%d: %.6f\n", ks[i], hb[i]);
    // also check the bsubu coverage
    cudaMemcpy(h_all, mw.d_bsubu, (p.ns - 1) * nZnT * sizeof(double), cudaMemcpyDeviceToHost);
    int nz2 = 0;
    for (int k = 0; k < nZnT; ++k) if (h_all[jMid * nZnT + k] == 0.0) ++nz2;
    printf("bsubu[jMid=%d] zeros: %d/%d\n", jMid, nz2, nZnT);
    delete[] h_all;

    fourierFree(fp); metricFree(mw); profilesFree(rp);

    // Assertions: a full-coverage kernel must leave no unwritten point on an
    // interior surface (zero is not a physical bsupu/bsubu value there).
    int bad = 0;
    if (nz != 0) { fprintf(stderr, "FAIL: bsupu coverage incomplete (%d zeros)\n", nz); bad = 1; }
    if (nz2 != 0) { fprintf(stderr, "FAIL: bsubu coverage incomplete (%d zeros)\n", nz2); bad = 1; }
    if (bad == 0) printf("test_geometry_iso: coverage OK\n");
    return bad;
}
