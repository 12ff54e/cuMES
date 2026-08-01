// output.cu — copy results from GPU and print.
#include "output.cuh"
#include "input.h"
#include <cstdio>

static void checkCuda(cudaError_t err, const char* tag) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error [%s]: %s\n", tag, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Save full spectral state as raw binary for Python analysis.
void outputSaveBinary(const SpectralState& st, const GridParams& p,
                       const char* filename) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", filename); return; }
    // Write header: ns, mnmax as ints
    int ns = p.ns, mnmax = p.mnmax;
    fwrite(&ns, sizeof(int), 1, fp);
    fwrite(&mnmax, sizeof(int), 1, fp);
    // Write each coefficient array (5 arrays, each ns*mnmax doubles)
    size_t nb = ns * mnmax * sizeof(double);
    auto* buf = new double[ns * mnmax];
    checkCuda(cudaMemcpy(buf, st.d_rmncc, nb, cudaMemcpyDeviceToHost), "cpy rmncc");
    fwrite(buf, sizeof(double), ns * mnmax, fp);
    checkCuda(cudaMemcpy(buf, st.d_zmnsc, nb, cudaMemcpyDeviceToHost), "cpy zmnsc");
    fwrite(buf, sizeof(double), ns * mnmax, fp);
    checkCuda(cudaMemcpy(buf, st.d_lmnsc, nb, cudaMemcpyDeviceToHost), "cpy lmnsc");
    fwrite(buf, sizeof(double), ns * mnmax, fp);
    checkCuda(cudaMemcpy(buf, st.d_rmnss, nb, cudaMemcpyDeviceToHost), "cpy rmnss");
    fwrite(buf, sizeof(double), ns * mnmax, fp);
    checkCuda(cudaMemcpy(buf, st.d_zmncs, nb, cudaMemcpyDeviceToHost), "cpy zmncs");
    fwrite(buf, sizeof(double), ns * mnmax, fp);
    delete[] buf;
    fclose(fp);
    printf("Saved binary state to %s\n", filename);
}

void outputPrint(const SpectralState& st, const GridParams& p, int niter,
                 bool converged, double fsqr, double fsqz, double fsql) {
    // Pull boundary-surface spectral coefficients back to host for inspection.
    // Column-major layout: index(surface=j, mode=m) = j + m * ns.
    // Use cudaMemcpy2D to read with stride ns between consecutive modes.
    int j = p.ns - 1;  // boundary surface
    auto* h_rmnc = new double[p.mnmax];
    auto* h_zmns = new double[p.mnmax];
    auto* h_lmnc = new double[p.mnmax];

    checkCuda(cudaMemcpy2D(h_rmnc, sizeof(double),
                           st.d_rmncc + j, p.ns * sizeof(double),
                           sizeof(double), p.mnmax,
                           cudaMemcpyDeviceToHost), "cpy rmnc out");
    checkCuda(cudaMemcpy2D(h_zmns, sizeof(double),
                           st.d_zmnsc + j, p.ns * sizeof(double),
                           sizeof(double), p.mnmax,
                           cudaMemcpyDeviceToHost), "cpy zmns out");
    checkCuda(cudaMemcpy2D(h_lmnc, sizeof(double),
                           st.d_lmnsc + j, p.ns * sizeof(double),
                           sizeof(double), p.mnmax,
                           cudaMemcpyDeviceToHost), "cpy lmnc out");

    // Also read axis (j=0) coefficients
    auto* h_rmnc_ax = new double[p.mnmax];
    auto* h_zmns_ax = new double[p.mnmax];
    auto* h_lmnc_ax = new double[p.mnmax];
    checkCuda(cudaMemcpy2D(h_rmnc_ax, sizeof(double),
                           st.d_rmncc, p.ns * sizeof(double),
                           sizeof(double), p.mnmax,
                           cudaMemcpyDeviceToHost), "cpy rmnc ax");
    checkCuda(cudaMemcpy2D(h_zmns_ax, sizeof(double),
                           st.d_zmnsc, p.ns * sizeof(double),
                           sizeof(double), p.mnmax,
                           cudaMemcpyDeviceToHost), "cpy zmns ax");
    checkCuda(cudaMemcpy2D(h_lmnc_ax, sizeof(double),
                           st.d_lmnsc, p.ns * sizeof(double),
                           sizeof(double), p.mnmax,
                           cudaMemcpyDeviceToHost), "cpy lmnc ax");

    // Read R_00 radial profile
    auto* h_rmnc_r = new double[p.ns];
    checkCuda(cudaMemcpy(h_rmnc_r, st.d_rmncc, p.ns * sizeof(double),
                         cudaMemcpyDeviceToHost), "cpy rmnc radial");

    printf("\n========================================\n");
    printf("  Solver Result\n");
    printf("========================================\n");
    printf("  Status:     %s\n", converged ? "CONVERGED" : "NOT CONVERGED");
    printf("  Iterations: %d\n", niter);
    printf("  FSQR:       %.3e\n", fsqr);
    printf("  FSQZ:       %.3e\n", fsqz);
    printf("  FSQL:       %.3e\n", fsql);
    printf("\n  R_00 radial profile:\n  j  |  R_00\n  ---+--------\n");
    for (int jj = 0; jj < p.ns; ++jj) {
        printf("  %2d | %10.6f\n", jj, h_rmnc_r[jj]);
    }
    printf("\n  Axis (j=0) and Boundary (j=%d):\n", j);
    printf("  Mode | m  n |   rmnc(ax)   rmnc(bdy)  zmns(ax)   zmns(bdy)\n");
    printf("  ------+-------+-------------------------------------------\n");
    for (int mode = 0; mode < p.mnmax && mode < 12; ++mode) {
        int mm = mode / kNtor;
        int nn = mode % kNtor;
        printf("  %4d | %d %d | %10.6f %10.6f %10.6f %10.6f\n",
               mode, mm, nn,
               h_rmnc_ax[mode], h_rmnc[mode],
               h_zmns_ax[mode], h_zmns[mode]);
    }
    delete[] h_rmnc_ax; delete[] h_zmns_ax; delete[] h_lmnc_ax;
    delete[] h_rmnc_r;

    delete[] h_rmnc; delete[] h_zmns; delete[] h_lmnc;
}
