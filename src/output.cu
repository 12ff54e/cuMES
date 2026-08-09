// output.cu — copy results from GPU and print.
//
// The on-disk state format (cumes_state.bin) stays double regardless of the
// computation type T — the Python comparison scripts (scripts/compare_*.py)
// parse doubles with struct.unpack("<d"). outputSaveBinary converts T->double
// on save.
#include "output.cuh"
#include "input.h"
#include <cstdio>
#include <cstring>   // strrchr
#include <strings.h>  // strcasecmp

static void checkCuda(cudaError_t err, const char* tag) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error [%s]: %s\n", tag, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Save full spectral state as raw binary for Python analysis.
// 6 coefficient arrays: rmncc zmnsc lmnsc rmnss zmncs lmncs, each ns*mnmax
// doubles on disk (converted from T).
template <typename T>
void outputSaveBinary(const SpectralState<T>& st, const GridParams<T>& p,
                      const char* filename) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", filename); return; }
    // Write header: ns, mnmax as ints
    int ns = p.ns, mnmax = p.mnmax;
    fwrite(&ns, sizeof(int), 1, fp);
    fwrite(&mnmax, sizeof(int), 1, fp);
    // Write each coefficient array (6 arrays, each ns*mnmax doubles on disk)
    size_t nb = ns * mnmax * sizeof(T);
    auto* buf = new T[ns * mnmax];
    auto* dbuf = new double[ns * mnmax];
    auto writeFam = [&](const T* d, const char* tag) {
        checkCuda(cudaMemcpy(buf, d, nb, cudaMemcpyDeviceToHost), tag);
        for (size_t i = 0; i < (size_t)ns * mnmax; ++i) dbuf[i] = (double)buf[i];
        fwrite(dbuf, sizeof(double), ns * mnmax, fp);
    };
    writeFam(st.d_rmncc, "cpy rmncc");
    writeFam(st.d_zmnsc, "cpy zmnsc");
    writeFam(st.d_lmnsc, "cpy lmnsc");
    writeFam(st.d_rmnss, "cpy rmnss");
    writeFam(st.d_zmncs, "cpy zmncs");
    writeFam(st.d_lmncs, "cpy lmncs");
    delete[] dbuf;
    delete[] buf;
    fclose(fp);
    printf("Saved binary state to %s\n", filename);
}

template <typename T>
void outputPrint(const SpectralState<T>& st, const GridParams<T>& p, int niter,
                 bool converged, T fsqr, T fsqz, T fsql) {
    // Pull boundary-surface spectral coefficients back to host for inspection.
    // Column-major layout: index(surface=j, mode=m) = j + m * ns.
    // Use cudaMemcpy2D to read with stride ns between consecutive modes.
    int j = p.ns - 1;  // boundary surface
    auto* h_rmnc = new T[p.mnmax];
    auto* h_zmns = new T[p.mnmax];
    auto* h_lmnc = new T[p.mnmax];

    checkCuda(cudaMemcpy2D(h_rmnc, sizeof(T),
                           st.d_rmncc + j, p.ns * sizeof(T),
                           sizeof(T), p.mnmax,
                           cudaMemcpyDeviceToHost), "cpy rmnc out");
    checkCuda(cudaMemcpy2D(h_zmns, sizeof(T),
                           st.d_zmnsc + j, p.ns * sizeof(T),
                           sizeof(T), p.mnmax,
                           cudaMemcpyDeviceToHost), "cpy zmns out");
    checkCuda(cudaMemcpy2D(h_lmnc, sizeof(T),
                           st.d_lmnsc + j, p.ns * sizeof(T),
                           sizeof(T), p.mnmax,
                           cudaMemcpyDeviceToHost), "cpy lmnc out");

    // Also read axis (j=0) coefficients
    auto* h_rmnc_ax = new T[p.mnmax];
    auto* h_zmns_ax = new T[p.mnmax];
    auto* h_lmnc_ax = new T[p.mnmax];
    checkCuda(cudaMemcpy2D(h_rmnc_ax, sizeof(T),
                           st.d_rmncc, p.ns * sizeof(T),
                           sizeof(T), p.mnmax,
                           cudaMemcpyDeviceToHost), "cpy rmnc ax");
    checkCuda(cudaMemcpy2D(h_zmns_ax, sizeof(T),
                           st.d_zmnsc, p.ns * sizeof(T),
                           sizeof(T), p.mnmax,
                           cudaMemcpyDeviceToHost), "cpy zmns ax");
    checkCuda(cudaMemcpy2D(h_lmnc_ax, sizeof(T),
                           st.d_lmnsc, p.ns * sizeof(T),
                           sizeof(T), p.mnmax,
                           cudaMemcpyDeviceToHost), "cpy lmnc ax");

    // Read R_00 radial profile
    auto* h_rmnc_r = new T[p.ns];
    checkCuda(cudaMemcpy(h_rmnc_r, st.d_rmncc, p.ns * sizeof(T),
                         cudaMemcpyDeviceToHost), "cpy rmnc radial");

    printf("\n========================================\n");
    printf("  Solver Result\n");
    printf("========================================\n");
    printf("  Status:     %s\n", converged ? "CONVERGED" : "NOT CONVERGED");
    printf("  Iterations: %d\n", niter);
    printf("  FSQR:       %.3e\n", (double)fsqr);
    printf("  FSQZ:       %.3e\n", (double)fsqz);
    printf("  FSQL:       %.3e\n", (double)fsql);
    printf("\n  R_00 radial profile:\n  j  |  R_00\n  ---+--------\n");
    for (int jj = 0; jj < p.ns; ++jj) {
        printf("  %2d | %10.6f\n", jj, (double)h_rmnc_r[jj]);
    }
    printf("\n  Axis (j=0) and Boundary (j=%d):\n", j);
    printf("  Mode | m  n |   rmncc(ax)  rmncc(bdy)  zmnsc(ax)  zmnsc(bdy)\n");
    printf("  ------+-------+-------------------------------------------\n");
    for (int mode = 0; mode < p.mnmax && mode < 12; ++mode) {
        int mm = mode / (p.ntor + 1);
        int nn = mode % (p.ntor + 1);
        printf("  %4d | %d %d | %10.6f %10.6f %10.6f %10.6f\n",
               mode, mm, nn,
               (double)h_rmnc_ax[mode], (double)h_rmnc[mode],
               (double)h_zmns_ax[mode], (double)h_zmns[mode]);
    }
    delete[] h_rmnc_ax; delete[] h_zmns_ax; delete[] h_lmnc_ax;
    delete[] h_rmnc_r;

    delete[] h_rmnc; delete[] h_zmns; delete[] h_lmnc;
}

// Format dispatcher. The path suffix decides the format: .nc -> NetCDF,
// .h5/.hdf5 -> HDF5, .bin -> binary at the given path. An unrecognized
// suffix or a missing argv[2] falls back to binary cumes_state.bin in the
// working directory (the pre-argv[2] behaviour), with a stderr warning.
// A known suffix whose backend is not compiled in is a hard error: the
// requested format cannot be produced, so we hint at the binary option and
// exit instead of silently writing something else.
template <typename T>
void outputSave(const SpectralState<T>& st, const GridParams<T>& p,
                const InputParams& ip, const SolverResult<T>& result,
                const char* path, const char* input_file) {
    // ip/result/input_file are only read by the backend writers; silence
    // -Wunused-parameter when both backends are compiled out.
    (void)ip; (void)result; (void)input_file;
    const char* ext = strrchr(path, '.');
    if (ext == nullptr) { ext = ""; }
    if (strcasecmp(ext, ".bin") == 0) {
        outputSaveBinary<T>(st, p, path);
        return;
    }
    if (strcasecmp(ext, ".nc") == 0) {
#ifdef CUMES_HAVE_NETCDF
        outputSaveNetcdf<T>(st, p, ip, result, path, input_file);
        return;
#else
        fprintf(stderr,
                "ERROR: %s: .nc output requested but cuMES was built "
                "without NetCDF support\n"
                "       (possible option: write a binary format - use a "
                ".bin suffix or omit argv[2])\n", path);
        exit(EXIT_FAILURE);
#endif
    } else if (strcasecmp(ext, ".h5") == 0 || strcasecmp(ext, ".hdf5") == 0) {
#ifdef CUMES_HAVE_HDF5
        outputSaveHdf5<T>(st, p, ip, result, path, input_file);
        return;
#else
        fprintf(stderr,
                "ERROR: %s: %s output requested but cuMES was built "
                "without HDF5 support\n"
                "       (possible option: write a binary format - use a "
                ".bin suffix or omit argv[2])\n", path, ext);
        exit(EXIT_FAILURE);
#endif
    } else {
        fprintf(stderr, "WARNING: %s: unrecognized output suffix '%s' - "
                        "writing binary cumes_state.bin\n", path, ext);
    }
    outputSaveBinary<T>(st, p, "cumes_state.bin");
}

// ---- Explicit instantiation (double + float) ----------------------------
template void outputSaveBinary<double>(const SpectralState<double>&, const GridParams<double>&, const char*);
template void outputSaveBinary<float>(const SpectralState<float>&, const GridParams<float>&, const char*);
template void outputPrint<double>(const SpectralState<double>&, const GridParams<double>&, int, bool, double, double, double);
template void outputPrint<float>(const SpectralState<float>&, const GridParams<float>&, int, bool, float, float, float);
template void outputSave<double>(const SpectralState<double>&, const GridParams<double>&, const InputParams&, const SolverResult<double>&, const char*, const char*);
template void outputSave<float>(const SpectralState<float>&, const GridParams<float>&, const InputParams&, const SolverResult<float>&, const char*, const char*);
