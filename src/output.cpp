// output.cpp — copy results from GPU and print.
//
// The on-disk state format (cumes_state.bin) stays double regardless of the
// computation type T — the Python comparison scripts (scripts/compare_*.py)
// parse doubles with struct.unpack("<d"). outputSaveBinary converts T->double
// on save.
//
// Atomic publication: the writer writes to a same-directory temp file, flushes
// and syncs it, closes it, then rename()s it over the target. A reader never
// sees a half-written state file, and a failure at any point leaves the target
// untouched (only the temp is removed). On POSIX rename() within one directory
// is atomic.
#include "output.cuh"
#include "input.h"
#include <cuda_runtime.h>  // cudaMemcpy/cudaMemcpy2D/cudaGetErrorString (host runtime API)
#include <cstdio>
#include <cstdlib>   // getpid
#include <cstring>   // strrchr
#include <string>
#include <strings.h>  // strcasecmp
#include <unistd.h>   // getpid, fsync, rename

// A same-directory temp path for `path` (so rename() stays on one filesystem).
static std::string tempPathFor(const char* path) {
    std::string p = path;
    p += ".tmp." + std::to_string((long)getpid());
    return p;
}

// Flush + fsync + close `fp`, then atomically rename `tmp` over `path`.
// Returns true on success; on failure prints and removes the temp, leaving
// `path` untouched. `fp` is always closed exactly once (a close that fails is
// not re-closed). Used by the atomic writers.
static bool publishAtomic(FILE* fp, const char* tmp, const char* path) {
    bool fail = false;
    if (fflush(fp) != 0) {
        fprintf(stderr, "output: fflush %s failed\n", tmp);
        fail = true;
    }
    if (!fail && fsync(fileno(fp)) != 0) {
        fprintf(stderr, "output: fsync %s failed\n", tmp);
        fail = true;
    }
    if (fclose(fp) != 0) {
        fprintf(stderr, "output: fclose %s failed\n", tmp);
        fail = true;
    }
    if (fail) { remove(tmp); return false; }
    if (rename(tmp, path) != 0) {
        fprintf(stderr, "output: rename %s -> %s failed\n", tmp, path);
        remove(tmp);
        return false;
    }
    return true;
}

#include "cumes/runtime/cuda_status.hpp"

// Compile-time summary of the output libraries linked into this build,
// for the disabled-backend error hints.
const char* linkedOutputLibraries() {
#if defined(CUMES_HAVE_NETCDF) && defined(CUMES_HAVE_HDF5)
    return "NetCDF, HDF5";
#elif defined(CUMES_HAVE_NETCDF)
    return "NetCDF";
#elif defined(CUMES_HAVE_HDF5)
    return "HDF5";
#else
    return "none (binary only)";
#endif
}
const char* linkedOutputSuffixes() {
#if defined(CUMES_HAVE_NETCDF) && defined(CUMES_HAVE_HDF5)
    return ".nc, .h5, .hdf5, .bin";
#elif defined(CUMES_HAVE_NETCDF)
    return ".nc, .bin";
#elif defined(CUMES_HAVE_HDF5)
    return ".h5, .hdf5, .bin";
#else
    return ".bin";
#endif
}

// Save full spectral state as raw binary for Python analysis.
// 6 coefficient arrays: rmncc zmnsc lmnsc rmnss zmncs lmncs, each ns*mnmax
// doubles on disk (converted from T). Returns false (after removing the
// partial file) when the file cannot be opened, written, or closed — the
// caller reports the run's output status in the CLI exit code.
template <typename T>
bool outputSaveBinary(const SpectralState<T>& st, const GridParams<T>& p,
                      const char* filename) {
    const std::string tmp = tempPathFor(filename);
    FILE* fp = fopen(tmp.c_str(), "wb");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", tmp.c_str()); return false; }
    // On any failure: close the temp, remove it, return false. The target
    // `filename` is never touched until the atomic publish.
    auto fail = [&](const char* tag) {
        fprintf(stderr, "outputSaveBinary: %s (%s)\n", tag, tmp.c_str());
        fclose(fp);
        remove(tmp.c_str());
        return false;
    };
    // Write header: ns, mnmax as ints
    int ns = p.ns, mnmax = p.mnmax;
    if (fwrite(&ns, sizeof(int), 1, fp) != 1) return fail("header ns");
    if (fwrite(&mnmax, sizeof(int), 1, fp) != 1) return fail("header mnmax");
    // Write each coefficient array (6 arrays, each ns*mnmax doubles on disk)
    size_t nb = ns * mnmax * sizeof(T);
    auto* buf = new T[ns * mnmax];
    auto* dbuf = new double[ns * mnmax];
    bool ok = true;
    auto writeFam = [&](const T* d, const char* tag) {
        cumes::check_cuda(cudaMemcpy(buf, d, nb, cudaMemcpyDeviceToHost), tag);
        for (size_t i = 0; i < (size_t)ns * mnmax; ++i) dbuf[i] = (double)buf[i];
        if (fwrite(dbuf, sizeof(double), ns * mnmax, fp) != (size_t)(ns * mnmax)) {
            ok = false;
        }
    };
    writeFam(st.d_rmncc, "cpy rmncc");
    writeFam(st.d_zmnsc, "cpy zmnsc");
    writeFam(st.d_lmnsc, "cpy lmnsc");
    writeFam(st.d_rmnss, "cpy rmnss");
    writeFam(st.d_zmncs, "cpy zmncs");
    writeFam(st.d_lmncs, "cpy lmncs");
    delete[] dbuf;
    delete[] buf;
    if (!ok) return fail("state write");
    // Flush + fsync + close + atomic rename. publishAtomic always closes fp
    // exactly once (a failing close is not re-closed) and removes the temp on
    // any failure, leaving `filename` untouched.
    if (!publishAtomic(fp, tmp.c_str(), filename)) return false;
    printf("Saved binary state to %s\n", filename);
    return true;
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

    cumes::check_cuda(cudaMemcpy2D(h_rmnc, sizeof(T),
                           st.d_rmncc + j, p.ns * sizeof(T),
                           sizeof(T), p.mnmax,
                           cudaMemcpyDeviceToHost), "cpy rmnc out");
    cumes::check_cuda(cudaMemcpy2D(h_zmns, sizeof(T),
                           st.d_zmnsc + j, p.ns * sizeof(T),
                           sizeof(T), p.mnmax,
                           cudaMemcpyDeviceToHost), "cpy zmns out");
    cumes::check_cuda(cudaMemcpy2D(h_lmnc, sizeof(T),
                           st.d_lmnsc + j, p.ns * sizeof(T),
                           sizeof(T), p.mnmax,
                           cudaMemcpyDeviceToHost), "cpy lmnc out");

    // Also read axis (j=0) coefficients
    auto* h_rmnc_ax = new T[p.mnmax];
    auto* h_zmns_ax = new T[p.mnmax];
    auto* h_lmnc_ax = new T[p.mnmax];
    cumes::check_cuda(cudaMemcpy2D(h_rmnc_ax, sizeof(T),
                           st.d_rmncc, p.ns * sizeof(T),
                           sizeof(T), p.mnmax,
                           cudaMemcpyDeviceToHost), "cpy rmnc ax");
    cumes::check_cuda(cudaMemcpy2D(h_zmns_ax, sizeof(T),
                           st.d_zmnsc, p.ns * sizeof(T),
                           sizeof(T), p.mnmax,
                           cudaMemcpyDeviceToHost), "cpy zmns ax");
    cumes::check_cuda(cudaMemcpy2D(h_lmnc_ax, sizeof(T),
                           st.d_lmnsc, p.ns * sizeof(T),
                           sizeof(T), p.mnmax,
                           cudaMemcpyDeviceToHost), "cpy lmnc ax");

    // Read R_00 radial profile
    auto* h_rmnc_r = new T[p.ns];
    cumes::check_cuda(cudaMemcpy(h_rmnc_r, st.d_rmncc, p.ns * sizeof(T),
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

// Preflight: does this build produce the format implied by `path`'s suffix?
// True for .bin (always) and for any compiled-in backend; false for a known
// suffix whose backend is not linked. Prints the same hint the old hard-exit
// path used. main calls this BEFORE creating the CUDA context / running any
// grid stage, so a requested-but-unlinked backend fails fast and cleanly
// instead of after thousands of iterations.
bool outputFormatAvailable(const char* path) {
    const char* ext = strrchr(path, '.');
    if (ext == nullptr) { ext = ""; }
    if (strcasecmp(ext, ".bin") == 0 || strcasecmp(ext, "") == 0) return true;
    if (strcasecmp(ext, ".nc") == 0) {
#ifdef CUMES_HAVE_NETCDF
        return true;
#else
        fprintf(stderr,
                "ERROR: %s: .nc output requested but cuMES was built "
                "without NetCDF support\n"
                "       (linked output libraries: %s; supported suffixes: "
                "%s; omit argv[2] for the binary fallback)\n",
                path, linkedOutputLibraries(), linkedOutputSuffixes());
        return false;
#endif
    }
    if (strcasecmp(ext, ".h5") == 0 || strcasecmp(ext, ".hdf5") == 0) {
#ifdef CUMES_HAVE_HDF5
        return true;
#else
        fprintf(stderr,
                "ERROR: %s: %s output requested but cuMES was built "
                "without HDF5 support\n"
                "       (linked output libraries: %s; supported suffixes: "
                "%s; omit argv[2] for the binary fallback)\n",
                path, ext, linkedOutputLibraries(), linkedOutputSuffixes());
        return false;
#endif
    }
    // Unknown suffix: the dispatcher warns and falls back to binary.
    return true;
}

// Format dispatcher. The path suffix decides the format: .nc -> NetCDF,
// .h5/.hdf5 -> HDF5, .bin -> binary at the given path. An unrecognized
// suffix or a missing argv[2] falls back to binary cumes_state.bin in the
// working directory (the pre-argv[2] behaviour), with a stderr warning.
// A known suffix whose backend is not compiled in returns false (the caller
// preflights via outputFormatAvailable before the solve, so this is only a
// belt-and-suspenders path) — never exit()s, so normal cleanup runs.
template <typename T>
bool outputSave(const SpectralState<T>& st, const GridParams<T>& p,
                const InputParams& ip, const SolverResult<T>& result,
                const char* path, const char* input_file) {
    // ip/result/input_file are only read by the backend writers; silence
    // -Wunused-parameter when both backends are compiled out.
    (void)ip; (void)result; (void)input_file;
    const char* ext = strrchr(path, '.');
    if (ext == nullptr) { ext = ""; }
    if (strcasecmp(ext, ".bin") == 0) {
        return outputSaveBinary<T>(st, p, path);
    }
    if (strcasecmp(ext, ".nc") == 0) {
#ifdef CUMES_HAVE_NETCDF
        return outputSaveNetcdf<T>(st, p, ip, result, path, input_file);
#else
        fprintf(stderr, "ERROR: %s: .nc output requested but cuMES was built "
                        "without NetCDF support\n", path);
        return false;
#endif
    }
    if (strcasecmp(ext, ".h5") == 0 || strcasecmp(ext, ".hdf5") == 0) {
#ifdef CUMES_HAVE_HDF5
        return outputSaveHdf5<T>(st, p, ip, result, path, input_file);
#else
        fprintf(stderr, "ERROR: %s: %s output requested but cuMES was built "
                        "without HDF5 support\n", path, ext);
        return false;
#endif
    }
    fprintf(stderr, "WARNING: %s: unrecognized output suffix '%s' - "
                    "writing binary cumes_state.bin\n", path, ext);
    return outputSaveBinary<T>(st, p, "cumes_state.bin");
}

// ---- Explicit instantiation (double + float) ----------------------------
template bool outputSaveBinary<double>(const SpectralState<double>&, const GridParams<double>&, const char*);
template bool outputSaveBinary<float>(const SpectralState<float>&, const GridParams<float>&, const char*);
template void outputPrint<double>(const SpectralState<double>&, const GridParams<double>&, int, bool, double, double, double);
template void outputPrint<float>(const SpectralState<float>&, const GridParams<float>&, int, bool, float, float, float);
template bool outputSave<double>(const SpectralState<double>&, const GridParams<double>&, const InputParams&, const SolverResult<double>&, const char*, const char*);
template bool outputSave<float>(const SpectralState<float>&, const GridParams<float>&, const InputParams&, const SolverResult<float>&, const char*, const char*);
