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
#include "cumes/io/output_spec.hpp"
#define CUMES_IO_DEVICE_STAGE 1  // opt in to FamilyStage (needs CUDA headers)
#include "cumes/io/writer_helpers.hpp"  // io_detail::tempPathFor/publishAtomic, FamilyStage
#include <cuda_runtime.h>  // cudaMemcpy/cudaMemcpy2D/cudaGetErrorString (host runtime API)
#include <cstdio>
#include <cstring>   // strrchr
#include <string>
#include <strings.h>  // strcasecmp

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

// Output-format availability preflight (completion plan step 2.5): this
// definition lives in the ADAPTER library — the only target with the
// CUMES_HAVE_NETCDF/CUMES_HAVE_HDF5 availability defines and the NetCDF/HDF5
// headers. The host I/O library (cumes_io_host) stays free of both.
bool cumes::output_format_available(cumes::OutputFormat fmt) {
    switch (fmt) {
        case cumes::OutputFormat::kBinary:
            return true;
        case cumes::OutputFormat::kNetCdf:
#ifdef CUMES_HAVE_NETCDF
            return true;
#else
            return false;
#endif
        case cumes::OutputFormat::kHdf5:
#ifdef CUMES_HAVE_HDF5
            return true;
#else
            return false;
#endif
    }
    return false;
}

// Save full spectral state as raw binary for Python analysis.
// 6 coefficient arrays: rmncc zmnsc lmnsc rmnss zmncs lmncs, each ns*mnmax
// doubles on disk (converted from T). Returns false (after removing the
// partial file) when the file cannot be opened, written, or closed — the
// caller reports the run's output status in the CLI exit code.
template <typename T>
bool outputSaveBinary(const cumes::SpectralStorage<T>& storage, const DeviceParams<T>& p,
                      const char* filename) {
    const std::string tmp = cumes::io_detail::tempPathFor(filename);
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
    // Write each coefficient array (6 arrays, each ns*mnmax doubles on disk).
    // The staging buffers are RAII (FamilyStage), and a CUDA copy failure is
    // reported instead of thrown, so the fail path (close temp, remove temp,
    // return false) runs even on a device fault.
    const auto n_opt = cumes::io_detail::familyCount(ns, mnmax);
    if (!n_opt) return fail("dimension product overflows size_t");
    const std::size_t n = *n_opt;
    cumes::io_detail::FamilyStage<T> stage(n);
    auto writeFam = [&](const T* d, const char* tag) {
        std::string reason;
        if (!stage.copy(d, tag, reason)) return fail(reason.c_str());
        if (fwrite(stage.data(), sizeof(double), n, fp) != n) {
            return fail("state write");
        }
        return true;
    };
    if (!writeFam(storage.family_ptr(cumes::SpectralComponent::Rcc), "cpy rmncc")) return false;
    if (!writeFam(storage.family_ptr(cumes::SpectralComponent::Zsc), "cpy zmnsc")) return false;
    if (!writeFam(storage.family_ptr(cumes::SpectralComponent::Lsc), "cpy lmnsc")) return false;
    if (!writeFam(storage.family_ptr(cumes::SpectralComponent::Rss), "cpy rmnss")) return false;
    if (!writeFam(storage.family_ptr(cumes::SpectralComponent::Zcs), "cpy zmncs")) return false;
    if (!writeFam(storage.family_ptr(cumes::SpectralComponent::Lcs), "cpy lmncs")) return false;
    // Flush + fsync + close + atomic rename. publishAtomic always closes fp
    // exactly once (a failing close is not re-closed) and removes the temp on
    // any failure, leaving `filename` untouched.
    const std::string err = cumes::io_detail::publishAtomic(fp, tmp, filename);
    if (!err.empty()) {
        fprintf(stderr, "outputSaveBinary: %s (%s)\n", err.c_str(), tmp.c_str());
        return false;
    }
    printf("Saved binary state to %s\n", filename);
    return true;
}

template <typename T>
void outputPrint(const cumes::SpectralStorage<T>& storage, const DeviceParams<T>& p, int niter,
                 bool converged, T fsqr, T fsqz, T fsql) {
    // Pull boundary-surface spectral coefficients back to host for inspection.
    // Column-major layout: index(surface=j, mode=m) = j + m * ns.
    // Use cudaMemcpy2D to read with stride ns between consecutive modes.
    int j = p.ns - 1;  // boundary surface
    auto* h_rmnc = new T[p.mnmax];
    auto* h_zmns = new T[p.mnmax];
    auto* h_lmnc = new T[p.mnmax];

    cumes::check_cuda(cudaMemcpy2D(h_rmnc, sizeof(T),
                           storage.family_ptr(cumes::SpectralComponent::Rcc) + j, p.ns * sizeof(T),
                           sizeof(T), p.mnmax,
                           cudaMemcpyDeviceToHost), "cpy rmnc out");
    cumes::check_cuda(cudaMemcpy2D(h_zmns, sizeof(T),
                           storage.family_ptr(cumes::SpectralComponent::Zsc) + j, p.ns * sizeof(T),
                           sizeof(T), p.mnmax,
                           cudaMemcpyDeviceToHost), "cpy zmns out");
    cumes::check_cuda(cudaMemcpy2D(h_lmnc, sizeof(T),
                           storage.family_ptr(cumes::SpectralComponent::Lsc) + j, p.ns * sizeof(T),
                           sizeof(T), p.mnmax,
                           cudaMemcpyDeviceToHost), "cpy lmnc out");

    // Also read axis (j=0) coefficients
    auto* h_rmnc_ax = new T[p.mnmax];
    auto* h_zmns_ax = new T[p.mnmax];
    auto* h_lmnc_ax = new T[p.mnmax];
    cumes::check_cuda(cudaMemcpy2D(h_rmnc_ax, sizeof(T),
                           storage.family_ptr(cumes::SpectralComponent::Rcc), p.ns * sizeof(T),
                           sizeof(T), p.mnmax,
                           cudaMemcpyDeviceToHost), "cpy rmnc ax");
    cumes::check_cuda(cudaMemcpy2D(h_zmns_ax, sizeof(T),
                           storage.family_ptr(cumes::SpectralComponent::Zsc), p.ns * sizeof(T),
                           sizeof(T), p.mnmax,
                           cudaMemcpyDeviceToHost), "cpy zmns ax");
    cumes::check_cuda(cudaMemcpy2D(h_lmnc_ax, sizeof(T),
                           storage.family_ptr(cumes::SpectralComponent::Lsc), p.ns * sizeof(T),
                           sizeof(T), p.mnmax,
                           cudaMemcpyDeviceToHost), "cpy lmnc ax");

    // Read R_00 radial profile
    auto* h_rmnc_r = new T[p.ns];
    cumes::check_cuda(cudaMemcpy(h_rmnc_r, storage.family_ptr(cumes::SpectralComponent::Rcc), p.ns * sizeof(T),
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

// The suffix-based dispatcher (outputSave) and its preflight
// (outputFormatAvailable) are GONE — completion plan step 2.1/2.2: the CLI
// resolves a typed OutputSpec up front (resolve_output_spec +
// cumes::output_format_available, defined here in the adapter library), and
// every backend consumes the single host EquilibriumSnapshot through the
// Writer interface. outputSaveBinary/outputPrint remain as the legacy
// device-reading reference (the byte-golden in test_io_golden.cu) and the
// console printout.

// ---- Explicit instantiation (double + float) ----------------------------
template bool outputSaveBinary<double>(const cumes::SpectralStorage<double>&, const DeviceParams<double>&, const char*);
template bool outputSaveBinary<float>(const cumes::SpectralStorage<float>&, const DeviceParams<float>&, const char*);
template void outputPrint<double>(const cumes::SpectralStorage<double>&, const DeviceParams<double>&, int, bool, double, double, double);
template void outputPrint<float>(const cumes::SpectralStorage<float>&, const DeviceParams<float>&, int, bool, float, float, float);
