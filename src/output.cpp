// output.cpp — copy results from GPU and print (console printout + the
// output-format availability preflight).
//
// The on-disk state files are written by the host-only v1 writers
// (cumes/io/writer.hpp); this TU keeps the device-reading console printout
// and the backend-availability preflight that main runs before any CUDA work.
#include "output.cuh"

#include "cumes/io/output_spec.hpp"
#include "cumes/runtime/cuda_status.hpp"

#include <cuda_runtime.h>  // cudaMemcpy2D (host runtime API)

#include <cstdio>
#include <string>
#include <vector>

// Output-format availability preflight (completion plan step 2.5): this
// definition lives in the ADAPTER library — the only target with the
// CUMES_HAVE_NETCDF/CUMES_HAVE_HDF5 availability defines and the NetCDF/HDF5
// headers. The host I/O library (cumes_io_host) stays free of both.
bool cumes::output_format_available(cumes::OutputFormat fmt) {
    switch (fmt) {
        case cumes::OutputFormat::BINARY:
            return true;
        case cumes::OutputFormat::NETCDF:
#ifdef CUMES_HAVE_NETCDF
            return true;
#else
            return false;
#endif
        case cumes::OutputFormat::HDF5:
#ifdef CUMES_HAVE_HDF5
            return true;
#else
            return false;
#endif
    }
    return false;
}

template <typename T>
void output_print(const cumes::SpectralStorage<T>& storage,
                  const DeviceParams<T>& p,
                  int niter,
                  bool converged,
                  T fsqr,
                  T fsqz,
                  T fsql) {
    // Pull boundary-surface spectral coefficients back to host for inspection.
    // Column-major layout: index(surface=j, mode=m) = j + m * ns.
    // Use cudaMemcpy2D to read with stride ns between consecutive modes.
    int j = p.ns - 1;  // boundary surface
    std::vector<T> h_rmnc(p.mnmax);
    std::vector<T> h_zmns(p.mnmax);
    std::vector<T> h_lmnc(p.mnmax);

    cumes::check_cuda(
        cudaMemcpy2D(h_rmnc.data(), sizeof(T),
                     storage.family_ptr(cumes::SpectralComponent::Rcc) + j,
                     p.ns * sizeof(T), sizeof(T), p.mnmax,
                     cudaMemcpyDeviceToHost),
        "cpy rmnc out");
    cumes::check_cuda(
        cudaMemcpy2D(h_zmns.data(), sizeof(T),
                     storage.family_ptr(cumes::SpectralComponent::Zsc) + j,
                     p.ns * sizeof(T), sizeof(T), p.mnmax,
                     cudaMemcpyDeviceToHost),
        "cpy zmns out");
    cumes::check_cuda(
        cudaMemcpy2D(h_lmnc.data(), sizeof(T),
                     storage.family_ptr(cumes::SpectralComponent::Lsc) + j,
                     p.ns * sizeof(T), sizeof(T), p.mnmax,
                     cudaMemcpyDeviceToHost),
        "cpy lmnc out");

    // Also read axis (j=0) coefficients
    std::vector<T> h_rmnc_ax(p.mnmax);
    std::vector<T> h_zmns_ax(p.mnmax);
    std::vector<T> h_lmnc_ax(p.mnmax);
    cumes::check_cuda(
        cudaMemcpy2D(h_rmnc_ax.data(), sizeof(T),
                     storage.family_ptr(cumes::SpectralComponent::Rcc),
                     p.ns * sizeof(T), sizeof(T), p.mnmax,
                     cudaMemcpyDeviceToHost),
        "cpy rmnc ax");
    cumes::check_cuda(
        cudaMemcpy2D(h_zmns_ax.data(), sizeof(T),
                     storage.family_ptr(cumes::SpectralComponent::Zsc),
                     p.ns * sizeof(T), sizeof(T), p.mnmax,
                     cudaMemcpyDeviceToHost),
        "cpy zmns ax");
    cumes::check_cuda(
        cudaMemcpy2D(h_lmnc_ax.data(), sizeof(T),
                     storage.family_ptr(cumes::SpectralComponent::Lsc),
                     p.ns * sizeof(T), sizeof(T), p.mnmax,
                     cudaMemcpyDeviceToHost),
        "cpy lmnc ax");

    // Read R_00 radial profile
    std::vector<T> h_rmnc_r(p.ns);
    cumes::check_cuda(
        cudaMemcpy(h_rmnc_r.data(),
                   storage.family_ptr(cumes::SpectralComponent::Rcc),
                   p.ns * sizeof(T), cudaMemcpyDeviceToHost),
        "cpy rmnc radial");

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
        printf("  %4d | %d %d | %10.6f %10.6f %10.6f %10.6f\n", mode, mm, nn,
               (double)h_rmnc_ax[mode], (double)h_rmnc[mode],
               (double)h_zmns_ax[mode], (double)h_zmns[mode]);
    }
}

// The suffix-based dispatcher (outputSave) and its preflight
// (outputFormatAvailable) are GONE — completion plan step 2.1/2.2: the CLI
// resolves a typed OutputSpec up front (resolve_output_spec +
// cumes::output_format_available, defined here in the adapter library), and
// every backend consumes the single host EquilibriumSnapshot through the
// Writer interface. output_print remains as the console printout.

// ---- Explicit instantiation (double + float) ----------------------------
template void output_print<double>(const cumes::SpectralStorage<double>&,
                                   const DeviceParams<double>&,
                                   int,
                                   bool,
                                   double,
                                   double,
                                   double);
template void output_print<float>(const cumes::SpectralStorage<float>&,
                                  const DeviceParams<float>&,
                                  int,
                                  bool,
                                  float,
                                  float,
                                  float);
