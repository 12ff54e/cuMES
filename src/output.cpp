// output.cpp — host-snapshot console printout + output-format availability
// preflight.
//
// The on-disk state files are written by the host-only v1 writers
// (cumes/io/writer.hpp); this TU keeps the device-reading console printout
// and the backend-availability preflight that main runs before any CUDA work.
#include "output.cuh"

#include "cumes/io/output_spec.hpp"

#include <cstdio>
#include <string>

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

void output_print(const cumes::EquilibriumSnapshot& snapshot,
                  int ntor,
                  int niter,
                  bool converged,
                  double fsqr,
                  double fsqz,
                  double fsql) {
    const int ns = snapshot.ns;
    const int mnmax = snapshot.mnmax;
    const int j = ns - 1;
    const auto& rmnc = snapshot.families[cumes::EquilibriumSnapshot::RMNCC];
    const auto& zmns = snapshot.families[cumes::EquilibriumSnapshot::ZMNSC];
    const auto at = [ns](const auto& family, int surface, int mode) {
        return family[static_cast<std::size_t>(surface) +
                      static_cast<std::size_t>(mode) *
                          static_cast<std::size_t>(ns)];
    };

    printf("\n========================================\n");
    printf("  Solver Result\n");
    printf("========================================\n");
    printf("  Status:     %s\n", converged ? "CONVERGED" : "NOT CONVERGED");
    printf("  Iterations: %d\n", niter);
    printf("  FSQR:       %.3e\n", (double)fsqr);
    printf("  FSQZ:       %.3e\n", (double)fsqz);
    printf("  FSQL:       %.3e\n", (double)fsql);
    printf("\n  R_00 radial profile:\n  j  |  R_00\n  ---+--------\n");
    for (int jj = 0; jj < ns; ++jj) {
        printf("  %2d | %10.6f\n", jj, at(rmnc, jj, 0));
    }
    printf("\n  Axis (j=0) and Boundary (j=%d):\n", j);
    printf("  Mode | m  n |   rmncc(ax)  rmncc(bdy)  zmnsc(ax)  zmnsc(bdy)\n");
    printf("  ------+-------+-------------------------------------------\n");
    for (int mode = 0; mode < mnmax && mode < 12; ++mode) {
        int mm = mode / (ntor + 1);
        int nn = mode % (ntor + 1);
        printf("  %4d | %d %d | %10.6f %10.6f %10.6f %10.6f\n", mode, mm, nn,
               at(rmnc, 0, mode), at(rmnc, j, mode), at(zmns, 0, mode),
               at(zmns, j, mode));
    }
}

// The suffix-based dispatcher (outputSave) and its preflight
// (outputFormatAvailable) are GONE — completion plan step 2.1/2.2: the CLI
// resolves a typed OutputSpec up front (resolve_output_spec +
// cumes::output_format_available, defined here in the adapter library), and
// every backend consumes the single host EquilibriumSnapshot through the
// Writer interface. output_print remains as the console printout.
