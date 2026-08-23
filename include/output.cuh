// output.cuh — copy results from GPU to host and print the console summary.
#ifndef CUMES_INCLUDE_OUTPUT_CUH_
#define CUMES_INCLUDE_OUTPUT_CUH_
#include "solver.cuh"  // SolverResult<T>
#include "vmec_types.h"
// (no include cycle: nothing in the input/solver/geometry/forces chain
//  includes output.cuh; only main.cu and output.cpp include this header)

// The console printout of the converged state (status, residuals, axis +
// boundary coefficients).
template <typename T>
void outputPrint(const cumes::SpectralStorage<T>& storage,
                 const DeviceParams<T>& p,
                 int niter,
                 bool converged,
                 T fsqr,
                 T fsqz,
                 T fsql);

// The on-disk writers are GONE from this header (completion plan steps
// 2.1/2.2): the CLI resolves a typed OutputSpec (cumes/io/output_spec.hpp)
// and every backend — binary, NetCDF, HDF5 — consumes the single host
// EquilibriumSnapshot through the Writer interface (cumes/io/writer.hpp; the
// NetCDF/HDF5 host adapters live in src/cumes/io/netcdf_writer.cpp /
// hdf5_writer.cpp and include their backend headers there, and only there).

#endif  // CUMES_INCLUDE_OUTPUT_CUH_
