// output.cuh — print the CLI console summary from a host snapshot.
#ifndef CUMES_INCLUDE_OUTPUT_CUH_
#define CUMES_INCLUDE_OUTPUT_CUH_
#include "cumes/io/equilibrium_snapshot.hpp"

// The console printout of the final state (status, residuals, axis + boundary
// coefficients). The solver facade has already performed the single D2H state
// transfer; printing must not retain a device-storage escape hatch.
void output_print(const cumes::EquilibriumSnapshot& snapshot,
                  int ntor,
                  int niter,
                  bool converged,
                  double fsqr,
                  double fsqz,
                  double fsql);

// The on-disk writers are GONE from this header (completion plan steps
// 2.1/2.2): the CLI resolves a typed OutputSpec (cumes/io/output_spec.hpp)
// and every backend — binary, NetCDF, HDF5 — consumes the single host
// EquilibriumSnapshot through the Writer interface (cumes/io/writer.hpp; the
// NetCDF/HDF5 host adapters live in src/cumes/io/netcdf_writer.cpp /
// hdf5_writer.cpp and include their backend headers there, and only there).

#endif  // CUMES_INCLUDE_OUTPUT_CUH_
