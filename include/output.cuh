// output.cuh — dump results from GPU to host and print / save.
#pragma once
#include "vmec_types.h"
#include "solver.cuh"    // SolverResult<T>
// (no include cycle: nothing in the input/solver/geometry/forces chain
//  includes output.cuh; only main.cu and output.cpp include this header)

namespace cumes { class ValidatedProblem; }

// The on-disk state format stays double regardless of T (the Python
// comparison scripts parse doubles); outputSaveBinary converts T -> double.
// All writers return true when the file was written and closed successfully,
// false on any failure (after cleaning up partial files). main folds the
// output status into the CLI exit code, so a run can never report success
// without a durable result.
template <typename T>
bool outputSaveBinary(const cumes::SpectralStorage<T>& storage, const DeviceParams<T>& p,
                      const char* filename);
template <typename T>
void outputPrint(const cumes::SpectralStorage<T>& storage, const DeviceParams<T>& p, int niter,
                 bool converged, T fsqr, T fsqz, T fsql);

// The NetCDF/HDF5 writers and the suffix dispatcher are GONE from this legacy
// header (completion plan steps 2.1/2.2): the CLI resolves a typed OutputSpec
// (cumes/io/output_spec.hpp) and every backend — binary, NetCDF, HDF5 —
// consumes the single host EquilibriumSnapshot through the Writer interface
// (cumes/io/writer.hpp; the NetCDF/HDF5 host adapters live in
// src/cumes/io/netcdf_writer.cpp / hdf5_writer.cpp and include their backend
// headers there, and only there). outputSaveBinary/outputPrint remain as the
// legacy device-reading reference (byte-golden in test_io_golden.cu) and the
// console printout.
