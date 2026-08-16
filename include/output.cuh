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
bool outputSaveBinary(const SpectralState<T>& st, const DeviceParams<T>& p,
                      const char* filename);
template <typename T>
void outputPrint(const SpectralState<T>& st, const DeviceParams<T>& p, int niter,
                 bool converged, T fsqr, T fsqz, T fsql);

#ifdef CUMES_HAVE_NETCDF
// Write state + grid params + convergence + full validated-problem provenance
// to a netCDF classic-3 file (NC_CLOBBER). Declared under the CMake define;
// the definition + explicit instantiation live in src/output_netcdf.cpp.
template <typename T>
bool outputSaveNetcdf(const SpectralState<T>& st, const DeviceParams<T>& p,
                      const cumes::ValidatedProblem& vp, const SolverResult<T>& result,
                      const char* path, const char* input_file);
#endif

#ifdef CUMES_HAVE_HDF5
// Same content as a serial HDF5 file (scalars as root-group attributes).
// Definition + explicit instantiation live in src/output_hdf5.cpp.
template <typename T>
bool outputSaveHdf5(const SpectralState<T>& st, const DeviceParams<T>& p,
                    const cumes::ValidatedProblem& vp, const SolverResult<T>& result,
                    const char* path, const char* input_file);
#endif

// Format dispatcher by path suffix (.nc/.h5/.hdf5/.bin). An unrecognized
// suffix falls back to binary cumes_state.bin in the working directory with
// a stderr warning. A known suffix whose backend is not compiled in returns
// false (the caller should have preflighted via outputFormatAvailable before
// running the solve). Always compiled (output.cpp).
template <typename T>
bool outputSave(const SpectralState<T>& st, const DeviceParams<T>& p,
                const cumes::ValidatedProblem& vp, const SolverResult<T>& result,
                const char* path, const char* input_file);

// Preflight: is the format implied by `path`'s suffix produced by this
// build? No side effects, no exit() — main calls this BEFORE creating the
// CUDA context / running any grid stage, so a requested-but-unlinked backend
// fails fast (and cleanly) instead of after thousands of iterations.
bool outputFormatAvailable(const char* path);
