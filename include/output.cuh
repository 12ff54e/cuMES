// output.cuh — dump results from GPU to host and print / save.
#pragma once
#include "vmec_types.h"
#include "input.h"       // InputParams (full provenance bundle)
#include "solver.cuh"    // SolverResult<T>
// (no include cycle: nothing in the input/solver/geometry/forces chain
//  includes output.cuh; only main.cu and output.cu include this header)

// The on-disk state format stays double regardless of T (the Python
// comparison scripts parse doubles); outputSaveBinary converts T -> double.
template <typename T>
void outputSaveBinary(const SpectralState<T>& st, const GridParams<T>& p,
                      const char* filename);
template <typename T>
void outputPrint(const SpectralState<T>& st, const GridParams<T>& p, int niter,
                 bool converged, T fsqr, T fsqz, T fsql);

#ifdef CUMES_HAVE_NETCDF
// Write state + grid params + convergence + full InputParams provenance to
// a netCDF classic-3 file (NC_CLOBBER). Declared under the CMake define;
// the definition + explicit instantiation live in src/output_netcdf.cu.
template <typename T>
void outputSaveNetcdf(const SpectralState<T>& st, const GridParams<T>& p,
                      const InputParams& ip, const SolverResult<T>& result,
                      const char* path, const char* input_file);
#endif

#ifdef CUMES_HAVE_HDF5
// Same content as a serial HDF5 file (scalars as root-group attributes).
// Definition + explicit instantiation live in src/output_hdf5.cu.
template <typename T>
void outputSaveHdf5(const SpectralState<T>& st, const GridParams<T>& p,
                    const InputParams& ip, const SolverResult<T>& result,
                    const char* path, const char* input_file);
#endif

// Format dispatcher by path suffix (.nc/.h5/.hdf5/.bin). Unrecognized
// suffix or a disabled backend falls back to binary cumes_state.bin in the
// working directory with a stderr warning. Always compiled (output.cu).
template <typename T>
void outputSave(const SpectralState<T>& st, const GridParams<T>& p,
                const InputParams& ip, const SolverResult<T>& result,
                const char* path, const char* input_file);
