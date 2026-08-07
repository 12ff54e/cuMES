// output.cuh — dump results from GPU to host and print / save.
#pragma once
#include "vmec_types.h"

// The on-disk state format stays double regardless of T (the Python
// comparison scripts parse doubles); outputSaveBinary converts T -> double.
template <typename T>
void outputSaveBinary(const SpectralState<T>& st, const GridParams<T>& p,
                      const char* filename);
template <typename T>
void outputPrint(const SpectralState<T>& st, const GridParams<T>& p, int niter,
                 bool converged, T fsqr, T fsqz, T fsql);
