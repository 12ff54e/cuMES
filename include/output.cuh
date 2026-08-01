// output.cuh — dump results from GPU to host and print / save.
#pragma once
#include "vmec_types.h"

void outputSaveBinary(const SpectralState& st, const GridParams& p,
                       const char* filename);
void outputPrint(const SpectralState& st, const GridParams& p, int niter,
                 bool converged, double fsqr, double fsqz, double fsql);
