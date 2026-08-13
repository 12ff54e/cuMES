#include "refine_impl.cuh"

// Explicit instantiation for double (cumes_cuda_double).
template void interpolateState<double>(SpectralState<double>&, const GridParams<double>&,
                                       const SpectralState<double>&, const GridParams<double>&);
