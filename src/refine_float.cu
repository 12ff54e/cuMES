#include "refine_impl.cuh"

// Explicit instantiation for float (cumes_cuda_float).
template void interpolateState<float>(SpectralState<float>&, const GridParams<float>&,
                                      const SpectralState<float>&, const GridParams<float>&);
