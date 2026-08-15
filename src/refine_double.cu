#include "refine_impl.cuh"

// Explicit instantiation for double (cumes_cuda_double).
template cumes::SpectralStorage<double> interpolateState<double>(
    const GridParams<double>&, const cumes::SpectralStorage<double>&, const GridParams<double>&, cudaStream_t);
