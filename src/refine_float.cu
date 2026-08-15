#include "refine_impl.cuh"

// Explicit instantiation for float (cumes_cuda_float).
template cumes::SpectralStorage<float> interpolateState<float>(
    const GridParams<float>&, const cumes::SpectralStorage<float>&, const GridParams<float>&, cudaStream_t);
