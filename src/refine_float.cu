#include "refine_impl.cuh"

// Explicit instantiation for float (cumes_cuda_float).
template cumes::SpectralStorage<float> interpolateState<float>(
    const DeviceParams<float>&, const cumes::SpectralStorage<float>&, const DeviceParams<float>&, cudaStream_t);
