#include "refine_impl.cuh"

// Explicit instantiation for double (cumes_cuda_double).
template cumes::SpectralStorage<double> interpolateState<double>(
    const DeviceParams<double>&, const cumes::SpectralStorage<double>&, const DeviceParams<double>&, cudaStream_t);
