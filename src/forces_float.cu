#include "forces_impl.cuh"

// Explicit instantiation for float (cumes_cuda_float).
template void computeForces<float>(const cumes::RealSpaceStorage<float>&, const DeviceParams<float>&, const cumes::RadialProfileViews<float>&, const MetricWorkspace<float>&, cudaStream_t);
