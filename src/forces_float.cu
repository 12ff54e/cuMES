#include "forces_impl.cuh"

// Explicit instantiation for float (cumes_cuda_float).
template void computeForces<float>(const cumes::RealSpaceStorage<float>&, const GridParams<float>&, const RadialProfiles<float>&, const MetricWorkspace<float>&, cudaStream_t);
