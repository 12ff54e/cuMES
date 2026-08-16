#include "forces_impl.cuh"

// Explicit instantiation for double (cumes_cuda_double).
template void computeForces<double>(const cumes::RealSpaceStorage<double>&, const GridParams<double>&, const RadialProfiles<double>&, const MetricWorkspace<double>&, cudaStream_t);
