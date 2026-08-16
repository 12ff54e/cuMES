#include "forces_impl.cuh"

// Explicit instantiation for double (cumes_cuda_double).
template void computeForces<double>(const cumes::RealSpaceStorage<double>&, const DeviceParams<double>&, const cumes::RadialProfileViews<double>&, const MetricWorkspace<double>&, cudaStream_t);
