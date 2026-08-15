#include "geometry_impl.cuh"

// Explicit instantiation for float (cumes_cuda_float).
template MetricWorkspace<float>  metricCreate<float>(const GridParams<float>&, cumes::DeviceArena*);
template void metricFree<float>(MetricWorkspace<float>&);
template void computeGeometry<float>(const FourierPlan<float>&, const GridParams<float>&, const RadialProfiles<float>&, MetricWorkspace<float>&, cudaStream_t);
template void computeForceNormPartials<float>(const GridParams<float>&, const MetricWorkspace<float>&, float*, float*, cudaStream_t);
template void computeJacobianStats<float>(const GridParams<float>&, const MetricWorkspace<float>&, float*, float*, cudaStream_t);
