#include "geometry_impl.cuh"

// Explicit instantiation for double (cumes_cuda_double).
template MetricWorkspace<double> metricCreate<double>(const GridParams<double>&, cumes::DeviceArena*);
template void metricFree<double>(MetricWorkspace<double>&);
template void computeGeometry<double>(const FourierPlan<double>&, const GridParams<double>&, const RadialProfiles<double>&, MetricWorkspace<double>&, cudaStream_t);
template void computeForceNormPartials<double>(const GridParams<double>&, const MetricWorkspace<double>&, double*, double*, cudaStream_t);
template void computeJacobianStats<double>(const GridParams<double>&, const MetricWorkspace<double>&, double*, double*, cudaStream_t);
