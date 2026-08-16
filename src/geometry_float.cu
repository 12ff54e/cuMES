#include "geometry_impl.cuh"

// Explicit instantiation for float (cumes_cuda_float).
template MetricWorkspace<float>  metricCreate<float>(const DeviceParams<float>&, cumes::DeviceArena*);
template void metricFree<float>(MetricWorkspace<float>&);
template void computeGeometry<float>(const cumes::RealSpaceStorage<float>&, const DeviceParams<float>&, const RadialProfiles<float>&, MetricWorkspace<float>&, cudaStream_t, bool);
template void computeBaseGeometry<float>(const cumes::RealSpaceStorage<float>&, const DeviceParams<float>&, const RadialProfiles<float>&, MetricWorkspace<float>&, cudaStream_t);
template void computeMagneticField<float>(const cumes::RealSpaceStorage<float>&, const DeviceParams<float>&, const RadialProfiles<float>&, MetricWorkspace<float>&, cudaStream_t, bool);
template void computeForceNormPartials<float>(const DeviceParams<float>&, const MetricWorkspace<float>&, float*, float*, cudaStream_t);
template void computeJacobianStats<float>(const DeviceParams<float>&, const MetricWorkspace<float>&, float*, cudaStream_t);

// GeometryOperator (owns the workspace; wraps computeGeometry + stats).
template class cumes::GeometryOperator<float>;
