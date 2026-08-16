#include "geometry_impl.cuh"

// Explicit instantiation for double (cumes_cuda_double).
template MetricWorkspace<double> metricCreate<double>(const DeviceParams<double>&, cumes::DeviceArena*);
template void metricFree<double>(MetricWorkspace<double>&);
template void computeGeometry<double>(const cumes::RealSpaceStorage<double>&, const DeviceParams<double>&, const cumes::RadialProfileViews<double>&, MetricWorkspace<double>&, cudaStream_t, bool);
template void computeBaseGeometry<double>(const cumes::RealSpaceStorage<double>&, const DeviceParams<double>&, const cumes::RadialProfileViews<double>&, MetricWorkspace<double>&, cudaStream_t);
template void computeMagneticField<double>(const cumes::RealSpaceStorage<double>&, const DeviceParams<double>&, const cumes::RadialProfileViews<double>&, MetricWorkspace<double>&, cudaStream_t, bool);
template void computeForceNormPartials<double>(const DeviceParams<double>&, const MetricWorkspace<double>&, double*, double*, cudaStream_t);
template void computeJacobianStats<double>(const DeviceParams<double>&, const MetricWorkspace<double>&, double*, cudaStream_t);

// GeometryOperator (owns the workspace; wraps computeGeometry + stats).
template class cumes::GeometryOperator<double>;
