#include "precon_impl.cuh"

// Explicit instantiation for double (cumes_cuda_double).
template PreconWorkspace<double> preconCreate<double>(const DeviceParams<double>&, cumes::DeviceArena*);
template void preconFree<double>(PreconWorkspace<double>&);
template void preconCompute<double>(const cumes::RealSpaceStorage<double>&, const int*, const int*, const DeviceParams<double>&, const cumes::RadialProfileViews<double>&, const MetricWorkspace<double>&, PreconWorkspace<double>&, cudaStream_t);
template void preconApply<double>(cumes::SpectralView<double, cumes::DecomposedResidualDomain>, const DeviceParams<double>&, const PreconWorkspace<double>&, const int*, const int*, cudaStream_t);

// Phase 8 tridiagonal backends (linkable from test_tridiagonal).
template class cumes::PcrBackend<double>;
template class cumes::ThomasBackend<double>;

// Preconditioner operator (owns the workspace; wraps preconCompute/Apply).
template class cumes::Preconditioner<double>;
