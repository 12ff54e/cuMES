#include "precon_impl.cuh"

// Explicit instantiation for float (cumes_cuda_float).
template PreconWorkspace<float>  preconCreate<float>(const GridParams<float>&, cumes::DeviceArena*);
template void preconFree<float>(PreconWorkspace<float>&);
template void preconCompute<float>(const FourierPlan<float>&, const GridParams<float>&, const RadialProfiles<float>&, const MetricWorkspace<float>&, PreconWorkspace<float>&, cudaStream_t);
template void preconApply<float>(cumes::SpectralView<float, cumes::DecomposedResidualDomain>, const GridParams<float>&, const PreconWorkspace<float>&, const int*, const int*, cudaStream_t);

// Phase 8 tridiagonal backends (linkable from test_tridiagonal).
template class cumes::PcrBackend<float>;
template class cumes::ThomasBackend<float>;

// Preconditioner operator (owns the workspace; wraps preconCompute/Apply).
template class cumes::Preconditioner<float>;
