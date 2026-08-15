#include "precon_impl.cuh"

// Explicit instantiation for float (cumes_cuda_float).
template PreconWorkspace<float>  preconCreate<float>(const GridParams<float>&, cumes::DeviceArena*);
template void preconFree<float>(PreconWorkspace<float>&);
template void preconCompute<float>(const FourierPlan<float>&, const GridParams<float>&, const RadialProfiles<float>&, const MetricWorkspace<float>&, PreconWorkspace<float>&, cudaStream_t);
template void preconApply<float>(cumes::SpectralView<float, cumes::DecomposedResidualDomain>, const GridParams<float>&, const PreconWorkspace<float>&, const int*, const int*, cudaStream_t);
