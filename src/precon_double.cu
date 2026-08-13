#include "precon_impl.cuh"

// Explicit instantiation for double (cumes_cuda_double).
template PreconWorkspace<double> preconCreate<double>(const GridParams<double>&);
template void preconFree<double>(PreconWorkspace<double>&);
template void preconCompute<double>(const FourierPlan<double>&, const GridParams<double>&, const RadialProfiles<double>&, const MetricWorkspace<double>&, PreconWorkspace<double>&);
template void preconApply<double>(double*, const GridParams<double>&, const PreconWorkspace<double>&, const int*, const int*);
