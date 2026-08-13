#include "forces_impl.cuh"

// Explicit instantiation for float (cumes_cuda_float).
template void computeForces<float>(const FourierPlan<float>&, const GridParams<float>&, const RadialProfiles<float>&, const MetricWorkspace<float>&);
