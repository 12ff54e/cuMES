#include "solver_impl.cuh"

// Explicit instantiation for float (cumes_cuda_float).
template SolverResult<float>  solverRun<float>(SpectralState<float>&, const GridParams<float>&, const RadialProfiles<float>&, FourierPlan<float>&, MetricWorkspace<float>&);
