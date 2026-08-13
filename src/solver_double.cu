#include "solver_impl.cuh"

// Explicit instantiation for double (cumes_cuda_double).
template SolverResult<double> solverRun<double>(SpectralState<double>&, const GridParams<double>&, const RadialProfiles<double>&, FourierPlan<double>&, MetricWorkspace<double>&);
