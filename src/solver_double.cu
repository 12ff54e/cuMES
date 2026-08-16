#include "solver_impl.cuh"

// Explicit instantiation for double (cumes_cuda_double).
template SolverResult<double> solverRun<double>(cumes::SpectralStorage<double>&, const GridParams<double>&, const RadialProfiles<double>&, FourierPlan<double>&, MetricWorkspace<double>&, cumes::DeviceArena*, cudaStream_t, cumes::SolverBench*);
