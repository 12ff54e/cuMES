#include "solver_impl.cuh"

// Explicit instantiation for float (cumes_cuda_float).
template SolverResult<float>  solverRun<float>(cumes::SpectralStorage<float>&, const GridParams<float>&, const RadialProfiles<float>&, FourierPlan<float>&, MetricWorkspace<float>&, cumes::DeviceArena*, cudaStream_t, cumes::SolverBench*, cumes::AxisymmetricOperator<float>*);
