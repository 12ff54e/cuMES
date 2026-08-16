#include "solver_impl.cuh"

// Explicit instantiation for float (cumes_cuda_float).
template SolverResult<float>  solverRun<float>(cumes::SpectralStorage<float>&, const GridParams<float>&, const RadialProfiles<float>&, cumes::ToroidalFftOperator<float>&, cumes::RealSpaceStorage<float>&, cumes::GeometryOperator<float>&, cumes::DeviceArena*, cudaStream_t, cumes::SolverBench*, cumes::SpectralOperator<float>*);
