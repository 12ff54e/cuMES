#include "solver_impl.cuh"

// Explicit instantiation for double (cumes_cuda_double).
template SolverResult<double> solverRun<double>(cumes::SpectralStorage<double>&, const GridParams<double>&, const RadialProfiles<double>&, cumes::ToroidalFftOperator<double>&, cumes::GeometryOperator<double>&, cumes::DeviceArena*, cudaStream_t, cumes::SolverBench*, cumes::AxisymmetricOperator<double>*);
