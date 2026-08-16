#include "solver_impl.cuh"

// Explicit instantiation for double (cumes_cuda_double).
template SolverResult<double> solverRun<double>(cumes::SpectralStorage<double>&, const GridParams<double>&, const cumes::Profiles<double>&, cumes::ToroidalFftOperator<double>&, cumes::RealSpaceStorage<double>&, cumes::GeometryOperator<double>&, cumes::DeviceArena*, cudaStream_t, cumes::SolverBench*, cumes::SpectralOperator<double>*);

// Stateless operators (migration steps 8/10): linkable from tests.
template class cumes::ResidualOperator<double>;
template class cumes::DescentOperator<double>;
