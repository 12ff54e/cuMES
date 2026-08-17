#include "solver_impl.cuh"

// Explicit instantiation for float (cumes_cuda_float).
template SolverResult<float>  solverRun<float>(cumes::SpectralStorage<float>&, const DeviceParams<float>&, const cumes::Profiles<float>&, cumes::ToroidalFftOperator<float>&, cumes::RealSpaceStorage<float>&, cumes::GeometryOperator<float>&, cumes::DeviceArena*, cudaStream_t, cumes::SolverBench*, cumes::SpectralOperator<float>*);

// Stateless operators (migration steps 8/10): linkable from tests.
template class cumes::ResidualOperator<float>;
template class cumes::DescentOperator<float>;

// Explicit instantiation pins every EquilibriumOperator member regardless of
// the optimizer's out-of-line-emission heuristics (see solver_double.cu).
template class cumes::EquilibriumOperator<float>;
