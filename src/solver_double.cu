#include "solver_impl.cuh"

// Explicit instantiation for double (cumes_cuda_double).
template SolverResult<double> solverRun<double>(cumes::SpectralStorage<double>&, const DeviceParams<double>&, const cumes::Profiles<double>&, cumes::ToroidalFftOperator<double>&, cumes::RealSpaceStorage<double>&, cumes::GeometryOperator<double>&, cumes::DeviceArena*, cudaStream_t, cumes::SolverBench*, cumes::SpectralOperator<double>*);

// Stateless operators (migration steps 8/10): linkable from tests.
template class cumes::ResidualOperator<double>;
template class cumes::DescentOperator<double>;

// The EquilibriumOperator is a real class template whose members are defined
// in solver_impl.cuh: an EXPLICIT instantiation pins every member (ctor,
// enqueue, ...) regardless of the optimizer's out-of-line-emission heuristics
// (at -O2 -g nvcc inlines the same-TU uses and drops the out-of-line ctor,
// leaving separate-TU callers like the benchmarks/tests unresolved).
template class cumes::EquilibriumOperator<double>;
