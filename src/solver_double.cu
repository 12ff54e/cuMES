#include "kernels/solver_impl.cuh"

// Explicit instantiation for double (cumes_cuda_double).
template SolverResult<double> solver_run<double>(
    cumes::SpectralStorage<double>&,
    const DeviceParams<double>&,
    const cumes::Profiles<double>&,
    cumes::ToroidalFftOperator<double>&,
    cumes::RealSpaceStorage<double>&,
    cumes::GeometryOperator<double>&,
    std::optional<std::reference_wrapper<cumes::DeviceArena>>,
    cudaStream_t,
    std::optional<std::reference_wrapper<cumes::SolverBench>>,
    std::optional<std::reference_wrapper<cumes::SpectralOperator<double>>>,
    std::optional<std::reference_wrapper<cumes::FreeBoundaryOperator<double>>>,
    bool);

// Stateless operators (migration steps 8/10): linkable from tests.
template class cumes::ResidualOperator<double>;
template class cumes::DescentOperator<double>;

// The EquilibriumOperator is a real class template whose members are defined
// in kernels/solver_impl.cuh: an EXPLICIT instantiation pins every member
// (ctor, enqueue, ...) regardless of the optimizer's out-of-line-emission
// heuristics (at -O2 -g nvcc inlines the same-TU uses and drops the out-of-line
// ctor, leaving separate-TU callers like the benchmarks/tests unresolved).
template class cumes::EquilibriumOperator<double>;
