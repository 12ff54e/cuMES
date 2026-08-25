#include "kernels/solver_impl.cuh"

// Explicit instantiation for float (cumes_cuda_float).
template SolverResult<float> solver_run<float>(
    cumes::SpectralStorage<float>&,
    const DeviceParams<float>&,
    const cumes::Profiles<float>&,
    cumes::ToroidalFftOperator<float>&,
    cumes::RealSpaceStorage<float>&,
    cumes::GeometryOperator<float>&,
    std::optional<std::reference_wrapper<cumes::DeviceArena>>,
    cudaStream_t,
    std::optional<std::reference_wrapper<cumes::SolverBench>>,
    std::optional<std::reference_wrapper<cumes::SpectralOperator<float>>>,
    std::optional<std::reference_wrapper<cumes::FreeBoundaryOperator<float>>>);

// Stateless operators (migration steps 8/10): linkable from tests.
template class cumes::ResidualOperator<float>;
template class cumes::DescentOperator<float>;

// Explicit instantiation pins every EquilibriumOperator member regardless of
// the optimizer's out-of-line-emission heuristics (see solver_double.cu).
template class cumes::EquilibriumOperator<float>;
