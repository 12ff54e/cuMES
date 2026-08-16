#include "constraint_impl.cuh"

// Explicit instantiation for double (cumes_cuda_double).
template ConstraintWorkspace<double> constraintCreate<double>(const GridParams<double>&, cumes::DeviceArena*);
template void constraintFree<double>(ConstraintWorkspace<double>&);
template void constraintResetRzCon0<double>(const GridParams<double>&, ConstraintWorkspace<double>&, const double*, cudaStream_t);
template void constraintCompute<double>(const GridParams<double>&, const FourierPlan<double>&, const PreconWorkspace<double>&, ConstraintWorkspace<double>&, const double*, bool, cudaStream_t);
template void constraintComputeAxisym<double>(const GridParams<double>&, const FourierPlan<double>&, const PreconWorkspace<double>&, ConstraintWorkspace<double>&, const double*, bool, cumes::AxisymmetricOperator<double>&, cudaStream_t);
template void constraintDealiasBandpass<double>(const GridParams<double>&, const FourierPlan<double>&, ConstraintWorkspace<double>&, cudaStream_t);

// ConstraintOperator (owns the workspace; wraps the two de-alias backends).
template class cumes::ConstraintOperator<double>;
