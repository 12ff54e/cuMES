#include "constraint_impl.cuh"

// Explicit instantiation for double (cumes_cuda_double).
template ConstraintWorkspace<double> constraintCreate<double>(const DeviceParams<double>&, cumes::DeviceArena*);
template void constraintFree<double>(ConstraintWorkspace<double>&);
template void constraintResetRzCon0<double>(const DeviceParams<double>&, ConstraintWorkspace<double>&, const double*, cudaStream_t);
template void constraintCompute<double>(const DeviceParams<double>&, const cumes::RealSpaceStorage<double>&, const FourierPlan<double>&, const PreconWorkspace<double>&, ConstraintWorkspace<double>&, const double*, bool, cudaStream_t);
template void constraintDealiasBandpass<double>(const DeviceParams<double>&, const FourierPlan<double>&, const double*, const double*, const double*, double*, cudaStream_t);

// ConstraintOperator (owns the workspace; wraps the two de-alias backends).
template class cumes::ConstraintOperator<double>;
