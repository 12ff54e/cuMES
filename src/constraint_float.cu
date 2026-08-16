#include "constraint_impl.cuh"

// Explicit instantiation for float (cumes_cuda_float).
template ConstraintWorkspace<float>  constraintCreate<float>(const DeviceParams<float>&, cumes::DeviceArena*);
template void constraintFree<float>(ConstraintWorkspace<float>&);
template void constraintResetRzCon0<float>(const DeviceParams<float>&, ConstraintWorkspace<float>&, const float*, cudaStream_t);
template void constraintCompute<float>(const DeviceParams<float>&, const cumes::RealSpaceStorage<float>&, const FourierPlan<float>&, const float*, const float*, ConstraintWorkspace<float>&, const float*, bool, cudaStream_t);
template void constraintDealiasBandpass<float>(const DeviceParams<float>&, const FourierPlan<float>&, const float*, const float*, const float*, float*, cudaStream_t);

// ConstraintOperator (owns the workspace; wraps the two de-alias backends).
template class cumes::ConstraintOperator<float>;
