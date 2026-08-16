#include "constraint_impl.cuh"

// Explicit instantiation for float (cumes_cuda_float).
template ConstraintWorkspace<float>  constraintCreate<float>(const GridParams<float>&, cumes::DeviceArena*);
template void constraintFree<float>(ConstraintWorkspace<float>&);
template void constraintResetRzCon0<float>(const GridParams<float>&, ConstraintWorkspace<float>&, const float*, cudaStream_t);
template void constraintCompute<float>(const GridParams<float>&, const FourierPlan<float>&, const PreconWorkspace<float>&, ConstraintWorkspace<float>&, const float*, bool, cudaStream_t);
template void constraintDealiasBandpass<float>(const GridParams<float>&, const FourierPlan<float>&, ConstraintWorkspace<float>&, cudaStream_t);
