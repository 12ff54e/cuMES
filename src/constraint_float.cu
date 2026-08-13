#include "constraint_impl.cuh"

// Explicit instantiation for float (cumes_cuda_float).
template ConstraintWorkspace<float>  constraintCreate<float>(const GridParams<float>&);
template void constraintFree<float>(ConstraintWorkspace<float>&);
template void constraintRzConCompute<float>(const GridParams<float>&, const FourierPlan<float>&, const SpectralState<float>&, ConstraintWorkspace<float>&, const float*);
template void constraintResetRzCon0<float>(const GridParams<float>&, ConstraintWorkspace<float>&, const float*);
template void constraintCompute<float>(const GridParams<float>&, const FourierPlan<float>&, const PreconWorkspace<float>&, ConstraintWorkspace<float>&, const float*, bool);
