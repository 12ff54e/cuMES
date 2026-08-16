#include "constraint_impl.cuh"

// Explicit instantiation for float (cumes_cuda_float).
template class cumes::ConstraintOperator<float>;
template void constraintDealiasBandpass<float>(const DeviceParams<float>&, const FourierPlan<float>&, const float*, const float*, const float*, float*, cudaStream_t);
