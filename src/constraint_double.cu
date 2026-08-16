#include "constraint_impl.cuh"

// Explicit instantiation for double (cumes_cuda_double).
template class cumes::ConstraintOperator<double>;
template void constraintDealiasBandpass<double>(const DeviceParams<double>&, const FourierPlan<double>&, const double*, const double*, const double*, double*, cudaStream_t);
