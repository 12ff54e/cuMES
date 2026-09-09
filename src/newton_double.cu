#include "kernels/newton_impl.cuh"

// Explicit instantiation for double (cumes_cuda_double).
template class cumes::CorrectionCoordinates<double>;
template class cumes::DeviceGmres<double>;
template class cumes::NewtonCorrection<double>;
