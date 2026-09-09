#include "kernels/newton_impl.cuh"

// Explicit instantiation for float (cumes_cuda_float).
template class cumes::CorrectionCoordinates<float>;
template class cumes::DeviceGmres<float>;
template class cumes::NewtonCorrection<float>;
