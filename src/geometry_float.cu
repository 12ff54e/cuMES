#include "kernels/geometry_impl.cuh"

// Explicit instantiation for float (cumes_cuda_float).
template class cumes::GeometryOperator<float>;
template class cumes::MagneticFieldOperator<float>;
