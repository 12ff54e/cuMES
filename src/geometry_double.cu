#include "kernels/geometry_impl.cuh"

// Explicit instantiation for double (cumes_cuda_double).
template class cumes::GeometryOperator<double>;
template class cumes::MagneticFieldOperator<double>;
