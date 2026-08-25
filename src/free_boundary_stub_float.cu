// free_boundary_stub_float.cu — explicit float instantiation of the stub
// (builds without deps/vacuum-field).
#include "free_boundary_stub_impl.cuh"

template class cumes::FreeBoundaryOperator<float>;
