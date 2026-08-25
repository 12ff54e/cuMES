// free_boundary_stub_double.cu — explicit double instantiation of the stub
// (builds without deps/vacuum-field).
#include "free_boundary_stub_impl.cuh"

template class cumes::FreeBoundaryOperator<double>;
