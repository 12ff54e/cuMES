#include "precon_impl.cuh"

// Explicit instantiation for float (cumes_cuda_float).
template class cumes::PcrBackend<float>;
template class cumes::ThomasBackend<float>;
template class cumes::Preconditioner<float>;
