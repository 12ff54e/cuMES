#include "precon_impl.cuh"

// Explicit instantiation for double (cumes_cuda_double).
// Phase 8 tridiagonal backends (linkable from test_tridiagonal).
template class cumes::PcrBackend<double>;
template class cumes::ThomasBackend<double>;
template class cumes::Preconditioner<double>;
