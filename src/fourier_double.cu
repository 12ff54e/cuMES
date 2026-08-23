#include "kernels/fourier_impl.cuh"

// Explicit instantiation for double (cumes_cuda_double).
template cumes::DeviceModeTable cumes::mode_table_create<double>(
    const DeviceParams<double>&,
    cumes::DeviceArena*);
template cumes::RealSpaceStorage<double> real_space_create<double>(
    const DeviceParams<double>&,
    cumes::DeviceArena*);
template void real_space_free<double>(cumes::RealSpaceStorage<double>&);

// ToroidalFftOperator (owns the cuFFT plans + transform scratch; a
// SpectralOperator backend).
template class cumes::ToroidalFftOperator<double>;
