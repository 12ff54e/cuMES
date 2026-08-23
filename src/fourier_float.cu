#include "kernels/fourier_impl.cuh"

// Explicit instantiation for float (cumes_cuda_float).
template cumes::DeviceModeTable cumes::mode_table_create<float>(
    const DeviceParams<float>&,
    cumes::DeviceArena*);
template cumes::RealSpaceStorage<float> real_space_create<float>(
    const DeviceParams<float>&,
    cumes::DeviceArena*);
template void real_space_free<float>(cumes::RealSpaceStorage<float>&);

// ToroidalFftOperator (owns the cuFFT plans + transform scratch; a
// SpectralOperator backend).
template class cumes::ToroidalFftOperator<float>;
