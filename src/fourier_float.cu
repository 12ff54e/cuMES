#include "kernels/fourier_impl.cuh"

// Explicit instantiation for float (cumes_cuda_float).
template cumes::DeviceModeTable cumes::modeTableCreate<float>(
    const DeviceParams<float>&,
    cumes::DeviceArena*);
template cumes::RealSpaceStorage<float> realSpaceCreate<float>(
    const DeviceParams<float>&,
    cumes::DeviceArena*);
template void realSpaceFree<float>(cumes::RealSpaceStorage<float>&);

// ToroidalFftOperator (owns the cuFFT plans + transform scratch; a
// SpectralOperator backend).
template class cumes::ToroidalFftOperator<float>;
